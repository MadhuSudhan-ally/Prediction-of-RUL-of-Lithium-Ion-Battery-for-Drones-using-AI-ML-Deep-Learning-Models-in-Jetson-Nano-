

import os, sys, time, json, argparse, math
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.models import load_model

# CONFIG (edit paths as needed)
BASE_DIR = "path"
os.makedirs(BASE_DIR, exist_ok=True)

X_MEMMAP = "path1"
Y_MEMMAP = "path2"
SCALER_RAW_NPZ = os.path.join(BASE_DIR, "scaler_raw.npz")   # raw 13-feature scaler used for train/eval
SCALER_FEAT_NPZ = os.path.join(BASE_DIR, "scaler_memmap.npz")  # existing expanded-feat scaler (ignored by Option B)
OUT_METRICS = os.path.join(BASE_DIR, "metrics_full02.json")
OUT_PRED_CSV = os.path.join(BASE_DIR, "predictions_full02.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model / training defaults
SEQ_LEN = 50
N_FEATS_RAW = 13
DTYPE = "float32"
BATCH_DEFAULT = 256
ENSEMBLES_DEFAULT = 3

# Utilities
def ensure_gpu():
    g = tf.config.list_physical_devices("GPU")
    if g:
        for dev in g:
            try:
                tf.config.experimental.set_memory_growth(dev, True)
            except Exception:
                pass
        print("Using GPU:", tf.config.list_logical_devices("GPU"))
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass
    else:
        print("No GPU found, using CPU")
        try:
            tf.keras.mixed_precision.set_global_policy("float32")
        except Exception:
            pass

def load_memmap_xy(x_path, y_path, seq_len=SEQ_LEN, n_features=N_FEATS_RAW, dtype=DTYPE):
    item = np.dtype(dtype).itemsize
    ex = os.path.getsize(x_path) // item
    N = ex // (seq_len * n_features)
    X = np.memmap(x_path, mode='r', dtype=dtype, shape=(N, seq_len, n_features))
    ey = os.path.getsize(y_path) // item
    y_raw = np.memmap(y_path, mode='r', dtype=dtype, shape=(ey,))[:N].astype('float32')
    y = y_raw.reshape(-1,1)
    print(f"Loaded memmaps: N={N}, X.shape={X.shape}, y.shape={y.shape}")
    return X, y, N

# (Optional) expanded feature function kept for reference
def compute_window_features(X_memmap, indices=None):
    N, L, Fraw = X_memmap.shape
    if indices is None:
        indices = np.arange(N)
    X_out = np.zeros((len(indices), L, Fraw*5 + 3), dtype='float32')
    for idx_i, i in enumerate(indices):
        w = np.asarray(X_memmap[i], dtype='float32')  # (L, Fraw)
        mean = np.nanmean(w, axis=0)
        std  = np.nanstd(w, axis=0)
        mn   = np.nanmin(w, axis=0)
        mx   = np.nanmax(w, axis=0)
        V = w[:,0] if Fraw>0 else np.zeros(L)
        I = w[:,1] if Fraw>1 else np.zeros(L)
        power = V * I
        dvdt = np.concatenate([[0.0], np.diff(V)])
        base = w
        stats_stack = np.tile(np.concatenate([mean,std,mn,mx]), (L,1))
        extra = np.tile([np.mean(power), np.std(power), np.mean(dvdt)], (L,1))
        arr = np.concatenate([base, stats_stack, extra], axis=1)
        X_out[idx_i,:arr.shape[0],:arr.shape[1]] = arr
    feat_names = []
    for j in range(Fraw):
        feat_names.append(f"f{j}_raw")
    for s in ['mean','std','min','max']:
        for j in range(Fraw):
            feat_names.append(f"f{j}_{s}")
    feat_names += ["power_mean","power_std","dvdt_mean"]
    return X_out, feat_names

# Model (CNN-LSTM)
def build_cnn_lstm(seq_len=SEQ_LEN, n_features=100):
    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.Conv1D(96, 5, padding='same', activation='relu')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.25)(x)
    mean = layers.Dense(1, name='rul_mean', dtype='float32')(x)
    q10  = layers.Dense(1, name='rul_q10',  dtype='float32')(x)
    q50  = layers.Dense(1, name='rul_q50',  dtype='float32')(x)
    q90  = layers.Dense(1, name='rul_q90',  dtype='float32')(x)
    return models.Model(inp, [mean,q10,q50,q90])

def pinball_loss(q):
    def loss(y, f):
        e = tf.cast(y, f.dtype) - f
        return tf.reduce_mean(tf.maximum(q*e, (q-1.0)*e))
    return loss

def compile_model_for_train(model, lr=1e-3, loss_weights=None):
    if loss_weights is None:
        loss_weights = {'rul_mean':1.0,'rul_q10':0.25,'rul_q50':0.5,'rul_q90':0.25}
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss={'rul_mean':'mae','rul_q10':pinball_loss(0.1),'rul_q50':pinball_loss(0.5),'rul_q90':pinball_loss(0.9)},
                  loss_weights=loss_weights)
    return model

# Ensemble training & predict
def train_ensemble(X_train, y_train, X_val=None, y_val=None, ensembles=3, epochs=10, batch_size=256, outdir=BASE_DIR):
    models_list = []
    for i in range(ensembles):
        print(f"Training ensemble member {i+1}/{ensembles}")
        m = build_cnn_lstm(seq_len=X_train.shape[1], n_features=X_train.shape[2])
        compile_model_for_train(m)
        ck = callbacks.ModelCheckpoint(os.path.join(outdir, f"ens_{i}.keras"), save_best_only=True, monitor='val_rul_mean_loss', mode='min')
        hist = m.fit(X_train, [y_train,y_train,y_train,y_train],
                     validation_data=(X_val,[y_val,y_val,y_val,y_val]) if (X_val is not None and y_val is not None) else None,
                     epochs=epochs, batch_size=batch_size, callbacks=[ck], verbose=2)
        m_best = load_model(os.path.join(outdir, f"ens_{i}.keras"), compile=False)
        models_list.append(m_best)
    return models_list

def ensemble_predict(models_list, X_eval, batch=1024):
    parts = []
    for m in models_list:
        out = m.predict(X_eval, batch_size=batch, verbose=0)
        if isinstance(out, (list,tuple)):
            arr = np.stack([np.ravel(o) for o in out], axis=1)
        else:
            arr = np.ravel(out).reshape(-1,1)
            arr = np.concatenate([arr,arr,arr,arr], axis=1)
        parts.append(arr)
    P = np.stack(parts, axis=1)
    return P

# metrics+plots
def compute_metrics_and_plots(P_med_mean, q10, q90, y_true, out_prefix=BASE_DIR):
    e = P_med_mean - y_true.reshape(-1)
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e*e)))
    coverage = float(np.mean((y_true.reshape(-1) >= q10) & (y_true.reshape(-1) <= q90)))
    metrics = {"MAE": mae, "RMSE": rmse, "P10-P90_coverage": coverage, "N_eval": int(len(y_true))}
    json.dump(metrics, open(os.path.join(out_prefix,"metrics_eval.json"),"w"), indent=2)
    print("Metrics:", metrics)
    plt.figure(figsize=(6,4)); plt.hist(P_med_mean, bins=40); plt.title("Predicted RUL (mean)"); plt.xlabel("RUL"); plt.ylabel("count"); plt.savefig(os.path.join(out_prefix,"pred_rul_hist.png")); plt.close()
    plt.figure(figsize=(6,4)); plt.hist(q90 - q10, bins=40); plt.title("Interval width q90-q10"); plt.xlabel("width"); plt.ylabel("count"); plt.savefig(os.path.join(out_prefix,"interval_width_hist.png")); plt.close()
    plt.figure(figsize=(6,4)); plt.scatter(P_med_mean, q90-q10, s=4, alpha=0.3); plt.title("Width vs Mean"); plt.xlabel("mean"); plt.ylabel("width"); plt.savefig(os.path.join(out_prefix,"width_vs_mean.png")); plt.close()
    plt.figure(figsize=(6,4)); plt.hist(e, bins=60); plt.title(f"Residuals (MAE={mae:.3f}, RMSE={rmse:.3f})"); plt.xlabel("mean - y_true"); plt.ylabel("count"); plt.savefig(os.path.join(out_prefix,"residuals_hist.png")); plt.close()
    return metrics

def compute_and_save_raw_scaler(x_memmap_path, seq_len=SEQ_LEN, n_features=N_FEATS_RAW, dtype=DTYPE, out_path=SCALER_RAW_NPZ, batch=131072):
    print("Computing raw scaler (this may take time)...")
    item = np.dtype(dtype).itemsize
    elems = os.path.getsize(x_memmap_path) // item
    N = elems // (seq_len * n_features)
    Xm = np.memmap(x_memmap_path, mode='r', dtype=dtype, shape=(N, seq_len, n_features))
    sum_ = np.zeros((n_features,), dtype=np.float64)
    sumsq = np.zeros((n_features,), dtype=np.float64)
    count = 0
    for s in range(0, N, batch):
        c = np.asarray(Xm[s:s+batch])
        sum_ += np.sum(c, axis=(0,1))
        sumsq += np.sum(c*c, axis=(0,1))
        count += np.prod(c.shape[:2])
    mean = sum_ / max(count,1)
    var = sumsq / max(count,1) - mean**2
    std = np.sqrt(np.maximum(var, 1e-12))
    np.savez(out_path, mean=mean, std=std)
    print("Saved raw scaler to", out_path)
    return mean, std

def safe_normalize_raw(Xarr, scaler_path=SCALER_RAW_NPZ):
    Xarr = np.asarray(Xarr, dtype='float32')
    if not os.path.exists(scaler_path):
        _mean, _std = compute_and_save_raw_scaler(X_MEMMAP)
    sc = np.load(scaler_path)
    mean = sc["mean"].reshape(1,1,-1)
    std  = sc["std"].reshape(1,1,-1)
    return (Xarr - mean)/np.maximum(std, 1e-6)

# Other utilities (export / latency)
def export_quantized_tflite(model_path, tflite_out):
    model = load_model(model_path, compile=False)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        open(tflite_out,"wb").write(tflite_model)
        print("Saved tflite:", tflite_out)
    except Exception as e:
        print("TFLite conversion failed:", e)

def latency_benchmark(model, X_sample, n_runs=100, batch_size=128):
    times = []
    for i in range(0, min(len(X_sample), n_runs*batch_size), batch_size):
        t0 = time.time()
        _ = model.predict(X_sample[i:i+batch_size], batch_size=batch_size, verbose=0)
        dt = time.time() - t0
        times.append(dt)
    if not times:
        return None
    avg_batch = np.mean(times)
    samples_per_sec = batch_size / avg_batch
    return {"avg_batch_sec": float(avg_batch), "samples_per_sec": float(samples_per_sec)}

# Main CLI
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="eval", help="mode: feature_prep / train / eval / ablation / optimize_jetson / latency")
    parser.add_argument("--ensembles", type=int, default=ENSEMBLES_DEFAULT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    parser.add_argument("--n_eval", type=int, default=100000)
    parser.add_argument("--max_samples", type=int, default=200000, help="limit samples for quick runs")
    parser.add_argument("--groups", type=str, default="", help="comma separated ablation groups")
    parser.add_argument("--model_path", type=str, default=os.path.join(BASE_DIR,"ens_0.keras"), help="model path for export")
    parser.add_argument("--tflite_out", type=str, default=os.path.join(BASE_DIR,"model_quant.tflite"))
    args = parser.parse_args(argv)

    ensure_gpu()

    if args.mode == "feature_prep":
        print("Feature prep (kept for reference). This script focuses on raw-feature flow (Option B).")
        Xm, y, N = load_memmap_xy(X_MEMMAP, Y_MEMMAP)
        ns = min(args.max_samples, N)
        Xsub = np.asarray(Xm[:ns])
        Xfeat, feat_names = compute_window_features(Xsub)
        np.savez(SCALER_FEAT_NPZ, mean=np.nanmean(Xfeat, axis=(0,1)), std=np.nanstd(Xfeat, axis=(0,1)))
        np.save(os.path.join(BASE_DIR,"Xfeat_sample.npy"), Xfeat)
        np.save(os.path.join(BASE_DIR,"y_sample.npy"), y[:ns])
        print("Saved sample features and expanded-feature scaler to", BASE_DIR)
        compute_and_save_raw_scaler(X_MEMMAP)
        return

    if args.mode == "train":
        Xm, y, N = load_memmap_xy(X_MEMMAP, Y_MEMMAP)
        ns = min(args.max_samples, N)
        print("Using", ns, "samples for training (subsample for speed).")
        Xn = np.asarray(Xm[:ns], dtype='float32')
        if not os.path.exists(SCALER_RAW_NPZ):
            compute_and_save_raw_scaler(X_MEMMAP)
        Xn = safe_normalize_raw(Xn, SCALER_RAW_NPZ)
        models = train_ensemble(Xn, y[:ns], ensembles=args.ensembles, epochs=args.epochs, batch_size=args.batch)
        print("Trained ensemble and saved to", BASE_DIR)
        return

    if args.mode == "eval":
        Xm, y, N = load_memmap_xy(X_MEMMAP, Y_MEMMAP)
        n_eval = min(args.n_eval, N)
        rng = np.random.default_rng(123)
        idx = rng.choice(N, size=n_eval, replace=False)
        Xeval = np.asarray(Xm[idx], dtype='float32')
        if not os.path.exists(SCALER_RAW_NPZ):
            compute_and_save_raw_scaler(X_MEMMAP)
        Xeval = safe_normalize_raw(Xeval, SCALER_RAW_NPZ)
        models_list = []
        for i in range(args.ensembles):
            p = os.path.join(BASE_DIR, f"ens_{i}.keras")
            if os.path.exists(p):
                models_list.append(load_model(p, compile=False))
        if not models_list:
            print("No ensemble found; train first or reduce ensembles option.")
            return
        P = ensemble_predict(models_list, Xeval, batch=1024)
        pm_med = np.median(P[:,:,0], axis=1)
        q10 = np.median(P[:,:,1], axis=1)
        q90 = np.median(P[:,:,3], axis=1)
        metrics = compute_metrics_and_plots(pm_med, q10, q90, y[idx], out_prefix=BASE_DIR)
        df = pd.DataFrame({"index": idx, "y_true": y[idx].reshape(-1), "mean": pm_med, "q10": q10, "q90": q90})
        df.to_csv(OUT_PRED_CSV, index=False)
        print("Saved predictions:", OUT_PRED_CSV)
        return

    if args.mode == "ablation":
        Xm, y, N = load_memmap_xy(X_MEMMAP, Y_MEMMAP)
        ns = min(args.max_samples, N)
        Xsub = np.asarray(Xm[:ns], dtype='float32')
        tab = run_ablation(Xsub, y[:ns], groups_to_remove=[g.strip() for g in args.groups.split(",") if g.strip()], samples=ns, ensembles=2, epochs=3)
        print("Ablation results saved to", os.path.join(BASE_DIR,"ablation_results.csv"))
        print(tab)
        return

    if args.mode == "optimize_jetson":
        export_quantized_tflite(args.model_path, args.tflite_out)
        return

    if args.mode == "latency":
        if not os.path.exists(args.model_path):
            print("model not found:", args.model_path); return
        model = load_model(args.model_path, compile=False)
        Xm, y, N = load_memmap_xy(X_MEMMAP, Y_MEMMAP)
        sample = np.asarray(Xm[:4096], dtype='float32')
        if not os.path.exists(SCALER_RAW_NPZ):
            compute_and_save_raw_scaler(X_MEMMAP)
        sample = safe_normalize_raw(sample, SCALER_RAW_NPZ)
        bench = latency_benchmark(model, sample, n_runs=20, batch_size=128)
        print("Latency bench:", bench)
        open(os.path.join(BASE_DIR,"latency_bench.json"),"w").write(json.dumps(bench))
        return

    print("Unknown mode. Use --mode flag. e.g., --mode train")

if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("-f")]
    main(argv)
