

import os, math, shutil, datetime, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# ---------------- CONFIG ----------------
DRIVE_DIR   = "path"  
LOCAL_DIR   = "/content/data"                
MODEL_DIR   = os.path.join(DRIVE_DIR, "models")

X_NAME = "X_sequences_memmap.dat"
Y_NAME = "y_labels_memmap.dat"

TOTAL_SEQS = 9_253_765
SEQ_LEN    = 50
N_FEATURES = 13          
DTYPE      = np.float32


LATENT_DIM = 64
BASE_LR    = 2e-4
VAL_RATIO  = 0.05
SEED       = 42

# Loss weights
WEIGHT_RECON   = 1.0
WEIGHT_KL      = 1e-4
WEIGHT_RUL     = 1.0
WEIGHT_PHYS    = 1.0   
WEIGHT_SMOOTH  = 1e-3  


#CONFIG
USE_SUBSET          = True
SUBSET_FRACTION     = 0.0005  
MAX_STEPS_PER_EPOCH = 20       
EPOCHS              = 5
BATCH_SIZE          = 512




# Mixed precision + XLA
tf.config.optimizer.set_jit(True)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Seeds
np.random.seed(SEED)
tf.random.set_seed(SEED)



def fast_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        print(f"Already on local SSD: {dst}")
        return
    print(f"Copying {os.path.basename(src)} -> local SSD ...")
    shutil.copyfile(src, dst)
    print("Done.")

X_DRIVE = os.path.join(DRIVE_DIR, X_NAME)
Y_DRIVE = os.path.join(DRIVE_DIR, Y_NAME)
X_LOCAL = os.path.join(LOCAL_DIR, X_NAME)
Y_LOCAL = os.path.join(LOCAL_DIR, Y_NAME)

fast_copy(X_DRIVE, X_LOCAL)
fast_copy(Y_DRIVE, Y_LOCAL)

#Open memmaps
X_mem = np.memmap(X_LOCAL, dtype=DTYPE, mode='r', shape=(TOTAL_SEQS, SEQ_LEN, N_FEATURES))
y_mem = np.memmap(Y_LOCAL, dtype=DTYPE, mode='r', shape=(TOTAL_SEQS,))
print("Opened memmaps:", X_mem.shape, y_mem.shape)

#Split / subset
rng = np.random.default_rng(SEED)
all_idx = np.arange(TOTAL_SEQS, dtype=np.int64)

if USE_SUBSET and 0.0 < SUBSET_FRACTION < 1.0:
    sub_n = int(TOTAL_SEQS * SUBSET_FRACTION)
    print(f" Using subset: {sub_n:,} samples (~{int(SUBSET_FRACTION*100)}%).")
    all_idx = rng.choice(all_idx, size=sub_n, replace=False)

val_n   = max(1, int(len(all_idx) * VAL_RATIO))
train_i = all_idx[:-val_n]
val_i   = all_idx[-val_n:]
print(f"Train: {len(train_i):,} | Val: {len(val_i):,}")

#Vectorized memmap fetch
def memmap_batch_fetch(idxs_np):
    Xb = X_mem[idxs_np]
    yb = y_mem[idxs_np]
    return Xb, yb

def tf_fetch(idxs):
    Xb, yb = tf.numpy_function(func=lambda a: memmap_batch_fetch(a),
                               inp=[idxs], Tout=[tf.float32, tf.float32])
    Xb.set_shape((None, SEQ_LEN, N_FEATURES))
    yb.set_shape((None,))
    return Xb, yb

FEATURE_NAMES = [
    'Test_Time(s)','Step_Time(s)','Cycle_Index',
    'Current(A)','Voltage(V)',
    'Charge_Capacity(Ah)','Discharge_Capacity(Ah)',
    'Charge_Energy(Wh)','Discharge_Energy(Wh)',
    'dV/dt(V/s)','Internal_Resistance(Ohm)',
    'AC_Impedance(Ohm)','ACI_Phase_Angle(Deg)'
]
CAPACITY_FEATURE = FEATURE_NAMES.index('Discharge_Capacity(Ah)')

#On-the-fly FEATURE AUGMENTATION
AUG_FEATURES = 6 * N_FEATURES + 32

def positional_encoding(T, d):
    pos = tf.range(T)[:, None]
    i   = tf.range(d)[None, :]
    angle = tf.cast(pos, tf.float32) / tf.pow(10000.0, (2.0 * tf.cast(i//2, tf.float32)) / tf.cast(d, tf.float32))
    s = tf.sin(angle[:, 0::2]); c = tf.cos(angle[:, 1::2])
    pe = tf.reshape(tf.stack([s, c], axis=-1), [T, -1])
    return pe  # [T, d]

def movavg(z, k: int):
    if k <= 1: return z
    z32 = tf.cast(z, tf.float32)
    z4   = tf.expand_dims(z32, axis=2)
    F   = tf.shape(z32)[-1]
    w4  = tf.ones([k, 1, F, 1], dtype=tf.float32) / float(k)
    out4 = tf.nn.depthwise_conv2d(z4, w4, strides=[1,1,1,1], padding="SAME")
    return tf.cast(tf.squeeze(out4, axis=2), z.dtype)

def local_std(z, k: int):
    if k <= 1: return tf.zeros_like(z)
    z_mean  = movavg(z, k)
    sq_mean = movavg(z * z, k)
    var     = tf.nn.relu(sq_mean - z_mean * z_mean)
    return tf.sqrt(var + 1e-6)

def augment_features(x):
    x32 = tf.cast(x, tf.float32)
    dx   = tf.concat([tf.zeros_like(x32[:, :1, :]), x32[:, 1:, :] - x32[:, :-1, :]], axis=1)
    ma3  = movavg(x32, 3)
    ma9  = movavg(x32, 9)
    lstd = local_std(x32, 9)
    xTF  = tf.transpose(x32, [0, 2, 1])
    r    = tf.signal.rfft(xTF)
    rmag = tf.abs(r)
    low  = tf.reduce_mean(rmag[..., :tf.minimum(8, tf.shape(rmag)[-1])], axis=-1)
    r_low = tf.tile(low[:, None, :], [1, tf.shape(x32)[1], 1])
    pe = positional_encoding(tf.shape(x32)[1], 32)
    pe = tf.tile(pe[None, ...], [tf.shape(x32)[0], 1, 1])
    x_aug = tf.concat([x32, dx, ma3, ma9, lstd, r_low, pe], axis=-1)
    c = tf.shape(x_aug)[-1]
    x_aug = tf.slice(x_aug, [0, 0, 0], [-1, -1, tf.minimum(c, AUG_FEATURES)])
    x_aug = tf.pad(x_aug, [[0, 0], [0, 0], [0, tf.maximum(0, AUG_FEATURES - c)]])
    return tf.cast(x_aug, x.dtype)

#Datasets
def make_supervised_ds(index_array, batch_size=BATCH_SIZE, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(index_array)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(index_array), 1_000_000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False).map(tf_fetch, num_parallel_calls=tf.data.AUTOTUNE)
    def to_targets(x_raw, y):
        x_aug = augment_features(x_raw)
        cap_true = x_raw[:, -1, CAPACITY_FEATURE]
        inputs = (x_aug, x_raw)
        targets = {"recon": x_raw, "rul_p10": y, "rul_p50": y, "rul_p90": y, "capacity_pred": cap_true}
        return inputs, targets
    return ds.map(to_targets, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

train_ds = make_supervised_ds(train_i, shuffle=True)
val_ds   = make_supervised_ds(val_i, shuffle=False)

#Model blocks
def transformer_block(x, heads=4, dim=32, mlp_mult=2, drop=0.1):
    y = layers.LayerNormalization(epsilon=1e-5)(x)
    y = layers.MultiHeadAttention(num_heads=heads, key_dim=dim, dropout=drop)(y, y)
    x = layers.Add()([x, y])
    y = layers.LayerNormalization(epsilon=1e-5)(x)
    y = layers.Dense(int(y.shape[-1]) * mlp_mult, activation="gelu")(y)
    y = layers.Dropout(drop)(y)
    y = layers.Dense(int(x.shape[-1]))(y)
    y = layers.Dropout(drop)(y)
    return layers.Add()([x, y])

class SmoothnessLoss(layers.Layer):
    def __init__(self, weight=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.constant(weight, dtype=tf.float32)
    def call(self, inputs):
        x, recon = inputs
        x32, r32 = tf.cast(x, tf.float32), tf.cast(recon[..., :tf.shape(x)[-1]], tf.float32)
        dx = x32[:, 1:, :] - x32[:, :-1, :]
        dr = r32[:, 1:, :] - r32[:, :-1, :]
        penalty = tf.reduce_mean(tf.nn.relu(tf.abs(dr) - tf.abs(dx)))
        self.add_loss(self.w * penalty)
        return recon[..., :tf.shape(x)[-1]]

class TileLayer(layers.Layer):
    def call(self, inputs):
        tensor, ref = inputs
        b = tf.shape(ref)[0]
        return tf.tile(tensor[None, ...], [b, 1, 1])

#Build model
def build_model():
    aug_in  = tf.keras.Input((SEQ_LEN, AUG_FEATURES), name="input_aug")
    orig_in = tf.keras.Input((SEQ_LEN, N_FEATURES),   name="orig_seq")

    # 1) TCN stack
    x = layers.Conv1D(128, 5, padding="causal", activation="relu", dtype='float16')(aug_in)
    for d in [1, 2, 4, 8]:
        y = layers.Conv1D(128, 5, padding="causal", dilation_rate=d, activation="relu", dtype='float16')(x)
        y = layers.Conv1D(128, 1, activation="relu", dtype='float16')(y)
        if d % 2 == 0:
            y = layers.Dropout(0.1, dtype='float16')(y)
        if x.shape[-1] != 128:
            x = layers.Conv1D(128, 1, dtype='float16')(x)
        x = layers.Add(dtype='float16')([x, y])

    # 2) BiGRU
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dtype='float16'))(x)

    # 3) Transformer encoder blocks with explicit PE (sine) — keep PE within layers
    pe = positional_encoding(SEQ_LEN, 32)                   # [L, 32] constant tensor
    pe_tiled = TileLayer()([tf.cast(pe, x.dtype), x])       # [B, L, 32] via a Keras layer
    x = layers.Concatenate(dtype='float16')([x, pe_tiled])
    for _ in range(2):
        x = transformer_block(x)

    # 4) Attention pooling — wrap TF reduce_sum in a Lambda layer
    scores  = layers.Dense(1, activation='tanh', dtype='float16')(x)
    weights = layers.Softmax(axis=1)(scores)
    pooled  = layers.Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1),
                            name="attn_pool")([x, weights])

    # 5) Latent representation
    z = layers.Dense(256, activation="gelu", dtype='float16')(pooled)
    z = layers.Dropout(0.2, dtype='float16')(z)

    # 6) Sequence decoder (reconstruct original channels) + smoothness add_loss
    d = layers.RepeatVector(SEQ_LEN)(z)
    d = layers.GRU(128, return_sequences=True)(d)
    recon_raw = layers.TimeDistributed(layers.Dense(AUG_FEATURES))(d)
    recon = SmoothnessLoss(WEIGHT_SMOOTH)([orig_in, recon_raw])
    recon = layers.Activation("linear", name="recon")(recon)

    # 7) RUL quantiles
    rul_h   = layers.Dense(128, activation="gelu")(z)
    rul_p10 = layers.Dense(1, name="rul_p10")(rul_h)
    rul_p50 = layers.Dense(1, name="rul_p50")(rul_h)
    rul_p90 = layers.Dense(1, name="rul_p90")(rul_h)

    # 8) Capacity head
    cap_h   = layers.Dense(128, activation="gelu")(z)
    cap_out = layers.Dense(1, name="capacity_pred")(cap_h)

    return Model([aug_in, orig_in],
                 [recon, rul_p10, rul_p50, rul_p90, cap_out],
                 name="PInA_RVAE_ADV")


model = build_model()
model.summary(line_length=140)

#Optimizer & losses
steps_per_epoch = min(math.ceil(len(train_i)/BATCH_SIZE), MAX_STEPS_PER_EPOCH)
total_steps = steps_per_epoch * EPOCHS

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(BASE_LR, total_steps, alpha=0.1)

def pinball(q):
    q = tf.constant(q, tf.float32)
    def loss(y_true, y_pred):
        e = tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)
        return tf.reduce_mean(tf.maximum(q*e, (q-1.0)*e))
    return loss

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss={"recon":"mse","rul_p10":pinball(.1),"rul_p50":pinball(.5),"rul_p90":pinball(.9),"capacity_pred":"mse"},
    loss_weights={"recon":WEIGHT_RECON,"rul_p10":0.5*WEIGHT_RUL,"rul_p50":WEIGHT_RUL,"rul_p90":0.5*WEIGHT_RUL,"capacity_pred":WEIGHT_PHYS},
    jit_compile=True, steps_per_execution=64
)

#Train
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)
