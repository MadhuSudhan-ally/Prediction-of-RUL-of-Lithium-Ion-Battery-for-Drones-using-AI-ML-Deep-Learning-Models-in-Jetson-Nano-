
import os
import argparse
import json
from collections import deque, defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd


class LongWindowCounter:
    """First pass: count number of windows produced from LONG CSV."""
    def __init__(self, seq_len: int, max_ids=200000):
        self.L = seq_len
        self.buff = defaultdict(lambda: 0)  
        self.max_ids = max_ids
        self.total = 0

    def push_block(self, df: pd.DataFrame, idcol: str, feat_cols: List[str], ycol: str = None):
        for gid, block in df.groupby(idcol, sort=False):
            prev = self.buff.get(gid, 0)
            cnt = prev + len(block)
            new_windows = max(0, cnt - (self.L - 1)) - max(0, prev - (self.L - 1))
            if new_windows > 0:
                self.total += new_windows
            self.buff[gid] = cnt
            if len(self.buff) > self.max_ids:
                for k in list(self.buff.keys())[:len(self.buff)-self.max_ids]:
                    self.buff.pop(k, None)

class LongWindowWriter:
    """Second pass: produce windows and write them into preallocated memmaps"""
    def __init__(self, seq_len: int, n_feats: int, max_ids=200000):
        self.L = seq_len
        self.F = n_feats
        self.buff = defaultdict(lambda: deque(maxlen=self.L))
        self.ybuf = defaultdict(lambda: deque(maxlen=self.L))
        self.max_ids = max_ids

    def push_block(self, df: pd.DataFrame, idcol: str, feat_cols: List[str], ycol: str = None):
        Xw, Yw = [], []
        for gid, block in df.groupby(idcol, sort=False):
            feats = block[feat_cols].to_numpy(dtype='float32', copy=False)
            for row in feats:
                self.buff[gid].append(row)
            if ycol is not None:
                yvals = pd.to_numeric(block[ycol], errors='coerce').to_numpy(dtype='float32', copy=False)
                for yy in yvals:
                    self.ybuf[gid].append(yy)
            while len(self.buff[gid]) >= self.L:
                arr = np.stack(self.buff[gid], axis=0)[-self.L:]  
                if arr.shape[1] < self.F:
                    pad = np.zeros((self.L, self.F - arr.shape[1]), dtype='float32')
                    arr = np.concatenate([arr, pad], axis=1)
                elif arr.shape[1] > self.F:
                    arr = arr[:, :self.F]
                Xw.append(arr)
                if ycol is not None and len(self.ybuf[gid]) >= 1:
                    Yw.append(float(self.ybuf[gid][-1]))
                else:
                    Yw.append(np.nan)
                self.buff[gid].popleft()
                if ycol is not None and self.ybuf[gid]:
                    self.ybuf[gid].popleft()
            if len(self.buff) > self.max_ids:
                for k in list(self.buff.keys())[:len(self.buff)-self.max_ids]:
                    self.buff.pop(k, None); self.ybuf.pop(k, None)
        return Xw, Yw

def detect_long_vs_wide(sample_df: pd.DataFrame, seq_len: int, n_features: int) -> str:
    if sample_df.shape[1] >= seq_len * n_features:
        return "wide"
    return "long"

def find_time_and_id_columns(cols: List[str]) -> Tuple[str, str]:
    cols_lower = [c.lower() for c in cols]
    time_candidates = [c for c in cols if c.lower() in {"t","time","step","step_index","time_step","timestamp","ts"}]
    id_candidates = [c for c in cols if c.lower() in {"id","seq","sequence","window_id","unit","uid","device"}]
    tcol = time_candidates[0] if time_candidates else None
    idcol = id_candidates[0] if id_candidates else None
    return idcol, tcol

# Main two-pass processing
def process_wide(csv_path: str, out_dir: str, seq_len: int, n_features: int, chunksize: int = 20000):
    """
    WIDE mode: each CSV row already contains flattened sequence features (seq_len * n_features).
    Optional column 'y_true' is used as label if present.
    """
    print("Processing in WIDE mode.")
    reader = pd.read_csv(csv_path, chunksize=chunksize)
    total_rows = 0
    for chunk in reader:
        arr = chunk.iloc[:, :seq_len * n_features].apply(pd.to_numeric, errors='coerce')
        valid_mask = ~arr.isna().any(axis=1)
        total_rows += int(valid_mask.sum())
    print("Windows to write:", total_rows)
    if total_rows == 0:
        raise RuntimeError("No valid rows found in WIDE mode (check seq_len/n_features and CSV).")

    # prepare memmaps
    X_path = os.path.join(out_dir, "X_sequences_memmap.dat")
    y_path = os.path.join(out_dir, "y_labels_memmap.dat")
    X_mm = np.memmap(X_path, mode='w+', dtype='float32', shape=(total_rows, seq_len, n_features))
    y_mm = np.memmap(y_path, mode='w+', dtype='float32', shape=(total_rows,))

    # second pass: write
    reader = pd.read_csv(csv_path, chunksize=chunksize)
    idx = 0
    for chunk in reader:
        arr = chunk.iloc[:, :seq_len * n_features].apply(pd.to_numeric, errors='coerce')
        valid_rows = arr.dropna(axis=0, how='any')
        if valid_rows.shape[0] == 0:
            continue
        N = valid_rows.shape[0]
        raw = valid_rows.to_numpy(dtype='float32', copy=False)
        try:
            X = raw.reshape(N, seq_len, n_features)
        except Exception:
            X = raw.reshape(N, n_features, seq_len).transpose(0,2,1)
        X_mm[idx:idx+N, :, :] = X
        if "y_true" in chunk.columns:
            yvals = pd.to_numeric(chunk.loc[valid_rows.index, "y_true"], errors='coerce').to_numpy(dtype='float32', copy=False)
            yvals = np.nan_to_num(yvals, nan=np.nan)
        else:
            yvals = np.full((N,), np.nan, dtype='float32')
        y_mm[idx:idx+N] = yvals
        idx += N
        print(f"Wrote {idx}/{total_rows} windows", end="\r")
    X_mm.flush(); y_mm.flush()
    print("\nWIDE processing complete.")
    return X_path, y_path, total_rows

def process_long(csv_path: str, out_dir: str, seq_len: int, n_features: int, chunksize: int = 200000):
    print("Processing in LONG mode (two-pass).")
    peek = pd.read_csv(csv_path, nrows=50)
    idcol, tcol = find_time_and_id_columns(peek.columns.tolist())
    if tcol is None:
        raise RuntimeError("Could not detect a time column for LONG format. Please include a time column named t/time/step/time_step/step_index/timestamp.")
    if idcol is None:
        print("[WARN] No id column detected; script will synthesize unit ids per block which may be incorrect for your dataset.")
    candidate_feat_cols = [c for c in peek.columns if c not in {idcol, tcol}]
    numeric_candidates = []
    for c in candidate_feat_cols:
        ser = pd.to_numeric(peek[c], errors='coerce')
        if ser.notna().any():
            numeric_candidates.append(c)
    if len(numeric_candidates) == 0:
        raise RuntimeError("No numeric feature columns found in LONG CSV.")
    print("Detected id column:", idcol, "time column:", tcol)
    print(f"Numeric feature candidates (first 20): {numeric_candidates[:20]}")
    feat_cols = numeric_candidates[:n_features]
    print("Using feature columns:", feat_cols)
    counter = LongWindowCounter(seq_len=seq_len)
    reader = pd.read_csv(csv_path, chunksize=chunksize)
    for chunk in reader:
        if tcol in chunk.columns:
            if not np.issubdtype(chunk[tcol].dtype, np.number):
                try:
                    chunk[tcol] = pd.to_datetime(chunk[tcol], errors='coerce').astype('int64') // 10**9
                except Exception:
                    pass
        if idcol is None:
            chunk["__id__"] = (chunk.index == chunk[tcol].min()).cumsum() - 1
            idc = "__id__"
        else:
            idc = idcol
        chunk = chunk.sort_values([idc, tcol], kind='stable')
        counter.push_block(chunk, idc, feat_cols, ycol="y_true" if "y_true" in chunk.columns else None)
    total_windows = counter.total
    print("Total windows to produce:", total_windows)
    if total_windows == 0:
        raise RuntimeError("No windows produced. Check seq_len and data continuity.")

    # allocate memmaps
    X_path = os.path.join(out_dir, "X_sequences_memmap.dat")
    y_path = os.path.join(out_dir, "y_labels_memmap.dat")
    X_mm = np.memmap(X_path, mode='w+', dtype='float32', shape=(total_windows, seq_len, n_features))
    y_mm = np.memmap(y_path, mode='w+', dtype='float32', shape=(total_windows,))

    # second pass: write windows
    writer = LongWindowWriter(seq_len=seq_len, n_feats=n_features)
    reader = pd.read_csv(csv_path, chunksize=chunksize)
    idx = 0
    for chunk in reader:
        if tcol in chunk.columns:
            if not np.issubdtype(chunk[tcol].dtype, np.number):
                try:
                    chunk[tcol] = pd.to_datetime(chunk[tcol], errors='coerce').astype('int64') // 10**9
                except Exception:
                    pass
        if idcol is None:
            chunk["__id__"] = (chunk.index == chunk[tcol].min()).cumsum() - 1
            idc = "__id__"
        else:
            idc = idcol
        for c in feat_cols:
            chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
        chunk = chunk.sort_values([idc, tcol], kind='stable')
        Xw, Yw = writer.push_block(chunk, idc, feat_cols, ycol="y_true" if "y_true" in chunk.columns else None)
        for arr, yv in zip(Xw, Yw):
            if arr.shape[1] < n_features:
                pad = np.zeros((seq_len, n_features - arr.shape[1]), dtype='float32')
                arr2 = np.concatenate([arr, pad], axis=1)
            else:
                arr2 = arr[:, :n_features]
            X_mm[idx] = arr2.astype('float32', copy=False)
            y_mm[idx] = np.float32(yv) if np.isfinite(yv) else np.float32(np.nan)
            idx += 1
            if idx % 10000 == 0:
                print(f"Wrote {idx}/{total_windows} windows", end="\r")
    X_mm.flush(); y_mm.flush()
    print(f"\nLONG processing complete. Wrote {idx} windows to {X_path}")
    return X_path, y_path, total_windows, feat_cols

# CLI and main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to combined CSV")
    p.add_argument("--outdir", required=True, help="Output directory for memmaps and metadata")
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--n_features", type=int, default=13)
    p.add_argument("--mode", choices=["auto","wide","long"], default="auto", help="Format mode (auto detect recommended)")
    p.add_argument("--chunksize", type=int, default=200000, help="CSV chunk size (rows)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # read small sample to detect format
    sample = pd.read_csv(args.csv, nrows=20)
    mode = args.mode
    if mode == "auto":
        mode = detect_long_vs_wide(sample, args.seq_len, args.n_features)
        print("Auto-detected mode:", mode)

    if mode == "wide":
        Xp, yp, N = process_wide(args.csv, args.outdir, args.seq_len, args.n_features, chunksize=args.chunksize)
        feat_cols = [f"f{i}" for i in range(args.n_features)]
    else:
        Xp, yp, N, feat_cols = process_long(args.csv, args.outdir, args.seq_len, args.n_features, chunksize=args.chunksize)

    metadata = {
        "X_path": os.path.abspath(Xp),
        "y_path": os.path.abspath(yp),
        "seq_len": args.seq_len,
        "n_features": args.n_features,
        "N_windows": int(N),
        "feature_columns_used": feat_cols
    }
    meta_path = os.path.join(args.outdir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to", meta_path)
    print("Done.")

if __name__ == "__main__":
    main()
