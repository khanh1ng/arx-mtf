"""Raw 5-minute bars -> per-stock npz cache, and the fixed universe."""
import os, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
from . import config as C


def _build_one(tic):
    out = os.path.join(C.CACHE, tic + ".npz")
    if os.path.exists(out):
        return tic
    fs = sorted(glob.glob(os.path.join(C.RAW_DATA, tic, "*.tsv.gz")))
    if not fs:
        return None
    df = pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    for k in ("open", "high", "low", "close", "volume"):
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    ts = df["timestamp"].dt.tz_convert("America/New_York")
    np.savez_compressed(
        out,
        day=ts.dt.strftime("%Y%m%d").astype(np.int32).to_numpy(),
        minute=(ts.dt.hour * 60 + ts.dt.minute).astype(np.int16).to_numpy(),
        o=df["open"].to_numpy(np.float32), h=df["high"].to_numpy(np.float32),
        l=df["low"].to_numpy(np.float32), c=df["close"].to_numpy(np.float32),
        v=df["volume"].fillna(0).to_numpy(np.float32))
    return tic


def build_cache():
    os.makedirs(C.CACHE, exist_ok=True)
    tics = sorted(d for d in os.listdir(C.RAW_DATA) if os.path.isdir(os.path.join(C.RAW_DATA, d)))
    with Pool(C.N_WORKERS) as p:
        done = [t for t in p.map(_build_one, tics) if t]
    full = [t for t in done if len(np.load(os.path.join(C.CACHE, t + ".npz"))["c"]) > C.MIN_BARS]
    np.save(os.path.join(C.CACHE, "universe.npy"), np.array(full))
    return full


def universe():
    return list(np.load(os.path.join(C.CACHE, "universe.npy")))


def load(tic, upto=None):
    """Regular-hours bars for one stock. `upto` truncates at a bar index (used by look-ahead tests)."""
    z = np.load(os.path.join(C.CACHE, tic + ".npz"))
    d = {k: z[k] for k in ("day", "minute", "o", "h", "l", "c", "v")}
    keep = (d["minute"] >= C.RTH[0]) & (d["minute"] <= C.RTH[1])
    d = {k: v[keep] for k, v in d.items()}
    if upto is not None:
        d = {k: v[: upto + 1] for k, v in d.items()}
    for k in ("o", "h", "l", "c", "v"):
        d[k] = d[k].astype(np.float64)
    d["key"] = d["day"].astype(np.int64) * 10000 + d["minute"].astype(np.int64)
    d["first"] = np.r_[True, d["day"][1:] != d["day"][:-1]]
    return d


def lagmat(a, lags):
    """Column k holds a[t - lags[k]] (lag 0 allowed); rows without history are NaN."""
    n = len(a)
    out = np.full((n, len(lags)), np.nan)
    for j, g in enumerate(lags):
        if g == 0:
            out[:, j] = a
        else:
            out[g:, j] = a[:-g]
    return out
