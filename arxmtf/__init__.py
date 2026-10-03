"""Multi-timeframe ARX direction model on 5-minute bars: estimation, backtest, costs, statistics."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # small-matrix linear algebra: one BLAS thread per process
