"""All fixed choices in one place. Changing anything here changes every result."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.expanduser(os.getenv("ARXMTF_RAW_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "5_mins")))
CACHE = os.path.join(ROOT, "cache")
RESULTS = os.path.join(ROOT, "results")
PERTIC = os.path.join(RESULTS, "pertic")
PAPER = os.path.join(ROOT, "paper")

MIN_BARS = 95_000            # stocks with fewer bars are dropped (incomplete coverage)
RTH = (570, 955)             # first and last 5-minute bar start, minutes after midnight (09:30, 15:55)
BARS_PER_DAY = 78
BPY = 78 * 252               # 5-minute bars per year, for annualising

L = 5                        # lags of r and x
HAC_LAGS = 10                # Newey-West bandwidth

# 5-minute walk-forward (estimation-protocol replication)
TRAIN_5M = 20_000
REEST_5M = 2_000
# unified protocol: same calendar span for every timeframe
TRAIN_DAYS = 256
REEST_DAYS = 25.6
SIGMA_DAYS = 20

TAUS = [("5m", 1), ("15m", 3), ("30m", 6), ("1h", 12), ("2h", 24), ("4h", 48)]
NB = {1: 78, 3: 26, 6: 13, 12: 7, 24: 4, 48: 2}   # tau-bars per full session

GATE_Q = 0.75                # confidence filter quantile, fixed before any run
GATE_BLOCK = 2_000           # filter threshold uses the previous 2,000 five-minute bars
LEARN_END = 20220101         # weights learned on data before this date; hold-out starts here

COST_LEVELS = (0.0, 0.25, 0.87)   # flat half-spreads (bps) used for comparison
SPREAD_LOOKBACK_DAYS = 20          # per-stock spread: trailing window of past days only
N_WORKERS = 8
