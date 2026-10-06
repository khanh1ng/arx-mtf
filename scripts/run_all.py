"""Run every stage in order. Usage: python scripts/run_all.py [--skip-cache]"""
import os, sys, subprocess, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ["scripts/build_cache.py", "scripts/run_estimation.py", "scripts/run_signals.py",
          "scripts/run_portfolio.py", "scripts/run_metrics.py", "scripts/run_stats.py", "scripts/run_risk.py",
          "scripts/run_anatomy.py", "scripts/run_extra.py", "scripts/run_sensitivity.py",
          "tests/test_regression.py", "tests/test_lookahead.py", "scripts/make_papers.py",
          "scripts/run_passive.py", "tests/test_passive.py", "scripts/report_passive.py"]
if __name__ == "__main__":
    skip = "--skip-cache" in sys.argv
    for st in STAGES:
        if skip and st.endswith("build_cache.py"):
            continue
        t = time.time(); print(f"== {st}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, st)], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"stage failed: {st}")
        print(f"   {time.time() - t:.0f}s", flush=True)
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                   cwd=os.path.join(ROOT, "paper"), stdout=subprocess.DEVNULL)
    print("done: paper/main.pdf")
