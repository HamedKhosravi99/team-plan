"""Episode-clustered uncertainty for the paired PARTNR comparisons.

Each base episode appears under several injections, so pairs that share a base episode may be dependent.  This script
resamples base episodes (carrying all their injections and both configurations) and reports a percentile interval for
the paired difference in task success, next to the unclustered paired SE and the exact McNemar p.

  python partnr/cluster_bootstrap.py results/partnr_runs/pilot7b_v2 [--ref ccrp] [--B 10000]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np


def _analyze():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("analyze_runs", os.path.join(here, "analyze_runs.py"))
    mod = importlib.util.module_from_spec(spec); argv = sys.argv; sys.argv = ["x", "y", "--table"]
    spec.loader.exec_module(mod); sys.argv = argv
    return mod


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--ref", default="ccrp")
    ap.add_argument("--B", type=int, default=10000); ap.add_argument("--injections", default="")
    a = ap.parse_args(); ar = _analyze(); df = ar._episode_frame(a.root)
    if a.injections:
        df = df[df["injection"].isin(a.injections.split(","))]
    rng = np.random.default_rng(0); key = ["injection", "episode_id"]
    r = df[df.config == a.ref].set_index(key)
    for other in sorted(set(df.config) - {a.ref}):
        o = df[df.config == other].set_index(key); common = r.index.intersection(o.index)
        if len(common) == 0:
            continue
        d = (r.loc[common, "task_state_success"].values - o.loc[common, "task_state_success"].values).astype(float)
        eps = np.array([e for _, e in common]); ids = np.unique(eps)
        by = {e: d[eps == e] for e in ids}
        sums = np.array([by[e].sum() for e in ids]); cnts = np.array([len(by[e]) for e in ids])
        boot = np.empty(a.B)
        for b in range(a.B):
            k = rng.integers(len(ids), size=len(ids))
            boot[b] = sums[k].sum() / cnts[k].sum()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        w, l = int((d > 0).sum()), int((d < 0).sum())
        print(f"{other:16s} pairs={len(d):3d} base_episodes={len(ids):2d} diff={d.mean():+.3f} "
              f"paired_SE={d.std(ddof=1) / np.sqrt(len(d)):.3f} cluster_95CI=[{lo:+.3f},{hi:+.3f}] "
              f"W/L={w}/{l} mcnemar_p={ar._mcnemar_p(w, l):.3f}")


if __name__ == "__main__":
    main()
