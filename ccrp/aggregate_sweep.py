"""Aggregate sweep results: python -m ccrp.aggregate_sweep results/sweep"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

POLS = ["never", "partner_blind", "ladder", "kappa_gate", "ambiguity_gate", "ccrp_E", "ccrp_S", "exact_no_query", "reference", "oracle"]


def main(path: str) -> None:
    files = sorted(glob.glob(os.path.join(path, "cfg_*.json")))
    if not files:
        print("no results in", path); return
    rows = [json.load(open(f)) for f in files]
    print(f"{len(rows)} configurations, n={rows[0]['n']} episodes each; reward = success - costs\n")
    hdr = f"{'struct':6s} {'K':>2s} {'T':>2s} {'c_take':>6s} {'p_bad':>5s} | " + " ".join(f"{p[:10]:>10s}" for p in POLS)
    print(hdr); print("-" * len(hdr))
    gaps = {p: [] for p in POLS}
    for r in rows:
        s = r["summary"]
        line = f"{r['structure']:6s} {r['K']:2d} {r['T']:2d} {r['cost_takeover']:6.2f} {r['prior_goal_bad']:5.2f} | "
        line += " ".join(f"{s[p]['reward']:10.3f}" if p in s else f"{'-':>10s}" for p in POLS)
        print(line)
        for p in POLS:
            if p in s:
                gaps[p].append(s["reference"]["reward"] - s[p]["reward"])
    print("\nmean reward gap to the reference planner (positive = worse than reference):")
    for p in POLS:
        g = np.array(gaps[p]); print(f"  {p:16s} {g.mean():+.3f}  (worse in {int((g > 0.005).sum())}/{len(g)} configs, better in {int((g < -0.005).sum())})")
    # where does CCRP beat the partner-blind planner by more than 2 SE?
    print("\nSUCCESS RATE by configuration:")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        s = r["summary"]
        line = f"{r['structure']:6s} {r['K']:2d} {r['T']:2d} {r['cost_takeover']:6.2f} {r['prior_goal_bad']:5.2f} | "
        print(line + " ".join(f"{s[p]['success']:10.2f}" if p in s else f"{'-':>10s}" for p in POLS))
    n = rows[0]["n"]
    print("\nccrp_E vs partner_blind (reward diff, success diff; z from binomial SE):")
    for r in rows:
        s = r["summary"]
        if "ccrp_E" in s and "partner_blind" in s:
            d = s["ccrp_E"]["reward"] - s["partner_blind"]["reward"]
            ds = s["ccrp_E"]["success"] - s["partner_blind"]["success"]
            pbar = 0.5 * (s["ccrp_E"]["success"] + s["partner_blind"]["success"])
            se = (2 * pbar * (1 - pbar) / n) ** 0.5 if 0 < pbar < 1 else 1e-9
            flag = " <==" if abs(ds) > 2 * se else ""
            print(f"  {r['structure']:6s} K={r['K']} T={r['T']} c={r['cost_takeover']:.2f} p={r['prior_goal_bad']:.1f}: reward {d:+.3f}, success {ds:+.3f} (z={ds/se:+.1f}){flag}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/sweep")
