"""Quick check: do queries (value of sample information prior + BO rollouts) change Team-Plan's reward?

  python -m ccrp.rl_query_check --setting long --n 90 --procs 9
"""
from __future__ import annotations

import argparse
import pickle
import time
from multiprocessing import Pool

import numpy as np

from . import rl
from .rl_experiment import settings


def _job(args):
    name, kind, seeds = args
    spec = settings()[name]
    V = pickle.load(open(f"results/rl/value_{name}.pkl", "rb"))
    pol = rl.make_policy(kind, V, budget=64, eps=0.02)
    out = []
    for s in seeds:
        r = rl.run(spec, pol, s)
        out.append((s, r.reward, sum(a.startswith("query") for a in r.actions), r.rollouts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="long"); ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--procs", type=int, default=9); ap.add_argument("--kinds", default="rl_greedy,teamplan,teamplan_q")
    a = ap.parse_args()
    res = {}
    with Pool(a.procs) as pool:
        for kind in a.kinds.split(","):
            t0 = time.time()
            rows = sorted(x for p in pool.map(_job, [(a.setting, kind, range(j, a.n, a.procs)) for j in range(a.procs)]) for x in p)
            arr = np.array(rows); res[kind] = arr
            print(f"{a.setting} {kind:12s} reward {arr[:, 1].mean():.3f} queries/ep {arr[:, 2].mean():.2f} "
                  f"rollouts/ep {arr[:, 3].mean():.0f} [{time.time() - t0:.0f}s]", flush=True)
    base = res.get("rl_greedy")
    if base is not None:
        for k, arr in res.items():
            if k != "rl_greedy":
                d = arr[:, 1] - base[:, 1]
                print(f"  {k} - rl_greedy: {d.mean():+.3f} +- {d.std(ddof=1) / np.sqrt(len(d)):.3f}")


if __name__ == "__main__":
    main()
