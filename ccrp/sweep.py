"""Parameter sweep for the synthetic policy comparison (E5), one configuration per invocation.

Usage: python -m ccrp.sweep --index I --n 40 --out results/sweep
Grid: structure x K x T x cost_takeover x prior_goal_bad.  Writes results/sweep/cfg_I.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

from .policies import (ccrp_policy, exact_policy, gated_policy, ladder_policy, myopic_policy, never_policy,
                       no_query_exact_policy, oracle_policy, run_episode)
from .synthetic import EnvSpec, formula_and, random_formula

GRID = list(itertools.product(("and", "mixed"), (4, 6), (3, 5), (0.06, 0.15), (0.2, 0.4)))
# v2: scarce budget (K > T), costlier takeovers, stronger and more uncertain common cause
GRID_V2 = list(itertools.product(("and", "mixed"), (6, 8), (2, 3, 4), (0.10, 0.20), (0.3, 0.5)))
GRIDS = {"v1": GRID, "v2": GRID_V2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="results/sweep"); ap.add_argument("--grid", default="v1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    grid = GRIDS[a.grid]
    structure, K, T, c_take, p_goal = grid[a.index % len(grid)]
    pols = {"never": never_policy, "partner_blind": myopic_policy, "exact_no_query": no_query_exact_policy,
            "ladder": ladder_policy, "kappa_gate": gated_policy("kappa", 0.15), "ambiguity_gate": gated_policy("ambiguity", 0.01),
            "ccrp_E": ccrp_policy("E"), "ccrp_S": ccrp_policy("S", n_sample=48), "reference": exact_policy,
            "oracle": oracle_policy}
    rows = []
    t0 = time.time()
    for seed in range(a.n):
        fm = formula_and(K) if structure == "and" else random_formula(K, np.random.default_rng(seed))
        spec = EnvSpec(fm, K, n_objects=2, n_skills=1, horizon=T,
                       prior_bad={"goal": p_goal, "obj": 0.3, "skill": 0.2},
                       costs={"takeover": c_take, "help": c_take / 2, "clarify": 0.06, "align": 0.05, "query": 0.02})
        for name, pol in pols.items():
            r = run_episode(spec, pol, seed)
            rows.append({"policy": name, "seed": seed, "success": r.success, "reward": r.reward, "regret": r.regret,
                         "n_queries": sum(x.startswith("query") for x in r.actions),
                         "n_repairs": sum(x.startswith(("clarify", "align")) for x in r.actions),
                         "n_takeovers": sum(x.startswith("takeover") for x in r.actions)})
    summ = {}
    for name in pols:
        sub = [r for r in rows if r["policy"] == name]
        summ[name] = {k: float(np.mean([r[k] for r in sub])) for k in ("success", "reward", "regret", "n_queries", "n_repairs", "n_takeovers")}
        summ[name]["reward_se"] = float(np.std([r["reward"] for r in sub]) / np.sqrt(len(sub)))
    out = {"index": a.index, "grid": a.grid, "structure": structure, "K": K, "T": T, "cost_takeover": c_take, "prior_goal_bad": p_goal,
           "n": a.n, "seconds": time.time() - t0, "summary": summ}
    with open(os.path.join(a.out, f"cfg_{a.index:02d}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "summary"}))
    for name, v in summ.items():
        print(f"  {name:16s} succ={v['success']:.2f} reward={v['reward']:.3f}+-{v['reward_se']:.3f} regret={v['regret']:.3f} "
              f"q={v['n_queries']:.2f} rep={v['n_repairs']:.2f} take={v['n_takeovers']:.2f}")


if __name__ == "__main__":
    main()
