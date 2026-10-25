"""Coupling sweep: does the price of partner-blindness grow with how strongly hidden causes couple responsibilities?

Theorem 4 says the partner-blind (mean-rate) value error is pi(1-pi) times the coupling strength.  Here the coupling
is varied directly while every responsibility's marginal failure rate stays fixed: K responsibilities share
n_objects object beliefs (n_objects = 1: one belief behind all of them; n_objects = K: one each).  For each
configuration we measure, with common random numbers:
  - the outcome gap between the hypothesis-conditional reference and the partner-blind planner,
  - along the reference trajectory, how often the partner-blind planner would choose a different action,
  - the partner-blind value error at the first decision (max over actions of |myopic - exact|).

Usage: python -m ccrp.coupling_sweep --n 200 --out results/coupling --procs 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from .core import INFORMATIONAL
from .policies import exact_policy, myopic_policy, never_policy, oracle_policy, run_episode
from .synthetic import EnvSpec, formula_and, formula_or

GRID = list(itertools.product(("and", "or"), (2, 3), (1, 2, 4), (0.15, 0.5)))
K = 4


def spec_for(structure: str, T: int, n_objects: int, p_obj: float) -> EnvSpec:
    fm = formula_and(K) if structure == "and" else formula_or(K)
    return EnvSpec(fm, K, n_objects=n_objects, n_skills=1, horizon=T,
                   prior_bad={"goal": 0.05, "obj": p_obj, "skill": 0.0})


class DisagreementLog:
    """Acts like the reference policy and records what the partner-blind planner would have done."""

    def __init__(self):
        self.decisions = 0; self.disagree = 0; self.errors = []

    def __call__(self, env, post, acts):
        model = env.model()
        cands = [a for a in acts if a.kind not in INFORMATIONAL]
        ex = [model.exact_advantage(a, post) for a in cands]
        my = [model.myopic_score(a, post) for a in cands]
        self.decisions += 1
        self.disagree += int(int(np.argmax(ex)) != int(np.argmax(my)))
        self.errors.append(float(np.max(np.abs(np.array(ex) - np.array(my)))))
        return cands[int(np.argmax(ex))]


def run_config(args):
    idx, n = args
    structure, T, n_objects, p_obj = GRID[idx]
    t0 = time.time()
    rows = {"reference": [], "partner_blind": [], "never": [], "oracle": []}
    log = DisagreementLog()
    first_err = None
    for seed in range(n):
        spec = spec_for(structure, T, n_objects, p_obj)
        for name, pol in (("reference", exact_policy), ("partner_blind", myopic_policy), ("never", never_policy),
                          ("oracle", oracle_policy)):
            r = run_episode(spec, pol, seed)
            rows[name].append((r.success, r.reward))
        before = len(log.errors)
        run_episode(spec, log, seed)
        if first_err is None and len(log.errors) > before:
            first_err = log.errors[before]
    ref = np.array(rows["reference"]); blind = np.array(rows["partner_blind"])
    d_succ = ref[:, 0] - blind[:, 0]; d_rew = ref[:, 1] - blind[:, 1]
    out = {"index": idx, "structure": structure, "K": K, "T": T, "n_objects": n_objects, "p_obj_bad": p_obj, "n": n,
           "seconds": time.time() - t0,
           "success": {k: float(np.mean([s for s, _ in v])) for k, v in rows.items()},
           "reward": {k: float(np.mean([w for _, w in v])) for k, v in rows.items()},
           "gap_success": float(d_succ.mean()), "gap_success_se": float(d_succ.std(ddof=1) / np.sqrt(n)),
           "gap_reward": float(d_rew.mean()), "gap_reward_se": float(d_rew.std(ddof=1) / np.sqrt(n)),
           "disagreement_rate": log.disagree / max(log.decisions, 1), "decisions": log.decisions,
           "value_error_first": first_err, "value_error_mean": float(np.mean(log.errors)) if log.errors else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--out", default="results/coupling")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(i, a.n) for i in range(len(GRID))]
    with Pool(a.procs) as pool:
        for out in pool.imap_unordered(run_config, jobs):
            with open(os.path.join(a.out, f"cfg_{out['index']:02d}.json"), "w") as f:
                json.dump(out, f, indent=1)
            print("%-3s T=%d n_obj=%d p=%.2f | ref %.3f blind %.3f gap %+.3f+-%.3f | disagree %.2f | err0 %.3f | %.0fs" % (
                out["structure"], out["T"], out["n_objects"], out["p_obj_bad"], out["success"]["reference"],
                out["success"]["partner_blind"], out["gap_success"], out["gap_success_se"], out["disagreement_rate"],
                out["value_error_first"] or 0.0, out["seconds"]), flush=True)


if __name__ == "__main__":
    main()
