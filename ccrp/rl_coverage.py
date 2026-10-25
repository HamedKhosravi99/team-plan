"""Empirical coverage of the confidence check's widths on fresh held-out states.

At one random epoch of each held-out episode (greedy V behavior, seeds disjoint from calibration), the table advantage
of the leading actions is compared with a high-precision paired-rollout reference (`ref` pairs per action).
Reports action-wise coverage |table - ref| <= w, simultaneous coverage over the compared actions, how often the check
fires, and how often a fired check picks an action within eps of the best reference.

  python -m ccrp.rl_coverage --settings validation,long,large-and6 --states 150 --ref 64 --procs 9
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np

from . import rl
from .rl_experiment import settings
from .synthetic import SyntheticEnv


def _job(args):
    name, seeds, ref, top, eps = args
    spec = settings()[name]
    V = pickle.load(open(f"results/rl/value_{name}.pkl", "rb"))
    pol = rl.make_policy("rl_greedy", V); out = []
    for seed in seeds:
        env = SyntheticEnv(spec, seed); post = env.reset(); rng = np.random.default_rng(seed + 11)
        stats = rl.DeliberationStats(); t_pick = int(rng.integers(spec.horizon))
        while not env.done():
            if env.state.t == t_pick:
                acts = rl.candidate_actions(env); T = rl.action_table(env, post, V, acts); mix = T @ post.probs
                i0 = next(i for i, a in enumerate(acts) if a.kind == "continue")
                order = [i for i in np.argsort(-mix) if i != i0][:top]
                refs = {i0: 0.0}
                for i in order:
                    d = [rl.paired_rollout(env, acts[i], acts[i0], post.hyps[int(rng.choice(len(post.probs), p=post.probs))], V, rng)
                         for _ in range(ref)]
                    refs[i] = float(np.mean(d))
                w = V.adv_width; lead = int(np.argmax(mix))
                fired = all(i == lead or mix[i] + w <= mix[lead] - w + eps for i in range(len(acts)))
                cov = {i: abs(mix[i] - refs[i]) <= w for i in refs if i != i0}
                best_ref = max(refs.values())
                out.append({"cov": list(cov.values()), "fired": bool(fired), "lead_in_ref": lead in refs,
                            "lead_eps_opt": (refs.get(lead, -9) >= best_ref - eps) if lead in refs else None})
                break
            post, _ = env.step(pol(env, post, stats, rng))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="validation,long,large-and6"); ap.add_argument("--states", type=int, default=150)
    ap.add_argument("--ref", type=int, default=64); ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.02); ap.add_argument("--procs", type=int, default=9)
    a = ap.parse_args(); res = {}
    with Pool(a.procs) as pool:
        for name in a.settings.split(","):
            seeds = list(range(7_000_000, 7_000_000 + a.states))
            rows = [r for p in pool.map(_job, [(name, seeds[j::a.procs], a.ref, a.top, a.eps) for j in range(a.procs)]) for r in p]
            cov = np.concatenate([r["cov"] for r in rows]); sim = np.mean([all(r["cov"]) for r in rows])
            fired = [r for r in rows if r["fired"]]
            ok = [r["lead_eps_opt"] for r in fired if r["lead_eps_opt"] is not None]
            res[name] = {"states": len(rows), "actionwise_coverage": float(cov.mean()), "simultaneous_coverage": float(sim),
                         "check_fired": float(len(fired) / len(rows)),
                         "fired_eps_optimal": float(np.mean(ok)) if ok else None, "n_fired": len(fired)}
            print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open("results/rl/coverage.json", "w"), indent=1)


if __name__ == "__main__":
    main()
