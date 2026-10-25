"""RL + certificate-gated best-arm deliberation on the synthetic benchmark.

  python -m ccrp.rl_experiment --out results/rl --episodes 3000 --eval 300 --procs 9
  python -m ccrp.rl_experiment --smoke            # tiny run to check that everything executes

Settings
  validation  K=6 AND, 2 object beliefs, 1 skill, horizon 4: exact multi-step planning is still feasible, so the learned
              value can be checked against the exact hypothesis-conditional planner;
  large-and   K=10 AND, 3 object beliefs, 2 skills, horizon 5 (64 hypotheses): exact planning is infeasible;
  large-mixed the same with a fixed random AND/OR formula.
Policies: never, partner-blind (one step), Team-Plan without RL (exact one step), Team-Plan with the learned value only
(greedy), full Team-Plan (learned value + certificate + best-arm rollouts), always-deep (best-arm rollouts at every
decision, no certificate), oracle (knows the hidden state, greedy on the learned value), and in the validation setting
the exact multi-step planner.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

from . import rl
from .policies import exact_policy
from .synthetic import EnvSpec, SyntheticEnv, formula_and, random_formula

POLICIES = ("never", "blind_one_step", "one_step", "rl_greedy", "teamplan", "always_deep", "oracle")


def settings():
    prior = {"goal": 0.2, "obj": 0.3, "skill": 0.2}
    return {
        # small enough for the exact multi-step planner (reference for the learned value)
        "validation": EnvSpec(random_formula(6, np.random.default_rng(1), p_and=0.5), 6, n_objects=2, n_skills=1,
                              horizon=4, base_rate=0.9, prior_bad=prior),
        # exact multi-step planning infeasible: 10 responsibilities, 64 partner hypotheses, 5 epochs
        "large-mixed": EnvSpec(random_formula(10, np.random.default_rng(3), p_and=0.5), 10, n_objects=3, n_skills=2,
                               horizon=5, base_rate=0.93, prior_bad=prior),
        "large-and6": EnvSpec(formula_and(10), 10, n_objects=3, n_skills=2, horizon=5, base_rate=0.97,
                              prior_bad={"goal": 0.1, "obj": 0.2, "skill": 0.1}),
        # the validation task with a longer horizon: evidence keeps arriving, so waiting before intervening has value
        "long": EnvSpec(random_formula(6, np.random.default_rng(1), p_and=0.5), 6, n_objects=2, n_skills=1,
                        horizon=10, base_rate=0.9, prior_bad=prior),
    }


def _exact_wrapper():
    def pol(env, post, stats, rng):
        stats.decisions += 1
        acts = [a for a in env.actions() if a.kind not in rl.INFO]
        return exact_policy(env, post, acts)
    return pol


def _full_info_wrapper():
    """Full-information reference: the exact hypothesis-conditional plan for the partner's true state."""
    from .core import Posterior
    def pol(env, post, stats, rng):
        stats.decisions += 1
        acts = [a for a in env.actions() if a.kind not in rl.INFO]
        return exact_policy(env, Posterior([env.truth], [1.0]), acts)
    return pol


def _eval_job(args):
    spec, kind, Vb, seeds, budget, width_scale, eps = args
    V = pickle.loads(Vb)
    pol = (_exact_wrapper() if kind == "exact" else _full_info_wrapper() if kind == "full_info"
           else rl.make_policy(kind, V, budget=budget, width_scale=width_scale, eps=eps))
    out = []
    for s in seeds:
        r = rl.run(spec, pol, s)
        out.append((s, r.success, r.reward, r.rollouts, r.certified, r.decisions))
    return out


def evaluate(spec, V, kinds, seeds, procs, budget, width_scale=1.0, eps=0.02):
    Vb = pickle.dumps(V)
    res = {}
    for kind in kinds:
        t0 = time.time()
        chunks = [seeds[i::procs] for i in range(procs)]
        with Pool(procs) as pool:
            parts = pool.map(_eval_job, [(spec, kind, Vb, c, budget, width_scale, eps) for c in chunks])
        rows = sorted([r for p in parts for r in p])
        res[kind] = {"seeds": [r[0] for r in rows], "success": [r[1] for r in rows], "reward": [r[2] for r in rows],
                     "rollouts": [r[3] for r in rows], "certified": [r[4] for r in rows], "decisions": [r[5] for r in rows],
                     "seconds": time.time() - t0}
    return res


def summarize(res, ref="teamplan"):
    lines = []
    ref_r = np.array(res[ref]["reward"]) if ref in res else None
    for kind, d in res.items():
        r = np.array(d["reward"]); n = len(r)
        line = {"policy": kind, "n": n, "reward": float(r.mean()), "reward_se": float(r.std(ddof=1) / np.sqrt(n)),
                "success": float(np.mean(d["success"])), "rollouts_per_episode": float(np.mean(d["rollouts"])),
                "certified_share": float(np.sum(d["certified"]) / max(np.sum(d["decisions"]), 1)),
                "seconds": d["seconds"]}
        if ref_r is not None and kind != ref:
            diff = ref_r - r
            line["teamplan_minus_this"] = float(diff.mean()); line["paired_se"] = float(diff.std(ddof=1) / np.sqrt(n))
        lines.append(line)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rl"); ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--iterations", type=int, default=3); ap.add_argument("--eval", type=int, default=300)
    ap.add_argument("--procs", type=int, default=9); ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--settings", default="validation,large-mixed,large-and6"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.episodes, a.iterations, a.eval, a.budget, a.settings = 400, 2, 27, 32, "validation,large-and6"
    os.makedirs(a.out, exist_ok=True)
    spath = os.path.join(a.out, "summary.json")          # merge with settings run earlier
    allres = json.load(open(spath)) if os.path.exists(spath) else {}
    for name in a.settings.split(","):
        spec = settings()[name]
        print(f"== {name}: K={spec.K}, hypotheses={len(spec.prior())}, horizon={spec.horizon}", flush=True)
        V, hist = rl.train_value(spec, iterations=a.iterations, episodes=a.episodes, procs=a.procs,
                                 log=lambda m: print(m, flush=True))
        cal = rl.calibrate_advantage(spec, V, n_states=60 if a.smoke else 180, ref_rollouts=16 if a.smoke else 32,
                                     procs=a.procs)
        print(f"  certificate calibration: {cal}", flush=True)
        hist.append({"calibration": cal})
        with open(os.path.join(a.out, f"value_{name}.pkl"), "wb") as f:
            pickle.dump(V, f)
        seeds = list(range(a.eval))
        kinds = list(POLICIES) + (["exact"] if name == "validation" else [])
        res = evaluate(spec, V, kinds, seeds, a.procs, a.budget)
        # certificate slack sweep: how compute trades against reward (eps = 0.02 is the default above)
        for e in (0.0, 0.05):
            r2 = evaluate(spec, V, ["teamplan"], seeds, a.procs, a.budget, eps=e)
            res[f"teamplan_eps{e}"] = r2["teamplan"]
        summ = summarize(res)
        allres[name] = {"training": hist, "summary": summ}
        for s in summ:
            extra = (f" | TP-this {s['teamplan_minus_this']:+.3f}+-{s['paired_se']:.3f}" if "teamplan_minus_this" in s else "")
            print(f"  {s['policy']:16s} reward {s['reward']:.3f}+-{s['reward_se']:.3f} success {s['success']:.3f} "
                  f"rollouts/ep {s['rollouts_per_episode']:7.1f} certified {s['certified_share']:.2f}{extra} [{s['seconds']:.0f}s]", flush=True)
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(allres, f, indent=1)
        with open(os.path.join(a.out, f"raw_{name}.json"), "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
