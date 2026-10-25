"""Re-evaluate deliberating policies with a saved learned value (same seeds as rl_experiment, so results stay paired).

  python -m ccrp.rl_eval --setting validation --out results/rl --rollouts belief --eval 200 --procs 9
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from . import rl
from .rl_experiment import evaluate, settings, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True); ap.add_argument("--out", default="results/rl")
    ap.add_argument("--rollouts", default="belief"); ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--procs", type=int, default=9); ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--eps", default="0.02,0.05")
    ap.add_argument("--rule", default="bayes"); ap.add_argument("--baselines", default="")
    ap.add_argument("--recalibrate", action="store_true"); ap.add_argument("--no-deep", action="store_true")
    a = ap.parse_args()
    rl.ROLLOUT["mode"] = a.rollouts; rl.DELIBERATION["rule"] = a.rule
    spec = settings()[a.setting]
    with open(os.path.join(a.out, f"value_{a.setting}.pkl"), "rb") as f:
        V = pickle.load(f)
    if a.recalibrate or V.adv_width is None:
        cal = rl.calibrate_advantage(spec, V, n_states=180, ref_rollouts=32, procs=a.procs)
    else:
        cal = {"adv_width": V.adv_width, "reused": True}
    print(f"{a.setting}: calibration: {cal}", flush=True)
    seeds = list(range(a.eval))
    raw_path = os.path.join(a.out, f"raw_{a.setting}.json")
    base = json.load(open(raw_path)) if os.path.exists(raw_path) else {}

    def save(name, d):                      # merge each policy as soon as it finishes
        base[name] = d
        json.dump(base, open(raw_path, "w"))
        r = np.array(d["reward"])
        print(f"  done {name}: reward {r.mean():.3f} rollouts/ep {np.mean(d['rollouts']):.1f} [{d['seconds']:.0f}s]", flush=True)

    for k in [k for k in a.baselines.split(",") if k]:
        save(k, evaluate(spec, V, [k], seeds, a.procs, a.budget)[k])
    eps_list = [float(x) for x in a.eps.split(",") if x]
    for e in eps_list:
        save(f"teamplan_{a.rule}_eps{e}", evaluate(spec, V, ["teamplan"], seeds, a.procs, a.budget, eps=e)["teamplan"])
    if not a.no_deep:
        save(f"always_deep_{a.rule}", evaluate(spec, V, ["always_deep"], seeds, a.procs, a.budget)["always_deep"])
    ref = f"teamplan_{a.rule}_eps{eps_list[0]}" if eps_list else ("teamplan" if "teamplan" in base else
                                                                   f"teamplan_{a.rule}_eps0.02")
    summ = summarize(base, ref=ref)
    for s in summ:
        extra = (f" | ref-this {s['teamplan_minus_this']:+.3f}+-{s['paired_se']:.3f}" if "teamplan_minus_this" in s else "")
        print(f"  {s['policy']:24s} reward {s['reward']:.3f}+-{s['reward_se']:.3f} success {s['success']:.3f} "
              f"rollouts/ep {s['rollouts_per_episode']:7.1f} certified {s['certified_share']:.2f}{extra}", flush=True)
    json.dump({"setting": a.setting, "rollouts": a.rollouts, "calibration": cal, "summary": summ},
              open(os.path.join(a.out, f"summary_{a.setting}_{a.rule}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
