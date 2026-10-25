"""Value-quality sweep: when does Bayesian optimization over rollouts pay?

The confidence check's width is calibrated against rollouts, so a worse learned value gets a wider width, more
unresolved decisions and more rollouts.  This sweep trains V on fewer simulated episodes and compares, on the same
seeded episodes as rl_experiment, the greedy learned value against Team-Plan (confidence check + BO).

  python -m ccrp.rl_sweep --setting long --episodes 100,300,1000 --iterations 3 --eval 300 --procs 9
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from . import rl
from .rl_experiment import evaluate, settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="long"); ap.add_argument("--out", default="results/rl")
    ap.add_argument("--episodes", default="100,300,1000"); ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--eval", type=int, default=300); ap.add_argument("--procs", type=int, default=9)
    ap.add_argument("--budget", type=int, default=64); ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--safe", action="store_true", help="high-confidence switch rule for BO")
    ap.add_argument("--reuse", action="store_true", help="evaluate saved value_<setting>_n<N>.pkl instead of training")
    a = ap.parse_args()
    os.environ["TP_BO_SAFE"] = "1" if a.safe else "0"          # spawned evaluation workers read it at import
    rl.ROLLOUT["mode"] = "belief"; rl.DELIBERATION["rule"] = "bayes"; rl.DELIBERATION["safe"] = a.safe
    spec = settings()[a.setting]; seeds = list(range(a.eval))
    path = os.path.join(a.out, f"sweep_{a.setting}{'_safe' if a.safe else ''}.json")
    out = json.load(open(path)) if os.path.exists(path) else []
    for n in [int(x) for x in a.episodes.split(",")]:
        vpath = os.path.join(a.out, f"value_{a.setting}_n{n}.pkl" if n != 3000 else f"value_{a.setting}.pkl")
        if a.reuse and os.path.exists(vpath):
            V = pickle.load(open(vpath, "rb")); hist = [{"mae": float("nan")}]; cal = {"adv_width": V.adv_width}
        else:
            V, hist = rl.train_value(spec, iterations=a.iterations, episodes=n, procs=a.procs, seed=101,
                                     log=lambda m: print(m, flush=True))
            cal = rl.calibrate_advantage(spec, V, n_states=180, ref_rollouts=32, procs=a.procs)
            pickle.dump(V, open(vpath, "wb"))
        res = evaluate(spec, V, ["rl_greedy", "teamplan"], seeds, a.procs, a.budget, eps=a.eps)
        g = np.array(res["rl_greedy"]["reward"]); t = np.array(res["teamplan"]["reward"]); d = t - g
        row = {"episodes": n, "iterations": a.iterations, "mae": hist[-1]["mae"], "adv_width": cal["adv_width"],
               "greedy": float(g.mean()), "teamplan": float(t.mean()),
               "teamplan_minus_greedy": float(d.mean()), "paired_se": float(d.std(ddof=1) / np.sqrt(len(d))),
               "rollouts_per_episode": float(np.mean(res["teamplan"]["rollouts"])),
               "confident_share": float(np.sum(res["teamplan"]["certified"]) / max(np.sum(res["teamplan"]["decisions"]), 1)),
               "raw": {k: v["reward"] for k, v in res.items()}}
        out = [r for r in out if r["episodes"] != n] + [row]
        json.dump(out, open(path, "w"), indent=1)
        print(f"n={n}: MAE {row['mae']:.3f} width {row['adv_width']:.3f} | greedy {row['greedy']:.3f} "
              f"Team-Plan {row['teamplan']:.3f} diff {row['teamplan_minus_greedy']:+.3f}+-{row['paired_se']:.3f} "
              f"rollouts/ep {row['rollouts_per_episode']:.0f} confident {row['confident_share']:.2f}", flush=True)


if __name__ == "__main__":
    main()
