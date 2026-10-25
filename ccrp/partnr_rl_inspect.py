"""What does the learned PARTNR value do differently from the v4 rule? Action counts and timing in the model."""
from __future__ import annotations

import pickle
from collections import Counter

import numpy as np

from .partnr_sim import Hyps, PSpec, PartnrSim, greedy_v_policy, v4_policy


def trace(sp, hy, pol, seeds):
    kinds, times, by_cause = Counter(), {}, Counter()
    for s in seeds:
        sim = PartnrSim(sp, hy, s); post = sim.reset(); c0 = sim.truth
        cause = ("drop" if hy.goal[c0] >= 0 else "") + ("obj" if sp.n_obj and hy.obj[c0].any() else "") + \
                ("skill" if hy.skill[c0] else "") + ("room" if hy.room[c0] else "") or "fine"
        while not sim.done():
            a = pol(sim, post, sim.s)
            if a.kind != "continue":
                k = a.kind if a.kind != "take" else ("take_all" if len(a.target) > 1 else "take_one")
                kinds[k] += 1; times.setdefault(k, []).append(sim.s.t); by_cause[(cause, k)] += 1
            post = sim.step(a)
    n = len(seeds)
    return {k: (round(v / n, 2), round(float(np.mean(times[k])), 1)) for k, v in kinds.items()}, by_cause


if __name__ == "__main__":
    V = pickle.load(open("results/partnr_rl/value_partnr.pkl", "rb"))
    for p, o in [(1, 2), (2, 3)]:
        sp = PSpec(p, o); hy = Hyps(sp); seeds = range(400)
        for name, pol in (("v4", v4_policy), ("learned", greedy_v_policy(V))):
            k, bc = trace(sp, hy, pol, seeds)
            print(f"P={p} own={o} {name:8s} per-episode count (mean epoch): {k}")
            print("    by true cause:", {f"{c}/{a}": round(v / 400, 2) for (c, a), v in sorted(bc.items()) if v >= 8})
