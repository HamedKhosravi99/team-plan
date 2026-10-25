"""Calibrate the true world of the PARTNR simulator (partnr_sim.WORLD) to PARTNR outcomes.

The planner's own model (its rates, likelihoods and the features of the learned value) stays fixed; only the world the
episodes run in changes.  Targets are the task-success rates of the deployed one-step rule (v4) and of the first
learned value (RL v1) under the goal and skill injections on PARTNR episodes 20-49 (runs ext7b_v4, ext7b_sota,
ext7b_sota2); the held-out episodes 50-79 are not used.  Each candidate world is scored by the squared z-distance
between simulated and observed success (common random numbers across candidates).

  python -m ccrp.partnr_calib --n 200 --refine 1000 --top 8 --procs 9 --out results/partnr_calib
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

from . import partnr_sim as ps
from .partnr_rl import EVAL_MIX

TARGETS = {("v4", "goal"): (0.493, 0.058), ("v4", "skill"): (0.500, 0.071),
           ("rl1", "goal"): (0.633, 0.089), ("rl1", "skill"): (0.400, 0.112),
           # share of episodes still deciding at epoch 9 (step 4200), goal and skill pooled
           ("v4", "late"): (0.40, 0.047), ("rl1", "late"): (0.32, 0.073),
           # share of episodes with a takeover
           ("v4", "goal_take"): (0.37, 0.06), ("v4", "skill_take"): (0.50, 0.08),
           ("rl1", "goal_take"): (0.27, 0.09), ("rl1", "skill_take"): (0.33, 0.13)}
LATE = 9
GRID = {"T_h": [19], "rate_scale": [0.75, 0.9], "persist": [0.1, 0.5, 1.0], "eta": [0.6, 0.75], "qs": [0.3],
        "own_rate": [0.3], "skill_relief": [0.0, 0.5, 1.0], "repair": [0.2, 0.35, 0.5, 0.85], "q0": [0.03]}
V1_PATH = os.path.join(os.path.dirname(__file__), "..", "partnr", "habitat_llm_ext", "data", "value_partnr.pkl")
_HY, _V = {}, {}
SHAPES = [k for k, w in EVAL_MIX.items() if w]
WEIGHTS = np.array([EVAL_MIX[k] for k in SHAPES], float) / sum(EVAL_MIX[k] for k in SHAPES)


def _hy(p, o):
    if (p, o) not in _HY:
        _HY[(p, o)] = ps.Hyps(ps.PSpec(p, o))
    return _HY[(p, o)]


def _policy(name):
    if name == "v4":
        return ps.v4_policy
    if name not in _V:
        _V[name] = pickle.load(open(V1_PATH if name == "rl1" else name, "rb"))
    return ps.greedy_v_policy(_V[name])


def _truth(hy, inj, rng):
    mask = hy.goal >= 0 if inj == "goal" else hy.skill == 1      # the injected cause, other components from the prior
    pr = np.where(mask, hy.prior, 0.0); pr /= pr.sum()
    return int(rng.choice(hy.H, p=pr))


def episode(pol, inj, seed):
    rng = np.random.default_rng(seed)
    p, o = SHAPES[int(rng.choice(len(SHAPES), p=WEIGHTS))]
    sp, hy = ps.PSpec(p, o), _hy(p, o)
    sim = ps.PartnrSim(sp, hy, seed); post = sim.reset(truth=_truth(hy, inj, rng)); prng = np.random.default_rng(seed + 7)
    n = 0; took = False
    while not sim.done():
        a = pol(sim, post, sim.s, prng); took |= a.kind == "take"
        post = sim.step(a); n += 1
    return sim.final_success(), float(n > LATE), float(took)


def _job(args):
    world, name, inj, seeds = args
    ps.WORLD.update(world); pol = _policy(name)
    return [episode(pol, inj, s) for s in seeds]


def score(pool, world, n, procs, cells=TARGETS):
    out = {}; late = {}
    for (name, inj) in [k for k in cells if k[1] in ("goal", "skill")]:
        seeds = list(range(500_000, 500_000 + n))
        parts = pool.map(_job, [(world, name, inj, seeds[j::procs]) for j in range(procs)])
        rows = [x for q in parts for x in q]
        out[(name, inj)] = float(np.mean([r[0] for r in rows])); late.setdefault(name, []).extend(r[1] for r in rows)
        out[(name, inj + "_take")] = float(np.mean([r[2] for r in rows]))
    for name, v in late.items():
        out[(name, "late")] = float(np.mean(v))
    loss = sum(((out[k] - m) / se) ** 2 for k, (m, se) in TARGETS.items() if k in out)
    return out, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--refine", type=int, default=1000)
    ap.add_argument("--top", type=int, default=8); ap.add_argument("--procs", type=int, default=9)
    ap.add_argument("--out", default="results/partnr_calib")
    ap.add_argument("--grid", default="", help="JSON dict overriding GRID entries (e.g. to extend past an edge)")
    ap.add_argument("--targets", default="", help="comma-separated target kinds to fit (e.g. goal,skill,late)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    grid = dict(GRID); grid.update(json.loads(a.grid) if a.grid else {})
    if a.targets:
        keep = set(a.targets.split(","))
        for k in [k for k in TARGETS if k[1] not in keep]:
            del TARGETS[k]
    keys = list(grid); cands = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]
    default = {k: ps.WORLD[k] for k in keys}
    rows = []
    with Pool(a.procs) as pool:
        t0 = time.time()
        pred, loss = score(pool, default, a.refine, a.procs)
        print(f"planner's own model as the world: {fmt(pred)} loss={loss:.2f} [{time.time() - t0:.0f}s]", flush=True)
        rows.append({"world": default, "pred": {f"{k[0]}/{k[1]}": v for k, v in pred.items()}, "loss": loss, "n": a.refine})
        coarse = []
        for i, w in enumerate(cands):
            pred, loss = score(pool, w, a.n, a.procs); coarse.append((loss, w, pred))
            if i % 12 == 0:
                print(f"[{i + 1}/{len(cands)}] {w} {fmt(pred)} loss={loss:.2f} [{time.time() - t0:.0f}s]", flush=True)
        coarse.sort(key=lambda x: x[0])
        for loss0, w, _ in coarse[:a.top]:
            pred, loss = score(pool, w, a.refine, a.procs)
            rows.append({"world": w, "pred": {f"{k[0]}/{k[1]}": v for k, v in pred.items()}, "loss": loss,
                         "coarse_loss": loss0, "n": a.refine})
            print(f"refined {w} {fmt(pred)} loss={loss:.2f} (coarse {loss0:.2f})", flush=True)
    best = min(rows[1:], key=lambda r: r["loss"])
    json.dump({"targets": {f"{k[0]}/{k[1]}": v for k, v in TARGETS.items()}, "rows": rows, "best": best,
               "coarse": [{"world": w, "loss": l} for l, w, _ in coarse]}, open(os.path.join(a.out, "calib.json"), "w"), indent=1)
    print("best", json.dumps(best))


def fmt(pred):
    return " ".join(f"{k[0]}/{k[1]}={v:.3f}" for k, v in pred.items())


if __name__ == "__main__":
    main()
