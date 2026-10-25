"""Learn a hypothesis-conditional value for PARTNR on the planner's own world model (ccrp/partnr_sim.py) and compare
it with the deployed v4 one-step rule, on episode shapes weighted like PARTNR's held-out episodes 50-149.

  python -m ccrp.partnr_rl --episodes 4000 --iterations 3 --eval 400 --procs 9 --out results/partnr_rl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

from .partnr_sim import WORLD, Hyps, PSpec, PValue, features, greedy_v_policy, run_episode, v4_policy

# (n_partner, n_own) -> weight: parity allocation of PARTNR episodes 50-149 (props p: partner p//2, own (p+1)//2)
EVAL_MIX = {(1, 1): 21, (1, 2): 31, (2, 2): 15, (2, 3): 23, (3, 3): 1, (3, 4): 0}
TRAIN_SPECS = [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4)]
_HYPS = {}
# Robust training: PARTNR_WORLDS names a JSON list of {world, weight} (the worlds consistent with PARTNR outcomes,
# ccrp.partnr_calib); every episode runs in one of them, drawn by weight from its seed.
WORLDS = json.load(open(os.environ["PARTNR_WORLDS"])) if os.environ.get("PARTNR_WORLDS") else None


def _set_world(seed):
    if WORLDS:
        r = np.random.default_rng(seed + 12_345)
        WORLD.update(WORLDS[int(r.choice(len(WORLDS), p=[w["weight"] for w in WORLDS]))]["world"])


def _table_repair():
    if WORLDS:
        return float(sum(w["weight"] * w["world"]["repair"] for w in WORLDS) / sum(w["weight"] for w in WORLDS))
    return WORLD["repair"]


def _hy(p, o):
    key = (p, o)
    if key not in _HYPS:
        _HYPS[key] = Hyps(PSpec(p, o))
    return _HYPS[key]


def _policy(kind, Vb):
    if kind == "v4":
        return v4_policy
    if kind == "never":
        from .partnr_sim import CONT
        return lambda sim, post, s, rng=None: CONT
    V = pickle.loads(Vb)
    return greedy_v_policy(V)


def _collect(args):
    kind, Vb, specs, seeds, eps = args
    pol = _policy(kind, Vb); X, y = [], []
    for (p, o), seed in zip(specs, seeds):
        sp = PSpec(p, o); rec = []; _set_world(seed)
        run_episode(sp, _hy(p, o), pol, seed, eps=eps, record=rec)
        for sim, st, h, g in rec:
            X.append(features(sim, st, np.array([h]))[0]); y.append(g)
    return np.array(X), np.array(y)


def _eval(args):
    kind, Vb, p, o, seeds = args
    pol = _policy(kind, Vb); out = []
    for s in seeds:
        _set_world(s); out.append(run_episode(PSpec(p, o), _hy(p, o), pol, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4000); ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--eval", type=int, default=400); ap.add_argument("--procs", type=int, default=9)
    ap.add_argument("--out", default="results/partnr_rl")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0); V = PValue(); data = []
    with Pool(a.procs) as pool:
        for it in range(a.iterations):
            kind = "v4" if it == 0 else "greedy"
            specs = [TRAIN_SPECS[i] for i in rng.integers(len(TRAIN_SPECS), size=a.episodes)]
            seeds = list(range(1_000_000 * (it + 1), 1_000_000 * (it + 1) + a.episodes))
            Vb = pickle.dumps(V) if it > 0 else None; t0 = time.time()
            parts = pool.map(_collect, [(kind, Vb, specs[j::a.procs], seeds[j::a.procs], 0.15) for j in range(a.procs)])
            X = np.concatenate([q[0] for q in parts]); y = np.concatenate([q[1] for q in parts])
            data = (data + [(X, y)])[-2:]
            V = PValue(); info = V.fit(np.concatenate([d[0] for d in data]), np.concatenate([d[1] for d in data]), seed=it)
            V.repair = _table_repair(); V.world = WORLDS or dict(WORLD)   # the world(s) it was learned in (repair enters its table)
            print(f"iteration {it} behavior={kind}: {info} [{time.time() - t0:.0f}s]", flush=True)
        pickle.dump(V, open(os.path.join(a.out, "value_partnr.pkl"), "wb"))
        Vb = pickle.dumps(V); res = {}
        for kind in ("never", "v4", "greedy"):
            res[kind] = {}
            for (p, o), w in EVAL_MIX.items():
                if w == 0:
                    continue
                seeds = list(range(a.eval))
                parts = pool.map(_eval, [(kind, Vb, p, o, seeds[j::a.procs]) for j in range(a.procs)])
                r = np.zeros(a.eval)
                for j, part in enumerate(parts):
                    r[j::a.procs] = part
                res[kind][f"{p},{o}"] = r.tolist()
    # weighted summary and paired differences against v4
    tot = sum(w for w in EVAL_MIX.values() if w)
    summary = {}
    for kind in res:
        m = sum(EVAL_MIX[tuple(map(int, k.split(",")))] * np.mean(v) for k, v in res[kind].items()) / tot
        summary[kind] = {"weighted_reward": float(m)}
        for k, v in res[kind].items():
            summary[kind][k] = float(np.mean(v))
    for kind in ("never", "greedy"):
        d_all, wts = [], []
        for k in res[kind]:
            d = np.array(res[kind][k]) - np.array(res["v4"][k])
            summary[kind][f"minus_v4_{k}"] = [float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))]
            d_all.append(d.mean()); wts.append(EVAL_MIX[tuple(map(int, k.split(",")))])
        summary[kind]["minus_v4_weighted"] = float(np.average(d_all, weights=wts))
    json.dump({"summary": summary, "raw": res}, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    for kind, s in summary.items():
        print(kind, json.dumps({k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in s.items()}))


if __name__ == "__main__":
    main()
