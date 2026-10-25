"""Team-Plan + RL in PARTNR: decisions from a hypothesis-conditional value learned on the planner's own world model
(ccrp/partnr_sim.py, trained by ccrp/partnr_rl.py) instead of the one-step table.

The live planner state (posterior over the model's hypotheses, done / taken-over flags, the partner's 'Done', budgets,
the simulator-step clock) is mapped onto the simulator's state; the actions the planner may take are scored with the
learned value (repairs mix repaired and unrepaired hypotheses, queries by their value of sample information) under
the posterior, and the best is returned as the model's Action.  Episodes the simulator does not cover (no partner
responsibility, more than 6, or more than 4 own ones) fall back to the v4 one-step decision.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ccrp.core import Action, CONTINUE
from ccrp.partnr_sim import (CONT, Hyps, PAction, PSpec, PState, PartnrSim, greedy_v_policy, value_table, voi)

_HYPS: Dict[Tuple[int, int], Hyps] = {}
_V = {}


def _value(path: str):
    if path not in _V:
        with open(path, "rb") as f:
            _V[path] = pickle.load(f)
    return _V[path]


def default_value_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "value_partnr.pkl")


def sim_view(planner) -> Optional[Tuple[PartnrSim, np.ndarray, Dict[str, Any]]]:
    """(simulator positioned at the planner's state, posterior over the simulator's hypotheses, index maps)."""
    m = planner.model
    P, O = len(m.partner), len(m.own)
    if P == 0 or P > 6 or O > 4:
        return None
    key = (P, O)
    if key not in _HYPS:
        _HYPS[key] = Hyps(PSpec(P, O))
    hy = _HYPS[key]; sp = hy.sp
    pos = {r.index: j for j, r in enumerate(m.partner)}
    obj_pos = {}
    for o in m.hyp_objects:
        j = next((j for j, r in enumerate(m.partner) if o in r.objects), None)
        if j is not None and j < sp.n_obj:
            obj_pos[o] = j
    # posterior: model hypothesis -> simulator hypothesis index
    lookup = {}
    for i in range(hy.H):
        lookup[(int(hy.goal[i]), tuple(int(x) for x in hy.obj[i]), int(hy.skill[i]), int(hy.room[i]))] = i
    post = np.zeros(hy.H)
    for h, p in zip(m.posterior.hyps, m.posterior.probs):
        d = dict(h.components)                      # components absent from the model (e.g. no room cause) are good
        g = d.get("goal"); gi = -1 if g in (None, "good") else pos.get(int(str(g).split(":")[1]), -1)
        ob = [0] * sp.n_obj
        for o, j in obj_pos.items():
            if d.get(f"obj:{o}") == "bad":
                ob[j] = 1
        k = (gi, tuple(ob), int(d.get("skill") == "bad"), int(d.get("room") == "bad"))
        post[lookup[k]] += p
    post = post / post.sum() if post.sum() > 0 else hy.prior.copy()
    gap = max(int(getattr(planner, "min_gap_steps", 400)), 1)
    t = int((planner.iter - int(getattr(planner, "min_iter_before", 600))) // gap)
    t = min(max(t, 0), sp.T - 1)
    used = set()
    for u in getattr(planner, "used_repairs", set()):
        if u == "goal" or u == "room":
            used.add(u)
        elif str(u).startswith("obj:") and str(u)[4:] in obj_pos:
            used.add(f"obj{obj_pos[str(u)[4:]]}")
    s = PState(t, np.array([r.done for r in m.partner], bool), np.array([r.taken_over for r in m.partner], bool),
               np.zeros(P, int), m.partner_quit_step is not None, sum(1 for r in m.own if not r.done),
               int(planner.interventions), int(planner.n_messages), used, 0.0, bool(getattr(m, "_untouched_flagged", False)))
    sim = PartnrSim(sp, hy, 0); sim.s = s; sim.post = post
    return sim, post, {"pos": pos, "obj_pos": obj_pos}


def to_sim_action(a: Action, maps: Dict[str, Any]) -> Optional[PAction]:
    pos, obj_pos = maps["pos"], maps["obj_pos"]
    if a.kind == "continue":
        return CONT
    if a.kind == "takeover":
        tg = a.target if isinstance(a.target, tuple) else (a.target,)
        return PAction("take", tuple(sorted(pos[k] for k in tg))) if all(k in pos for k in tg) else None
    if a.kind == "clarify":
        return PAction("clarify", "goal")
    if a.kind == "align":
        if a.target == "room":
            return PAction("align", "room")
        o = str(a.target)[4:] if str(a.target).startswith("obj:") else None
        return PAction("align", f"obj{obj_pos[o]}") if o in obj_pos else None
    if a.kind == "query" and a.target in pos:
        return PAction("query", pos[a.target])
    return None


def rl_decide(planner, acts: List[Action], path: Optional[str] = None) -> Optional[Tuple[Action, Dict[str, Any]]]:
    """Best of the planner's candidate actions under the learned value, or None to fall back to the v4 decision."""
    view = sim_view(planner)
    if view is None:
        return None
    sim, post, maps = view
    if path and not os.path.isabs(path):            # e.g. data/value_partnr_v2.pkl, inside this package
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    V = _value(path or default_value_path())
    pairs = [(a, to_sim_action(a, maps)) for a in acts]
    pairs = [(a, b) for a, b in pairs if b is not None]
    phys = [(a, b) for a, b in pairs if b.kind != "query"]
    if not any(b.kind == "continue" for _, b in phys):
        phys = [(CONTINUE, CONT)] + phys
    T = value_table(sim, V, sim.s, [b for _, b in phys])
    mix = T @ post
    vals = [(float(v), a) for (a, _), v in zip(phys, mix)]
    vals += [(voi(sim, post, T, b), a) for a, b in pairs if b.kind == "query"]
    best_v, best = max(vals, key=lambda x: x[0])
    info = {"mode": "ccrp_rl", "epoch": int(sim.s.t), "value": best_v,
            "mixture": {str(a): round(v, 4) for v, a in vals}}
    return best, info
