"""Generative simulator of the PARTNR responsibility model (the world model of the deployed Team-Plan v4 planner),
used to learn a hypothesis-conditional value for PARTNR by fitted Monte-Carlo policy iteration and to run model
rollouts for Bayesian optimization.

It mirrors habitat_llm_ext/responsibility_model.py with the v4 configuration (conf/planner/ccrp_robot_planner.yaml):
  hypotheses   goal in {good, drop:i}, a belief per partner object (at most 3), skill, shared room belief;
               prior bad 0.2 (goal, split over the partner's responsibilities), 0.1 per object, 0.15 skill, 0.1 room;
  rates        0.85, x0.3 for a wrong belief (object or room), x0.35 for a broken skill, 0.05 if the goal drops it;
               taken over 0.75; own responsibilities 0.9; the task is a conjunction;
  actions      continue; take over one partner responsibility or all open ones (cost 0.08 each, plus 0.06 per own
               responsibility still open; only once the robot's own part is done or half the budget is gone; at most
               two takeovers unless the partner has quit); clarify the goal, align an object or the room (cost 0.03,
               repair succeeds with 0.85, once each); query a responsibility (cost 0.03, answer truthful with 0.8);
               at most two messages; one decision per epoch (400 simulator steps), epochs from step 600 to 8000;
  evidence     the planner's own likelihood ratios: the partner acting on a responsibility's object (x1/3 for 'dropped'
               or 'wrong belief', at most three action kinds per responsibility), unexplained Pick/Place failures (x4
               for a broken skill) and successes (x1/2), responsibilities untouched at mid-budget (x3 dropped, x2 wrong
               belief); the partner says Done when every responsibility it believes it has is done, and prematurely with
               probability 0.03 per epoch (0.08 with a broken skill), as LLM partners do.
After the partner's Done the robot gets one more decision; continuing with its own part finished ends the
episode (the v4 planner stops monitoring then).
Partner progress: an open responsibility completes with a constant hazard per epoch such that it completes by the end
with the model's rate; once the partner has quit nothing more of its part gets done.
"""
from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

COST_TAKE, COST_MSG, OPP_OWN, ETA, REPAIR, TRUTHFUL = 0.08, 0.03, 0.06, 0.75, 0.85, 0.8
BASE, GOAL_DROP, WRONG, SKILL, OWN = 0.85, 0.05, 0.3, 0.35, 0.9
PRIOR = {"goal": 0.2, "obj": 0.1, "skill": 0.15, "room": 0.1}
LR_MENTION, LR_FAIL, LR_SUCC, LR_UNT_DROP, LR_UNT_OBJ = 3.0, 4.0, 2.0, 3.0, 2.0
P_MENTION_FINE, P_MENTION_BAD, P_SKILL_EVENT, P_FAIL_BAD, P_FAIL_GOOD, P_OWN_DONE = 0.30, 0.10, 0.5, 0.6, 0.15, 0.3
MAX_TAKES, MAX_MSGS, MAX_KINDS = 2, 2, 3
P_QUIT_EARLY, P_QUIT_EARLY_SKILL = 0.03, 0.08     # LLM partners also say Done with work open (more often when failing)
# The true world, which may differ from the planner's model above (calibrated to PARTNR outcomes by ccrp.partnr_calib):
#   rate_scale  multiplies the partner's completion rates;  q0, qs  premature-Done hazards (skill fine / broken);
#   persist     multiplies those hazards once the robot has sent any message (an addressed partner keeps trying);
#   eta, repair  takeover and repair success;  T_h  epochs over which the partner does its work (None: the planner's
#   horizon; real partners finish or stop much sooner);  own_rate  per-epoch completion of the robot's own work (takeovers
#   wait for it);  skill_relief  after a message, a partner with a broken skill retries: its skill factor moves this far
#   towards 1.  The defaults reproduce the planner's model exactly.
WORLD = {"rate_scale": 1.0, "q0": P_QUIT_EARLY, "qs": P_QUIT_EARLY_SKILL, "persist": 1.0, "eta": ETA, "repair": REPAIR,
         "T_h": None, "own_rate": P_OWN_DONE, "skill_relief": 0.0}
if os.environ.get("PARTNR_WORLD"):                  # spawned worker processes inherit the calibrated world this way
    WORLD.update(json.loads(os.environ["PARTNR_WORLD"]))


@dataclass
class PSpec:
    n_partner: int
    n_own: int
    T: int = 19                     # epochs: steps 600, 1000, ..., 7800
    room_known: float = 0.7         # P(the room message can be sent / is complete)

    @property
    def n_obj(self) -> int:
        return min(self.n_partner, 3)


class Hyps:
    """All hypotheses of a PSpec as arrays: goal (-1 = good, else the dropped index), obj bits, skill, room."""

    def __init__(self, sp: PSpec):
        self.sp = sp
        P, O = sp.n_partner, sp.n_obj
        rows = []
        for g in [-1] + list(range(P)):
            for ob in itertools.product((0, 1), repeat=O):
                for sk in (0, 1):
                    for rm in (0, 1):
                        rows.append((g, ob, sk, rm))
        self.H = len(rows)
        self.goal = np.array([r[0] for r in rows]); self.obj = np.array([r[1] for r in rows]).reshape(self.H, O)
        self.skill = np.array([r[2] for r in rows]); self.room = np.array([r[3] for r in rows])
        pg = np.where(self.goal < 0, 1 - PRIOR["goal"], PRIOR["goal"] / max(P, 1))
        po = np.prod(np.where(self.obj == 1, PRIOR["obj"], 1 - PRIOR["obj"]), axis=1) if O else np.ones(self.H)
        self.prior = pg * po * np.where(self.skill == 1, PRIOR["skill"], 1 - PRIOR["skill"]) * \
            np.where(self.room == 1, PRIOR["room"], 1 - PRIOR["room"])
        self.prior /= self.prior.sum()
        # wrong belief about responsibility r's object: its object (r < 3) or the shared room
        wb = np.zeros((self.H, P), bool)
        for r in range(P):
            wb[:, r] = (self.room == 1) | ((self.obj[:, r] == 1) if r < O else False)
        self.wrong = wb
        self.dropped = np.stack([self.goal == r for r in range(P)], axis=1) if P else np.zeros((self.H, 0), bool)
        rate = np.full((self.H, P), BASE)
        rate = np.where(self.dropped, GOAL_DROP, rate)
        rate = rate * np.where(wb, WRONG, 1.0) * np.where(self.skill[:, None] == 1, SKILL, 1.0)
        self.rate0 = rate                                   # P(done by the end | h), no intervention
        self.fine = ~(self.dropped | wb)                    # acts on r's object at the 'fine' rate
        self.bad_for = self.dropped | wb | (self.skill[:, None] == 1)
        # index of the hypothesis with one component repaired
        key = {(int(g), tuple(int(x) for x in o), int(s), int(m)): i
               for i, (g, o, s, m) in enumerate(zip(self.goal, self.obj, self.skill, self.room))}
        def rep(i, comp):
            g, o, s, m = int(self.goal[i]), [int(x) for x in self.obj[i]], int(self.skill[i]), int(self.room[i])
            if comp == "goal": g = -1
            elif comp == "room": m = 0
            else: o[int(comp[3:])] = 0
            return key[(g, tuple(o), s, m)]
        self.repaired = {c: np.array([rep(i, c) for i in range(self.H)])
                         for c in ["goal", "room"] + [f"obj{k}" for k in range(O)]}
        self.is_bad_comp = {"goal": self.goal >= 0, "room": self.room == 1}
        for k in range(O):
            self.is_bad_comp[f"obj{k}"] = self.obj[:, k] == 1


@dataclass
class PState:
    t: int = 0
    done: np.ndarray = None
    taken: np.ndarray = None
    kinds: np.ndarray = None          # action kinds seen per partner responsibility (planner's mention updates)
    quit: bool = False
    own_left: int = 0
    takes: int = 0
    msgs: int = 0
    used: set = field(default_factory=set)
    cost: float = 0.0
    flagged: bool = False             # mid-budget evidence applied


@dataclass(frozen=True)
class PAction:
    kind: str                          # continue | take | clarify | align | query
    target: object = None              # take: tuple of indices; align: 'obj<k>' | 'room'; query: index
    def __str__(self): return f"{self.kind}({self.target})" if self.target is not None else self.kind


CONT = PAction("continue")


class PartnrSim:
    def __init__(self, sp: PSpec, hy: Optional[Hyps] = None, seed: int = 0):
        self.sp = sp; self.hy = hy or Hyps(sp); self.rng = np.random.default_rng(seed)

    # ---- episode ------------------------------------------------------------------------------------------
    def reset(self, truth: Optional[int] = None):
        sp, hy = self.sp, self.hy
        self.truth = int(self.rng.choice(hy.H, p=hy.prior)) if truth is None else truth
        self.s = PState(0, np.zeros(sp.n_partner, bool), np.zeros(sp.n_partner, bool), np.zeros(sp.n_partner, int),
                        False, sp.n_own)
        self.post = hy.prior.copy()
        return self.post

    def hazard(self, h: np.ndarray) -> np.ndarray:
        """Per-epoch completion hazard of each partner responsibility under hypothesis rows h (H' x P)."""
        return 1.0 - np.power(1.0 - np.clip(self.hy.rate0[h], 0, 0.999999), 1.0 / self.sp.T)

    def remaining(self, s: PState, h) -> np.ndarray:
        """P(r is done at the end | h, state): 1 done, ETA taken, 0 open after quitting, else remaining hazard."""
        h = np.atleast_1d(h)
        left = self.sp.T - s.t
        q = 1.0 - np.power(1.0 - self.hazard(h), max(left, 0))
        if s.quit:
            q = np.zeros_like(q)
        q = np.where(s.done[None, :], 1.0, q)
        return np.where(s.taken[None, :], ETA, q)

    def stop_value(self, s: PState, h) -> np.ndarray:
        return np.prod(self.remaining(s, h), axis=1) * OWN ** self.sp.n_own

    def actions(self, s: Optional[PState] = None, queries: bool = True) -> List[PAction]:
        s = s or self.s; sp = self.sp
        open_ = [r for r in range(sp.n_partner) if not (s.done[r] or s.taken[r])]
        acts = [CONT]
        can_take = (s.own_left == 0 or s.t >= sp.T // 2) and (s.takes < MAX_TAKES or s.quit)
        if can_take:
            acts += [PAction("take", (r,)) for r in open_]
            if len(open_) >= 2:
                acts.append(PAction("take", tuple(open_)))
        if not s.quit and s.msgs < MAX_MSGS and open_:
            if "goal" not in s.used:
                acts.append(PAction("clarify", "goal"))
            for k in range(sp.n_obj):
                if f"obj{k}" not in s.used and (k in open_):
                    acts.append(PAction("align", f"obj{k}"))
            if "room" not in s.used:
                acts.append(PAction("align", "room"))
            if queries:
                acts += [PAction("query", r) for r in open_]
        return acts

    def cost(self, a: PAction, s: PState) -> float:
        if a.kind == "take":
            return COST_TAKE * len(a.target) + OPP_OWN * s.own_left
        return COST_MSG if a.kind in ("clarify", "align", "query") else 0.0

    # ---- dynamics (true world) and the planner's posterior updates --------------------------------------------
    def step(self, a: PAction):
        sp, hy, s, rng = self.sp, self.hy, self.s, self.rng
        s.cost += self.cost(a, s)
        if a.kind == "take":
            s.taken[list(a.target)] = True
            if not s.quit:
                s.takes += 1
        elif a.kind in ("clarify", "align"):
            comp = "goal" if a.kind == "clarify" else a.target
            s.msgs += 1; s.used.add(comp)
            r = REPAIR * (sp.room_known if comp == "room" else 1.0)
            r_world = WORLD["repair"] * (sp.room_known if comp == "room" else 1.0)
            if hy.is_bad_comp[comp][self.truth] and rng.random() < r_world:
                self.truth = int(hy.repaired[comp][self.truth])
            self._post_repair(comp, r)          # the planner updates with its model's repair probability
        elif a.kind == "continue" and s.quit and s.own_left == 0:
            # the v4 planner stops monitoring once the partner has said Done: continuing ends the episode
            s.t = sp.T
            return self.post
        elif a.kind == "query":
            s.msgs += 1
            bad = hy.bad_for[self.truth, a.target]
            y_trouble = (rng.random() < TRUTHFUL) == bool(bad)
            lik = np.where(hy.bad_for[:, a.target], TRUTHFUL, 1 - TRUTHFUL)
            self._update(lik if y_trouble else 1 - lik)
        self._advance()
        return self.post

    def _update(self, lik: np.ndarray) -> None:
        w = self.post * lik
        self.post = w / w.sum()

    def _post_repair(self, comp: str, r: float) -> None:
        hy = self.hy; bad = hy.is_bad_comp[comp]
        new = self.post * np.where(bad, 1 - r, 1.0)
        np.add.at(new, hy.repaired[comp][bad], self.post[bad] * r)
        self.post = new / new.sum()

    def _advance(self) -> None:
        """One epoch of the partner's work and the evidence the planner receives."""
        sp, hy, s, rng, c = self.sp, self.hy, self.s, self.rng, self.truth
        if not s.quit:
            if WORLD["rate_scale"] == 1.0 and not WORLD["T_h"] and not WORLD["skill_relief"]:
                haz = self.hazard(np.array([c]))[0]
            else:
                rw = self.hy.rate0[c] * WORLD["rate_scale"]
                if WORLD["skill_relief"] and hy.skill[c] and s.msgs > 0:
                    rw = rw * (SKILL + WORLD["skill_relief"] * (1.0 - SKILL)) / SKILL
                rw = np.clip(rw, 0, 0.999999)
                haz = 1.0 - np.power(1.0 - rw, 1.0 / (WORLD["T_h"] or sp.T))
            for r in range(sp.n_partner):
                if s.done[r] or s.taken[r]:
                    continue
                # the partner acts on r's object (one more action kind) -> planner's mention update
                if s.kinds[r] < MAX_KINDS and rng.random() < (P_MENTION_FINE if hy.fine[c, r] else P_MENTION_BAD):
                    s.kinds[r] += 1
                    self._update(np.where(~hy.fine[:, r], 1.0 / LR_MENTION, 1.0))
                if rng.random() < haz[r]:
                    s.done[r] = True
                    if s.kinds[r] == 0:
                        s.kinds[r] = 1; self._update(np.where(~hy.fine[:, r], 1.0 / LR_MENTION, 1.0))
            if rng.random() < P_SKILL_EVENT:          # a Pick/Place ends
                fail = rng.random() < (P_FAIL_BAD if hy.skill[c] else P_FAIL_GOOD)
                self._update(np.where(hy.skill == 1, LR_FAIL if fail else 1.0 / LR_SUCC, 1.0))
            believed = [r for r in range(sp.n_partner) if hy.goal[c] != r and not s.taken[r]]
            if all(s.done[r] for r in believed):
                s.quit = True                          # 'Done': whatever it does not believe it has stays open
            elif rng.random() < (WORLD["qs"] if hy.skill[c] else WORLD["q0"]) * (WORLD["persist"] if s.msgs > 0 else 1.0):
                s.quit = True                          # premature 'Done' (gives up, or thinks it has finished)
        if s.own_left > 0:
            s.own_left -= int(rng.binomial(s.own_left, WORLD["own_rate"]))
        s.t += 1
        if not s.flagged and s.t >= sp.T // 2:
            s.flagged = True
            for r in range(sp.n_partner):
                if s.kinds[r] == 0 and not (s.done[r] or s.taken[r]):
                    self._update(np.where(hy.dropped[:, r], LR_UNT_DROP, 1.0) * np.where(hy.wrong[:, r], LR_UNT_OBJ, 1.0))

    def done(self) -> bool:
        return self.s.t >= self.sp.T

    def final_value(self) -> float:
        """Expected success at the end (Rao-Blackwellized over takeover outcomes and own work) minus costs."""
        return self.final_success() - self.s.cost

    def final_success(self) -> float:
        """Expected task success at the end in the true world (takeovers succeed with WORLD['eta'])."""
        s = self.s
        if WORLD["eta"] == ETA and not WORLD["T_h"]:
            return float(self.stop_value(s, self.truth)[0])
        q = np.where(s.done, 1.0, np.where(s.taken, WORLD["eta"], 0.0))   # at the end, open work stays open
        return float(np.prod(q) * OWN ** self.sp.n_own)


# ------------------------------------------------------------------------------------------------------------------
# value features and the v4 one-step rule
# ------------------------------------------------------------------------------------------------------------------
PMAX = 6


def features(sim: PartnrSim, s: PState, h: np.ndarray) -> np.ndarray:
    """Hypothesis-conditional features for rows h (permutation-invariant over responsibilities, padded to PMAX)."""
    sp, hy = sim.sp, sim.hy
    q = sim.remaining(s, h)                                           # H' x P
    open_ = ~(s.done | s.taken)
    R = np.prod(q, axis=1) * OWN ** sp.n_own
    qo = np.where(open_[None, :], q, 2.0)                            # sort the open ones, pad with 2 (absent)
    srt = np.sort(qo, axis=1)[:, :PMAX]
    if srt.shape[1] < PMAX:
        srt = np.concatenate([srt, np.full((len(h), PMAX - srt.shape[1]), 2.0)], axis=1)
    prod_other = R[:, None] / np.clip(q, 1e-6, None)
    room = np.sum(np.where(open_[None, :], (ETA - q) * prod_other, 0.0), axis=1)
    n_open = open_.sum(); left = (sp.T - s.t) / sp.T
    comp = np.stack([hy.goal[h] >= 0, (hy.obj[h].sum(axis=1) if sp.n_obj else np.zeros(len(h))), hy.skill[h],
                     hy.room[h]], axis=1).astype(float)
    avail = np.array([("goal" not in s.used), sum(f"obj{k}" not in s.used for k in range(sp.n_obj)),
                      ("room" not in s.used), MAX_MSGS - s.msgs, (MAX_TAKES - s.takes) if not s.quit else 9,
                      float(s.quit), float(s.own_left == 0 or s.t >= sp.T // 2), s.own_left, sp.n_own, n_open,
                      int(s.taken.sum()), int(s.done.sum()), left], float)
    return np.concatenate([R[:, None], srt, room[:, None], comp, np.tile(avail, (len(h), 1))], axis=1)


def one_step_table(sim: PartnrSim, s: PState, acts: Sequence[PAction]) -> np.ndarray:
    """The deployed v4 rule (tier E, one step): advantages per action and hypothesis with the model's static rates
    (0.85-based, no time), A(take S) = (eta^|S| - prod p) * L_S - cost, repairs r * (cont(h') - cont(h)) - cost."""
    sp, hy = sim.sp, sim.hy
    p = np.where(s.done[None, :], 1.0, np.where(s.taken[None, :], ETA, 0.0 if s.quit else hy.rate0))  # H x P
    own = OWN ** sp.n_own
    cont = np.prod(p, axis=1) * own
    T = np.zeros((len(acts), hy.H))
    for i, a in enumerate(acts):
        if a.kind == "take":
            S = list(a.target); pS = np.prod(p[:, S], axis=1)
            others = [j for j in range(sp.n_partner) if j not in S]
            L = (np.prod(p[:, others], axis=1) if others else np.ones(hy.H)) * own   # Birnbaum importance of S
            T[i] = (ETA ** len(S) - pS) * L - sim.cost(a, s)
        elif a.kind in ("clarify", "align"):
            comp = "goal" if a.kind == "clarify" else a.target
            r = REPAIR * (sp.room_known if comp == "room" else 1.0)
            T[i] = np.where(hy.is_bad_comp[comp], r * (cont[hy.repaired[comp]] - cont), 0.0) - COST_MSG
        elif a.kind == "query":
            T[i] = -COST_MSG
    return T


def voi(sim: PartnrSim, post: np.ndarray, T: np.ndarray, a: PAction) -> float:
    """Value of sample information of a query from a table (see rl.query_value)."""
    lik = np.where(sim.hy.bad_for[:, a.target], TRUTHFUL, 1 - TRUTHFUL)
    base = float(np.max(T @ post)); val = 0.0
    for l in (lik, 1 - lik):
        py = float(post @ l)
        if py > 1e-12:
            val += py * float(np.max(T @ (post * l / py)))
    return val - base - COST_MSG


def v4_policy(sim: PartnrSim, post: np.ndarray, s: PState, rng=None) -> PAction:
    acts = sim.actions(s)
    phys = [a for a in acts if a.kind != "query"]
    T = one_step_table(sim, s, phys); mix = T @ post
    vals = list(mix) + [voi(sim, post, T, a) for a in acts if a.kind == "query"]
    allacts = phys + [a for a in acts if a.kind == "query"]
    i = int(np.argmax(vals))
    return allacts[i] if vals[i] > 0 else CONT


# ------------------------------------------------------------------------------------------------------------------
# learned value: V(z, h) = R + f(features), fitted Monte-Carlo policy iteration with the true hypothesis
# ------------------------------------------------------------------------------------------------------------------
class PValue:
    def __init__(self):
        self.reg = None; self.adv_width = None

    def value(self, sim: PartnrSim, s: PState, h: np.ndarray) -> np.ndarray:
        X = features(sim, s, h)
        if s.t >= sim.sp.T or self.reg is None:
            return X[:, 0] - s.cost * 0
        return X[:, 0] + self.reg.predict(X)

    def fit(self, X, y, seed=0):
        from sklearn.ensemble import HistGradientBoostingRegressor
        rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); cut = int(0.9 * len(y))
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=seed)
        reg.fit(X[idx[:cut]], y[idx[:cut]] - X[idx[:cut], 0])
        res = np.abs(y[idx[cut:]] - (X[idx[cut:], 0] + reg.predict(X[idx[cut:]])))
        res0 = np.abs(y[idx[cut:]] - X[idx[cut:], 0])
        self.reg = reg
        return {"n": int(len(y)), "mae": float(res.mean()), "mae_stop_now": float(res0.mean())}


def _clone_state(s: PState) -> PState:
    return PState(s.t, s.done.copy(), s.taken.copy(), s.kinds.copy(), s.quit, s.own_left, s.takes, s.msgs,
                  set(s.used), s.cost, s.flagged)


def _apply_deterministic(sim: PartnrSim, s: PState, a: PAction) -> PState:
    """State after the robot's action, before the partner's epoch (for the value look-ahead)."""
    n = _clone_state(s)
    if a.kind == "continue" and n.quit and n.own_left == 0:
        n.t = sim.sp.T                                   # the episode ends (see PartnrSim.step)
        return n
    if a.kind == "take":
        n.taken[list(a.target)] = True
        if not n.quit:
            n.takes += 1
    elif a.kind in ("clarify", "align", "query"):
        n.msgs += 1
        if a.kind != "query":
            n.used.add("goal" if a.kind == "clarify" else a.target)
    n.t += 1
    return n


def value_table(sim: PartnrSim, V: PValue, s: PState, acts: Sequence[PAction]) -> np.ndarray:
    """T[a, h] = V(after a, h') - cost(a) - V(after continue, h), repairs mixing repaired / unrepaired h."""
    hy = sim.hy; H = np.arange(hy.H)
    base_s = _apply_deterministic(sim, s, CONT); base = V.value(sim, base_s, H)
    T = np.zeros((len(acts), hy.H))
    for i, a in enumerate(acts):
        if a.kind == "continue":
            continue
        n = _apply_deterministic(sim, s, a)
        if a.kind == "take":
            T[i] = V.value(sim, n, H) - sim.cost(a, s) - base
        elif a.kind in ("clarify", "align"):
            comp = "goal" if a.kind == "clarify" else a.target
            # a value learned in a calibrated world carries that world's repair success (V.repair)
            r = getattr(V, "repair", REPAIR) * (sim.sp.room_known if comp == "room" else 1.0)
            vn = V.value(sim, n, H); vrep = vn[hy.repaired[comp]]
            T[i] = np.where(hy.is_bad_comp[comp], r * vrep + (1 - r) * vn, vn) - COST_MSG - base
        else:                       # query: its value is information, scored separately (voi)
            T[i] = V.value(sim, n, H) - COST_MSG - base
    return T


def greedy_v_policy(V: PValue, queries: bool = True):
    def pol(sim: PartnrSim, post: np.ndarray, s: PState, rng=None) -> PAction:
        acts = sim.actions(s, queries=queries)
        phys = [a for a in acts if a.kind != "query"]
        T = value_table(sim, V, s, phys); mix = T @ post
        vals = list(mix) + [voi(sim, post, T, a) for a in acts if a.kind == "query"]
        allacts = phys + [a for a in acts if a.kind == "query"]
        return allacts[int(np.argmax(vals))]
    return pol


def rollout_policy(name: str, V: Optional[PValue] = None):
    return v4_policy if name == "v4" else greedy_v_policy(V)


def run_episode(sp: PSpec, hy: Hyps, policy, seed: int, eps: float = 0.0, record=None) -> float:
    sim = PartnrSim(sp, hy, seed); post = sim.reset(); rng = np.random.default_rng(seed + 7)
    visits = []
    while not sim.done():
        s = sim.s
        if record is not None:
            visits.append((_clone_state(s), sim.truth))
        acts = sim.actions(s)
        a = acts[int(rng.integers(len(acts)))] if rng.random() < eps else policy(sim, post, s, rng)
        post = sim.step(a)
    G = sim.final_value()
    if record is not None:
        final_cost = sim.s.cost
        for st, h in visits:
            # return-to-go from this epoch: final success minus the costs paid from here on, given the truth then
            record.append((sim, st, h, G + st.cost))
    return G


SPECS = [PSpec(p, o) for p in (1, 2, 3, 4, 5, 6) for o in (0, 1, 2, 3)]
