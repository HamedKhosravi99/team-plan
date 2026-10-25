"""Multi-step base policy for the synthetic environment.

The planner's table needs Q-factors under an explicit base policy (method document, section 3).  A base
policy that never intervenes again ("continue forever") makes the exact planner one-step greedy, and in
conjunctive tasks it undervalues complementary rescues.  Here the base policy is the *hypothesis-conditional
optimal plan*: given a hypothesis c and the remaining epochs n, it chooses the set of interventions (at most
one per epoch) that maximizes expected final team value minus costs.  Q(a | c) is the value of taking a now
and then following that plan; the advantage is A(a | c) = Q(a | c) - Q(continue | c).

The partner-blind counterpart plans with the posterior-mean success rates (per-responsibility marginal
beliefs, no correlation across responsibilities): the natural "per-task belief" baseline.

Values are exact (enumeration over intervention subsets and repair outcomes) and vectorized over subsets.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import Action, Hypothesis, Posterior
from .synthetic import EnvSpec, EpisodeState, Node


def reliability_batch(node: Node, P: np.ndarray) -> np.ndarray:
    """Reliability of a read-once formula for a batch of rate vectors P [B, K]."""
    if node.op == "leaf":
        return P[:, node.leaf]
    vals = [reliability_batch(c, P) for c in node.children]
    if node.op == "and":
        out = np.ones(P.shape[0])
        for v in vals:
            out = out * v
        return out
    out = np.ones(P.shape[0])
    for v in vals:
        out = out * (1.0 - v)
    return 1.0 - out


class BasePolicyPlanner:
    """Exact hypothesis-conditional plans for the synthetic environment."""

    def __init__(self, spec: EnvSpec, state: EpisodeState, epochs_left: int):
        self.spec = spec
        self.state = state
        self.n = epochs_left
        self._cache: Dict[Any, float] = {}
        K = spec.K
        self._unaided = [k for k in range(K) if not state.taken_over[k]]
        # all takeover subsets of the unaided responsibilities, as 0/1 masks [S, K]
        masks = []
        for r in range(len(self._unaided) + 1):
            for T in itertools.combinations(self._unaided, r):
                m = np.zeros(K); m[list(T)] = 1.0
                masks.append(m)
        self._masks = np.array(masks) if masks else np.zeros((1, K))
        self._mask_sizes = self._masks.sum(axis=1)
        self._take_cost = spec.costs["takeover"]

    # rates ------------------------------------------------------------------------
    def base_rates(self, h: Hypothesis) -> np.ndarray:
        sp, st = self.spec, self.state
        p = np.array([sp.unaided_rate(k, h) for k in range(sp.K)])
        for k in range(sp.K):
            if st.taken_over[k]:
                p[k] = sp.eta_takeover
            elif st.helped[k]:
                p[k] = p[k] + (1 - p[k]) * sp.help_boost
        return p

    def _value_given_rates(self, p: np.ndarray, n: int, extra_cost: float = 0.0) -> float:
        """Best expected value with at most n takeovers (no repairs) from rate vector p."""
        ok = self._mask_sizes <= n
        M = self._masks[ok]
        P = M * self.spec.eta_takeover + (1 - M) * p[None, :]
        vals = reliability_batch(self.spec.formula, P) - self._take_cost * self._mask_sizes[ok] - extra_cost
        return float(vals.max())

    # exact, hypothesis-conditional --------------------------------------------------
    def repairable(self, h: Hypothesis) -> List[str]:
        return [name for name in h.names if h.get(name) == "bad"]

    def value_star(self, h: Hypothesis, n: int) -> float:
        """V*(c, n): optimal expected value over takeover subsets and repair subsets, at most n actions."""
        key = ("V", h, n)
        if key in self._cache:
            return self._cache[key]
        if n <= 0:
            v = float(self.spec.formula.reliability(self.base_rates(h)))
            self._cache[key] = v
            return v
        best = -np.inf
        reps = self.repairable(h)
        r = self.spec.repair_success
        for m in range(0, min(len(reps), n) + 1):
            for R in itertools.combinations(reps, m):
                cost_R = sum(self.spec.costs["clarify" if c == "goal" else "align"] for c in R)
                budget = n - m
                # expectation over repair outcomes (independent successes)
                total = 0.0
                for outcome in itertools.product((0, 1), repeat=m):
                    pr = 1.0; h2 = h
                    for comp, s in zip(R, outcome):
                        pr *= r if s else (1 - r)
                        if s:
                            h2 = h2.with_component(comp, "good")
                    total += pr * self._value_given_rates(self.base_rates(h2), budget)
                best = max(best, total - cost_R)
        self._cache[key] = best
        return best

    def q_value(self, a: Action, h: Hypothesis) -> float:
        """Q(a | c): take a now, then follow the optimal plan for the remaining n-1 epochs."""
        sp = self.spec; n1 = self.n - 1
        if a.kind == "continue":
            return self.value_star(h, n1)
        if a.kind == "takeover":
            st2 = _copy_state(self.state); st2.taken_over[a.target] = True
            return BasePolicyPlanner(sp, st2, n1).value_star(h, n1) - a.cost
        if a.kind == "help":
            st2 = _copy_state(self.state); st2.helped[a.target] = True
            return BasePolicyPlanner(sp, st2, n1).value_star(h, n1) - a.cost
        if a.kind in ("clarify", "align"):
            if h.get(a.target) == "bad":
                h2 = h.with_component(a.target, "good"); r = sp.repair_success
                return r * self.value_star(h2, n1) + (1 - r) * self.value_star(h, n1) - a.cost
            return self.value_star(h, n1) - a.cost
        raise ValueError(a.kind)

    def advantage(self, a: Action, h: Hypothesis) -> float:
        return self.q_value(a, h) - self.q_value(Action("continue"), h)

    def succ_prob_under_continue(self, k: int, h: Hypothesis) -> float:
        """P(k succeeds | continue, base policy): approximated by the base rate unless the optimal plan takes k over."""
        return float(self.base_rates(h)[k])

    # partner-blind: posterior-mean rates ----------------------------------------------
    def mean_rates(self, post: Posterior, h_transform=None) -> np.ndarray:
        tot = np.zeros(self.spec.K)
        for h, p in zip(post.hyps, post.probs):
            tot += p * self.base_rates(h_transform(h) if h_transform else h)
        return tot

    def blind_q(self, a: Action, post: Posterior) -> float:
        """Q under per-responsibility marginal beliefs (no correlation)."""
        sp = self.spec; n1 = self.n - 1
        if a.kind == "continue":
            return self._blind_value_star(post, n1)
        if a.kind in ("takeover", "help"):
            st2 = _copy_state(self.state)
            (st2.taken_over if a.kind == "takeover" else st2.helped)[a.target] = True
            return BasePolicyPlanner(sp, st2, n1)._blind_value_star(post, n1) - a.cost
        if a.kind in ("clarify", "align"):
            post2 = _posterior_after_repair(post, a.target, sp.repair_success)
            return self._blind_value_star(post2, n1) - a.cost
        raise ValueError(a.kind)

    def _blind_value_star(self, post: Posterior, n: int) -> float:
        """Plan with mean rates; repairs modeled as deterministic improvements of the mean rates."""
        key = ("B", tuple(post.hyps), tuple(np.round(post.probs, 10)), n)
        if key in self._cache:
            return self._cache[key]
        if n <= 0:
            v = float(self.spec.formula.reliability(self.mean_rates(post)))
            self._cache[key] = v
            return v
        comps = post.hyps[0].names
        best = -np.inf
        for m in range(0, min(len(comps), n) + 1):
            for R in itertools.combinations(comps, m):
                post2 = post
                cost_R = 0.0
                for c in R:
                    post2 = _posterior_after_repair(post2, c, self.spec.repair_success)
                    cost_R += self.spec.costs["clarify" if c == "goal" else "align"]
                best = max(best, self._value_given_rates(self.mean_rates(post2), n - m) - cost_R)
        self._cache[key] = best
        return best

    def blind_advantage(self, a: Action, post: Posterior) -> float:
        return self.blind_q(a, post) - self.blind_q(Action("continue"), post)


def _copy_state(st: EpisodeState) -> EpisodeState:
    return EpisodeState(list(st.taken_over), list(st.helped), st.t, st.total_cost, [])


def _posterior_after_repair(post: Posterior, comp: str, r: float) -> Posterior:
    new: Dict[Hypothesis, float] = {}
    for h, p in zip(post.hyps, post.probs):
        if h.get(comp) == "bad":
            g = h.with_component(comp, "good")
            new[g] = new.get(g, 0.0) + p * r
            new[h] = new.get(h, 0.0) + p * (1 - r)
        else:
            new[h] = new.get(h, 0.0) + p
    hyps = list(new)
    return Posterior(hyps, np.array([new[h] for h in hyps]))
