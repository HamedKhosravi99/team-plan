"""Synthetic responsibility environment with a hidden partner state (common cause).

* Task value: a read-once AND/OR formula over K partner responsibilities (coherent structure).
* Partner hidden components: goal understanding, per-object world-model beliefs, per-skill competence.
  Each responsibility depends on one object and one skill; a bad component multiplies its success rate.
* Epochs: each epoch our agent takes one action and then observes one noisy progress signal per unaided
  responsibility (conditionally independent given the hidden state).  Taking over a responsibility removes
  its signal (evidence censoring).
* Actions: continue, takeover(k), help(k), clarify('goal'), align('obj_m'), query(k).
* Final reward: f(X) minus action costs, with X realized at the end from the (intervened) success rates.

`ExactModel` implements `ConditionalModel` in closed form (Lemma 1 with Birnbaum importances), so the
planner's table is exact and the reference planner needs no rollouts.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import (Action, CONTINUE, ConditionalModel, Hypothesis, Posterior)

INFORMATIONAL_KINDS = ("query", "inspect")


# ----------------------------------------------------------------------------
# read-once formulas
# ----------------------------------------------------------------------------
@dataclass
class Node:
    op: str                       # 'leaf' | 'and' | 'or'
    leaf: Optional[int] = None
    children: List["Node"] = field(default_factory=list)

    def leaves(self) -> List[int]:
        if self.op == "leaf":
            return [self.leaf]
        return [l for c in self.children for l in c.leaves()]

    def reliability(self, p: np.ndarray) -> float:
        if self.op == "leaf":
            return float(p[self.leaf])
        vals = [c.reliability(p) for c in self.children]
        if self.op == "and":
            return float(np.prod(vals))
        return float(1.0 - np.prod([1.0 - v for v in vals]))

    def evaluate(self, x: np.ndarray) -> int:
        if self.op == "leaf":
            return int(x[self.leaf])
        vals = [c.evaluate(x) for c in self.children]
        return int(all(vals)) if self.op == "and" else int(any(vals))

    def parent_op(self, k: int, parent: Optional[str] = None) -> Optional[str]:
        if self.op == "leaf":
            return parent if self.leaf == k else None
        for c in self.children:
            r = c.parent_op(k, self.op)
            if r is not None:
                return r
        return None

    def describe(self) -> str:
        if self.op == "leaf":
            return f"x{self.leaf}"
        sep = " & " if self.op == "and" else " | "
        return "(" + sep.join(c.describe() for c in self.children) + ")"


def birnbaum(f: Node, p: np.ndarray, k: int) -> float:
    p1 = p.copy(); p1[k] = 1.0
    p0 = p.copy(); p0[k] = 0.0
    return f.reliability(p1) - f.reliability(p0)


def formula_and(K: int) -> Node:
    return Node("and", children=[Node("leaf", leaf=k) for k in range(K)])


def formula_or(K: int) -> Node:
    return Node("or", children=[Node("leaf", leaf=k) for k in range(K)])


def formula_chain(K: int) -> Node:
    """Prerequisite chain x0 -> x1 -> ... -> x_{K-1}: the team succeeds iff all succeed (series)."""
    return formula_and(K)


def random_formula(K: int, rng: np.random.Generator, p_and: float = 0.5, max_arity: int = 3) -> Node:
    leaves = list(rng.permutation(K))
    nodes: List[Node] = [Node("leaf", leaf=int(k)) for k in leaves]
    while len(nodes) > 1:
        m = int(min(len(nodes), rng.integers(2, max_arity + 1)))
        group, nodes = nodes[:m], nodes[m:]
        op = "and" if rng.random() < p_and else "or"
        nodes.append(Node(op, children=group))
        rng.shuffle(nodes)
    return nodes[0]


# ----------------------------------------------------------------------------
# environment specification
# ----------------------------------------------------------------------------
@dataclass
class EnvSpec:
    formula: Node
    K: int
    n_objects: int = 2
    n_skills: int = 1
    obj_of: Optional[List[int]] = None          # responsibility -> object index
    skill_of: Optional[List[int]] = None        # responsibility -> skill index
    base_rate: float = 0.85                     # unaided success rate with all components good
    goal_factor: float = 0.25                   # multiplier when the goal is misread
    obj_factor: float = 0.2                     # multiplier when the object belief is wrong
    skill_factor: float = 0.3                   # multiplier when the skill is broken
    eta_takeover: float = 0.97                  # success rate under takeover
    help_boost: float = 0.5                     # help: p -> p + (1-p)*boost
    repair_success: float = 0.9                 # clarify/align success probability
    prior_bad: Dict[str, float] = field(default_factory=lambda: {"goal": 0.3, "obj": 0.3, "skill": 0.2})
    signal_stall_good: float = 0.15             # P(stall signal | responsibility's components all good)
    signal_stall_bad: float = 0.55              # P(stall signal | some component bad)
    query_noise: float = 0.2                    # P(wrong answer) for a query about k
    horizon: int = 6
    costs: Dict[str, float] = field(default_factory=lambda: {"takeover": 0.08, "help": 0.05, "clarify": 0.06,
                                                              "align": 0.05, "query": 0.02})

    def __post_init__(self) -> None:
        if self.obj_of is None:
            self.obj_of = [k % self.n_objects for k in range(self.K)]
        if self.skill_of is None:
            self.skill_of = [k % self.n_skills for k in range(self.K)]

    # hypothesis space -----------------------------------------------------------
    def component_names(self) -> List[str]:
        return ["goal"] + [f"obj{m}" for m in range(self.n_objects)] + [f"skill{s}" for s in range(self.n_skills)]

    def prior(self) -> Posterior:
        names = self.component_names()
        hyps, probs = [], []
        for bits in itertools.product((0, 1), repeat=len(names)):
            d = {}; pr = 1.0
            for name, b in zip(names, bits):
                key = "goal" if name == "goal" else ("obj" if name.startswith("obj") else "skill")
                pb = self.prior_bad[key]
                d[name] = "bad" if b else "good"
                pr *= pb if b else (1 - pb)
            hyps.append(Hypothesis.make(**d)); probs.append(pr)
        return Posterior(hyps, probs)

    def components_of(self, k: int) -> List[str]:
        return ["goal", f"obj{self.obj_of[k]}", f"skill{self.skill_of[k]}"]

    def unaided_rate(self, k: int, h: Hypothesis) -> float:
        p = self.base_rate
        if h.get("goal") == "bad":
            p *= self.goal_factor
        if h.get(f"obj{self.obj_of[k]}") == "bad":
            p *= self.obj_factor
        if h.get(f"skill{self.skill_of[k]}") == "bad":
            p *= self.skill_factor
        return p

    def any_bad(self, k: int, h: Hypothesis) -> bool:
        return any(h.get(c) == "bad" for c in self.components_of(k))


# ----------------------------------------------------------------------------
# episode state
# ----------------------------------------------------------------------------
@dataclass
class EpisodeState:
    taken_over: List[bool]
    helped: List[bool]
    t: int = 0
    total_cost: float = 0.0
    log: List[Dict[str, Any]] = field(default_factory=list)


class SyntheticEnv:
    def __init__(self, spec: EnvSpec, seed: int = 0):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.truth: Optional[Hypothesis] = None
        self.state: Optional[EpisodeState] = None
        self.posterior: Optional[Posterior] = None

    # lifecycle -----------------------------------------------------------------
    def reset(self, truth: Optional[Hypothesis] = None) -> Posterior:
        prior = self.spec.prior()
        if truth is None:
            idx = self.rng.choice(len(prior), p=prior.probs)
            truth = prior.hyps[idx]
        self.truth = truth
        self.state = EpisodeState([False] * self.spec.K, [False] * self.spec.K)
        self.posterior = prior
        # common random numbers: outcomes depend on the final state, not on the order or number of draws
        K, T = self.spec.K, self.spec.horizon + 1
        self.u_final = self.rng.random(K)
        self.u_signal = self.rng.random((T, K))
        self.u_answer = self.rng.random((T, K))
        self.u_repair = {name: self.rng.random(T) for name in self.spec.component_names()}
        return self.posterior

    def actions(self) -> List[Action]:
        sp, st = self.spec, self.state
        acts = [CONTINUE]
        for k in range(sp.K):
            if not st.taken_over[k]:
                acts.append(Action("takeover", k, sp.costs["takeover"]))
                if not st.helped[k]:
                    acts.append(Action("help", k, sp.costs["help"]))
        acts.append(Action("clarify", "goal", sp.costs["clarify"]))
        for m in range(sp.n_objects):
            acts.append(Action("align", f"obj{m}", sp.costs["align"]))
        for k in range(sp.K):
            if not st.taken_over[k]:
                acts.append(Action("query", k, sp.costs["query"]))
        return acts

    def model(self, epochs_left: Optional[int] = None) -> "ExactModel":
        n = self.spec.horizon - self.state.t if epochs_left is None else epochs_left
        return ExactModel(self.spec, self.state, max(n, 1))

    # dynamics ------------------------------------------------------------------
    def step(self, a: Action) -> Tuple[Posterior, Dict[str, Any]]:
        sp, st, rng = self.spec, self.state, self.rng
        info: Dict[str, Any] = {"action": str(a), "t": st.t}
        st.total_cost += a.cost
        if a.kind == "takeover":
            st.taken_over[a.target] = True
        elif a.kind == "help":
            st.helped[a.target] = True
        elif a.kind in ("clarify", "align"):
            if self.truth.get(a.target) == "bad" and self.u_repair[a.target][st.t] < sp.repair_success:
                self.truth = self.truth.with_component(a.target, "good")
                info["repaired"] = True
            # the agent knows the repair attempt happened; posterior moves mass accordingly
            self._posterior_after_repair(a.target)
        elif a.kind == "query":
            y = self._answer(a.target, self.truth, self.u_answer[st.t, a.target])
            info["answer"] = y
            model = self.model()
            self.posterior.update(lambda h, a=a, y=y: math.log(max(model.answer_model(a, h).get(y, 0.0), 1e-12)))
        # progress signals for unaided responsibilities
        signals = {}
        for k in range(sp.K):
            if st.taken_over[k]:
                continue
            stall = self.u_signal[st.t, k] < (sp.signal_stall_bad if sp.any_bad(k, self.truth) else sp.signal_stall_good)
            signals[k] = stall
        self.posterior.update(lambda h: sum(
            math.log(sp.signal_stall_bad if sp.any_bad(k, h) else sp.signal_stall_good) if stall
            else math.log(1 - (sp.signal_stall_bad if sp.any_bad(k, h) else sp.signal_stall_good))
            for k, stall in signals.items()))
        info["signals"] = signals
        st.t += 1
        st.log.append(info)
        return self.posterior, info

    def _posterior_after_repair(self, comp: str) -> None:
        """Repair succeeds with probability r on 'bad' hypotheses: move mass bad -> good."""
        r = self.spec.repair_success
        post = self.posterior
        new: Dict[Hypothesis, float] = {}
        for h, p in zip(post.hyps, post.probs):
            if h.get(comp) == "bad":
                new[h.with_component(comp, "good")] = new.get(h.with_component(comp, "good"), 0.0) + p * r
                new[h] = new.get(h, 0.0) + p * (1 - r)
            else:
                new[h] = new.get(h, 0.0) + p
        hyps = list(new); probs = np.array([new[h] for h in hyps])
        self.posterior = Posterior(hyps, probs)

    def _answer(self, k: int, truth: Hypothesis, u: float) -> str:
        bad = self.spec.any_bad(k, truth)
        truthful = "trouble" if bad else "fine"
        if u < self.spec.query_noise:
            return "fine" if truthful == "trouble" else "trouble"
        return truthful

    def done(self) -> bool:
        return self.state.t >= self.spec.horizon

    def finish(self) -> Dict[str, Any]:
        """Realize final statuses and the team reward."""
        sp, st = self.spec, self.state
        p = ExactModel(sp, st, 1).rates(self.truth)
        x = (self.u_final < p).astype(int)
        success = sp.formula.evaluate(x)
        return {"success": success, "reward": success - st.total_cost, "cost": st.total_cost, "x": x.tolist()}


# ----------------------------------------------------------------------------
# exact conditional model
# ----------------------------------------------------------------------------
class ExactModel(ConditionalModel):
    """Hypothesis-conditional quantities under the multi-step base policy of planning.py.

    epochs_left = 1 recovers the one-step model (base policy = continue): uplift x Birnbaum importance.
    """

    def __init__(self, spec: EnvSpec, state: EpisodeState, epochs_left: int = 1):
        from .planning import BasePolicyPlanner
        self.spec = spec
        self.state = state
        self.n = epochs_left
        self.planner = BasePolicyPlanner(spec, state, epochs_left)

    def rates(self, h: Hypothesis) -> np.ndarray:
        sp, st = self.spec, self.state
        p = np.array([sp.unaided_rate(k, h) for k in range(sp.K)])
        for k in range(sp.K):
            if st.taken_over[k]:
                p[k] = sp.eta_takeover
            elif st.helped[k]:
                p[k] = p[k] + (1 - p[k]) * sp.help_boost
        return p

    def rate_after(self, action: Action, h: Hypothesis) -> float:
        p = self.rates(h)[action.target]
        if action.kind == "takeover":
            return self.spec.eta_takeover
        if action.kind == "help":
            return p + (1 - p) * self.spec.help_boost
        return p

    # ConditionalModel interface --------------------------------------------------
    def uplift(self, action: Action, h: Hypothesis) -> float:
        return self.rate_after(action, h) - self.rates(h)[action.target]

    def consequence(self, k: Any, h: Hypothesis) -> float:
        """Effective consequence under the base policy: A(takeover(k) | c) + cost, divided by the uplift.
        With one epoch left this is the Birnbaum importance (Lemma 1, one-step)."""
        if self.n <= 1:
            return birnbaum(self.spec.formula, self.rates(h), k)
        a = Action("takeover", k, self.spec.costs["takeover"])
        u = self.uplift(a, h)
        if abs(u) < 1e-9:
            return birnbaum(self.spec.formula, self.rates(h), k)
        return (self.planner.advantage(a, h) + a.cost) / u

    def continuation_value(self, h: Hypothesis) -> float:
        """Q(continue | c) under the base policy (= final reliability when one epoch is left)."""
        if self.n <= 1:
            return self.spec.formula.reliability(self.rates(h))
        return self.planner.q_value(CONTINUE, h)

    def repair(self, action: Action, h: Hypothesis) -> Tuple[Hypothesis, float]:
        if h.get(action.target) == "bad":
            return h.with_component(action.target, "good"), self.spec.repair_success
        return h, 0.0

    def answer_model(self, action: Action, h: Hypothesis) -> Dict[Any, float]:
        bad = self.spec.any_bad(action.target, h)
        q = self.spec.query_noise
        return {"trouble": (1 - q) if bad else q, "fine": q if bad else (1 - q)}

    def slot_cost(self, action: Action, h: Hypothesis) -> float:
        """V*(c, n) - V*(c, n-1): what one epoch of acting is worth under c (0 when one epoch is left)."""
        if self.n <= 1:
            return 0.0
        return max(self.planner.value_star(h, self.n) - self.planner.value_star(h, self.n - 1), 0.0)

    def logical_role(self, k: Any) -> Optional[str]:
        op = self.spec.formula.parent_op(k)
        return {"and": "conjunctive", "or": "disjunctive"}.get(op) if op else None

    def sign_law_applies(self, k: Any) -> bool:
        """All responsibilities share the same hidden components => success rates are one scalar function of c."""
        return self.spec.n_objects == 1 and self.spec.n_skills == 1

    # exact advantages for evaluation ---------------------------------------------
    def exact_advantage(self, action: Action, posterior: Posterior) -> float:
        """Expected advantage over continue under the posterior and the base policy (exact)."""
        if action.kind == "continue":
            return 0.0
        if action.kind in INFORMATIONAL_KINDS:
            return float("nan")
        if self.n <= 1:
            tot = 0.0
            for h, p in zip(posterior.hyps, posterior.probs):
                if action.kind in ("takeover", "help"):
                    tot += p * (self.uplift(action, h) * self.consequence(action.target, h))
                else:
                    h2, r = self.repair(action, h)
                    tot += p * r * (self.continuation_value(h2) - self.continuation_value(h))
            return tot - action.cost
        return float(sum(p * self.planner.advantage(action, h) for h, p in zip(posterior.hyps, posterior.probs)))

    def myopic_score(self, action: Action, posterior: Posterior) -> float:
        """Partner-blind score.  One step: marginal uplift x marginal consequence - cost.  Multi-step: plan with
        per-responsibility posterior-mean success rates (no correlation across responsibilities)."""
        if action.kind == "continue":
            return 0.0
        if self.n <= 1:
            if action.kind not in ("takeover", "help"):
                return self.exact_advantage(action, posterior)
            u = sum(p * self.uplift(action, h) for h, p in zip(posterior.hyps, posterior.probs))
            L = sum(p * self.consequence(action.target, h) for h, p in zip(posterior.hyps, posterior.probs))
            return u * L - action.cost
        return self.planner.blind_advantage(action, posterior)


def components_of_query(spec: EnvSpec):
    def f(a: Action) -> Sequence[str]:
        return spec.components_of(a.target) if a.kind == "query" else []
    return f
