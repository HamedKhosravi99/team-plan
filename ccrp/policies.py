"""Policies for the synthetic benchmark: the CCRP planner and the baselines of the method document (section 14)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .core import (Action, CONTINUE, CCRPPlanner, Decision, PlannerConfig, Posterior, PHYSICAL, INFORMATIONAL,
                   fill_table, certify, ladder)
from .synthetic import EnvSpec, ExactModel, SyntheticEnv, components_of_query

Policy = Callable[[SyntheticEnv, Posterior, List[Action]], Action]


@dataclass
class PolicyStats:
    deliberations: int = 0
    epochs: int = 0
    certified: int = 0


# ----------------------------------------------------------------------------
# reference and partner-blind planners
# ----------------------------------------------------------------------------
def exact_policy(env: SyntheticEnv, post: Posterior, acts: List[Action]) -> Action:
    """Reference planner: exact table (tier E) including value of information for queries."""
    model = env.model()
    planner = CCRPPlanner(model, PlannerConfig(tier="E", use_query_targeting=False), components_of_query(env.spec))
    return planner.decide(acts, post).action


def ccrp_policy(tier: str = "E", eps: float = 0.0, query_targeting: bool = True, n_sample: int = 64,
                stats: Optional[PolicyStats] = None) -> Policy:
    def pol(env: SyntheticEnv, post: Posterior, acts: List[Action]) -> Action:
        model = env.model()
        planner = CCRPPlanner(model, PlannerConfig(tier=tier, eps=eps, use_query_targeting=query_targeting,
                                                   n_sample=n_sample), components_of_query(env.spec), rng=env.rng)
        d = planner.decide(acts, post)
        if stats is not None:
            stats.epochs += 1; stats.certified += int(d.certified)
        return d.action
    return pol


def myopic_policy(env: SyntheticEnv, post: Posterior, acts: List[Action]) -> Action:
    """Partner-blind: marginal uplift x marginal consequence for physical actions; no queries."""
    model = env.model()
    cands = [a for a in acts if a.kind not in INFORMATIONAL]
    scores = [model.myopic_score(a, post) for a in cands]
    return cands[int(np.argmax(scores))]


def never_policy(env: SyntheticEnv, post: Posterior, acts: List[Action]) -> Action:
    return CONTINUE


def oracle_policy(env: SyntheticEnv, post: Posterior, acts: List[Action]) -> Action:
    """Upper bound: plans with the true hidden state (same base policy, no uncertainty)."""
    from .planning import BasePolicyPlanner
    bp = BasePolicyPlanner(env.spec, env.state, max(env.spec.horizon - env.state.t, 1))
    cands = [a for a in acts if a.kind not in INFORMATIONAL]
    scores = [bp.q_value(a, env.truth) for a in cands]
    return cands[int(np.argmax(scores))]


def no_query_exact_policy(env: SyntheticEnv, post: Posterior, acts: List[Action]) -> Action:
    model = env.model()
    cands = [a for a in acts if a.kind not in INFORMATIONAL]
    scores = [model.exact_advantage(a, post) for a in cands]
    return cands[int(np.argmax(scores))]


# ----------------------------------------------------------------------------
# gated planners: cheap myopic score unless a gate fires, then the exact planner
# ----------------------------------------------------------------------------
def coupling_matrix(spec: EnvSpec, strength: float = 0.8) -> np.ndarray:
    """Task-graph coupling for the kappa baseline: siblings under the same AND/OR node are coupled."""
    K = spec.K
    C = np.eye(K)

    def visit(node):
        if node.op == "leaf":
            return
        leaves_by_child = [c.leaves() for c in node.children]
        for i, li in enumerate(leaves_by_child):
            for j, lj in enumerate(leaves_by_child):
                if i != j:
                    for a in li:
                        for b in lj:
                            C[a, b] = max(C[a, b], strength / (len(li) * len(lj)) ** 0.5)
        for c in node.children:
            visit(c)
    visit(spec.formula)
    return C


def kappa_exposure(spec: EnvSpec, gamma: float = 0.9, H: int = 4, strength: float = 0.8) -> np.ndarray:
    C = coupling_matrix(spec, strength)
    D = np.diag(np.diag(C))
    ell = np.ones(spec.K)
    X = sum((gamma ** h) * (np.linalg.matrix_power(C, h) - np.linalg.matrix_power(D, h)) for h in range(1, H + 1))
    return X @ ell


def ambiguity(model: ExactModel, action: Action, post: Posterior) -> float:
    """Common-cause ambiguity proxy: posterior SD(uplift) x SD(consequence) (Theorem 1 radius)."""
    u = np.array([model.uplift(action, h) for h in post.hyps]); L = np.array([model.consequence(action.target, h) for h in post.hyps])
    p = post.probs
    sdu = np.sqrt(max(float((u ** 2) @ p - (u @ p) ** 2), 0.0)); sdl = np.sqrt(max(float((L ** 2) @ p - (L @ p) ** 2), 0.0))
    return sdu * sdl


def gated_policy(gate: str, threshold: float, stats: Optional[PolicyStats] = None) -> Policy:
    """gate in {'kappa', 'ambiguity', 'entropy', 'always', 'never'}: when the gate fires, use the exact planner
    (deliberation); otherwise the myopic score."""
    def pol(env: SyntheticEnv, post: Posterior, acts: List[Action]) -> Action:
        model = env.model()
        cands = [a for a in acts if a.kind not in INFORMATIONAL]
        phys = [a for a in cands if a.kind in PHYSICAL]
        if gate == "always":
            fire = True
        elif gate == "never":
            fire = False
        elif gate == "kappa":
            kap = kappa_exposure(env.spec)
            # delta = size of the initial perturbation (posterior-mean uplift), kappa = interdependence exposure
            fire = any(kap[a.target] * abs(sum(p * model.uplift(a, h) for h, p in zip(post.hyps, post.probs))) > threshold
                       for a in phys)
        elif gate == "ambiguity":
            fire = any(ambiguity(model, a, post) > threshold for a in phys)
        elif gate == "entropy":
            p = post.probs; ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
            fire = ent > threshold
        else:
            raise ValueError(gate)
        if stats is not None:
            stats.epochs += 1; stats.deliberations += int(fire)
        if fire:
            scores = [model.exact_advantage(a, post) for a in cands]
        else:
            scores = [model.myopic_score(a, post) for a in cands]
        return cands[int(np.argmax(scores))]
    return pol


# ----------------------------------------------------------------------------
# escalation ladder as a standalone policy (binary collapse on the goal component)
# ----------------------------------------------------------------------------
def ladder_policy(env: SyntheticEnv, post: Posterior, acts: List[Action]) -> Action:
    """Collapse the posterior to suspicion about the goal; for the responsibility with the largest rescue value,
    apply Proposition 10 with hypothesis-conditional quantities computed from the exact model."""
    model = env.model(); sp = env.spec
    pi_bad = post.component_mass("goal", "bad")
    good = [(h, p) for h, p in zip(post.hyps, post.probs) if h.get("goal") == "good"]
    bad = [(h, p) for h, p in zip(post.hyps, post.probs) if h.get("goal") == "bad"]

    def cond(fn, group):
        z = sum(p for _, p in group)
        return sum(p * fn(h) for h, p in group) / z if z > 0 else 0.0
    takeovers = [a for a in acts if a.kind == "takeover"]
    if not takeovers:
        return CONTINUE
    best, best_val = CONTINUE, 0.0
    fbar_gap = cond(model.continuation_value, good) - cond(model.continuation_value, bad)
    clarify = next((a for a in acts if a.kind == "clarify"), None)
    for a in takeovers:
        h_g = cond(lambda h: model.uplift(a, h) * model.consequence(a.target, h), good)
        h_b = cond(lambda h: model.uplift(a, h) * model.consequence(a.target, h), bad)
        val = pi_bad * h_b + (1 - pi_bad) * h_g - a.cost
        if val > best_val:
            best, best_val = a, val
    if clarify is not None:
        rep = pi_bad * sp.repair_success * fbar_gap - clarify.cost
        if rep > best_val:
            best, best_val = clarify, rep
    return best


# ----------------------------------------------------------------------------
# episode runner
# ----------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    success: int
    reward: float
    cost: float
    regret: float                      # sum over epochs of exact advantage(best) - exact advantage(chosen)
    actions: List[str]
    per_epoch: List[Dict[str, Any]]


def run_episode(spec: EnvSpec, policy: Policy, seed: int, truth=None) -> EpisodeResult:
    env = SyntheticEnv(spec, seed)
    post = env.reset(truth)
    regret = 0.0; actions = []; per_epoch = []
    while not env.done():
        acts = env.actions()
        model = env.model()
        a = policy(env, post, acts)
        # per-epoch regret against the exact one-step advantages (queries scored by exact VoI through the planner)
        ex = {str(b): model.exact_advantage(b, post) for b in acts if b.kind not in INFORMATIONAL}
        best = max(ex.values())
        chosen = ex.get(str(a), None)
        if chosen is None:   # informational action: value it with the exact planner's table
            planner = CCRPPlanner(model, PlannerConfig(tier="E", use_query_targeting=False), components_of_query(spec))
            d = planner.decide(acts, post)
            tab = d.table; chosen = float(tab.mixture()[tab.idx(a)])
        regret += max(best - chosen, 0.0)
        per_epoch.append({"t": env.state.t, "action": str(a), "pi_goal_bad": post.component_mass("goal", "bad"),
                          "regret": max(best - chosen, 0.0)})
        actions.append(str(a))
        post, _ = env.step(a)
    out = env.finish()
    return EpisodeResult(out["success"], out["reward"], out["cost"], regret, actions, per_epoch)
