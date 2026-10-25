"""Common-cause-certified responsibility planning (CCRP): core algorithms.

Implements Algorithms 1-6 of pseudocode.md on top of an environment-specific
`ConditionalModel`:

  * Posterior            - cutset of partner hypotheses and its posterior
  * fill_table           - action x hypothesis table of conditional advantages (Lemma 1)
  * certify              - tiers E (enumeration), M (Frechet), S (sampled, empirical Bernstein)
  * deliberate           - stratified best-action identification / hypothesis refinement
  * query_targets        - coupling shares (Theorem 8) and diagnosticity (Proposition 9)
  * ladder               - escalation ladder thresholds (Proposition 10)
  * informational_price  - Proposition 11

Python 3.8+, numpy only.  Status is binary: index 1 = success, 0 = failure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SUCC, FAIL = 1, 0


# ----------------------------------------------------------------------------
# hypotheses and posterior
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Hypothesis:
    """A point of the cutset: an assignment to the partner's hidden components."""
    components: Tuple[Tuple[str, Any], ...]

    @staticmethod
    def make(**kw: Any) -> "Hypothesis":
        return Hypothesis(tuple(sorted(kw.items())))

    def get(self, name: str) -> Any:
        for k, v in self.components:
            if k == name:
                return v
        raise KeyError(name)

    def with_component(self, name: str, value: Any) -> "Hypothesis":
        d = dict(self.components)
        d[name] = value
        return Hypothesis(tuple(sorted(d.items())))

    @property
    def names(self) -> List[str]:
        return [k for k, _ in self.components]


class Posterior:
    """Discrete posterior over hypotheses (optionally pruned with a residual mass)."""

    def __init__(self, hyps: Sequence[Hypothesis], probs: Optional[Sequence[float]] = None):
        self.hyps: List[Hypothesis] = list(hyps)
        p = np.ones(len(self.hyps)) if probs is None else np.asarray(probs, dtype=float)
        self.probs: np.ndarray = p / p.sum()
        self.residual: float = 0.0  # mass pruned away (handled by tier S)

    def __len__(self) -> int:
        return len(self.hyps)

    def index(self, h: Hypothesis) -> int:
        return self.hyps.index(h)

    def update(self, loglik: Callable[[Hypothesis], float]) -> None:
        ll = np.array([loglik(h) for h in self.hyps])
        w = self.probs * np.exp(ll - ll.max())
        self.probs = w / w.sum()

    def prune(self, n_max: int) -> None:
        if len(self.hyps) <= n_max:
            return
        keep = np.argsort(-self.probs)[:n_max]
        self.residual += 1.0 - self.probs[keep].sum()
        self.hyps = [self.hyps[i] for i in keep]
        self.probs = self.probs[keep] / self.probs[keep].sum()

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(len(self.hyps), size=n, p=self.probs)

    def component_values(self, name: str) -> List[Any]:
        vals: List[Any] = []
        for h in self.hyps:
            v = h.get(name)
            if v not in vals:
                vals.append(v)
        return vals

    def component_mass(self, name: str, value: Any) -> float:
        return float(sum(p for h, p in zip(self.hyps, self.probs) if h.get(name) == value))

    def mass(self, pred: Callable[[Hypothesis], bool]) -> float:
        return float(sum(p for h, p in zip(self.hyps, self.probs) if pred(h)))

    def copy(self) -> "Posterior":
        q = Posterior(self.hyps, self.probs.copy())
        q.residual = self.residual
        return q


# ----------------------------------------------------------------------------
# actions
# ----------------------------------------------------------------------------
PHYSICAL = ("help", "takeover", "selfexec")
CAUSE_LEVEL = ("clarify", "align")
INFORMATIONAL = ("query", "inspect")


@dataclass(frozen=True)
class Action:
    kind: str                     # continue | help | takeover | clarify | align | query | inspect | replan
    target: Any = None            # responsibility id (physical/informational) or component name (cause-level)
    cost: float = 0.0

    def __str__(self) -> str:
        return self.kind if self.target is None else f"{self.kind}({self.target})"


CONTINUE = Action("continue")


# ----------------------------------------------------------------------------
# environment-specific model (implemented by the synthetic env and the PARTNR adapter)
# ----------------------------------------------------------------------------
class ConditionalModel:
    """Quantities the planner needs, conditional on a hypothesis.  Override in subclasses."""

    # physical interventions ---------------------------------------------------
    def uplift(self, action: Action, h: Hypothesis) -> float:
        """Delta p_k(a | c): increase in P(status_k = success) caused by the action, given c."""
        raise NotImplementedError

    def consequence(self, k: Any, h: Hypothesis) -> float:
        """L_k(c) = v_k(succ | c) - v_k(fail | c): conditional Birnbaum importance of k."""
        raise NotImplementedError

    def consequence_variance(self, k: Any, h: Hypothesis) -> float:
        """Monte-Carlo variance of the consequence estimate (0 when exact)."""
        return 0.0

    def opportunity_cost(self, action: Action, h: Hypothesis) -> float:
        return 0.0

    def more_rollouts(self, k: Any, h: Hypothesis, n: int) -> None:
        """Spend n rollouts refining consequence(k, h).  No-op for exact models."""
        return None

    # cause-level interventions ------------------------------------------------
    def continuation_value(self, h: Hypothesis) -> float:
        """V(z | c): value of continuing under hypothesis c."""
        raise NotImplementedError

    def repair(self, action: Action, h: Hypothesis) -> Tuple[Hypothesis, float]:
        """(repaired hypothesis c', success probability r(c))."""
        raise NotImplementedError

    # informational interventions ----------------------------------------------
    def answer_model(self, action: Action, h: Hypothesis) -> Dict[Any, float]:
        """P(y | c) for the answer y to a query/inspection, given c."""
        raise NotImplementedError

    def behavior_change_value(self, action: Action, h: Hypothesis) -> float:
        """Lemma-1 term if being asked changes the partner's behavior on k (default 0)."""
        return 0.0

    def derail_prob(self, action: Action) -> float:
        """Probability that a message sends the partner into failing actions (0: messages are harmless)."""
        return 0.0

    def derail_value(self, h: Hypothesis) -> float:
        """Continuation value under h once the partner has been derailed (default: unchanged)."""
        return self.continuation_value(h)

    def slot_cost(self, action: Action, h: Hypothesis) -> float:
        """Value of the epoch an informational action consumes under hypothesis h (default 0): with a finite
        budget of epochs, asking now means one fewer epoch for acting."""
        return 0.0

    # structure ----------------------------------------------------------------
    def logical_role(self, k: Any) -> Optional[str]:
        """'conjunctive' | 'disjunctive' | None (unknown): used by tier M."""
        return None

    def sign_law_applies(self, k: Any) -> bool:
        """True when uplift and consequence of k are monotone functions of one scalar partner quality
        (Theorem 2 assumptions), so that tier M may use the Frechet endpoint as exact (Corollary 7.1)."""
        return False

    def decoupling_radius(self, action: Action) -> float:
        """beta_t(a) when consequences come from a decoupled approximation (default 0)."""
        return 0.0


# ----------------------------------------------------------------------------
# action-hypothesis table
# ----------------------------------------------------------------------------
@dataclass
class Table:
    actions: List[Action]
    posterior: Posterior
    A: np.ndarray                       # [n_actions, n_hyps] conditional advantages
    U: np.ndarray                       # conditional uplift (physical actions; nan otherwise)
    V: np.ndarray                       # conditional consequence L_k(c) (physical actions; nan otherwise)
    var: np.ndarray                     # Monte-Carlo variance of A entries (0 when exact)
    derived: np.ndarray                 # True for informational actions (column constant across c)
    info: Dict[str, Any] = field(default_factory=dict)

    def mixture(self) -> np.ndarray:
        return self.A @ self.posterior.probs

    def idx(self, action: Action) -> int:
        return self.actions.index(action)

    @property
    def continue_index(self) -> int:
        return self.idx(CONTINUE)


def fill_table(actions: Sequence[Action], posterior: Posterior, model: ConditionalModel,
               query_candidates: Optional[Sequence[Action]] = None) -> Table:
    """Algorithm 2.  Informational actions are valued by value of information over the table."""
    acts = list(actions)
    base = [a for a in acts if a.kind not in INFORMATIONAL]
    info = [a for a in acts if a.kind in INFORMATIONAL]
    if query_candidates is not None:
        info = [a for a in info if a in set(query_candidates)]
    acts = base + info
    n_a, n_c = len(acts), len(posterior)
    A = np.zeros((n_a, n_c)); U = np.full((n_a, n_c), np.nan); V = np.full((n_a, n_c), np.nan)
    var = np.zeros((n_a, n_c)); derived = np.zeros(n_a, dtype=bool)
    for i, a in enumerate(base):
        for j, h in enumerate(posterior.hyps):
            if a.kind == "continue":
                A[i, j] = 0.0
            elif a.kind in PHYSICAL:
                u = model.uplift(a, h); L = model.consequence(a.target, h)
                U[i, j], V[i, j] = u, L
                A[i, j] = u * L - a.cost - model.opportunity_cost(a, h)
                var[i, j] = (u ** 2) * model.consequence_variance(a.target, h)
            elif a.kind in CAUSE_LEVEL:
                h2, r = model.repair(a, h)
                d, v0 = model.derail_prob(a), model.continuation_value(h)
                A[i, j] = (1 - d) * r * (model.continuation_value(h2) - v0) + d * (model.derail_value(h) - v0) - a.cost
            elif a.kind == "replan":
                A[i, j] = getattr(model, "replan_value")(h) - model.continuation_value(h) - a.cost
            else:
                raise ValueError(a.kind)
    n_base = len(base)
    Abase = A[:n_base]
    for t, a in enumerate(info):
        i = n_base + t
        derived[i] = True
        A[i, :] = value_of_information(a, Abase, posterior, model) - a.cost \
            + np.array([model.behavior_change_value(a, h) - model.slot_cost(a, h)
                        + model.derail_prob(a) * (model.derail_value(h) - model.continuation_value(h)) for h in posterior.hyps])
    return Table(acts, posterior, A, U, V, var, derived)


def value_of_information(a: Action, Abase: np.ndarray, posterior: Posterior, model: ConditionalModel) -> float:
    """E_y[max_a' sum_c pi(c|y) A(a'|c)] - max_a' sum_c pi(c) A(a'|c)."""
    probs = posterior.probs
    answers: Dict[Any, np.ndarray] = {}
    for j, h in enumerate(posterior.hyps):
        for y, p in model.answer_model(a, h).items():
            answers.setdefault(y, np.zeros(len(probs)))[j] = p
    prior_best = float((Abase @ probs).max())
    voi = 0.0
    for y, lik in answers.items():
        joint = probs * lik
        py = joint.sum()
        if py <= 0:
            continue
        voi += py * float((Abase @ (joint / py)).max())
    return max(voi - prior_best, 0.0)


# ----------------------------------------------------------------------------
# certificates (Algorithm 3)
# ----------------------------------------------------------------------------
def coupled_expectation(u: np.ndarray, pu: np.ndarray, v: np.ndarray, pv: np.ndarray, reverse: bool) -> float:
    """E[UV] under the comonotone (reverse=False) or countermonotone (reverse=True) coupling of two
    discrete marginals (Theorem 7).  Exact quantile matching by merging sorted masses."""
    iu = np.argsort(u); iv = np.argsort(v if not reverse else -v)
    u, pu, v, pv = u[iu], pu[iu], v[iv], pv[iv]
    i = j = 0; ru, rv = pu[0], pv[0]; tot = 0.0
    while i < len(u) and j < len(v):
        m = min(ru, rv)
        tot += m * u[i] * v[j]
        ru -= m; rv -= m
        if ru <= 1e-15:
            i += 1; ru = pu[i] if i < len(u) else 0.0
        if rv <= 1e-15:
            j += 1; rv = pv[j] if j < len(v) else 0.0
    return float(tot)


def empirical_bernstein(x: np.ndarray, alpha: float, b: float) -> float:
    n = len(x)
    if n < 2:
        return float("inf")
    var = float(np.var(x, ddof=1))
    return math.sqrt(2 * var * math.log(4 / alpha) / n) + 7 * b * math.log(4 / alpha) / (3 * (n - 1))


@dataclass
class Certificate:
    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    tier: str
    certified: Optional[int]        # index of the certified action, or None
    unresolved: List[int]           # indices that can still be optimal


def certify(table: Table, tier: str = "E", alpha: float = 0.05, q_alpha: float = 1.0, b_range: Optional[float] = None,
            eps: float = 0.0, n_sample: int = 64, rng: Optional[np.random.Generator] = None,
            model: Optional[ConditionalModel] = None, z: float = 1.96) -> Certificate:
    """Tier E: exact mixture +- Monte-Carlo width.  Tier M: Frechet interval from marginals of uplift and
    consequence (endpoint when the logical role is known).  Tier S: sampled hypotheses, empirical Bernstein."""
    probs = table.posterior.probs
    n_a = len(table.actions)
    mean = table.mixture()
    lo = mean.copy(); hi = mean.copy()
    if tier == "E":
        sd = np.sqrt((table.var * (probs[None, :] ** 2)).sum(axis=1))
        w = z * sd
    elif tier == "M":
        w = np.zeros(n_a)
        for i, a in enumerate(table.actions):
            if a.kind in PHYSICAL:
                u, v = table.U[i], table.V[i]
                e_lo = coupled_expectation(u, probs, v, probs, reverse=True)
                e_hi = coupled_expectation(u, probs, v, probs, reverse=False)
                oc = float(((table.A[i] - u * v + a.cost)) @ probs)   # -opportunity cost (mixture)
                role = model.logical_role(a.target) if (model is not None and model.sign_law_applies(a.target)) else None
                if role == "conjunctive":
                    lo[i] = hi[i] = e_lo - a.cost + oc
                elif role == "disjunctive":
                    lo[i] = hi[i] = e_hi - a.cost + oc
                else:
                    lo[i], hi[i] = e_lo - a.cost + oc, e_hi - a.cost + oc
                mean[i] = 0.5 * (lo[i] + hi[i])
    elif tier == "S":
        rng = rng or np.random.default_rng(0)
        idx = table.posterior.sample(n_sample, rng)
        b = b_range if b_range is not None else float(np.nanmax(np.abs(table.A)) * 2 + 1e-9)
        w = np.zeros(n_a)
        for i in range(n_a):
            x = table.A[i, idx]
            mean[i] = float(x.mean())
            w[i] = 0.0 if table.derived[i] else empirical_bernstein(x, alpha, b)
        lo = mean.copy(); hi = mean.copy()
    else:
        raise ValueError(tier)
    if tier != "M":
        w = q_alpha * w
        if model is not None:
            w = w + np.array([model.decoupling_radius(a) for a in table.actions])
        ci = table.continue_index
        lo = mean - w; hi = mean + w
        lo[ci] = hi[ci] = 0.0
    best_lo = lo.max()
    unresolved = [i for i in range(n_a) if hi[i] >= best_lo - 1e-12]
    certified = None
    for i in range(n_a):
        others = np.delete(hi, i)
        if len(others) == 0 or lo[i] >= others.max() - eps:
            certified = i
            break
    return Certificate(lo, hi, mean, tier, certified, unresolved)


# ----------------------------------------------------------------------------
# deliberation (Algorithm 4)
# ----------------------------------------------------------------------------
@dataclass
class Decision:
    action: Action
    certified: bool
    tier: str
    rollouts_used: int
    hyps_added: int
    table: Table
    cert: Certificate


def deliberate(table: Table, cert: Certificate, model: ConditionalModel, budget: int, eps: float = 0.0,
               tier: str = "E", alpha: float = 0.05, rollouts_per_round: int = 8,
               rng: Optional[np.random.Generator] = None) -> Tuple[Certificate, int]:
    """Stratified best-action identification: spend rollouts on the leader and challenger, allocated over
    hypotheses by posterior mass x spread (Neyman).  Returns the final certificate and rollouts used."""
    used = 0
    probs = table.posterior.probs
    while budget > 0 and cert.certified is None:
        unresolved = cert.unresolved
        leader = max(unresolved, key=lambda i: cert.mean[i])
        others = [i for i in unresolved if i != leader]
        if not others:
            break
        challenger = max(others, key=lambda i: cert.hi[i])
        progressed = False
        for i in (leader, challenger):
            a = table.actions[i]
            if a.kind not in PHYSICAL:
                continue
            sd = np.sqrt(np.maximum(table.var[i], 0.0))
            alloc = probs * sd
            if alloc.sum() <= 0:
                continue
            alloc = np.maximum(np.round(rollouts_per_round * alloc / alloc.sum()), 0).astype(int)
            for j, n in enumerate(alloc):
                if n <= 0 or budget <= 0:
                    continue
                n = int(min(n, budget))
                h = table.posterior.hyps[j]
                model.more_rollouts(a.target, h, n)
                L = model.consequence(a.target, h); u = table.U[i, j]
                table.V[i, j] = L
                table.A[i, j] = u * L - a.cost - model.opportunity_cost(a, h)
                table.var[i, j] = (u ** 2) * model.consequence_variance(a.target, h)
                budget -= n; used += n; progressed = True
        if not progressed:
            break
        cert = certify(table, tier=tier, alpha=alpha, eps=eps, model=model, rng=rng)
    return cert, used


# ----------------------------------------------------------------------------
# query targeting (Algorithm 5)
# ----------------------------------------------------------------------------
def coupling_shares(table: Table, action_index: int) -> Dict[str, float]:
    """First-order Hoeffding terms iota_i = sum_z pi_i(z) (E[U|z]-E[U]) (E[L|z]-E[L]) for the physical action
    at action_index (Theorem 8)."""
    post = table.posterior
    u, v = table.U[action_index], table.V[action_index]
    probs = post.probs
    EU, EV = float(u @ probs), float(v @ probs)
    out: Dict[str, float] = {}
    for name in post.hyps[0].names:
        tot = 0.0
        for z in post.component_values(name):
            mask = np.array([h.get(name) == z for h in post.hyps])
            pz = probs[mask].sum()
            if pz <= 0:
                continue
            mu = float((u[mask] @ probs[mask]) / pz); mv = float((v[mask] @ probs[mask]) / pz)
            tot += pz * (mu - EU) * (mv - EV)
        out[name] = tot
    return out


def bhattacharyya(p: Dict[Any, float], q: Dict[Any, float]) -> float:
    keys = set(p) | set(q)
    return float(sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys))


def diagnosticity(action: Action, component: str, posterior: Posterior, model: ConditionalModel) -> float:
    """1 - BC between the answer laws under the two most probable values of the component (binary case);
    for more values, the posterior-weighted average pairwise 1 - BC."""
    vals = posterior.component_values(component)
    laws = {}
    for z in vals:
        mass = 0.0; law: Dict[Any, float] = {}
        for h, p in zip(posterior.hyps, posterior.probs):
            if h.get(component) != z:
                continue
            mass += p
            for y, py in model.answer_model(action, h).items():
                law[y] = law.get(y, 0.0) + p * py
        if mass > 0:
            laws[z] = {y: v / mass for y, v in law.items()}
    zs = list(laws)
    if len(zs) < 2:
        return 0.0
    tot = 0.0; wsum = 0.0
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            w = posterior.component_mass(component, zs[i]) * posterior.component_mass(component, zs[j])
            tot += w * (1.0 - bhattacharyya(laws[zs[i]], laws[zs[j]])); wsum += w
    return tot / wsum if wsum > 0 else 0.0


def query_targets(table: Table, posterior: Posterior, model: ConditionalModel, informational: Sequence[Action],
                  components_of_query: Callable[[Action], Sequence[str]], top_m: int = 2) -> List[Action]:
    """Rank hidden components by |coupling share| for the leading physical intervention, then pick the most
    diagnostic query for each of the top components."""
    phys = [i for i, a in enumerate(table.actions) if a.kind in PHYSICAL]
    if not phys or not informational:
        return list(informational)
    mix = table.mixture()
    lead = max(phys, key=lambda i: mix[i])
    shares = coupling_shares(table, lead)
    top = sorted(shares, key=lambda c: -abs(shares[c]))[:top_m]
    chosen: List[Action] = []
    for comp in top:
        cands = [q for q in informational if comp in components_of_query(q)]
        if not cands:
            continue
        best = max(cands, key=lambda q: diagnosticity(q, comp, posterior, model))
        if best not in chosen:
            chosen.append(best)
    return chosen or list(informational)


# ----------------------------------------------------------------------------
# escalation ladder (Algorithm 6 / Proposition 10)
# ----------------------------------------------------------------------------
@dataclass
class Ladder:
    order: List[str]
    pi_rescue_repair: float
    pi_rescue_continue: float
    pi_repair_continue: float

    def act(self, pi_bad: float, h_g: float, h_b: float, fbar_gap: float, c_s: float, c_c: float) -> str:
        vals = {"continue": 0.0, "rescue": pi_bad * h_b + (1 - pi_bad) * h_g - c_s, "repair": pi_bad * fbar_gap - c_c}
        return max(vals, key=lambda k: vals[k])


def ladder(h_g: float, h_b: float, R_k: float, I_k: float, fbar_gap: float, c_s: float, c_c: float) -> Ladder:
    inf = float("inf")
    pi_rr = (c_c - c_s + I_k) / R_k if R_k > 0 else inf
    pi_rc = (c_s - h_g) / (h_b - h_g) if abs(h_b - h_g) > 1e-12 else inf
    pi_pc = c_c / fbar_gap if fbar_gap > 0 else inf
    order = ["continue", "rescue", "repair"] if h_b >= h_g else ["rescue", "continue", "repair"]
    return Ladder(order, pi_rr, pi_rc, pi_pc)


# ----------------------------------------------------------------------------
# informational price (Proposition 11)
# ----------------------------------------------------------------------------
def informational_price(sigma: float, odds0: float, bc_all: float, bc_k: float) -> float:
    """Increase of the type-g worst-case regret bound when responsibility k's signal is removed."""
    if bc_k <= 0 or bc_all / bc_k >= 1.0:
        return float("inf")
    return 0.5 * sigma * math.sqrt(odds0) * (1.0 / (1.0 - bc_all / bc_k) - 1.0 / (1.0 - bc_all))


# ----------------------------------------------------------------------------
# one decision epoch (Algorithm 1)
# ----------------------------------------------------------------------------
@dataclass
class PlannerConfig:
    tier: str = "E"
    alpha: float = 0.05
    q_alpha: float = 1.0
    eps: float = 0.0
    budget: int = 0
    n_sample: int = 64
    use_query_targeting: bool = True
    top_m: int = 2


class CCRPPlanner:
    def __init__(self, model: ConditionalModel, config: Optional[PlannerConfig] = None,
                 components_of_query: Optional[Callable[[Action], Sequence[str]]] = None,
                 rng: Optional[np.random.Generator] = None):
        self.model = model
        self.cfg = config or PlannerConfig()
        self.components_of_query = components_of_query
        self.rng = rng or np.random.default_rng(0)

    def decide(self, actions: Sequence[Action], posterior: Posterior) -> Decision:
        cfg = self.cfg
        info = [a for a in actions if a.kind in INFORMATIONAL]
        base = [a for a in actions if a.kind not in INFORMATIONAL]
        table = fill_table(base, posterior, self.model)
        if info:
            cands = info
            asks = [a for a in info if a.kind != "inspect"]
            if cfg.use_query_targeting and self.components_of_query is not None and len(asks) > 1:
                # targeting chooses which question to put to the partner; passive inspection (watching for evidence
                # that arrives anyway) costs the partner nothing and always stays a candidate
                cands = query_targets(table, posterior, self.model, asks, self.components_of_query, cfg.top_m)
                cands = list(cands) + [a for a in info if a.kind == "inspect"]
            table = fill_table(base + list(info), posterior, self.model, query_candidates=cands)
        cert = certify(table, tier=cfg.tier, alpha=cfg.alpha, q_alpha=cfg.q_alpha, eps=cfg.eps,
                       n_sample=cfg.n_sample, rng=self.rng, model=self.model)
        used = 0
        if cert.certified is None and cfg.budget > 0:
            cert, used = deliberate(table, cert, self.model, cfg.budget, eps=cfg.eps, tier=cfg.tier,
                                    alpha=cfg.alpha, rng=self.rng)
        if cert.certified is not None:
            i = cert.certified; certified = True
        else:
            i = max(cert.unresolved, key=lambda j: cert.mean[j]); certified = False
        return Decision(table.actions[i], certified, cfg.tier, used, 0, table, cert)
