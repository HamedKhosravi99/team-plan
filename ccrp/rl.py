"""Learned hypothesis-conditional value (RL) and certificate-gated best-arm deliberation (BAI) on the synthetic benchmark.

This is the scalable path of the method: when the number of responsibilities makes exact multi-step planning
infeasible, the hypothesis-conditional value V(z, c) is learned by fitted Monte-Carlo policy iteration in simulation,
where the partner's true hidden state c is known (privileged training).  Online, each epoch:

  1. value table  T[a, c] = Q(a | c) - Q(continue | c) by one-step lookahead on V (exact at the last epoch);
  2. certificate   the posterior-mixture advantage of each action +- a width w calibrated on held-out value errors;
  3. deliberation  if the leader is not certified, LUCB best-arm identification spends rollouts (hypothesis sampled
                   from the posterior, hypothesis-conditional greedy rollout policy, paired random numbers) only on the
                   unresolved actions, up to a budget.

V(z, c) = R(p(z, c)) + f(phi(z, c)), where R is the exact reliability of the task formula at the current success rates
(the value of stopping now) and f, a gradient-boosted regressor, learns what the remaining epochs are worth.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import Action, CONTINUE, Hypothesis, Posterior
from .synthetic import EnvSpec, ExactModel, Node, SyntheticEnv

INFO = ("query", "inspect")


# ----------------------------------------------------------------------------------------------------------------
# vectorized task value
# ----------------------------------------------------------------------------------------------------------------
def reliability_rows(node: Node, P: np.ndarray) -> np.ndarray:
    """Reliability of a read-once formula for each row of success rates P (N x K)."""
    if node.op == "leaf":
        return P[:, node.leaf]
    vals = [reliability_rows(c, P) for c in node.children]
    if node.op == "and":
        out = vals[0].copy()
        for v in vals[1:]:
            out *= v
        return out
    out = 1.0 - vals[0]
    for v in vals[1:]:
        out *= (1.0 - v)
    return 1.0 - out


class Encoder:
    """Hypotheses -> unaided success-rate vectors, and (flags, hypothesis, epochs left) -> features."""

    def __init__(self, spec: EnvSpec):
        self.spec = spec
        self._cache: Dict[Hypothesis, np.ndarray] = {}

    def unaided(self, h: Hypothesis) -> np.ndarray:
        u = self._cache.get(h)
        if u is None:
            u = np.array([self.spec.unaided_rate(k, h) for k in range(self.spec.K)])
            self._cache[h] = u
        return u

    def rates(self, U: np.ndarray, taken: np.ndarray, helped: np.ndarray) -> np.ndarray:
        """U: N x K unaided rates; taken/helped: N x K booleans."""
        sp = self.spec
        P = np.where(helped, U + (1.0 - U) * sp.help_boost, U)
        return np.where(taken, sp.eta_takeover, P)

    def features(self, P: np.ndarray, R: np.ndarray, taken: np.ndarray, helped: np.ndarray, n: np.ndarray) -> np.ndarray:
        sp = self.spec
        B = np.empty_like(P)
        for k in range(sp.K):                          # Birnbaum importance of each responsibility (vectorized)
            P1 = P.copy(); P1[:, k] = 1.0; P0 = P.copy(); P0[:, k] = 0.0
            B[:, k] = reliability_rows(sp.formula, P1) - reliability_rows(sp.formula, P0)
        room = np.where(taken, 0.0, sp.eta_takeover - P) * B          # what a takeover could still add, per responsibility
        return np.column_stack([R, n / sp.horizon, P, B, room, taken.astype(float), helped.astype(float)])


class ValueModel:
    """V(z, c) = R + f(features); f fitted to Monte-Carlo returns-to-go minus R."""

    def __init__(self, spec: EnvSpec):
        self.spec = spec
        self.enc = Encoder(spec)
        self.reg = None
        self.width = 0.0          # conformal width of |G - V| (set by fit)
        self.adv_width = None     # conformal width of advantage errors against a rollout reference (calibrate_advantage)

    def value_rows(self, U, taken, helped, n) -> np.ndarray:
        P = self.enc.rates(U, taken, helped)
        R = reliability_rows(self.spec.formula, P)
        out = R.copy()
        live = n > 0
        if self.reg is not None and live.any():
            X = self.enc.features(P[live], R[live], taken[live], helped[live], n[live])
            out[live] = R[live] + self.reg.predict(X)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0, quantile: float = 0.9) -> Dict[str, float]:
        from sklearn.ensemble import HistGradientBoostingRegressor
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y)); cut = int(0.9 * len(y))
        tr, va = idx[:cut], idx[cut:]
        R = X[:, 0]
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=seed)
        reg.fit(X[tr], y[tr] - R[tr])
        res = np.abs(y[va] - (R[va] + reg.predict(X[va])))
        res0 = np.abs(y[va] - R[va])
        self.reg = reg
        self.width = float(np.quantile(res, quantile))
        return {"n": int(len(y)), "mae": float(res.mean()), "mae_no_learning": float(res0.mean()), "width": self.width}


# ----------------------------------------------------------------------------------------------------------------
# action values under the posterior
# ----------------------------------------------------------------------------------------------------------------
def candidate_actions(env: SyntheticEnv) -> List[Action]:
    return [a for a in env.actions() if a.kind not in INFO]


def action_table(env: SyntheticEnv, post: Optional[Posterior], V: ValueModel, acts: Sequence[Action],
                 hyps: Optional[Sequence[Hypothesis]] = None) -> np.ndarray:
    """T[a, j] = Q(a | c_j) - Q(continue | c_j), with Q(a | c) = V(next state, c) - cost(a).  One batched call."""
    sp, st = env.spec, env.state
    hyps = list(post.hyps if hyps is None else hyps)
    H = len(hyps)
    n_next = sp.horizon - st.t - 1
    U = np.stack([V.enc.unaided(h) for h in hyps])
    tk0 = np.tile(np.array(st.taken_over, bool), (H, 1)); hp0 = np.tile(np.array(st.helped, bool), (H, 1))
    blocks_U, blocks_tk, blocks_hp, plan = [U], [tk0], [hp0], []
    off = H
    for i, a in enumerate(acts):
        if a.kind in ("takeover", "help"):
            tk, hp = tk0.copy(), hp0.copy()
            (tk if a.kind == "takeover" else hp)[:, a.target] = True
            blocks_U.append(U); blocks_tk.append(tk); blocks_hp.append(hp)
            plan.append((i, "phys", off, None)); off += H
        elif a.kind in ("clarify", "align"):
            bad = np.array([h.get(a.target) == "bad" for h in hyps])
            if bad.any():
                Ug = np.stack([V.enc.unaided(h.with_component(a.target, "good")) for h, b in zip(hyps, bad) if b])
                blocks_U.append(Ug); blocks_tk.append(tk0[bad]); blocks_hp.append(hp0[bad])
                plan.append((i, "repair", off, bad)); off += int(bad.sum())
            else:
                plan.append((i, "repair", off, bad))
    Uall = np.concatenate(blocks_U); tkall = np.concatenate(blocks_tk); hpall = np.concatenate(blocks_hp)
    vals = V.value_rows(Uall, tkall, hpall, np.full(len(Uall), n_next))
    base = vals[:H]
    T = np.zeros((len(acts), H))
    for i, kind, o, bad in plan:
        a = acts[i]
        if kind == "phys":
            T[i] = vals[o:o + H] - a.cost - base
        else:
            v = base.copy()
            nb = int(bad.sum())
            if nb:
                r = sp.repair_success
                v[bad] = r * vals[o:o + nb] + (1 - r) * base[bad]
            T[i] = v - a.cost - base
    return T


# ----------------------------------------------------------------------------------------------------------------
# rollouts (hypothesis-conditional, paired random numbers) and best-arm identification
# ----------------------------------------------------------------------------------------------------------------
def query_value(env: SyntheticEnv, post: Posterior, T: np.ndarray, q_act: Action) -> float:
    """Value of a query from the hypothesis-conditional table, minus its cost.

    A table T[a, c] conditioned on the partner's true state c gives every query the value -cost (QMDP's blindness to
    information).  The answer y to a query is informative about c, though, and the value of that sample information
    is exactly computable from the same table:
        VSI(q) = sum_y P(y) max_a sum_c pi(c | y) T[a, c]  -  max_a sum_c pi(c) T[a, c]   >= 0,
    with P(y | c) the answer model.  (T contains the continue row, so both maxima are >= 0.)  This is the myopic value
    of the answer: it assumes the agent acts on it at once; belief rollouts estimate the full multi-step value.
    """
    sp = env.spec; qn = sp.query_noise
    bad = np.array([sp.any_bad(q_act.target, h) for h in post.hyps], dtype=bool)
    lik_trouble = np.where(bad, 1.0 - qn, qn)
    base = float(np.max(T @ post.probs))
    val = 0.0
    for lik in (lik_trouble, 1.0 - lik_trouble):
        py = float(post.probs @ lik)
        if py <= 1e-12:
            continue
        val += py * float(np.max(T @ (post.probs * lik / py)))
    return val - base - q_act.cost


def greedy_given(env_like: SyntheticEnv, h: Hypothesis, V: ValueModel) -> Action:
    acts = candidate_actions(env_like)
    T = action_table(env_like, None, V, acts, hyps=[h])
    return acts[int(np.argmax(T[:, 0]))]


def rollout(env: SyntheticEnv, first: Action, h: Hypothesis, V: ValueModel, u_rep: np.ndarray) -> float:
    """Return (expected success at the end - costs from now on) when the truth is h, `first` is taken now and the
    hypothesis-conditional greedy policy on V follows.  u_rep: uniforms for repair outcomes (paired across actions)."""
    sp = env.spec
    sim = SyntheticEnv.__new__(SyntheticEnv)
    sim.spec = sp
    sim.state = type(env.state)(list(env.state.taken_over), list(env.state.helped), env.state.t, 0.0, [])
    truth = h; cost = 0.0
    a = first
    while True:
        cost += a.cost
        if a.kind == "takeover":
            sim.state.taken_over[a.target] = True
        elif a.kind == "help":
            sim.state.helped[a.target] = True
        elif a.kind in ("clarify", "align"):
            if truth.get(a.target) == "bad" and u_rep[sim.state.t] < sp.repair_success:
                truth = truth.with_component(a.target, "good")
        sim.state.t += 1
        if sim.state.t >= sp.horizon:
            break
        a = greedy_given(sim, truth, V)
    p = ExactModel(sp, sim.state, 1).rates(truth)
    return sp.formula.reliability(p) - cost


def rollout_belief(env: SyntheticEnv, first: Action, h: Hypothesis, V: ValueModel, seed: int) -> float:
    """Simulate the agent's actual future when the truth is h: `first` now, then progress signals drawn from h, Bayes
    updates of the posterior, and greedy decisions on the posterior-mixture table of V (the base policy).  Returns the
    expected final success minus the costs paid from now on.  `seed` fixes the random numbers (paired across actions)."""
    import copy
    sp = env.spec
    sim = SyntheticEnv(sp, seed)
    sim.truth = h
    sim.state = type(env.state)(list(env.state.taken_over), list(env.state.helped), env.state.t, 0.0, [])
    sim.posterior = Posterior(list(env.posterior.hyps), np.array(env.posterior.probs, dtype=float).copy())
    K, T = sp.K, sp.horizon + 1
    r = np.random.default_rng(seed)
    sim.u_final = r.random(K); sim.u_signal = r.random((T, K)); sim.u_answer = r.random((T, K))
    sim.u_repair = {name: r.random(T) for name in sp.component_names()}
    a = first
    while True:
        sim.step(a)
        if sim.done():
            break
        acts = candidate_actions(sim)
        mix = action_table(sim, sim.posterior, V, acts) @ sim.posterior.probs
        a = acts[int(np.argmax(mix))]
    p = ExactModel(sp, sim.state, 1).rates(sim.truth)
    return sp.formula.reliability(p) - sim.state.total_cost


ROLLOUT = {"mode": "belief"}          # "belief" (agent's actual future) or "qmdp" (hypothesis-conditional greedy)


def paired_rollout(env, a, a0, h, V, rng) -> float:
    if ROLLOUT["mode"] == "belief":
        seed = int(rng.integers(2**31 - 1))
        return rollout_belief(env, a, h, V, seed) - rollout_belief(env, a0, h, V, seed)
    u = rng.random(env.spec.horizon + 1)
    return rollout(env, a, h, V, u) - rollout(env, a0, h, V, u)


@dataclass
class DeliberationStats:
    rollouts: int = 0
    certified: int = 0
    decisions: int = 0


def eb_radius(x: np.ndarray, delta: float, b: float) -> float:
    n = len(x)
    if n < 2:
        return float("inf")
    v = float(np.var(x, ddof=1))
    return math.sqrt(2 * v * math.log(3 / delta) / n) + 3 * b * math.log(3 / delta) / n


def best_arm(env: SyntheticEnv, post: Posterior, V: ValueModel, acts: Sequence[Action], prior_mean: np.ndarray,
             cand: Sequence[int], budget: int, rng: np.random.Generator, per_round: int = 4,
             delta: float = 0.1, eps: float = 0.0) -> Tuple[int, int]:
    """LUCB over the candidate actions: paired rollout differences against continue.  Returns (choice, rollouts)."""
    i_cont = next(i for i, a in enumerate(acts) if a.kind == "continue")
    samples: Dict[int, List[float]] = {i: [] for i in cand}
    used = 0; b = 1.0 + max(a.cost for a in acts)
    while used < budget:
        means = {i: (np.mean(samples[i]) if samples[i] else prior_mean[i]) for i in cand}
        rad = {i: eb_radius(np.array(samples[i]), delta, b) for i in cand}
        leader = max(cand, key=lambda i: means[i])
        others = [i for i in cand if i != leader]
        if not others:
            break
        challenger = max(others, key=lambda i: means[i] + rad[i])
        if means[leader] - rad[leader] >= means[challenger] + rad[challenger] - eps:
            break
        for i in (leader, challenger):
            for _ in range(per_round):
                if used >= budget:
                    break
                h = post.hyps[int(rng.choice(len(post.probs), p=post.probs))]
                d = 0.0 if i == i_cont else paired_rollout(env, acts[i], acts[i_cont], h, V, rng)
                samples[i].append(d); used += 2 if i != i_cont else 0
    means = {i: (np.mean(samples[i]) if samples[i] else prior_mean[i]) for i in cand}
    return max(cand, key=lambda i: means[i]), used


def bayes_best_arm(env: SyntheticEnv, post: Posterior, V: ValueModel, acts: Sequence[Action], prior_mean: np.ndarray,
                   cand: Sequence[int], budget: int, rng: np.random.Generator, eps: float = 0.0, stop_early: bool = True,
                   per_round: int = 4, conf: float = 1.2816) -> Tuple[int, int]:
    """Bayesian best-arm identification with the learned table as the prior (the Bayesian-optimization view of
    deliberation).  Each action's advantage over continue has prior N(table value, s0^2), s0 from the certificate's
    calibration; paired belief rollouts are Gaussian observations with pooled variance.  Continue is known exactly (0).
    Samples go to the leader and to the challenger most likely to beat it; with stop_early the loop ends once that
    probability is below 10% (eps-best); the choice is the posterior mean."""
    i_cont = next(i for i, a in enumerate(acts) if a.kind == "continue")
    s0 = max((V.adv_width or 0.05) / 1.645, 1e-3)
    n = {i: 0 for i in cand}; tot = {i: 0.0 for i in cand}; allx: List[float] = []
    used = 0

    def post_mv(i, sr2):
        if i == i_cont:
            return 0.0, 0.0
        prec = 1.0 / s0 ** 2 + n[i] / sr2
        return (prior_mean[i] / s0 ** 2 + tot[i] / sr2) / prec, 1.0 / prec

    while used < budget:
        sr2 = max(float(np.var(allx, ddof=1)) if len(allx) >= 6 else 0.15 ** 2, 0.02 ** 2)
        mv = {i: post_mv(i, sr2) for i in cand}
        leader = max(cand, key=lambda i: mv[i][0])
        others = [i for i in cand if i != leader]
        if not others:
            break
        def zbeat(i):                       # standardized margin by which i could beat the leader by more than eps
            v = mv[i][1] + mv[leader][1]
            return (mv[i][0] - mv[leader][0] - eps) / math.sqrt(v) if v > 0 else -np.inf
        challenger = max(others, key=zbeat)
        if stop_early and zbeat(challenger) <= -conf:
            break
        sampled = False
        for i in (leader, challenger):
            if i == i_cont:
                continue
            for _ in range(per_round):
                if used >= budget:
                    break
                h = post.hyps[int(rng.choice(len(post.probs), p=post.probs))]
                x = paired_rollout(env, acts[i], acts[i_cont], h, V, rng)
                n[i] += 1; tot[i] += x; allx.append(x); used += 2; sampled = True
        if not sampled:
            break
    sr2 = max(float(np.var(allx, ddof=1)) if len(allx) >= 6 else 0.15 ** 2, 0.02 ** 2)
    best = max(cand, key=lambda i: post_mv(i, sr2)[0])
    if DELIBERATION.get("safe"):
        # high-confidence improvement: keep the learned table's choice unless the posterior gives the rollouts' best
        # action probability >= DELIBERATION["safe_conf"] of beating it (each harmful switch has probability <= 1 - conf)
        table_lead = max(cand, key=lambda i: prior_mean[i])
        if best != table_lead:
            (mb, vb), (ml, vl) = post_mv(best, sr2), post_mv(table_lead, sr2)
            z = (mb - ml) / math.sqrt(vb + vl) if vb + vl > 0 else float("inf")
            if z < DELIBERATION.get("safe_z", 1.2816):
                best = table_lead
    return best, used


# rule: "bayes" (learned table as prior) or "lucb" (rollout means only); safe: high-confidence switch from the table
# (read from the environment so that spawned worker processes see the same setting as the parent)
DELIBERATION = {"rule": os.environ.get("TP_BO_RULE", "bayes"), "safe": os.environ.get("TP_BO_SAFE", "0") == "1",
                "safe_z": float(os.environ.get("TP_BO_SAFE_Z", "1.2816"))}


# ----------------------------------------------------------------------------------------------------------------
# policies: (env, posterior, stats, rng) -> Action
# ----------------------------------------------------------------------------------------------------------------
def make_policy(kind: str, V: Optional[ValueModel] = None, budget: int = 64, width_scale: float = 1.0, eps: float = 0.02):
    # "<kind>_q": the same policy with query actions, valued by their value of sample information (query_value)
    use_q = kind.endswith("_q")
    if use_q:
        kind = kind[:-2]

    def pol(env: SyntheticEnv, post: Posterior, stats: DeliberationStats, rng: np.random.Generator) -> Action:
        acts = candidate_actions(env)
        stats.decisions += 1
        if kind == "never":
            return CONTINUE
        if kind in ("one_step", "blind_one_step"):
            m = ExactModel(env.spec, env.state, 1)
            sc = [m.exact_advantage(a, post) if kind == "one_step" else m.myopic_score(a, post) for a in acts]
            return acts[int(np.argmax(sc))]
        if kind == "oracle":
            return greedy_given(env, env.truth, V)
        T = action_table(env, post, V, acts)
        mix = T @ post.probs
        if use_q:
            q_acts = [a for a in env.actions() if a.kind == "query"]
            if q_acts:
                mix = np.concatenate([mix, [query_value(env, post, T, a) for a in q_acts]])
                acts = list(acts) + q_acts
        if kind == "rl_greedy":
            return acts[int(np.argmax(mix))]
        w = width_scale * (V.adv_width if V.adv_width is not None else V.width * math.sqrt(2.0))
        lead = int(np.argmax(mix))
        # epsilon-certificate: an action stays in play only if it could beat the leader by more than eps
        unresolved = [i for i in range(len(acts)) if i == lead or mix[i] + w > mix[lead] - w + eps]
        if kind == "teamplan" and len(unresolved) <= 1:
            stats.certified += 1
            return acts[lead]
        cand = unresolved if kind == "teamplan" else list(range(len(acts)))      # always_deep: every action, every epoch
        if DELIBERATION["rule"] == "bayes":
            choice, used = bayes_best_arm(env, post, V, acts, mix, cand, budget, rng, eps=eps if kind == "teamplan" else 0.0,
                                          stop_early=(kind == "teamplan"))
        else:
            choice, used = best_arm(env, post, V, acts, mix, cand, budget, rng, eps=eps if kind == "teamplan" else 0.0)
        stats.rollouts += used
        return acts[choice]
    return pol


@dataclass
class Result:
    success: int
    reward: float
    rollouts: int
    certified: int
    decisions: int
    actions: List[str] = field(default_factory=list)


def run(spec: EnvSpec, policy, seed: int) -> Result:
    env = SyntheticEnv(spec, seed)
    post = env.reset()
    stats = DeliberationStats(); rng = np.random.default_rng(10_000 + seed); acts = []
    while not env.done():
        a = policy(env, post, stats, rng)
        acts.append(str(a))
        post, _ = env.step(a)
    out = env.finish()
    return Result(out["success"], out["reward"], stats.rollouts, stats.certified, stats.decisions, acts)


# ----------------------------------------------------------------------------------------------------------------
# training: fitted Monte-Carlo policy iteration with privileged hidden state
# ----------------------------------------------------------------------------------------------------------------
def collect(spec: EnvSpec, behavior, seeds: Sequence[int], eps: float, V: Optional[ValueModel]) -> Tuple[np.ndarray, np.ndarray]:
    """Run the behavior policy (epsilon-greedy) and record (features of the visited state under the true hidden state,
    return-to-go).  The return-to-go is Rao-Blackwellized: expected final success under the final truth minus the
    costs paid from that epoch on."""
    enc = V.enc if V is not None else Encoder(spec)
    X, y = [], []
    for seed in seeds:
        env = SyntheticEnv(spec, seed); post = env.reset(); rng = np.random.default_rng(seed + 7)
        stats = DeliberationStats(); visits = []
        while not env.done():
            st = env.state
            visits.append((np.array(st.taken_over, bool), np.array(st.helped, bool), env.truth, spec.horizon - st.t,
                           st.total_cost))
            acts = candidate_actions(env)
            a = acts[int(rng.integers(len(acts)))] if rng.random() < eps else behavior(env, post, stats, rng)
            post, _ = env.step(a)
        final = ExactModel(spec, env.state, 1).rates(env.truth)
        G_end = spec.formula.reliability(final) - env.state.total_cost
        for taken, helped, truth, n, cost_before in visits:
            U = enc.unaided(truth)[None, :]
            P = enc.rates(U, taken[None, :], helped[None, :])
            R = reliability_rows(spec.formula, P)
            X.append(enc.features(P, R, taken[None, :], helped[None, :], np.array([n]))[0])
            y.append(G_end + cost_before)          # return-to-go from this epoch: success - future costs
    return np.array(X), np.array(y)


def _collect_job(args):
    spec, kind, Vbytes, seeds, eps = args
    import pickle
    V = pickle.loads(Vbytes) if Vbytes is not None else None
    beh = make_policy(kind, V)
    return collect(spec, beh, seeds, eps, V)


def train_value(spec: EnvSpec, iterations: int = 3, episodes: int = 3000, eps: float = 0.15, procs: int = 8,
                seed: int = 0, log: Callable[[str], None] = print) -> Tuple[ValueModel, List[Dict[str, float]]]:
    import pickle
    V = ValueModel(spec); hist = []; data: List[Tuple[np.ndarray, np.ndarray]] = []
    for it in range(iterations):
        kind = "one_step" if it == 0 else "rl_greedy"
        seeds = list(range(1_000_000 * (it + 1) + seed, 1_000_000 * (it + 1) + seed + episodes))
        chunks = [seeds[i::procs] for i in range(procs)]
        Vb = pickle.dumps(V) if it > 0 else None
        t0 = time.time()
        with Pool(procs) as pool:
            parts = pool.map(_collect_job, [(spec, kind, Vb, c, eps) for c in chunks])
        X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
        data.append((X, y)); data = data[-2:]                     # latest two iterations (on-policy-ish)
        Xa = np.concatenate([d[0] for d in data]); ya = np.concatenate([d[1] for d in data])
        V = ValueModel(spec); info = V.fit(Xa, ya, seed=seed + it)
        info.update({"iteration": it, "behavior": kind, "seconds": time.time() - t0})
        hist.append(info); log(f"  RL iteration {it}: behavior={kind} samples={info['n']} MAE {info['mae']:.4f} "
                              f"(stop-now baseline {info['mae_no_learning']:.4f}) width {info['width']:.3f} [{info['seconds']:.0f}s]")
    return V, hist


# ----------------------------------------------------------------------------------------------------------------
# certificate calibration: advantage errors of the learned table against a high-budget rollout reference
# ----------------------------------------------------------------------------------------------------------------
def _calib_job(args):
    import pickle
    spec, Vb, seeds, ref_rollouts, top = args
    V = pickle.loads(Vb); out = []
    pol = make_policy("rl_greedy", V)
    for seed in seeds:
        env = SyntheticEnv(spec, seed); post = env.reset(); rng = np.random.default_rng(seed + 99)
        stats = DeliberationStats()
        t_pick = int(rng.integers(spec.horizon))            # one exchangeable decision epoch per episode
        while not env.done():
            if env.state.t == t_pick:
                acts = candidate_actions(env)
                T = action_table(env, post, V, acts); mix = T @ post.probs
                i_cont = next(i for i, a in enumerate(acts) if a.kind == "continue")
                order = [i for i in np.argsort(-mix) if i != i_cont][:top]
                for i in order:
                    d = []
                    for _ in range(ref_rollouts):
                        h = post.hyps[int(rng.choice(len(post.probs), p=post.probs))]
                        d.append(paired_rollout(env, acts[i], acts[i_cont], h, V, rng))
                    out.append(abs(float(mix[i]) - float(np.mean(d))))
                break
            post, _ = env.step(pol(env, post, stats, rng))
    return out


def calibrate_advantage(spec: EnvSpec, V: ValueModel, n_states: int = 180, ref_rollouts: int = 32, top: int = 3,
                        quantile: float = 0.9, procs: int = 8, seed: int = 5_000_000) -> Dict[str, float]:
    """Split-conformal width for the certificate: at one random epoch of held-out episodes, compare the learned
    advantages of the leading actions with a paired-rollout reference; the width is the `quantile` of the errors."""
    import pickle
    seeds = list(range(seed, seed + n_states)); chunks = [seeds[i::procs] for i in range(procs)]
    with Pool(procs) as pool:
        parts = pool.map(_calib_job, [(spec, pickle.dumps(V), c, ref_rollouts, top) for c in chunks])
    err = np.array([e for p in parts for e in p])
    V.adv_width = float(np.quantile(err, quantile))
    return {"n_errors": int(len(err)), "median": float(np.median(err)), "adv_width": V.adv_width}
