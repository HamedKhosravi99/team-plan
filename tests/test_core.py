"""Smoke tests for the CCRP core against brute force on the synthetic environment."""
import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ccrp.core import (Action, CONTINUE, Hypothesis, Posterior, certify, coupled_expectation, coupling_shares,
                       fill_table, ladder, value_of_information, informational_price)
from ccrp.synthetic import EnvSpec, ExactModel, SyntheticEnv, formula_and, formula_or, random_formula, birnbaum

rng = np.random.default_rng(0)


def small_env(K=3, formula=None, seed=0):
    spec = EnvSpec(formula or formula_and(K), K, n_objects=2, n_skills=1, horizon=4)
    env = SyntheticEnv(spec, seed); post = env.reset()
    return spec, env, post


def test_lemma1_table_equals_bruteforce_improvement():
    """Table mixture of takeover(k) == E_c[ fbar(p with p_k -> eta) - fbar(p) ] - cost (exact identity)."""
    spec, env, post = small_env(K=4, formula=random_formula(4, np.random.default_rng(3)))
    model = env.model(epochs_left=1); acts = env.actions()
    table = fill_table([a for a in acts if a.kind != "query"], post, model)
    mix = table.mixture()
    for a in table.actions:
        if a.kind != "takeover":
            continue
        brute = 0.0
        for h, p in zip(post.hyps, post.probs):
            rates = model.rates(h); r2 = rates.copy(); r2[a.target] = spec.eta_takeover
            brute += p * (spec.formula.reliability(r2) - spec.formula.reliability(rates))
        assert abs(mix[table.idx(a)] - (brute - a.cost)) < 1e-12


def test_birnbaum_is_derivative_of_reliability():
    f = random_formula(5, np.random.default_rng(1)); p = rng.random(5)
    for k in range(5):
        eps = 1e-6; p1 = p.copy(); p1[k] += eps
        num = (f.reliability(p1) - f.reliability(p)) / eps
        assert abs(num - birnbaum(f, p, k)) < 1e-4


def test_frechet_coupling_matches_bruteforce_and_monotone_exactness():
    n = 6; pz = rng.dirichlet(np.ones(n))
    u = np.sort(rng.random(n))[::-1]; v = np.sort(rng.random(n))            # countermonotone pair
    exact = float((pz * u * v).sum())
    lo = coupled_expectation(u, pz, v, pz, reverse=True); hi = coupled_expectation(u, pz, v, pz, reverse=False)
    assert abs(exact - lo) < 1e-12 and lo <= hi + 1e-12
    v2 = np.sort(rng.random(n))[::-1]                                         # comonotone pair
    assert abs(float((pz * u * v2).sum()) - coupled_expectation(u, pz, v2, pz, reverse=False)) < 1e-12


def test_tier_M_endpoint_equals_truth_on_conjunctive_task():
    spec = EnvSpec(formula_and(3), 3, n_objects=1, n_skills=1, horizon=4)   # one scalar partner quality
    env = SyntheticEnv(spec, 0); post = env.reset()
    model = env.model(epochs_left=1)
    acts = [a for a in env.actions() if a.kind in ("continue", "takeover")]
    table = fill_table(acts, post, model)
    cert = certify(table, tier="M", model=model)
    for i, a in enumerate(table.actions):
        if a.kind == "takeover":
            assert abs(cert.lo[i] - table.mixture()[i]) < 1e-9 and abs(cert.hi[i] - table.mixture()[i]) < 1e-9


def test_tier_M_interval_contains_truth_generally():
    for seed in range(20):
        f = random_formula(4, np.random.default_rng(seed))
        spec, env, post = small_env(K=4, formula=f, seed=seed)
        model = env.model()
        acts = [a for a in env.actions() if a.kind in ("continue", "takeover")]
        table = fill_table(acts, post, model)
        # ignore the logical-role shortcut: check the raw interval
        class NoRole(type(model)):
            def logical_role(self, k): return None
        model2 = NoRole(spec, env.state, 1)
        cert = certify(table, tier="M", model=model2)
        mix = table.mixture()
        for i, a in enumerate(table.actions):
            if a.kind == "takeover":
                assert cert.lo[i] - 1e-9 <= mix[i] <= cert.hi[i] + 1e-9


def test_tier_S_covers_tier_E_mixture():
    spec, env, post = small_env(K=4, formula=random_formula(4, np.random.default_rng(7)))
    model = env.model(); acts = [a for a in env.actions() if a.kind in ("continue", "takeover", "clarify")]
    table = fill_table(acts, post, model); mix = table.mixture()
    miss = 0; trials = 100
    for s in range(trials):
        cert = certify(table, tier="S", alpha=0.1, n_sample=200, rng=np.random.default_rng(s))
        miss += int(any((mix[i] < cert.lo[i] - 1e-12) or (mix[i] > cert.hi[i] + 1e-12) for i in range(len(acts)) if acts[i].kind != "continue"))
    assert miss / trials <= 0.1 + 0.05


def test_certified_action_is_the_argmax():
    spec, env, post = small_env(K=3, formula=formula_and(3))
    model = env.model(); acts = [a for a in env.actions() if a.kind != "query"]
    table = fill_table(acts, post, model)
    cert = certify(table, tier="E")
    assert cert.certified is not None and cert.certified == int(np.argmax(table.mixture()))


def test_voi_nonnegative_and_zero_when_uninformative():
    spec, env, post = small_env(K=3)
    model = env.model(); acts = [a for a in env.actions() if a.kind != "query"]
    table = fill_table(acts, post, model)
    q = Action("query", 0, 0.0)
    assert value_of_information(q, table.A, post, model) >= 0.0
    class Uninformative(type(model)):
        def answer_model(self, action, h): return {"fine": 0.5, "trouble": 0.5}
    assert abs(value_of_information(q, table.A, post, Uninformative(spec, env.state, model.n))) < 1e-12


def test_coupling_shares_match_first_order_hoeffding():
    spec, env, post = small_env(K=3)
    model = env.model(); acts = [a for a in env.actions() if a.kind == "takeover"]
    table = fill_table([CONTINUE] + acts, post, model)
    i = 1; u, v = table.U[i], table.V[i]; p = post.probs
    shares = coupling_shares(table, i)
    for name, val in shares.items():
        EU, EV = u @ p, v @ p; tot = 0.0
        for z in post.component_values(name):
            m = np.array([h.get(name) == z for h in post.hyps]); pz = p[m].sum()
            tot += pz * ((u[m] @ p[m]) / pz - EU) * ((v[m] @ p[m]) / pz - EV)
        assert abs(tot - val) < 1e-12
    # a component the action's responsibility does not depend on has zero share
    unused = [c for c in post.hyps[0].names if c not in spec.components_of(acts[0].target)]
    for c in unused:
        assert abs(shares[c]) < 1e-12


def test_ladder_thresholds_match_envelope():
    K, pg, pb, cs, cc = 4, 0.8, 0.4, 0.05, 0.07
    h = lambda p: (1 - p) * p ** (K - 1)
    R = pg ** (K - 1) - pb ** (K - 1); I = h(pg); gap = pg ** K - pb ** K
    L = ladder(h(pg), h(pb), R, I, gap, cs, cc)
    for pi in np.linspace(0, 1, 101):
        vals = {"continue": 0.0, "rescue": pi * h(pb) + (1 - pi) * h(pg) - cs, "repair": pi * gap - cc}
        assert L.act(pi, h(pg), h(pb), gap, cs, cc) == max(vals, key=vals.get)
        if abs(pi - L.pi_rescue_repair) > 1e-6:
            assert (vals["repair"] > vals["rescue"]) == (pi > L.pi_rescue_repair)


def test_informational_price_nonnegative_and_monotone():
    p0 = informational_price(1.0, 1.0, 0.5, 0.9)
    p1 = informational_price(1.0, 1.0, 0.5, 0.7)      # removing a more informative signal costs more
    assert 0 <= p0 <= p1


def test_multistep_table_matches_planner_advantages():
    """With several epochs left, table entries equal Q(a|c) - Q(continue|c) under the base policy."""
    spec, env, post = small_env(K=3, formula=formula_and(3))
    model = env.model()            # epochs_left = horizon = 4
    acts = [a for a in env.actions() if a.kind in ("continue", "takeover", "clarify", "align")]
    table = fill_table(acts, post, model)
    for i, a in enumerate(table.actions):
        for j, h in enumerate(post.hyps):
            assert abs(table.A[i, j] - model.planner.advantage(a, h)) < 1e-9, (str(a), table.A[i, j], model.planner.advantage(a, h))


def test_multistep_reduces_to_one_step_when_one_epoch_left():
    spec, env, post = small_env(K=3, formula=formula_and(3))
    m1 = env.model(epochs_left=1); a = Action("takeover", 0, spec.costs["takeover"])
    from ccrp.planning import BasePolicyPlanner
    bp = BasePolicyPlanner(spec, env.state, 1)
    for h in post.hyps:
        one_step = m1.uplift(a, h) * birnbaum(spec.formula, m1.rates(h), 0) - a.cost
        assert abs(bp.advantage(a, h) - one_step) < 1e-9


def test_episode_runs_end_to_end():
    from ccrp.policies import run_episode, exact_policy, myopic_policy, ccrp_policy, ladder_policy, gated_policy
    spec = EnvSpec(formula_and(4), 4, horizon=4)
    for pol in (exact_policy, myopic_policy, ccrp_policy("S"), ladder_policy, gated_policy("kappa", 0.5)):
        r = run_episode(spec, pol, seed=1)
        assert r.success in (0, 1) and len(r.actions) == 4 and r.regret >= -1e-9


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception:
                fails += 1; print("FAIL", name); traceback.print_exc()
    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
