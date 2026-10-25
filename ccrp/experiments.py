"""Synthetic-benchmark experiments E1-E6 (method document, section 15; predictions P1-P8).

Usage:  python -m ccrp.experiments --exp all --out results --quick
Each experiment writes a CSV and a PNG into the output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from .core import Action, CONTINUE, CCRPPlanner, PlannerConfig, fill_table, certify, INFORMATIONAL
from .policies import (PolicyStats, ambiguity, ccrp_policy, exact_policy, gated_policy, kappa_exposure,
                       ladder_policy, myopic_policy, never_policy, no_query_exact_policy, oracle_policy, run_episode)
from .synthetic import EnvSpec, ExactModel, SyntheticEnv, formula_and, formula_or, random_formula, components_of_query


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ----------------------------------------------------------------------------
# E1: sign law.  Local score minus exact advantage for takeovers, conjunctive vs disjunctive tasks.
# ----------------------------------------------------------------------------
def e1_sign_law(out: str, n: int, K: int = 4) -> Dict[str, Any]:
    rows = []
    for kind, fm in (("conjunctive", formula_and(K)), ("disjunctive", formula_or(K))):
        spec = EnvSpec(fm, K, n_objects=1, n_skills=1, horizon=3)
        for seed in range(n):
            env = SyntheticEnv(spec, seed); post = env.reset()
            # random evidence: advance a couple of epochs with 'continue' to diversify posteriors
            for _ in range(int(env.rng.integers(0, 3))):
                post, _ = env.step(CONTINUE)
            model = env.model(epochs_left=1)
            for a in [b for b in env.actions() if b.kind == "takeover"]:
                exact = model.exact_advantage(a, post); myo = model.myopic_score(a, post)
                rows.append({"structure": kind, "seed": seed, "k": a.target, "exact": exact, "myopic": myo,
                             "error": myo - exact, "pi_bad": post.component_mass("goal", "bad")})
    write_csv(os.path.join(out, "e1_sign_law.csv"), rows)
    summ = {}
    for kind in ("conjunctive", "disjunctive"):
        err = np.array([r["error"] for r in rows if r["structure"] == kind])
        summ[kind] = {"mean_error(myopic-exact)": float(err.mean()), "frac_over": float((err > 1e-12).mean()),
                      "frac_under": float((err < -1e-12).mean()), "n": int(len(err))}
    plot_e1(out, rows)
    return summ


def plot_e1(out: str, rows: List[Dict[str, Any]]) -> None:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for kind, col in (("conjunctive", "C3"), ("disjunctive", "C0")):
        x = [r["pi_bad"] for r in rows if r["structure"] == kind]; y = [r["error"] for r in rows if r["structure"] == kind]
        ax.scatter(x, y, s=10, alpha=0.6, c=col, label=kind)
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("suspicion P(goal misread)"); ax.set_ylabel("myopic - exact")
    ax.set_title("E1 sign law: local score over/under-estimates rescues"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out, "e1_sign_law.png"), dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
# E2: front-loading.  |myopic - exact| over epochs as evidence about the partner accumulates.
# ----------------------------------------------------------------------------
def e2_front_loading(out: str, n: int, K: int = 4, T: int = 8) -> Dict[str, Any]:
    spec = EnvSpec(formula_and(K), K, n_objects=1, n_skills=1, horizon=T)
    rows = []
    for seed in range(n):
        env = SyntheticEnv(spec, seed); post = env.reset()
        while not env.done():
            model = env.model(epochs_left=1); a = Action("takeover", 0, spec.costs["takeover"])
            rows.append({"seed": seed, "t": env.state.t, "abs_error": abs(model.myopic_score(a, post) - model.exact_advantage(a, post)),
                         "pi_var": post.component_mass("goal", "bad") * (1 - post.component_mass("goal", "bad"))})
            post, _ = env.step(CONTINUE)
    write_csv(os.path.join(out, "e2_front_loading.csv"), rows)
    by_t = {}
    for r in rows:
        by_t.setdefault(r["t"], []).append(r["abs_error"])
    curve = {t: float(np.mean(v)) for t, v in sorted(by_t.items())}
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.2)); ax.plot(list(curve), list(curve.values()), marker="o")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean |myopic - exact|"); ax.set_title("E2 front-loading of the local error")
    fig.tight_layout(); fig.savefig(os.path.join(out, "e2_front_loading.png"), dpi=150); plt.close(fig)
    return {"mean_abs_error_by_epoch": curve}


# ----------------------------------------------------------------------------
# E3: escalation ladder.  Optimal action vs suspicion for small and large conjunctive tasks.
# ----------------------------------------------------------------------------
def e3_ladder(out: str, Ks=(2, 6)) -> Dict[str, Any]:
    rows = []
    for K in Ks:
        spec = EnvSpec(formula_and(K), K, n_objects=1, n_skills=1, horizon=3,
                       prior_bad={"goal": 0.3, "obj": 0.0, "skill": 0.0})
        env = SyntheticEnv(spec, 0); env.reset()
        model = env.model(epochs_left=1)          # Proposition 10 is a one-step statement
        hyps = spec.prior().hyps
        good = [h for h in hyps if h.get("goal") == "good"][0]; bad = [h for h in hyps if h.get("goal") == "bad"][0]
        take = Action("takeover", 0, spec.costs["takeover"]); clar = Action("clarify", "goal", spec.costs["clarify"])
        for pi in np.linspace(0, 1, 101):
            vals = {"continue": 0.0}
            vals["rescue"] = pi * model.uplift(take, bad) * model.consequence(0, bad) + (1 - pi) * model.uplift(take, good) * model.consequence(0, good) - take.cost
            vals["repair"] = pi * spec.repair_success * (model.continuation_value(good) - model.continuation_value(bad)) - clar.cost
            rows.append({"K": K, "pi": float(pi), **vals, "best": max(vals, key=vals.get)})
    write_csv(os.path.join(out, "e3_ladder.csv"), rows)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(Ks), figsize=(4 * len(Ks), 3.2), sharey=False)
    summ = {}
    for ax, K in zip(np.atleast_1d(axes), Ks):
        sub = [r for r in rows if r["K"] == K]
        for key, col in (("rescue", "C1"), ("repair", "C2")):
            ax.plot([r["pi"] for r in sub], [r[key] for r in sub], label=key, c=col)
        ax.axhline(0, color="k", lw=0.8, label="continue"); ax.set_title(f"K={K}: order " + " -> ".join(dict.fromkeys(r["best"] for r in sub)))
        ax.set_xlabel("suspicion pi"); ax.legend(fontsize=7)
        summ[K] = list(dict.fromkeys(r["best"] for r in sub))
    fig.tight_layout(); fig.savefig(os.path.join(out, "e3_ladder.png"), dpi=150); plt.close(fig)
    return {"rung_order_by_K": summ}


# ----------------------------------------------------------------------------
# E4: gating.  Does common-cause ambiguity predict myopic != exact better than kappa / entropy?
# ----------------------------------------------------------------------------
def e4_gating(out: str, n: int, K: int = 5) -> Dict[str, Any]:
    rows = []
    rng = np.random.default_rng(0)
    for seed in range(n):
        fm = random_formula(K, np.random.default_rng(seed))
        spec = EnvSpec(fm, K, n_objects=2, n_skills=1, horizon=4)
        env = SyntheticEnv(spec, seed); post = env.reset()
        kap = kappa_exposure(spec)
        for _ in range(int(rng.integers(0, 3))):
            post, _ = env.step(CONTINUE)
        model = env.model(epochs_left=1)
        cands = [a for a in env.actions() if a.kind not in INFORMATIONAL]
        ex = np.array([model.exact_advantage(a, post) for a in cands]); my = np.array([model.myopic_score(a, post) for a in cands])
        disagree = int(cands[int(ex.argmax())] != cands[int(my.argmax())])
        phys = [a for a in cands if a.kind == "takeover"]
        p = post.probs; ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        rows.append({"seed": seed, "formula": fm.describe(), "disagree": disagree,
                     "ambiguity": max(ambiguity(model, a, post) for a in phys),
                     "kappa": max(kap[a.target] * abs(sum(pp * model.uplift(a, h) for h, pp in zip(post.hyps, post.probs))) for a in phys),
                     "entropy": ent, "regret_myopic": float(ex.max() - ex[int(my.argmax())])})
    write_csv(os.path.join(out, "e4_gating.csv"), rows)
    lab = np.array([r["disagree"] for r in rows])
    summ = {k: auroc(np.array([r[k] for r in rows]), lab) for k in ("ambiguity", "kappa", "entropy")}
    summ["disagreement_rate"] = float(lab.mean()); summ["n"] = len(rows)
    return {"auroc_for_predicting_local!=exact": summ}


# ----------------------------------------------------------------------------
# E5: policies.  Episode reward / regret / deliberation counts across policies.
# ----------------------------------------------------------------------------
def e5_policies(out: str, n: int, K: int = 5, T: int = 4, structure: str = "mixed") -> Dict[str, Any]:
    pols = {
        "never": never_policy,
        "myopic(partner-blind)": myopic_policy,
        "exact-no-query": no_query_exact_policy,
        "ladder": ladder_policy,
        "kappa-gate": gated_policy("kappa", 0.15),
        "ambiguity-gate": gated_policy("ambiguity", 0.01),
        "ccrp(E)": ccrp_policy("E"),
        "ccrp(S)": ccrp_policy("S", n_sample=48),
        "exact+VoI(reference)": exact_policy,
        "oracle(knows partner)": oracle_policy,
    }
    rows = []
    for seed in range(n):
        fm = formula_and(K) if structure == "and" else (formula_or(K) if structure == "or" else random_formula(K, np.random.default_rng(seed)))
        spec = EnvSpec(fm, K, n_objects=2, n_skills=1, horizon=T, costs={"takeover": 0.12, "help": 0.06, "clarify": 0.06, "align": 0.05, "query": 0.02})
        for name, pol in pols.items():
            t0 = time.time(); r = run_episode(spec, pol, seed); dt = time.time() - t0
            rows.append({"policy": name, "seed": seed, "success": r.success, "reward": r.reward, "cost": r.cost, "regret": r.regret,
                         "n_queries": sum(a.startswith("query") for a in r.actions), "n_repairs": sum(a.startswith(("clarify", "align")) for a in r.actions),
                         "n_takeovers": sum(a.startswith("takeover") for a in r.actions), "seconds": dt})
    write_csv(os.path.join(out, f"e5_policies_{structure}.csv"), rows)
    summ = {}
    for name in pols:
        sub = [r for r in rows if r["policy"] == name]
        summ[name] = {k: float(np.mean([r[k] for r in sub])) for k in ("success", "reward", "regret", "n_queries", "n_repairs", "n_takeovers")}
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.4)); names = list(pols)
    ax.bar(range(len(names)), [summ[nm]["reward"] for nm in names], yerr=[np.std([r["reward"] for r in rows if r["policy"] == nm]) / np.sqrt(n) for nm in names])
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7); ax.set_ylabel("mean reward")
    ax.set_title(f"E5 policies ({structure}, K={K}, T={T}, n={n})"); fig.tight_layout()
    fig.savefig(os.path.join(out, f"e5_policies_{structure}.png"), dpi=150); plt.close(fig)
    return summ


# ----------------------------------------------------------------------------
# E6: query targeting.  Coupling-share targeting vs entropy targeting vs no queries.
# ----------------------------------------------------------------------------
def e6_queries(out: str, n: int, K: int = 5, T: int = 4) -> Dict[str, Any]:
    pols = {"ccrp(E)+coupling-share": ccrp_policy("E", query_targeting=True),
            "ccrp(E)+all-queries": ccrp_policy("E", query_targeting=False),
            "exact-no-query": no_query_exact_policy}
    rows = []
    for seed in range(n):
        spec = EnvSpec(random_formula(K, np.random.default_rng(seed)), K, n_objects=2, n_skills=2, horizon=T)
        for name, pol in pols.items():
            r = run_episode(spec, pol, seed)
            rows.append({"policy": name, "seed": seed, "reward": r.reward, "success": r.success, "regret": r.regret,
                         "n_queries": sum(a.startswith("query") for a in r.actions)})
    write_csv(os.path.join(out, "e6_queries.csv"), rows)
    return {name: {k: float(np.mean([r[k] for r in rows if r["policy"] == name])) for k in ("reward", "success", "regret", "n_queries")} for name in pols}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all"); ap.add_argument("--out", default="results"); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n = 12 if args.quick else args.n
    todo = ["e1", "e2", "e3", "e4", "e5", "e6"] if args.exp == "all" else args.exp.split(",")
    summary: Dict[str, Any] = {}
    for e in todo:
        t0 = time.time()
        if e == "e1": summary["e1_sign_law"] = e1_sign_law(args.out, n)
        if e == "e2": summary["e2_front_loading"] = e2_front_loading(args.out, n)
        if e == "e3": summary["e3_ladder"] = e3_ladder(args.out)
        if e == "e4": summary["e4_gating"] = e4_gating(args.out, 4 * n)
        if e == "e5":
            for st in ("and", "mixed"):
                summary[f"e5_policies_{st}"] = e5_policies(args.out, n, structure=st)
        if e == "e6": summary["e6_queries"] = e6_queries(args.out, n)
        print(f"[{e}] done in {time.time() - t0:.1f}s")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
