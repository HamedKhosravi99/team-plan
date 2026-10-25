"""Recompute the numbers of the paper's figures and tables from the result files in results/.

    python scripts/reproduce.py all            # every artifact below, written to results/derived/*.json
    python scripts/reproduce.py figure2        # one artifact: figure2 | figure3 | figure_c1 | table_d1 | appendix_d |
                                               #               coverage | theory

Everything runs on a CPU in under a minute and reads only files in this repository. The PARTNR numbers come from the
per-episode outcome files of the four reported runs, the synthetic numbers from the saved sweep outputs (Figure 3 is
recomputed exactly from the decision model).
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PARTNR = os.path.join(ROOT, "results", "partnr")
SYNTH = os.path.join(ROOT, "results", "synthetic")
DERIVED = os.path.join(ROOT, "results", "derived")


def _analyze_runs():
    """partnr/analyze_runs.py as a module (it is also a command-line script)."""
    spec = importlib.util.spec_from_file_location("analyze_runs", os.path.join(ROOT, "partnr", "analyze_runs.py"))
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["analyze_runs", "-", "--table"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


def _save(name: str, obj) -> None:
    os.makedirs(DERIVED, exist_ok=True)
    with open(os.path.join(DERIVED, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=1)


# --------------------------------------------------------------------------------------------------- PARTNR (Figure 2)
# (row label, run folder, Team-Plan configuration, baseline configuration, failures)
FIGURE2_ROWS = [
    ("ProAgent-style", "existing_agents", "ccrp", "proagent", None),
    ("Hypothetical-Minds-style", "existing_agents", "ccrp", "hypminds", None),
    ("CoELA-style", "existing_agents_rerun", "ccrp", "coela", None),
    ("Ask-when-unsure", "existing_agents_rerun", "ccrp", "introspect", None),
    ("ReAct + dialogue", "existing_agents", "ccrp", "react_dialog", None),
    ("LLawCo-style", "agents_2026", "ccrp_laws", "llawco", None),
    ("ProACT-style", "agents_2026", "ccrp_laws", "proact", None),
    ("one-step table", "learned_value_heldout", "ccrp_rl2", "ccrp", None),
    ("one-step table, goal failures", "learned_value_heldout", "ccrp_rl2", "ccrp", "goal"),
]


def _paired(df, ref: str, other: str, failure=None):
    """Paired success of ref and other on the episodes both completed or crashed (a crash counts as a failure)."""
    if failure:
        df = df[df.injection == failure]
    key = ["injection", "episode_id"]
    r = df[df.config == ref].set_index(key)["task_state_success"]
    o = df[df.config == other].set_index(key)["task_state_success"]
    common = r.index.intersection(o.index)
    return r.loc[common].values.astype(float), o.loc[common].values.astype(float), np.array([e for _, e in common])


def _cluster_ci(d: np.ndarray, episodes: np.ndarray, B: int = 100000, seed: int = 0):
    """95% percentile interval of the mean paired difference, resampling base episodes with all their failures.
    With 30 base episodes the bootstrap distribution sits on a grid of 1/60, so bounds near a grid step need many
    resamples to settle; 100,000 resamples give the same bounds for every seed tried except one 1.7-point step."""
    ids = np.unique(episodes)
    sums = np.array([d[episodes == e].sum() for e in ids]); cnts = np.array([(episodes == e).sum() for e in ids])
    rng = np.random.default_rng(seed); boot = np.empty(B)
    for b in range(B):
        k = rng.integers(len(ids), size=len(ids))
        boot[b] = sums[k].sum() / cnts[k].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def figure2() -> list:
    ar = _analyze_runs(); frames = {}; rows = []
    for label, run, ref, other, failure in FIGURE2_ROWS:
        df = frames.setdefault(run, ar._episode_frame(os.path.join(PARTNR, run)))
        r, o, eps = _paired(df, ref, other, failure)
        d = r - o; w, l = int((d > 0).sum()), int((d < 0).sum())
        lo, hi = _cluster_ci(d, eps)
        rows.append({"comparison": label, "run": run, "team_plan_config": ref, "baseline_config": other,
                     "failures": failure or "goal and skill", "pairs": int(len(d)),
                     "team_plan_success": float(r.mean()), "baseline_success": float(o.mean()),
                     "gain_points": 100 * float(d.mean()), "ci95_points": [100 * lo, 100 * hi],
                     "wins": w, "losses": l, "mcnemar_p": ar._mcnemar_p(w, l)})
    _save("figure2", rows)
    print(f"{'Figure 2':32s} {'n':>3s} {'Team-Plan':>9s} {'baseline':>8s} {'gain [95% interval]':>21s} {'W/L':>6s} {'p':>6s}")
    for x in rows:
        ci = "[{}, {}]".format(*(f"{v:+.0f}" if round(v) else "0" for v in x["ci95_points"]))
        print(f"{x['comparison']:32s} {x['pairs']:3d} {100 * x['team_plan_success']:8.0f}% {100 * x['baseline_success']:7.0f}% "
              f"{x['gain_points']:+7.0f} {ci:>13s} {x['wins']:3d}/{x['losses']:<2d} {x['mcnemar_p']:6.3f}")
    return rows


# ------------------------------------------------------------------------------------------- PARTNR (Appendix D)
def _decision_kinds(run: str, config: str) -> dict:
    """(failure, episode) -> what Team-Plan's decision model did: 'takeover', 'messages' or 'none'."""
    acts = defaultdict(set)
    for js in glob.glob(os.path.join(PARTNR, run, f"{config}__*", "shard*", "ccrp_decisions_stream.jsonl")):
        failure = js.split(os.sep)[-3].split("__")[1]
        for line in open(js):
            if line.strip():
                r = json.loads(line)
                acts[(failure, str(r.get("episode_id")))].add(str(r.get("action", "")).split("(")[0])
    out = {}
    for key, a in acts.items():
        out[key] = "takeover" if "takeover" in a else ("messages" if a & {"clarify", "align", "query"} else "none")
    return out


def table_d1() -> dict:
    """Discordant pairs of Team-Plan and ReAct + dialogue, split by Team-Plan's intervention in the episode."""
    ar = _analyze_runs(); df = ar._episode_frame(os.path.join(PARTNR, "existing_agents"))
    kinds = _decision_kinds("existing_agents", "ccrp")
    key = ["injection", "episode_id"]
    r = df[df.config == "ccrp"].set_index(key)["task_state_success"]
    o = df[df.config == "react_dialog"].set_index(key)["task_state_success"]
    table = {k: [0, 0] for k in ("takeover", "messages", "none")}
    for idx in r.index.intersection(o.index):
        diff = r.loc[idx] - o.loc[idx]
        if diff != 0:
            table[kinds.get((idx[0], str(idx[1])), "none")][0 if diff > 0 else 1] += 1
    labels = {"takeover": "Took over at least one assignment", "messages": "Sent messages only",
              "none": "No intervention"}
    out = {labels[k]: {"wins": v[0], "losses": v[1]} for k, v in table.items()}
    out["All discordant pairs"] = {"wins": sum(v[0] for v in table.values()), "losses": sum(v[1] for v in table.values())}
    _save("table_d1", out)
    print("Table D1, wins/losses against ReAct + dialogue")
    for k, v in out.items():
        print(f"  {k:36s} {v['wins']}/{v['losses']}")
    return out


def appendix_d() -> dict:
    """Share of the task completed in the first comparison, and agreement of two identical Team-Plan runs."""
    ar = _analyze_runs()
    first = ar._episode_frame(os.path.join(PARTNR, "existing_agents"))
    share = {c: float(first[first.config == c]["task_percent_complete"].mean())
             for c in ("ccrp", "react_dialog", "proagent", "hypminds")}
    rerun = ar._episode_frame(os.path.join(PARTNR, "existing_agents_rerun"))
    key = ["injection", "episode_id"]
    a = first[first.config == "ccrp"].set_index(key)["task_state_success"]
    b = rerun[rerun.config == "ccrp"].set_index(key)["task_state_success"]
    common = a.index.intersection(b.index); d = a.loc[common].values - b.loc[common].values
    same = {f: float(np.mean([a.loc[i] == b.loc[i] for i in common if i[0] == f])) for f in ("goal", "skill")}
    out = {"task_share_completed": share,
           "identical_runs": {"pairs": int(len(common)), "success_first": float(a.loc[common].mean()),
                              "success_second": float(b.loc[common].mean()), "disagreements": int((d != 0).sum()),
                              "won_by_first_only": int((d > 0).sum()), "won_by_second_only": int((d < 0).sum()),
                              "same_outcome_share": same}}
    _save("appendix_d", out)
    print("Appendix D, share of the task completed: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in share.items()))
    r = out["identical_runs"]
    print(f"Appendix D, identical runs: {r['pairs']} pairs, success {100 * r['success_first']:.0f}% and "
          f"{100 * r['success_second']:.0f}%, disagree on {r['disagreements']} ({r['won_by_first_only']} against "
          f"{r['won_by_second_only']}), same outcome goal {100 * same['goal']:.0f}% skill {100 * same['skill']:.0f}%")
    return out


# ----------------------------------------------------------------------------------------------- synthetic tasks
def figure3() -> list:
    """One-step score of taking over the first assignment at the prior, cause-blind minus cause-aware (exact)."""
    from ccrp.coupling_sweep import spec_for
    from ccrp.synthetic import SyntheticEnv
    rows = []
    for structure in ("and", "or"):
        for q in (0.15, 0.5):
            for beliefs in (1, 2, 4):
                env = SyntheticEnv(spec_for(structure, 2, beliefs, q), seed=0); post = env.reset()
                takeover = next(a for a in env.actions() if a.kind == "takeover" and a.target == 0)
                model = env.model(epochs_left=1)
                bias = model.myopic_score(takeover, post) - model.exact_advantage(takeover, post)
                rows.append({"task": structure.upper(), "q": q, "object_beliefs": beliefs, "bias": float(bias)})
    _save("figure3", rows)
    print("Figure 3, cause-blind minus cause-aware score of a first-decision takeover (1, 2, 4 beliefs)")
    for structure in ("AND", "OR"):
        for q in (0.15, 0.5):
            vals = [r["bias"] for r in rows if r["task"] == structure and r["q"] == q]
            print(f"  {structure:3s} q={q:<4} " + "  ".join(f"{v:+.3f}" for v in vals))
    return rows


FIGURE_C1_ROWS = [  # (row label, key in raw_<setting>.json; Team-Plan comes from the safe-switch sweep)
    ("Never intervene", "never"), ("Cause-blind, one step", "blind_one_step"),
    ("Cause-aware, exact one-step", "one_step"), ("QMDP-style multi-step", "exact"),
    ("Learned value, greedy", "rl_greedy"), ("Team-Plan", None), ("BO at every decision", "always_deep*"),
    ("Full information", "full_info")]
SETTINGS = [("validation", "Short horizon"), ("long", "Long horizon"), ("large-and6", "Large")]


def figure_c1() -> dict:
    out = {}
    for s, name in SETTINGS:
        raw = json.load(open(os.path.join(SYNTH, "rl", f"raw_{s}.json")))
        sweep = next(e for e in json.load(open(os.path.join(SYNTH, "rl", f"sweep_{s}_safe.json"))) if e["episodes"] == 3000)
        cells = {}
        for label, key in FIGURE_C1_ROWS:
            if key is None:
                cells[label] = {"reward": sweep["teamplan"], "rollouts": sweep["rollouts_per_episode"]}
                continue
            if key.endswith("*"):
                key = key[:-1] + ("_bayes" if key[:-1] + "_bayes" in raw else "")
            r = raw[key]
            cells[label] = {"reward": float(np.mean(r["reward"])), "rollouts": float(np.mean(r["rollouts"])),
                            "success": float(np.mean(r["success"]))}
        out[name] = cells
    raw = json.load(open(os.path.join(SYNTH, "rl", "raw_long.json")))
    g = np.array(raw["rl_greedy"]["reward"])
    gaps = {}
    for key in ("one_step", "exact"):
        d = g - np.array(raw[key]["reward"]); gaps[key] = [float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))]
    out["long_horizon_caption"] = {"greedy_minus_one_step": gaps["one_step"], "greedy_minus_qmdp": gaps["exact"],
                                   "greedy_successes": int(np.sum(raw["rl_greedy"]["success"])),
                                   "one_step_successes": int(np.sum(raw["one_step"]["success"])),
                                   "episodes": len(g)}
    _save("figure_c1", out)
    print("Figure C1, reward (rollouts per episode)")
    print(f"  {'':30s}" + "".join(f"{name:>16s}" for _, name in SETTINGS))
    for label, _ in FIGURE_C1_ROWS:
        cells = []
        for _, name in SETTINGS:
            c = out[name][label]; v = f"{c['reward']:.3f}".lstrip("0")
            cells.append(v + (f" ({c['rollouts']:.0f})" if c["rollouts"] > 0 else ""))
        print(f"  {label:30s}" + "".join(f"{c:>16s}" for c in cells))
    c = out["long_horizon_caption"]
    print(f"  long horizon: greedy minus one-step {c['greedy_minus_one_step'][0]:+.3f} (SE {c['greedy_minus_one_step'][1]:.3f}),"
          f" minus QMDP-style {c['greedy_minus_qmdp'][0]:+.3f} (SE {c['greedy_minus_qmdp'][1]:.3f}),"
          f" successes {c['greedy_successes']} vs {c['one_step_successes']} of {c['episodes']}")
    return out


def coverage() -> dict:
    c = json.load(open(os.path.join(SYNTH, "rl", "coverage.json")))
    fired = sum(v["n_fired"] for v in c.values())
    within = sum(round(v["n_fired"] * v["fired_eps_optimal"]) for v in c.values())
    out = {"per_setting": c, "checks_passed": fired, "chosen_within_eps": within}
    _save("coverage", out)
    print("Appendix A, coverage of the error widths: " + "; ".join(
        f"{s} action-wise {100 * v['actionwise_coverage']:.0f}%, simultaneous {100 * v['simultaneous_coverage']:.0f}%,"
        f" check passed {100 * v['check_fired']:.1f}%" for s, v in c.items()) + f"; within eps in {within} of {fired}")
    return out


def theory() -> dict:
    """Appendix C theory checks: sign of the cause-blind error, its decline over epochs, the disagreement AUROC and
    the reward gap of the cause-aware planner in the four-assignment tasks."""
    from sklearn.metrics import roc_auc_score
    rows = list(csv.DictReader(open(os.path.join(SYNTH, "theory", "e1_sign_law.csv"))))
    sign = {}
    for structure, want in (("conjunctive", 1), ("disjunctive", -1)):
        errs = [float(r["myopic"]) - float(r["exact"]) for r in rows if r["structure"] == structure]
        sign[structure] = float(np.mean([np.sign(e) == want for e in errs if abs(e) > 1e-12]))
    by_t = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(SYNTH, "theory", "e2_front_loading.csv"))):
        by_t[int(r["t"])].append(float(r["abs_error"]))
    mae = [float(np.mean(by_t[t])) for t in sorted(by_t)]
    gate = list(csv.DictReader(open(os.path.join(SYNTH, "theory", "e4_gating.csv"))))
    auroc = float(roc_auc_score([int(r["disagree"]) for r in gate], [float(r["ambiguity"]) for r in gate]))
    gaps = [json.load(open(f))["gap_reward"] for f in sorted(glob.glob(os.path.join(SYNTH, "coupling", "cfg_*.json")))]
    out = {"predicted_sign_share": sign, "mean_abs_error_by_epoch": mae, "disagreement_auroc": auroc,
           "cause_aware_minus_cause_blind_reward": {"min": float(min(gaps)), "max": float(max(gaps)),
                                                     "combinations": len(gaps)}}
    _save("theory", out)
    print(f"Appendix C theory checks: predicted sign in {100 * sign['conjunctive']:.0f}% (AND) and "
          f"{100 * sign['disjunctive']:.0f}% (OR) of cases; mean absolute error by epoch "
          + ", ".join(f"{v:.3f}" for v in mae) + f"; AUROC {auroc:.2f}; reward gap {min(gaps):+.3f} to {max(gaps):+.3f}"
          f" over {len(gaps)} combinations")
    return out


ARTIFACTS = {"figure2": figure2, "figure3": figure3, "figure_c1": figure_c1, "table_d1": table_d1,
             "appendix_d": appendix_d, "coverage": coverage, "theory": theory}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifact", choices=["all", *ARTIFACTS])
    a = ap.parse_args()
    for name in (ARTIFACTS if a.artifact == "all" else [a.artifact]):
        ARTIFACTS[name](); print()


if __name__ == "__main__":
    main()
