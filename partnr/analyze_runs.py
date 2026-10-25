"""Summarize PARTNR runs: python code/partnr/analyze_runs.py <runs_root>

For every run directory (hydra output) below runs_root, reports the episode metrics written by partnr-planner
(task_state_success, task_percent_complete, steps) from the CSV logs under results/, plus the CCRP decision log
(ccrp_decisions.jsonl): interventions per episode, certified share, posterior accuracy on the injected hidden
state.  Column names are matched loosely because the exact CSV schema comes from the benchmark code.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


def metric_columns(df) -> dict:
    out = {}
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ("success", "percent", "step", "replan", "explanation")):
            if df[c].dtype.kind in "fib":
                out[c] = float(df[c].mean())
    return out


def decisions_summary(path: str) -> dict:
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if not recs:
        return {}
    n_int = [r.get("interventions", 0) for r in recs]
    kinds = {}
    certified = 0; total = 0; post_acc = []
    for r in recs:
        for d in r.get("decisions", []):
            total += 1; certified += int(bool(d.get("certified", False)))
            a = str(d.get("action", "")).split("(")[0]
            if a != "continue":
                kinds[a] = kinds.get(a, 0) + 1
        inj = r.get("injection") or {}
        pb = r.get("final_posterior_goal_bad")
        if pb is not None and inj:
            truth_bad = 1.0 if inj.get("dropped") else 0.0
            post_acc.append(1.0 - abs(truth_bad - pb))
    return {"episodes_logged": len(recs), "interventions_per_episode": float(np.mean(n_int)),
            "certified_share": certified / total if total else None, "intervention_kinds": kinds,
            "posterior_accuracy_goal": float(np.mean(post_acc)) if post_acc else None}


def main(root: str) -> None:
    run_dirs = sorted({os.path.dirname(os.path.dirname(p)) for p in glob.glob(os.path.join(root, "**", "results", "*.csv"), recursive=True)})
    run_dirs += sorted({os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "ccrp_decisions.jsonl"), recursive=True)} - set(run_dirs))
    if not run_dirs:
        print("no runs under", root); return
    for d in run_dirs:
        print(f"\n=== {os.path.relpath(d, root)} ===")
        for csv in sorted(glob.glob(os.path.join(d, "results", "*.csv"))):
            if pd is None:
                print("  (pandas not available)"); break
            try:
                df = pd.read_csv(csv)
            except Exception as e:
                print(f"  {os.path.basename(csv)}: unreadable ({e})"); continue
            m = metric_columns(df)
            print(f"  {os.path.basename(csv)}: {len(df)} rows; " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
        dl = os.path.join(d, "ccrp_decisions.jsonl")
        if os.path.exists(dl):
            print("  decisions:", json.dumps(decisions_summary(dl)))


if __name__ == "__main__" and not (len(sys.argv) > 2 and sys.argv[2] in ("--table", "--paired", "--interventions")):
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")


def pilot_table(root: str) -> None:
    """Aggregate episode logs across shards into a (config x injection) table (crashed episodes count as failures)."""
    if pd is None:
        print("pandas required"); return
    df = _episode_frame(root)
    if df is None:
        print("no episode logs yet"); return
    for c in ("sim_step_count", "replanning_count_0", "replanning_count_1"):
        if c not in df.columns:
            df[c] = np.nan
    g = df.groupby(["config", "injection"])
    out = g.agg(n=("task_state_success", "size"), success=("task_state_success", "mean"),
                pct_complete=("task_percent_complete", "mean"), sim_steps=("sim_step_count", "mean"),
                replans_robot=("replanning_count_0", "mean"), replans_human=("replanning_count_1", "mean"),
                crashed=("errored", "sum")).reset_index()
    out["success_se"] = (out["success"] * (1 - out["success"]) / out["n"].clip(lower=1)) ** 0.5
    order = {"react_silent": 0, "react_dialog": 1, "never": 2, "partner_blind": 3, "ccrp": 4, "always_takeover": 5}
    out["o"] = out["config"].map(order).fillna(9); out = out.sort_values(["injection", "o"]).drop(columns="o")
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    # decision logs
    for js in sorted(glob.glob(os.path.join(root, "*", "shard*", "ccrp_decisions_stream.jsonl"))):
        cell = js.split(os.sep)[-3]
        recs = [json.loads(l) for l in open(js) if l.strip()]
        kinds = {}
        for r in recs:
            a = str(r.get("action", "")).split("(")[0]; kinds[a] = kinds.get(a, 0) + 1
        cert = sum(1 for r in recs if r.get("certified")); 
        print(f"  decisions {cell}: {len(recs)} total, certified {cert}, by kind {kinds}")


def _episode_frame(root: str, errors_as_failures: bool = True):
    """One row per (config, injection, episode).  Episodes that crashed inside partnr-planner are not in the CSV;
    their stats/<id>.json says success=false.  With errors_as_failures they enter as failures (success 0,
    percent complete 0) for whichever config crashed - the same rule for every config."""
    rows = []
    for shard in glob.glob(os.path.join(root, "*", "shard*")):
        cell = shard.split(os.sep)[-2]
        cfg, inj = cell.split("__") if "__" in cell else (cell, "none")
        seen = set()
        for csv in glob.glob(os.path.join(shard, "results", "episode_result_log.csv")):
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            df["config"] = cfg; df["injection"] = inj; df["errored"] = 0
            rows.append(df); seen |= set(str(e) for e in df["episode_id"])
        if errors_as_failures:
            err = []
            for st in glob.glob(os.path.join(shard, "results", "*", "stats", "*.json")):
                try:
                    d = json.load(open(st))
                except Exception:
                    continue
                eid = os.path.basename(st)[:-5]
                if not d.get("success", True) and eid not in seen:
                    err.append({"episode_id": int(eid) if eid.isdigit() else eid, "task_state_success": 0.0,
                                "task_percent_complete": 0.0, "config": cfg, "injection": inj, "errored": 1})
                    seen.add(eid)
            # runs copied without stats/ keep the planner's stdout, which logs every crash as a traceback followed by
            # "Skipping evaluating episode: <id>"
            log = os.path.join(shard, "stdout.log")
            if os.path.exists(log):
                with open(log, errors="replace") as fh:
                    for eid in re.findall(r"Skipping evaluating episode: (\S+)", fh.read()):
                        if eid not in seen:
                            err.append({"episode_id": int(eid) if eid.isdigit() else eid, "task_state_success": 0.0,
                                        "task_percent_complete": 0.0, "config": cfg, "injection": inj, "errored": 1})
                            seen.add(eid)
            if err:
                rows.append(pd.DataFrame(err))
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    df["episode_id"] = df["episode_id"].astype(str)
    # resumed shards append again: keep the last record per episode (a completed run wins over a crash record)
    df = df.sort_values("errored", ascending=False, kind="stable")
    return df.drop_duplicates(subset=["config", "injection", "episode_id"], keep="last")


def _mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs (b: ref succeeds only, c: other succeeds only)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_table(root: str, ref: str = "ccrp") -> None:
    """ref vs every other config on the episodes both ran (same episode, same injection): mean paired difference in
    success and percent-complete, its standard error, discordant pairs and the exact McNemar p-value."""
    if pd is None:
        print("pandas required"); return
    df = _episode_frame(root)
    if df is None:
        print("no episode logs yet"); return
    out = []
    for inj in sorted(df["injection"].unique()) + ["ALL"]:
        d = df if inj == "ALL" else df[df["injection"] == inj]
        key = ["injection", "episode_id"]
        r = d[d["config"] == ref].set_index(key)
        for other in sorted(set(d["config"]) - {ref}):
            o = d[d["config"] == other].set_index(key)
            common = r.index.intersection(o.index)
            if len(common) == 0:
                continue
            ds = r.loc[common, "task_state_success"].values - o.loc[common, "task_state_success"].values
            dp = r.loc[common, "task_percent_complete"].values - o.loc[common, "task_percent_complete"].values
            b = int((ds > 0).sum()); c = int((ds < 0).sum()); n = len(common)
            se = float(ds.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            sep = float(dp.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            out.append({"injection": inj, "vs": other, "n": n, "ref_success": float(r.loc[common, "task_state_success"].mean()),
                        "other_success": float(o.loc[common, "task_state_success"].mean()), "diff": float(ds.mean()), "se": se,
                        "ref_wins": b, "other_wins": c, "mcnemar_p": _mcnemar_p(b, c),
                        "diff_pct": float(dp.mean()), "se_pct": sep})
    if not out:
        print("no common episodes yet"); return
    pd.set_option("display.width", 220)
    print(f"paired comparison: {ref} minus other (same episodes)")
    print(pd.DataFrame(out).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def interventions_table(root: str) -> None:
    """Interventions per episode by (config, injection): messages (clarify / align / query), takeovers, inspections,
    share of episodes with a takeover.  Episodes without any decision record count as zero interventions."""
    if pd is None:
        print("pandas required"); return
    df = _episode_frame(root)
    if df is None:
        print("no episode logs yet"); return
    counts = {}
    for js in glob.glob(os.path.join(root, "*", "shard*", "ccrp_decisions_stream.jsonl")):
        cell = js.split(os.sep)[-3]
        cfg, inj = cell.split("__") if "__" in cell else (cell, "none")
        for l in open(js):
            if not l.strip():
                continue
            r = json.loads(l); a = str(r.get("action", "")).split("(")[0]
            key = (cfg, inj, str(r.get("episode_id")))
            c = counts.setdefault(key, {"messages": 0, "takeovers": 0, "inspect": 0})
            if a in ("clarify", "align", "query"):
                c["messages"] += 1
            elif a == "takeover":
                c["takeovers"] += 1
            elif a == "inspect":
                c["inspect"] += 1
    rows = []
    for (cfg, inj), g in df.groupby(["config", "injection"]):
        if cfg.startswith("react"):
            continue
        per = [counts.get((cfg, inj, str(e)), {"messages": 0, "takeovers": 0, "inspect": 0}) for e in g["episode_id"]]
        rows.append({"config": cfg, "injection": inj, "n": len(per),
                     "messages_per_ep": float(np.mean([c["messages"] for c in per])),
                     "takeovers_per_ep": float(np.mean([c["takeovers"] for c in per])),
                     "share_with_takeover": float(np.mean([c["takeovers"] > 0 for c in per])),
                     "inspect_per_ep": float(np.mean([c["inspect"] for c in per]))})
    if rows:
        pd.set_option("display.width", 200)
        print(pd.DataFrame(rows).sort_values(["config", "injection"]).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "--table":
    pilot_table(sys.argv[1])

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "--interventions":
    interventions_table(sys.argv[1])

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "--paired":
    paired_table(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else "ccrp")
