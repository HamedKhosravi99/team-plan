# Results

Outputs of the runs behind the paper's figures and tables. `scripts/reproduce.py` reads these files and writes the
paper's numbers to `derived/`.

## `partnr/`

One folder per reported PARTNR run, with one cell per agent and partner failure:

```
<run>/<agent>__<failure>/shard<k>/
  results/episode_result_log.csv          one row per completed episode, including episode_id, instruction,
                                          task_state_success (0 or 1), task_percent_complete, sim_step_count and the
                                          planning calls of each robot (replanning_count_0, replanning_count_1)
  results/<dataset>/stats/<episode>.json  benchmark statistics per episode, including episodes that crashed
  ccrp_decisions*.jsonl                   Team-Plan only: one record per decision (action, posterior, check passed)
  partner_model*.jsonl                    reimplemented agents only: their reasoning-module outputs
  stdout.log                              agents_2026 only: planner output, whose crash marker is
                                          "Skipping evaluating episode: <id>"
```

| Run | Contents |
|---|---|
| `existing_agents` | Team-Plan (`ccrp`), ProAgent-style, Hypothetical-Minds-style and ReAct + dialogue, episodes 20 to 49 |
| `existing_agents_rerun` | a second run of the same Team-Plan configuration, CoELA-style and ask-when-unsure, episodes 20 to 49 |
| `agents_2026` | Team-Plan with LLawCo's laws (`ccrp_laws`), LLawCo-style and ProACT-style, episodes 20 to 49 |
| `learned_value_heldout` | Team-Plan with the learned value (`ccrp_rl2`) and with the one-step table (`ccrp`), held-out episodes 50 to 79 |
| `calibration` | fit of the PARTNR simulator to the tuning episodes (`calib.json`) and the 32 worlds the learned value was trained in (`plausible_worlds.json`) |

## `synthetic/`

| Folder | Files | Produced by |
|---|---|---|
| `theory` | `e1_sign_law.csv` (cause-blind and cause-aware scores), `e2_front_loading.csv` (error by epoch), `e4_gating.csv` (disagreement and its predictors) | `ccrp.experiments` |
| `coupling` | `cfg_<i>.json`, one per combination of task, shared beliefs, prior `q` and horizon, with success, reward and their paired gaps | `ccrp.coupling_sweep` |
| `rl` | `raw_<setting>.json` (per-episode reward, success and rollouts of each policy on fixed seeds), `sweep_<setting>_safe.json` (Team-Plan with the safe switch, by training budget), `coverage.json`, `value_<setting>.pkl` (learned values) | `ccrp.rl_experiment`, `ccrp.rl_eval`, `ccrp.rl_sweep`, `ccrp.rl_coverage` |

The settings `validation`, `long` and `large-and6` are the paper's short horizon, long horizon and large settings.

## `derived/`

One JSON file per artifact (`figure2`, `figure3`, `figure_c1`, `table_d1`, `appendix_d`, `coverage`, `theory`),
written by `python scripts/reproduce.py all`.
