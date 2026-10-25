#!/usr/bin/env bash
# Run one shard of one PARTNR cell (one agent under one partner failure) with partnr-planner.
#
#   PARTNR_PLANNER=/path/to/partnr-planner bash partnr/run_shard.sh AGENT FAILURE SHARD SHARDS N_EP EP_OFFSET OUT_DIR
#
#   AGENT      react_dialog | ccrp | ccrp_laws | ccrp_rl2 | proagent | hypminds | coela | introspect | llawco | proact
#   FAILURE    goal | skill
#   SHARD      shard index in [0, SHARDS); episodes are split across shards by index modulo SHARDS
#   N_EP       number of validation episodes, starting at EP_OFFSET (the paper uses 30 episodes from 20 or from 50)
#   OUT_DIR    results root; the shard writes to OUT_DIR/<AGENT>__<FAILURE>/shard<SHARD>
#
# Optional environment variables
#   MODEL             Hugging Face model id for both robots (default Qwen/Qwen2.5-7B-Instruct)
#   SEED_BY           set to episode_id to seed each injected failure by its episode (used for every reported run)
#   INJ_SEED_MOD      injection seed = SHARD mod INJ_SEED_MOD, so runs with more shards inject the same failures
#   PARTNR_LLM_DTYPE  bf16 (default) or fp16 on GPUs without native bfloat16
#
# Both configurations of a paired comparison should run on the same GPU type, since greedy decoding still differs
# across hardware.
set -euo pipefail
AGENT=${1:?agent}; FAILURE=${2:?failure}; SHARD=${3:?shard}; SHARDS=${4:?shards}; N_EP=${5:?n_ep}
EP_OFFSET=${6:?ep_offset}; OUT_ROOT=${7:?out_dir}
PP=${PARTNR_PLANNER:?set PARTNR_PLANNER to a partnr-planner checkout}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
OUT_ROOT=$(mkdir -p "$OUT_ROOT" && cd "$OUT_ROOT" && pwd)
OUT=$OUT_ROOT/${AGENT}__${FAILURE}/shard${SHARD}
mkdir -p "$OUT"

# GPUs without native bfloat16 (compute capability < 8) run the language model in fp16
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true); CAPMAJ=${CAP%%.*}
if [ -n "$CAPMAJ" ] && [ "$CAPMAJ" -lt 8 ] 2>/dev/null; then export PARTNR_LLM_DTYPE=fp16; else export PARTNR_LLM_DTYPE=${PARTNR_LLM_DTYPE:-bf16}; fi

cd "$PP"
bash "$CODE/partnr/install_ext.sh" "$PP" "$CODE" >/dev/null

# episodes [EP_OFFSET, EP_OFFSET + N_EP) of the validation split, written once as their own dataset file
SUF=${N_EP}; [ "$EP_OFFSET" != "0" ] && SUF=${N_EP}o${EP_OFFSET}
DATA=data/datasets/partnr_episodes/v0_0/val_main${SUF}.json.gz
if [ ! -f "$DATA" ]; then python - "$N_EP" "$EP_OFFSET" "$SUF" <<'EOF'
import gzip, json, os, sys
n, off, suf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
d = json.load(gzip.open("data/versioned_data/partnr_episodes/v0_0/val.json.gz"))
out = dict(d); out["episodes"] = d["episodes"][off:off + n]
dst = "data/versioned_data/partnr_episodes/v0_0/val_main%s.json.gz" % suf; tmp = "%s.tmp.%d" % (dst, os.getpid())
with gzip.open(tmp, "wt") as f:
    json.dump(out, f)
os.replace(tmp, dst)
EOF
fi

PLAN=evaluation.agents.agent_0.planner.plan_config
case $AGENT in
  react_dialog) CFG=baselines/react_dialog.yaml; EXTRA="";;
  # Team-Plan with its one-step table
  ccrp)         CFG=baselines/ccrp_decentralized.yaml; EXTRA="$PLAN.ccrp.mode=ccrp";;
  # Team-Plan with LLawCo's laws as decision-model terms (own-loop cause, direct execution, message derailment)
  ccrp_laws)    CFG=baselines/ccrp_decentralized.yaml; EXTRA="$PLAN.ccrp.mode=ccrp $PLAN.ccrp.model.self_model=true $PLAN.ccrp.model.derail_prob=0.06";;
  # Team-Plan with the learned value, trained across the 32 calibrated simulator worlds
  ccrp_rl2)     CFG=baselines/ccrp_decentralized.yaml; EXTRA="$PLAN.ccrp.mode=ccrp_rl $PLAN.ccrp.rl_value_path=data/value_partnr_v2.pkl";;
  proagent|hypminds|coela|introspect|llawco|proact)
                CFG=baselines/partner_model_decentralized.yaml; EXTRA="$PLAN.partner_model.mode=$AGENT";;
  *) echo "unknown agent: $AGENT" >&2; exit 2;;
esac

ISEED=$SHARD; [ -n "${INJ_SEED_MOD:-}" ] && ISEED=$((SHARD % INJ_SEED_MOD))
INJ="evaluation.agents.agent_1.planner.plan_config.injection.kind=$FAILURE evaluation.agents.agent_1.planner.plan_config.injection.seed=$ISEED"
[ -n "${SEED_BY:-}" ] && INJ="$INJ evaluation.agents.agent_1.planner.plan_config.injection.seed_by=$SEED_BY"

# a re-queued shard keeps its finished episodes
RES=""; ls "$OUT"/results/*/planner-log/*.json >/dev/null 2>&1 && RES="+resume=True"
COMMON="habitat.dataset.data_path=$DATA num_runs_per_episode=1 evaluation.save_video=False hydra.run.dir=$OUT"
COMMON="$COMMON +episode_mod_filter=[$SHARD,$SHARDS] $RES"
COMMON="$COMMON $PLAN.constrained_generation=False evaluation.agents.agent_1.planner.plan_config.constrained_generation=False"
LLM="$PLAN.llm.generation_params.engine=$MODEL $PLAN.llm.inference_mode=hf"
LLM="$LLM evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=$MODEL"
LLM="$LLM evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf"

echo "agent=$AGENT failure=$FAILURE shard=$SHARD/$SHARDS episodes=[$EP_OFFSET,$((EP_OFFSET + N_EP))) out=$OUT"
python -m habitat_llm.examples.planner_demo --config-name "$CFG" $COMMON $LLM $EXTRA $INJ > "$OUT/stdout.log" 2>&1
echo "done, crashed episodes: $(grep -c 'Skipping evaluating episode' "$OUT/stdout.log" || true)"
