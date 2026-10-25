#!/bin/bash
# Install the Team-Plan extension (internal name CCRP) into a partnr-planner checkout.
# Usage: bash install_ext.sh /path/to/partnr-planner /path/to/team-plan
set -euo pipefail
PP=${1:?partnr-planner path}
CODE=${2:?code dir (contains ccrp/ and partnr/)}
# Many jobs start at once and share one checkout: install only when the source changed (content hash), under a
# lock, and swap directories atomically (an rm -rf / cp race between jobs left a broken package behind once).
HASH=$( (cd "$CODE" && find ccrp partnr/habitat_llm_ext partnr/conf -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.pkl" \) -print0 \
        | sort -z | xargs -0 cat) | md5sum | cut -d" " -f1 )
if [ "$(cat "$PP/.ext_installed" 2>/dev/null)" != "$HASH" ]; then
  exec 9>"$PP/.install_ext.lock"; flock -w 600 9 || echo "install_ext: lock timeout, installing anyway"
  if [ "$(cat "$PP/.ext_installed" 2>/dev/null)" != "$HASH" ]; then
    for pkg in habitat_llm_ext ccrp; do
      src="$CODE/partnr/$pkg"; [ "$pkg" = "ccrp" ] && src="$CODE/ccrp"
      tmp="$PP/.$pkg.new.$$"; rm -rf "$tmp"; cp -r "$src" "$tmp"; find "$tmp" -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null
      [ -d "$PP/$pkg" ] && mv "$PP/$pkg" "$PP/.$pkg.old.$$"
      mv "$tmp" "$PP/$pkg"; rm -rf "$PP/.$pkg.old.$$" 2>/dev/null || true
    done
    # hydra configs (single files, copied to a temp name and renamed into place)
    for f in instruct/zero_shot_dialog_prompt.yaml planner/message_aware_react_planner.yaml planner/ccrp_robot_planner.yaml \
             baselines/ccrp_decentralized.yaml baselines/react_dialog.yaml \
             planner/partner_model_robot_planner.yaml baselines/partner_model_decentralized.yaml; do
      cp "$CODE/partnr/conf/$f" "$PP/habitat_llm/conf/$f.tmp.$$" && mv "$PP/habitat_llm/conf/$f.tmp.$$" "$PP/habitat_llm/conf/$f"
    done
    echo "$HASH" > "$PP/.ext_installed"
    echo "installed CCRP extension into $PP ($HASH)"
  fi
  flock -u 9 2>/dev/null || true
else
  echo "CCRP extension up to date in $PP ($HASH)"
fi

# The public release leaves HFModel.init_local_model unimplemented (inference_mode: hf fails).  Add a plain
# transformers loader to the Llama class (idempotent).
python3 - "$PP/habitat_llm/llm/llama.py" <<'PY'
import sys, re
p = sys.argv[1]; s = open(p).read()
if "def init_local_model" not in s:
    method = '''    _LOCAL_MODELS = {}   # engine -> (tokenizer, model); the two planners share one copy of the weights

    def init_local_model(self):
        """Load tokenizer and causal LM from generation_params.engine (HF hub id or local path), once per process."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        engine = self.generation_params.engine
        if engine not in Llama._LOCAL_MODELS:
            tok = AutoTokenizer.from_pretrained(engine)
            mdl = AutoModelForCausalLM.from_pretrained(engine, torch_dtype=torch.bfloat16, device_map="auto")
            mdl.eval()
            Llama._LOCAL_MODELS[engine] = (tok, mdl)
        self.tokenizer, self.model = Llama._LOCAL_MODELS[engine]

'''
    anchor = "    def generate_hf_llm("
    assert anchor in s, "anchor not found in llama.py"
    s = s.replace(anchor, method + anchor, 1)
    open(p, "w").write(s)
    print("patched llama.py with init_local_model")
else:
    print("llama.py already has init_local_model")
PY

# Model dtype from PARTNR_LLM_DTYPE (bf16 default; fp16 on GPUs without native bf16, e.g. Turing).  Idempotent.
python3 - "$PP/habitat_llm/llm/llama.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = 'mdl = AutoModelForCausalLM.from_pretrained(engine, torch_dtype=torch.bfloat16, device_map="auto")'
new = ('import os\n            dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[os.environ.get("PARTNR_LLM_DTYPE", "bf16")]\n'
       '            mdl = AutoModelForCausalLM.from_pretrained(engine, torch_dtype=dt, device_map="auto")')
if old in s:
    open(p, "w").write(s.replace(old, new)); print("patched llama.py: dtype from PARTNR_LLM_DTYPE")
PY

# The evaluation loop runs len(dataset) iterations of a cycling episode iterator; with +episode_mod_filter the loop
# count can exceed the filtered subset, so a shard reruns its episodes (and overwrites their results).  Stop once an
# episode comes around again.  Idempotent.
python3 - "$PP/habitat_llm/examples/planner_demo.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = """            for _ in range(num_episodes):
                # Get episode id
                episode_id = env_interface.env.env.env._env.current_episode.episode_id
"""
new = """            seen_ids = set()
            for _ in range(num_episodes):
                # Get episode id
                episode_id = env_interface.env.env.env._env.current_episode.episode_id
                # the episode iterator cycles: stop once an episode comes around again
                if episode_id in seen_ids:
                    break
                seen_ids.add(episode_id)
"""
if "seen_ids = set()" not in s:
    assert old in s, "planner_demo loop not found"
    open(p, "w").write(s.replace(old, new, 1)); print("patched planner_demo.py: one pass per episode")
PY
