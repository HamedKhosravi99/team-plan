"""Robot planners that model the partner, or their own uncertainty, with the language model: re-implementations, in
PARTNR, of the core mechanism of six recent methods, used as baselines for Team-Plan.

  proagent  ProAgent (Zhang et al., AAAI 2024): at each planning step the LLM infers the teammate's current intention,
            first correcting its previous belief against what the teammate actually did since ("belief correction"),
            and the agent then plans its own skill given that intention.
  hypminds  Hypothetical Minds (Cross et al., ICLR 2025): a theory-of-mind module keeps a small bank of natural-language
            hypotheses about the other agent, scores each against its observed behaviour, proposes new ones, refines
            the bank, and the agent plans conditioned on the best-scoring hypothesis.
  coela     CoELA (Zhang et al., ICLR 2024): a memory module summarises task progress, the agent's own and the other
            agent's actions and the dialogue; a communication module drafts a message; a reasoning module chooses one
            high-level plan among "continue my own part", "send the message" and "do the other agent's part X".
            Sending is CoELA's communication action, so a chosen message is posted directly (at most max_messages).
  introspect  Introspective planning / Ask-to-Act style help-seeking (Liang et al., NeurIPS 2024; Ramrakhya et al.,
            2025): the LLM reasons about which of the robot's options are consistent with the situation, the options
            are scored by the model's next-token probabilities (multiple-choice, as in KnowNo-style conformal
            planning), and when more than one option has probability >= ask_threshold the robot asks the other agent
            a clarifying question (at most max_questions); otherwise it is told its confident choice.  This is help
            triggered by the robot's own uncertainty, not by a model of the partner's failure.
  llawco    LLawCo (Zhou, Gan and Cherian, ICML 2026), its prompting variant ("w/o SFT"): the laws of cooperation the
            authors report for a Qwen backbone on PARTNR are put in the robot's prompt, and the robot is asked to ground
            each action in the most relevant law.  No partner-model module runs.
  proact    ProACT (Yang, Xu, Pei and Wang, 2026): at each decision point the LLM diagnoses whether the collaboration
            shows an intervention-worthy breakdown (forgotten responsibility, uncertainty, underspecified plan,
            conflict, loop, goal drift, idle partner, risky action), decides whether to speak or stay silent, and when
            speaking writes one concise message with the matching skill; the message is posted directly (at most
            max_messages).

All see exactly the partner evidence Team-Plan sees: the division of work, the partner's action history (actions and
their results) and the partner's messages.  The module's output is appended to the robot's ReAct prompt at replanning
points (at most every `every_steps` simulator steps, after `min_steps`); the robot then acts with its usual tools,
including SendMessage and doing a partner's object itself.  There is no execution layer and no monitoring: deciding
whether and how to help is left to the language model, as in the original methods.

plan_config.partner_model: {mode: proagent|hypminds|coela|introspect|llawco|proact, every_steps, min_steps, history_len, max_tokens,
                           n_hypotheses, max_messages, max_questions, ask_threshold}
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from habitat_llm_ext.allocation import allocation_text
from habitat_llm_ext.tools.general.message_board import board
from .message_aware_planner import MessageAwareReactPlanner


def _log_dir() -> str:
    try:
        from hydra.core.hydra_config import HydraConfig
        return HydraConfig.get().runtime.output_dir
    except Exception:
        return os.environ.get("CCRP_LOG_DIR", ".")


PROAGENT_PROMPT = """{system_tag}You analyse a teammate in a shared household task, to help a robot plan.{eot_tag}{user_tag}Task: {task}

{division}

The other agent's recent actions and their results (oldest first):
{history}

Messages from the other agent: {messages}

Your previous belief about the other agent's intention: {previous}

Answer in exactly this format:
Correction: <compare the previous belief with what the other agent actually did since; say if it was right and correct it>
Intention: <what the other agent is trying to do now, and which of its parts it will and will not complete>
Advice: <should the robot continue its own part, send the other agent a message (say what), or do one of the other agent's parts itself (say which)? One sentence.>{eot_tag}{assistant_tag}"""

HYPMINDS_PROMPT = """{system_tag}You model the mind of a teammate in a shared household task, to help a robot plan.{eot_tag}{user_tag}Task: {task}

{division}

The other agent's recent actions and their results (oldest first):
{history}

Messages from the other agent: {messages}

Current hypotheses about the other agent (what it believes, intends or is able to do), with scores from 0 to 10 for how well they explain its behaviour so far:
{bank}

1. Re-score every current hypothesis against the behaviour above.
2. Propose up to {n_new} new hypotheses that would better explain the behaviour (for example: it forgot part of its task, it believes an object is somewhere else, its actions are failing, it is doing fine).
3. Pick the best hypothesis and say what the robot should do about it.

Answer in exactly this format:
Score H<k>: <0-10>   (one line per current hypothesis)
New: <hypothesis>     (one line per new hypothesis)
Best: <the best hypothesis>
Advice: <should the robot continue its own part, send the other agent a message (say what), or do one of the other agent's parts itself (say which)? One sentence.>{eot_tag}{assistant_tag}"""

COELA_PROMPT = """{system_tag}You are the reasoning module of a cooperative robot that shares a household task with another agent. Your memory of the episode is below.{eot_tag}{user_tag}Task: {task}

{division}

The robot's own recent actions and their results (oldest first):
{own_history}

The other agent's recent actions and their results (oldest first):
{history}

Dialogue so far (oldest first):
{dialogue}

1. Progress: summarise which parts of the task are done, which are still open, and who is working on what.
2. Message: write the single most useful message to send the other agent now, or "none" if no message is needed.
3. Plan: choose exactly one of: continue my own part | send the message | do the other agent's part <which>.

Answer in exactly this format:
Progress: <one or two sentences>
Message: <the message, or none>
Plan: <one of the three plans>{eot_tag}{assistant_tag}"""

INTROSPECT_OPTIONS = [("A", "continue its own part"),
                      ("B", "send the other agent a message (a reminder of its part, or where an object is)"),
                      ("C", "do one of the other agent's open parts itself")]

INTROSPECT_PROMPT = """{system_tag}You help a robot decide whether it should step in on another agent's part of a shared household task.{eot_tag}{user_tag}Task: {task}

{division}

The other agent's recent actions and their results (oldest first):
{history}

Dialogue so far (oldest first):
{dialogue}

Options for the robot now:
{options}

Reason about which options are consistent with what the other agent is doing and what is still open, then write the question the robot would ask the other agent if it were unsure.
Answer in exactly this format:
Reasoning: <two sentences>
Question: <one short question to the other agent>{eot_tag}{assistant_tag}"""

INTROSPECT_CHOICE = "{user_tag}Which option should the robot take now? Answer with a single letter: A, B or C.{eot_tag}{assistant_tag}"

# LLawCo (ICML 2026): the laws reported for the Qwen backbone on PARTNR; the instruction follows its law-guided reasoning
LLAWCO_LAWS = ["Plan Before Acting", "Use Clear and Effective Communication", "Avoid Infinite Loops",
               "Transport Independently", "Confirm Completion"]
LLAWCO_NOTE = ("Laws of cooperation for this task: " + "; ".join(f"({i + 1}) {l}" for i, l in enumerate(LLAWCO_LAWS))
               + ". Before each action, state which law is most relevant and justify the action by it.")

PROACT_PROMPT = """{system_tag}You are a proactive collaboration assistant for a robot that shares a household task with another agent. You speak only when it helps.{eot_tag}{user_tag}Task: {task}

{division}

The other agent's recent actions and their results (oldest first):
{history}

Dialogue so far (oldest first):
{dialogue}

1. Diagnose: does the collaboration show an intervention-worthy breakdown? Choose one: none, forgotten responsibility, uncertainty, underspecified plan, conflict, loop, goal drift, idle partner, risky action.
2. Decide: should the robot speak now, or stay silent so as not to interrupt productive work?
3. If speaking, write one concise message to the other agent with the matching skill (constraint reminder, clarification question, plan completion suggestion, conflict mediation, goal refocusing, loop-breaking guidance, participation balancing, risk flagging).

Answer in exactly this format:
Breakdown: <one of the categories>
Decision: <speak or silent>
Message: <the message, or none>{eot_tag}{assistant_tag}"""


class PartnerModelRobotPlanner(MessageAwareReactPlanner):
    def __init__(self, plan_config, env_interface):
        # defaults first: the base constructor calls reset() before the rest of this constructor runs
        self.pm_mode = "proagent"; self.iter = 0; self.last_analysis = -10**9; self.belief = "none yet"
        self.bank: List[List[Any]] = []; self.analyses: List[Dict[str, Any]] = []; self.episode_id = None
        self.n_sent = 0; self.n_asked = 0; self.laws_given = False
        super().__init__(plan_config, env_interface)
        c = dict(plan_config.get("partner_model", {}))
        self.pm_mode = str(c.get("mode", "proagent"))
        self.every_steps = int(c.get("every_steps", 400))      # same spacing as Team-Plan's interventions
        self.min_steps = int(c.get("min_steps", 600))          # let the partner start, as Team-Plan does
        self.history_len = int(c.get("history_len", 12))
        self.max_tokens = int(c.get("max_tokens", 220))
        self.n_hypotheses = int(c.get("n_hypotheses", 3))
        self.max_messages = int(c.get("max_messages", 2))        # CoELA's direct messages (Team-Plan's cap)
        self.max_questions = int(c.get("max_questions", 2))      # clarifying questions of the introspective planner
        self.ask_threshold = float(c.get("ask_threshold", 0.25))  # an option enters the prediction set at p >= this

    def reset(self):
        if getattr(self, "analyses", None):
            self._flush_log()
        super().reset()
        self.iter = 0; self.last_analysis = -10**9; self.belief = "none yet"; self.bank = []; self.analyses = []
        self.n_sent = 0; self.n_asked = 0; self.laws_given = False

    # ---- evidence (identical to what Team-Plan's model reads) --------------------------------------------------
    def _partner_uid(self) -> int:
        return 1 - int(self.agents[0].uid)

    def _history_text(self, uid: int = None) -> str:
        uid = self._partner_uid() if uid is None else uid
        hist = self.env_interface.agent_action_history.get(uid, [])[-self.history_len:]
        lines = []
        for el in hist:
            a = el.action or ()
            name = a[0] if a else "?"; arg = a[1] if len(a) > 1 else ""
            resp = " ".join(str(el.response or "(running)").split())[:90]
            lines.append(f"- {name}[{arg}] -> {resp}")
        return "\n".join(lines) if lines else "- (no actions yet)"

    def _messages_text(self) -> str:
        msgs = [t for _, t, _ in board.all_for(self.agents[0].uid)]
        return " | ".join(f'"{m}"' for m in msgs[-4:]) if msgs else "none"

    def _dialogue_text(self) -> str:
        me = int(self.agents[0].uid)
        lines = [f'- {"robot" if s == me else "other agent"}: "{t}"' for s, t, _ in board.messages[-8:]]
        return "\n".join(lines) if lines else "- (no messages yet)"

    def _option_probs(self, prompt: str, letters: List[str]) -> List[float]:
        """Next-token probabilities of the option letters after `prompt` (multiple-choice scoring), renormalised.
        Uses the planner's own HF model; without one, the letter of a greedy answer gets probability 1."""
        tok, mdl = getattr(self.llm, "tokenizer", None), getattr(self.llm, "model", None)
        if tok is None or mdl is None or not hasattr(mdl, "device"):
            self.last_scorer = "greedy"
            ans = self._generate(prompt).strip()[:3].upper()
            return [1.0 if ans.startswith(L) else 0.0 for L in letters] if any(ans.startswith(L) for L in letters) \
                else [1.0 / len(letters)] * len(letters)
        import torch
        self.last_scorer = "logits"
        ids = tok(prompt, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            p = torch.softmax(mdl(**ids).logits[0, -1].float(), dim=-1)
        out = []
        for L in letters:
            cand = {t[0] for t in (tok.encode(v, add_special_tokens=False) for v in (L, " " + L)) if len(t) == 1}
            out.append(float(sum(p[c] for c in cand)))
        z = sum(out)
        return [x / z for x in out] if z > 0 else [1.0 / len(letters)] * len(letters)

    # ---- the partner-modeling module ----------------------------------------------------------------------------
    def _fmt(self, template: str, **kw) -> str:
        llm = self.planner_config.llm
        try:
            system_tag = llm.get("system_tag", "") or ""
        except Exception:
            system_tag = ""
        return template.format(system_tag=system_tag, user_tag=llm.user_tag, eot_tag=llm.eot_tag,
                               assistant_tag=llm.assistant_tag, **kw)

    def _generate(self, prompt: str) -> str:
        """One completion of the module, stopped at the end-of-turn tag; anything the model writes after its own
        turn (e.g. an invented next user turn) is cut off so it never reaches the robot's prompt."""
        eot = str(self.planner_config.llm.eot_tag or "") or None
        try:
            out = str(self.llm.generate(prompt, eot, self.max_tokens))
        except TypeError:
            out = str(self.llm.generate(prompt))
        cuts = [i for i in (out.find(t) for t in ([eot] if eot else []) + ["<|", "\nHuman:", "\nUser:"]) if i >= 0]
        return out[:min(cuts)] if cuts else out

    def _analyse(self, instruction: str) -> str:
        uid = self.agents[0].uid
        common = dict(task=instruction, division=allocation_text(self.env_interface, uid), history=self._history_text(),
                      messages=self._messages_text())
        if self.pm_mode == "coela":
            out = self._generate(self._fmt(COELA_PROMPT, own_history=self._history_text(int(uid)),
                                           dialogue=self._dialogue_text(), **common))
            progress = self._field(out, "Progress"); message = self._field(out, "Message"); plan = self._field(out, "Plan")
            sent = ""
            if (plan.lower().startswith("send") and message and message.lower().strip(' ."') != "none"
                    and self.n_sent < self.max_messages):
                board.post(uid, message); self.n_sent += 1; sent = message.strip().strip('"')
            if sent:
                progress = progress.rstrip(". ")
                note = (f"Cooperation planner: progress: {progress or 'unclear'}. You have sent the other agent the "
                        f'message "{sent}". Now continue.')
            else:
                note = (f"Cooperation planner: progress: {progress.rstrip('. ') or 'unclear'}. "
                        f"Chosen plan: {plan.rstrip('. ') or 'continue my own part'}.")
            out += f"\n[sent: {sent}]" if sent else ""
        elif self.pm_mode == "introspect":
            options = "\n".join(f"{L}) {t}" for L, t in INTROSPECT_OPTIONS)
            first = self._fmt(INTROSPECT_PROMPT, dialogue=self._dialogue_text(), options=options, **common)
            gen = self._generate(first)
            reasoning = self._field(gen, "Reasoning"); question = self._field(gen, "Question")
            eot = str(self.planner_config.llm.eot_tag or "")
            choice = first + f"Reasoning: {reasoning}\nQuestion: {question}{eot}" + self._fmt(INTROSPECT_CHOICE)
            letters = [L for L, _ in INTROSPECT_OPTIONS]
            probs = self._option_probs(choice, letters)
            pset = [L for L, q in sorted(zip(letters, probs), key=lambda x: -x[1]) if q >= self.ask_threshold]
            best = letters[max(range(len(letters)), key=lambda i: probs[i])]
            text = dict(INTROSPECT_OPTIONS)
            asked = ""
            if len(pset) > 1 and question and self.n_asked < self.max_questions:
                board.post(uid, question); self.n_asked += 1; asked = question.strip().strip('"')
                note = (f"Introspective planner: unsure between: {'; '.join(text[L] for L in pset)}. You have asked "
                        f'the other agent: "{asked}". Use its answer before acting on its parts.')
            else:
                note = f"Introspective planner: confident choice: {text[best]}."
            out = gen + (f"\n[scorer: {getattr(self, 'last_scorer', '?')} probs: "
                         f"{dict(zip(letters, [float(f'{q:.6g}') for q in probs]))} set: {pset} asked: {asked}]")
        elif self.pm_mode == "proact":
            out = self._generate(self._fmt(PROACT_PROMPT, dialogue=self._dialogue_text(), **common))
            breakdown = self._field(out, "Breakdown") or "none"; decision = self._field(out, "Decision").lower()
            message = self._field(out, "Message"); sent = ""
            if (decision.startswith("speak") and message and message.lower().strip(' ."') != "none"
                    and self.n_sent < self.max_messages):
                board.post(uid, message); self.n_sent += 1; sent = message.strip().strip('"')
            if sent:
                note = (f"Collaboration check: breakdown: {breakdown.rstrip('. ')}. You have sent the other agent the "
                        f'message "{sent}". Now continue.')
            else:
                note = f"Collaboration check: breakdown: {breakdown.rstrip('. ')}. No message needed; continue."
            out += f"\n[sent: {sent}]" if sent else ""
        elif self.pm_mode == "hypminds":
            bank = "\n".join(f"H{k + 1}: {h} (score {sc:.0f})" for k, (h, sc, _) in enumerate(self.bank)) or "(none yet)"
            out = self._generate(self._fmt(HYPMINDS_PROMPT, bank=bank, n_new=self.n_hypotheses, **common))
            self._update_bank(out)
            best = self._field(out, "Best") or (self.bank[0][0] if self.bank else "")
            advice = self._field(out, "Advice")
            note = (f"Teammate model (hypotheses about the other agent): most likely, {best}. "
                    f"Suggested response: {advice}")
        else:
            out = self._generate(self._fmt(PROAGENT_PROMPT, previous=self.belief, **common))
            intention = self._field(out, "Intention"); advice = self._field(out, "Advice")
            if intention:
                self.belief = intention
            note = (f"Teammate analysis: the other agent's intention is: {intention or 'unclear'}. "
                    f"Suggested response: {advice}")
        self.analyses.append({"iter": self.iter, "mode": self.pm_mode, "output": out[:1200], "note": note,
                              "bank": [list(b) for b in self.bank]})
        try:
            with open(os.path.join(_log_dir(), "partner_model_stream.jsonl"), "a") as f:
                f.write(json.dumps({"episode_id": self.episode_id, **self.analyses[-1]}, default=str) + "\n")
        except Exception:
            pass
        return note

    @staticmethod
    def _field(text: str, key: str) -> str:
        m = re.search(rf"^\s*{key}\s*:\s*(.+)$", text, flags=re.M | re.I)
        return " ".join(m.group(1).split())[:300] if m else ""

    def _update_bank(self, out: str) -> None:
        """Hypothetical Minds' refinement: running average of each hypothesis' score, new ones enter with a neutral
        score, keep the best n_hypotheses."""
        for m in re.finditer(r"Score\s*H(\d+)\s*:\s*(\d+(?:\.\d+)?)", out, flags=re.I):
            k = int(m.group(1)) - 1
            if 0 <= k < len(self.bank):
                h = self.bank[k]; h[2] += 1; h[1] = h[1] + (min(float(m.group(2)), 10.0) - h[1]) / h[2]
        for m in re.finditer(r"^\s*New\s*:\s*(.+)$", out, flags=re.M | re.I):
            text = " ".join(m.group(1).split())[:200]
            if text and all(text.lower() != b[0].lower() for b in self.bank):
                self.bank.append([text, 5.0, 1])
        best = self._field(out, "Best")
        for b in self.bank:
            if best and b[0].lower() == best.lower():
                b[1] = max(b[1], 6.0)
        self.bank.sort(key=lambda b: -b[1])
        self.bank = self.bank[:self.n_hypotheses]

    def _flush_log(self) -> None:
        try:
            with open(os.path.join(_log_dir(), "partner_model.jsonl"), "a") as f:
                f.write(json.dumps({"episode_id": self.episode_id, "mode": self.pm_mode, "n_analyses": len(self.analyses),
                                    "injection": getattr(board, "extra", {}).get("injection"),
                                    "analyses": self.analyses}, default=str) + "\n")
        except Exception as e:
            print(f"[partner_model] could not write log: {e}")
        self.analyses = []

    # ---- main ------------------------------------------------------------------------------------------------------
    def get_next_action(self, instruction, observations, world_graph, verbose: bool = False):
        if self.episode_id is None or self.iter == 0:
            try:
                self.episode_id = getattr(self.env_interface.env.env.env._env.current_episode, "episode_id", None)
            except Exception:
                self.episode_id = None
        self.iter += 1
        at_replan = self.replan_required or self.curr_prompt == ""
        if self.pm_mode == "llawco":
            # LLawCo without fine-tuning: the laws enter the prompt once, and no partner-model module runs
            if at_replan and self.curr_prompt and not self.laws_given:
                llm = self.planner_config.llm
                self.curr_prompt += f"{llm.user_tag}{LLAWCO_NOTE}\n{llm.eot_tag}{llm.assistant_tag}"
                self.trace += LLAWCO_NOTE + "\n"; self.laws_given = True
                self.analyses.append({"iter": self.iter, "mode": "llawco", "output": LLAWCO_NOTE, "note": LLAWCO_NOTE,
                                      "bank": []})
            return super().get_next_action(instruction, observations, world_graph, verbose=verbose)
        if (at_replan and self.curr_prompt and not self.is_done and self.iter >= self.min_steps
                and self.iter - self.last_analysis >= self.every_steps):
            self.last_analysis = self.iter
            note = self._analyse(self.instruction_override or instruction)
            llm = self.planner_config.llm
            self.curr_prompt += f"{llm.user_tag}{note}\n{llm.eot_tag}{llm.assistant_tag}"
            self.trace += note + "\n"
        out = super().get_next_action(instruction, observations, world_graph, verbose=verbose)
        if out[2] and self.analyses:
            self._flush_log()
        return out
