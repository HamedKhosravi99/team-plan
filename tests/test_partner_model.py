"""Offline tests for the ProAgent / Hypothetical Minds baselines (partner_model_planner.py)."""
import types

import partnr_sim as ps
from partnr_sim import AttrDict, HistEl, World


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies); self.prompts = []

    def generate(self, prompt, stop=None, max_length=None):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "Intention: unclear\nAdvice: continue"


def make(mode, replies, partner_actions=(), steps=3000, robot_script=None, setup=None, **pm):
    from habitat_llm_ext.planner.partner_model_planner import PartnerModelRobotPlanner
    from habitat_llm_ext.tools.general.message_board import board
    w = World()
    cfg = AttrDict({"llm": {"user_tag": "<u>", "eot_tag": "<e>", "assistant_tag": "<a>", "system_tag": "<s>"},
                    "replanning_threshold": 50,
                    "partner_model": {"mode": mode, "every_steps": 400, "min_steps": 600, **pm}})
    pl = PartnerModelRobotPlanner(cfg, w)
    pl.agents = [types.SimpleNamespace(uid=0, tools={})]
    pl.reset(); board.reset()
    pl.llm = FakeLLM(replies)
    if setup:
        setup(pl)
    pl.llm_script = list(robot_script or [("Explore", "kitchen_1", None)] * 40)
    acts = sorted(partner_actions)
    for t in range(steps):
        while acts and acts[0][0] == t:
            _, a, r = acts.pop(0); w.agent_action_history[1].append(HistEl(a, r))
        low, info, done = pl.get_next_action("Set the table", {}, w.world_graph)
        if done:
            break
    return pl, w


def test_proagent_injects_analysis_and_corrects_belief():
    replies = ["Correction: none\nIntention: the partner is fetching the fork\nAdvice: continue your own part",
               "Correction: wrong, it never touched the bread\nIntention: it forgot the bread\nAdvice: remind it about bread_5"]
    pl, _ = make("proagent", replies, partner_actions=[(10, ("Navigate", "fork_0", None), "Successful execution!")])
    assert len(pl.llm.prompts) >= 2
    assert "Navigate[fork_0]" in pl.llm.prompts[0]                       # partner evidence reaches the module
    assert "the partner is fetching the fork" in pl.llm.prompts[1]       # belief correction sees the previous belief
    assert pl.belief == "it forgot the bread" or "unclear" in pl.belief
    assert "Teammate analysis" in pl.curr_prompt and "remind it about bread_5" in pl.curr_prompt


def test_no_analysis_before_min_steps_and_spacing():
    pl, _ = make("proagent", [], steps=900)
    iters = [a["iter"] for a in pl.analyses] if pl.analyses else []
    assert all(i >= 600 for i in iters)
    assert all(b - a >= 400 for a, b in zip(iters, iters[1:]))


def test_hypminds_bank_scores_and_refines():
    replies = ["New: it forgot the bread\nNew: it is doing fine\nBest: it forgot the bread\nAdvice: remind it about bread_5",
               "Score H1: 9\nScore H2: 1\nNew: its grasp is failing\nBest: it forgot the bread\nAdvice: remind it about bread_5"]
    pl, _ = make("hypminds", replies)
    texts = [b[0] for b in pl.bank]
    assert texts[0] == "it forgot the bread"
    assert len(pl.bank) <= 3
    assert "Teammate model" in pl.curr_prompt
    assert "H1: it forgot the bread" in pl.llm.prompts[1]                # the bank is fed back for re-scoring


def test_output_is_cut_at_end_of_turn():
    junk = ("Score H1: 0\nNew: it is still searching\nBest: it is still searching\n"
            "Advice: continue your part<e>Human: Can you explain how you scored this?\nAdvice: take over everything")
    pl, _ = make("hypminds", [junk, junk])
    assert pl.llm.prompts                                         # the module ran
    assert "Human:" not in pl.curr_prompt and "take over everything" not in pl.curr_prompt
    assert "continue your part" in pl.curr_prompt


def test_coela_sends_drafted_message_at_most_max_messages():
    from habitat_llm_ext.tools.general.message_board import board
    reply = ("Progress: plates done, the fork is still open and nobody is on it\nMessage: Please bring fork_0 to the table\n"
             "Plan: send the message")
    pl, _ = make("coela", [reply] * 6, partner_actions=[(10, ("Navigate", "fork_0", None), "Successful execution!")])
    sent = [t for s, t, _ in board.messages if s == 0]
    assert sent and len(sent) <= 2 and sent[0] == "Please bring fork_0 to the table"
    assert "Navigate[fork_0]" in pl.llm.prompts[0]                       # the memory sees the partner's actions
    assert "Cooperation planner" in pl.curr_prompt and "You have sent the other agent" in pl.curr_prompt


def test_coela_continue_plan_posts_nothing():
    from habitat_llm_ext.tools.general.message_board import board
    reply = "Progress: all on track\nMessage: none\nPlan: continue my own part"
    pl, _ = make("coela", [reply] * 6)
    assert not [t for s, t, _ in board.messages if s == 0]
    assert "Chosen plan: continue my own part" in pl.curr_prompt


def test_introspect_asks_when_unsure_and_acts_when_confident():
    from habitat_llm_ext.tools.general.message_board import board
    gen = "Reasoning: the fork has not been touched; it may have been forgotten.\nQuestion: Are you going to bring the fork?"
    unsure = make("introspect", [gen] * 6, setup=lambda pl: setattr(pl, "_option_probs", lambda p, L: [0.5, 0.4, 0.1]))[0]
    asked = [t for s, t, _ in board.messages if s == 0]
    assert asked and asked[0] == "Are you going to bring the fork?" and len(asked) <= 2
    assert "unsure between" in unsure.curr_prompt
    assert "Reasoning:" in unsure.llm.prompts[0] and "A) continue its own part" in unsure.llm.prompts[0]
    sure = make("introspect", [gen] * 6, setup=lambda pl: setattr(pl, "_option_probs", lambda p, L: [0.9, 0.05, 0.05]))[0]
    assert not [t for s, t, _ in board.messages if s == 0]
    assert "confident choice: continue its own part" in sure.curr_prompt


def test_llawco_puts_the_laws_in_the_prompt_once():
    pl, _ = make("llawco", [])
    assert pl.curr_prompt.count("Laws of cooperation for this task") == 1
    assert "Avoid Infinite Loops" in pl.curr_prompt and not pl.llm.prompts      # no partner-model module runs


def test_proact_speaks_only_on_a_breakdown():
    from habitat_llm_ext.tools.general.message_board import board
    speak = "Breakdown: forgotten responsibility\nDecision: speak\nMessage: Please also bring the bread to the table"
    pl, _ = make("proact", [speak] * 6, partner_actions=[(10, ("Navigate", "fork_0", None), "Successful execution!")])
    sent = [t for s, t, _ in board.messages if s == 0]
    assert sent and sent[0] == "Please also bring the bread to the table" and len(sent) <= 2
    assert "Collaboration check: breakdown: forgotten responsibility" in pl.curr_prompt
    quiet = "Breakdown: none\nDecision: silent\nMessage: none"
    pl, _ = make("proact", [quiet] * 6)
    assert not [t for s, t, _ in board.messages if s == 0] and "No message needed" in pl.curr_prompt
