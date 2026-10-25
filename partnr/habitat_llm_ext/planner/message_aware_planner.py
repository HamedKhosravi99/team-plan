"""A ZeroShotReactPlanner that (i) has the SendMessage tool, (ii) reads the message board at every replanning
step, and (iii) for the human agent, applies the configured failure injection (hidden common cause).

Prompt template: conf/instruct/zero_shot_dialog_prompt.yaml ({responsibilities} from the allocation protocol).
plan_config.injection: {kind: none|goal|world|skill|mixed, seed: int, q_skill: float}
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from habitat_llm.planner.llm_planner import LLMPlanner
from habitat_llm.planner.zero_shot_react_planner import ZeroShotReactPlanner

from habitat_llm_ext.tools.general.message_board import board
from habitat_llm_ext.tools.general.send_message_tool import SendMessageTool
from habitat_llm_ext.allocation import allocation_text
from habitat_llm_ext.injection import FlakySkill, corrupt_instruction, plan_injection


class MessageAwareReactPlanner(ZeroShotReactPlanner):
    def __init__(self, plan_config, env_interface):
        # defaults first: the base constructor calls reset() before the rest of this constructor runs
        self.injection_kind = "none"; self.injection_seed = 0; self.injection_q = 0.5; self.injection_seed_by = "counter"
        self.n_messages_read = 0; self.instruction_override = None; self.injected_truth = None
        self.episode_counter = 0; self._episode_initialized = False
        super().__init__(plan_config, env_interface)
        inj = plan_config.get("injection", None)
        self.injection_kind = str(inj.get("kind", "none")) if inj is not None else "none"
        self.injection_seed = int(inj.get("seed", 0)) if inj is not None else 0
        self.injection_q = float(inj.get("q_skill", 0.5)) if inj is not None else 0.5
        # counter: the n-th episode this process runs (v1/v2 pilots); episode_id: the dataset episode id, so that the
        # same episode gets the same injection in every config and after a resume
        self.injection_seed_by = str(inj.get("seed_by", "counter")) if inj is not None else "counter"

    # tools --------------------------------------------------------------------------------
    @property
    def agents(self):
        return LLMPlanner.agents.fget(self)

    @agents.setter
    def agents(self, agents):
        LLMPlanner.agents.fset(self, agents)
        for agent in self._agents:
            tools = getattr(agent, "tools", None)
            if isinstance(tools, dict) and "SendMessage" not in tools:
                tools["SendMessage"] = SendMessageTool(agent_uid_arg=agent.uid)

    def reset(self):
        super().reset()
        agents = getattr(self, "_agents", None) or []
        if agents and agents[0].uid == getattr(self.env_interface, "robot_agent_uid", 0):
            board.reset()                       # the robot's planner resets first in the runner; harmless if not
        self.n_messages_read = 0
        self.instruction_override = None
        self.injected_truth = None
        self._episode_initialized = False
        self.episode_counter = getattr(self, "episode_counter", 0) + 1

    # injection (human planner only) -----------------------------------------------------------
    def _init_episode(self, instruction: str) -> None:
        self._episode_initialized = True
        uid = self.agents[0].uid
        if self.injection_kind == "none" or uid == getattr(self.env_interface, "robot_agent_uid", 0):
            return
        key = self.episode_counter
        if self.injection_seed_by == "episode_id":
            try:
                key = int(self.env_interface.env.env.env._env.current_episode.episode_id)
            except Exception:
                key = self.episode_counter
        truth = plan_injection(self.injection_kind, self.injection_seed * 1000 + key,
                               self.env_interface, uid, self.injection_q)
        if truth.get("dropped") or truth.get("wrong_object") or truth.get("wrong_objects"):
            self.instruction_override = corrupt_instruction(instruction, truth)
        if truth.get("skill_q", 0) > 0:
            tools = self.agents[0].tools
            for name in list(tools):
                if name in ("Pick", "Place"):
                    tools[name] = FlakySkill(tools[name], truth["skill_q"], self.injection_seed * 7 + key)
        self.injected_truth = truth
        board.extra["injection"] = truth   # ground truth for logging only

    # prompt variables ---------------------------------------------------------------------------
    def prepare_prompt(self, input_instruction: str, world_graph, **kwargs) -> Tuple[str, Dict[str, Any]]:
        if self.instruction_override is not None:
            input_instruction = self.instruction_override
        if "{responsibilities}" in self.prompt:
            kwargs = dict(kwargs); kwargs["should_format"] = False
            _, params = super().prepare_prompt(input_instruction, world_graph, **kwargs)
            dropped = (self.injected_truth or {}).get("dropped")
            params["responsibilities"] = allocation_text(self.env_interface, self.agents[0].uid,
                                                         exclude_index=dropped[0] if dropped else None)
            return self.prompt.format(**params), params
        return super().prepare_prompt(input_instruction, world_graph, **kwargs)

    # messages ------------------------------------------------------------------------------------
    def _inject_messages(self) -> int:
        # only at a natural replanning point: interrupting a running skill leaves the agent without a response
        if not (self.replan_required or self.curr_prompt == "") or self.curr_prompt == "":
            return 0
        msgs = board.unread(self.agents[0].uid)
        if not msgs:
            return 0
        lines = "".join(f'\nMessage from the other agent: "{t}"' for _, t, _ in msgs)
        self.curr_prompt += (f"{self.planner_config.llm.user_tag}{lines}\n"
                             f"{self.planner_config.llm.eot_tag}{self.planner_config.llm.assistant_tag}")
        self.trace += lines + "\n"
        self.n_messages_read += len(msgs)
        return len(msgs)

    def get_next_action(self, instruction, observations, world_graph, verbose: bool = False):
        if not self._episode_initialized:
            self._init_episode(instruction)
        if self.instruction_override is not None:
            instruction = self.instruction_override
        self._inject_messages()
        out = super().get_next_action(instruction, observations, world_graph, verbose=verbose)
        if self.injected_truth is not None and isinstance(out[1], dict):
            out[1]["injection"] = self.injected_truth
        board.tick()
        return out
