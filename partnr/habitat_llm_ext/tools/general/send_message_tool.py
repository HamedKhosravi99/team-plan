"""SendMessage[text]: post a natural-language message to the other agent."""
from __future__ import annotations

from typing import List, Tuple

from habitat_llm.tools import Tool

from .message_board import board


class SendMessageTool(Tool):
    def __init__(self, skill_config=None, name_arg: str = "SendMessage", agent_uid_arg: int = 0):
        name = getattr(skill_config, "name", name_arg) if skill_config is not None else name_arg
        super().__init__(name, agent_uid_arg)
        self.skill_config = skill_config
        self._description = getattr(skill_config, "description", None) if skill_config is not None else None
        self.last_text = ""
        self.env_interface = None

    @property
    def description(self) -> str:
        return self._description or (
            'Used for sending a short natural-language message to the other agent, for example to ask what they '
            'are doing, to tell them where an object is, or to correct a misunderstanding of the task. '
            'Example (SendMessage[Are you bringing the cup to the dining table?])'
        )

    @property
    def argument_types(self) -> List[str]:
        return ["text"]

    # hooks the agent/planner may call on any tool
    def get_state_description(self) -> str:
        return f'Sending a message: "{self.last_text}"' if self.last_text else "Sending a message"

    def set_environment(self, env) -> None:
        self.env_interface = env

    def to(self, device):
        return self

    def process_high_level_action(self, input_query: str, observations) -> Tuple[None, str]:
        text = str(input_query).strip().strip('"').strip("'")
        if not text:
            return None, "SendMessage failed: empty message."
        board.post(self.agent_uid, text)
        self.last_text = text
        return None, f'Message sent to the other agent: "{text}"'
