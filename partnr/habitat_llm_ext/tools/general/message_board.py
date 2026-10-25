"""A process-wide message board shared by the two agents' planners.

partnr-planner has no inter-agent communication; this is the dialogue channel of the experiments.
Messages are (sender_uid, text, planner_step).  Each planner tracks what it has already read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class MessageBoard:
    messages: List[Tuple[int, str, int]] = field(default_factory=list)
    read_upto: Dict[int, int] = field(default_factory=dict)
    step: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)      # per-episode side information (e.g. injected truth)

    def reset(self) -> None:
        self.messages.clear()
        self.read_upto.clear()
        self.extra.clear()
        self.step = 0

    def post(self, sender_uid: int, text: str) -> None:
        self.messages.append((int(sender_uid), text.strip(), self.step))

    def unread(self, reader_uid: int) -> List[Tuple[int, str, int]]:
        start = self.read_upto.get(int(reader_uid), 0)
        out = [m for m in self.messages[start:] if m[0] != int(reader_uid)]
        self.read_upto[int(reader_uid)] = len(self.messages)
        return out

    def all_for(self, reader_uid: int) -> List[Tuple[int, str, int]]:
        return [m for m in self.messages if m[0] != int(reader_uid)]

    def tick(self) -> None:
        self.step += 1


board = MessageBoard()
