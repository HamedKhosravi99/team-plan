"""Partner failure injection (hidden common causes), applied to the human agent's planner.

Injections, each a hidden component of the partner:
  goal      - the partner receives an instruction with one of its responsibilities removed (misread goal);
  world     - the partner is told that one task object is in a wrong room (wrong world model, one responsibility);
  world_all - the partner is told that all of its task objects are in one wrong room (a shared wrong belief that
              couples all of its responsibilities: the high-coupling condition);
  skill     - the partner's Pick/Place fail on each invocation with probability q (broken skill).
The injected ground truth is recorded for calibration and for scoring the robot's posterior.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

from .allocation import allocate, describe_proposition, _names


def _rooms(world_graph) -> List[str]:
    try:
        return [r.name for r in world_graph.get_all_rooms()]
    except Exception:
        return []


def _object_room(world_graph, obj_name: str) -> Optional[str]:
    """Room of an object: object -> furniture -> room (WorldGraph.get_room_for_entity only resolves furniture)."""
    try:
        for o in world_graph.get_all_objects():
            if o.name == obj_name:
                furn = world_graph.find_furniture_for_object(o)
                if furn is None:
                    return None
                return getattr(world_graph.get_room_for_entity(furn), "name", None)
    except Exception:
        return None
    return None


def plan_injection(kind: str, seed: int, env_interface, human_uid: int, q_skill: float = 0.5) -> Dict[str, Any]:
    """Decide the injection for the current episode.  Returns a dict with the ground truth and the pieces the
    human planner needs (instruction edit, wrong-location note, skill failure probability)."""
    rng = random.Random(seed)
    alloc = allocate(env_interface)
    partner_props = alloc.get(human_uid, [])
    truth: Dict[str, Any] = {"kind": kind, "dropped": None, "wrong_object": None, "wrong_room": None, "skill_q": 0.0}
    kinds = [kind] if kind != "mixed" else [rng.choice(["goal", "world", "skill"])]
    for k in kinds:
        if k == "goal" and partner_props:
            i, p = rng.choice(partner_props)
            truth["dropped"] = (i, describe_proposition(p, env_interface))
        elif k in ("world", "world_all") and partner_props:
            wg = env_interface.world_graph[human_uid]
            full = getattr(env_interface, "full_world_graph", None) or wg      # ground truth for the true rooms
            objs = []
            for _, p in partner_props:
                for o in _names((getattr(p, "args", None) or p.get("args", {})).get("object_handles"), env_interface):
                    if o not in objs:
                        objs.append(o)
            rooms = _rooms(full) or _rooms(wg)
            rooms_of = {o: _object_room(full, o) for o in objs}
            known = [o for o in objs if rooms_of[o]]                 # objects whose true room is resolvable
            if known and rooms:
                chosen = [rng.choice(known)] if k == "world" else known
                true_rooms = {o: rooms_of[o] for o in chosen}
                wrong = [r for r in rooms if r not in set(true_rooms.values())]
                if wrong:
                    truth["wrong_room"] = rng.choice(wrong); truth["true_room"] = true_rooms[chosen[0]]
                    if k == "world":
                        truth["wrong_object"] = chosen[0]
                    else:
                        truth["wrong_objects"] = chosen; truth["true_rooms"] = true_rooms
        elif k == "skill":
            truth["skill_q"] = q_skill
    return truth


def corrupt_instruction(instruction: str, truth: Dict[str, Any]) -> str:
    """Remove the clause that mentions the dropped responsibility's object; add the wrong-location note."""
    text = instruction
    if truth.get("dropped"):
        desc = truth["dropped"][1]
        # object base names ("plate_1" -> "plate"), which is how instructions refer to objects
        words = [re.sub(r"_\d+$", "", w) for w in re.findall(r"[a-z]+(?:_[a-z]+)*_\d+", desc.lower())]
        words = [w.replace("_", " ") for w in words] or [w for w in re.findall(r"[a-z]{4,}", desc.lower())
                                                           if w not in ("put", "the", "on", "inside", "bring", "place", "next", "make")]
        clauses = re.split(r"(?<=[.;])\s+|,\s*(?:and\s+|then\s+)?|\s+and\s+(?=(?:put|place|bring|move|then|the)\b)", text)
        clauses = [c for c in clauses if c and c.strip()]
        kept = [c for c in clauses if not any(w in c.lower() for w in words[:1])]
        if 0 < len(kept) < len(clauses):
            text = ", then ".join(c.strip().rstrip(".") for c in kept) + "."
            text = text[0].upper() + text[1:]
    if truth.get("wrong_object"):
        text += f" Note: the {truth['wrong_object']} is in the {truth['wrong_room']}."
    if truth.get("wrong_objects"):
        names = truth["wrong_objects"]
        listed = names[0] if len(names) == 1 else ", the ".join(names[:-1]) + " and the " + names[-1]
        verb = "is" if len(names) == 1 else ("are both" if len(names) == 2 else "are all")
        text += f" Note: the {listed} {verb} in the {truth['wrong_room']}."
    return text


class FlakySkill:
    """Wraps a motor-skill tool so that each invocation fails with probability q.

    process_high_level_action is called on every simulator step while a skill runs, so the failure is drawn once
    when an invocation starts (no invocation active) and the invocation ends when a response is returned."""

    def __init__(self, tool, q: float, seed: int):
        self._tool = tool; self._q = q; self._rng = random.Random(seed); self.failures = 0; self.invocations = 0
        self._active = False

    def __getattr__(self, name):
        return getattr(self._tool, name)

    def process_high_level_action(self, input_query, observations):
        if not self._active:
            self._active = True; self.invocations += 1
            if self._rng.random() < self._q:
                self.failures += 1; self._active = False
                return None, f"Unexpected failure! - {self._tool.name}[{input_query}] failed."
        low, response = self._tool.process_high_level_action(input_query, observations)
        if response:
            self._active = False
        return low, response
