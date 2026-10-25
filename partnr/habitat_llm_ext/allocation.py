"""Allocation protocol: which evaluation propositions belong to which agent.

Deterministic split by proposition index parity (robot even, human odd), stated in both prompts.
Propositions are rendered in natural language from their function name and arguments.
"""
from __future__ import annotations

from typing import Optional, Any, Dict, List, Tuple


def current_episode(env_interface):
    return env_interface.env.env.env._env.current_episode


_NAME_CACHE: Dict[int, Dict[str, str]] = {}


def handle_to_name(env_interface, handle: str) -> str:
    """Map a simulator handle (e.g. '9dfbc1...cf_:0000', 'B075HR7LD2_:0000') to the readable world-graph name the
    planners use (e.g. 'candle_0', 'table_5').  Uses the ground-truth graph for naming only."""
    h = str(handle)
    if env_interface is None:
        return h.split("_:")[0]
    cache = _NAME_CACHE.setdefault(id(env_interface), {})
    if h in cache:
        return cache[h]
    name = None
    try:
        g = getattr(env_interface, "full_world_graph", None)
        nodes = list(g.get_all_objects()) + list(g.get_all_furnitures()) if g is not None else []
        base = h.split("_:")[0]
        for n in nodes:
            sh = getattr(n, "sim_handle", None)
            if sh is not None and (sh == h or str(sh).split("_:")[0] == base):
                name = n.name; break
    except Exception:
        name = None
    if name is None:
        try:   # fall back to the object category from the metadata (e.g. 'candle')
            mi = getattr(getattr(env_interface, "perception", None), "metadata_interface", None)
            cat = mi.get_object_category(h.split("_:")[0].split(".")[0]) if mi is not None else None
            name = cat or h.split("_:")[0]
        except Exception:
            name = h.split("_:")[0]
    cache[h] = name
    return name


def _names(handles: Any, env_interface=None) -> List[str]:
    if handles is None:
        return []
    if isinstance(handles, str):
        handles = [handles]
    out: List[str] = []
    for h in handles:
        n = handle_to_name(env_interface, h)
        if n not in out:
            out.append(n)
    return out


def describe_proposition(prop, env_interface=None) -> str:
    fn = getattr(prop, "function_name", None) or prop.get("function_name")
    args = getattr(prop, "args", None) or prop.get("args", {})
    objs = _names(args.get("object_handles"), env_interface)
    recs = _names(args.get("receptacle_handles"), env_interface)
    rooms = _names(args.get("room_ids"), env_interface)
    n = args.get("number", 1)
    o = " or ".join(objs) if objs else "an object"
    if fn == "is_on_top":
        return f"put {o} on {' or '.join(recs)}"
    if fn == "is_inside":
        return f"put {o} inside {' or '.join(recs)}"
    if fn == "is_in_room":
        return f"bring {o} to the {' or '.join(rooms)}"
    if fn == "is_next_to":
        return f"place {o} next to {' or '.join(recs) or 'the reference object'}"
    if fn == "is_on_floor":
        return f"put {o} on the floor"
    if fn in ("is_clean", "is_dirty", "is_filled", "is_empty", "is_powered_on", "is_powered_off"):
        return f"make {o} {fn[3:].replace('_', ' ')}"
    return f"{fn}({o})"


def allocate(env_interface, robot_uid: int = 0) -> Dict[int, List[Tuple[int, Any]]]:
    """Return {agent_uid: [(proposition_index, proposition), ...]}."""
    ep = current_episode(env_interface)
    props = list(getattr(ep, "evaluation_propositions", []))
    out = {robot_uid: [], 1 - robot_uid: []}
    for i, p in enumerate(props):
        out[robot_uid if i % 2 == 0 else 1 - robot_uid].append((i, p))
    return out


def allocation_text(env_interface, agent_uid: int, exclude_index: Optional[int] = None) -> str:
    """The division-of-work paragraph for agent_uid.  exclude_index drops one of the agent's own responsibilities
    (used by the goal injection so that the partner's whole picture of the task lacks the dropped clause)."""
    alloc = allocate(env_interface)
    mine = [describe_proposition(p, env_interface) for i, p in alloc.get(agent_uid, []) if i != exclude_index]
    theirs = [describe_proposition(p, env_interface) for _, p in alloc.get(1 - agent_uid, [])]
    s = "Division of work. You are responsible for: " + ("; ".join(mine) if mine else "nothing specific") + ". "
    s += "The other agent is responsible for: " + ("; ".join(theirs) if theirs else "nothing specific") + ". "
    s += "Do your part first; you may help with the other agent's part only if it is clearly not being done."
    return s
