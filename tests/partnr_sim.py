"""Offline harness for the PARTNR robot planner: stubbed habitat_llm, a PARTNR-like world graph (object -> furniture
-> room; get_room_for_entity raises for objects), a fake ReAct base planner with a scripted LLM, fake skills, and
a runner that mimics partnr's evaluation loop.  No simulator or GPU needed.
"""
from __future__ import annotations

import os
import sys
import types

import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CCRP_LOG_DIR", tempfile.mkdtemp(prefix="ccrp_test_logs_"))   # keep decision logs out of the repo
sys.path[:0] = [os.path.join(HERE, "..", "partnr"), os.path.join(HERE, "..")]

# ---- stub habitat_llm -------------------------------------------------------------------------------------------
for name in ("habitat_llm", "habitat_llm.tools", "habitat_llm.planner", "habitat_llm.planner.llm_planner",
             "habitat_llm.planner.zero_shot_react_planner"):
    sys.modules.setdefault(name, types.ModuleType(name))


class _Tool:
    def __init__(self, name_arg, agent_uid_arg=0):
        self.name = name_arg; self.agent_uid = agent_uid_arg


sys.modules["habitat_llm.tools"].Tool = _Tool
sys.modules["habitat_llm.tools"].PerceptionTool = _Tool

SKILL_STEPS = {"Explore": 300, "Rearrange": 120, "Wait": 600, "SendMessage": 1, "Navigate": 50, "Pick": 20, "Place": 20}


class FakeBase:
    """Minimal stand-in for partnr's LLMPlanner: scripted LLM, skills that take SKILL_STEPS steps."""

    def __init__(self, plan_config, env_interface):
        self.planner_config = plan_config; self.env_interface = env_interface; self.prompt = "{responsibilities}"
        self._agents = []; self.llm_script = []; self.reset()

    agents = property(lambda self: self._agents, lambda self, v: setattr(self, "_agents", v))

    def reset(self):
        self.curr_prompt = ""; self.trace = ""; self.replan_required = True; self.is_done = False
        self.replanning_count = 0; self.last_high_level_actions = {}; self._running = None

    def prepare_prompt(self, instruction, wg, **kw):
        return "PROMPT", {}

    def process_high_level_actions(self, hla, obs):
        uid = list(hla)[0]; a = hla[uid]
        if a[0] == "Navigate" and not self.env_interface.has_node(str(a[1])):     # the 7B robot's usual mistake
            self._running = None
            return {uid: "low"}, {uid: f"Node with name {a[1]} not present in the graph"}
        if self._running is None or self._running[0] != a:
            self._running = [a, 0]
        self._running[1] += 1
        if self._running[1] >= SKILL_STEPS.get(a[0], 10):
            self._running = None
            if a[0] == "Explore":
                self.env_interface.explore(str(a[1]))
            if a[0] == "Rearrange":
                obj = a[1].split(",")[0].strip()
                if obj in self.env_interface.fail_once:
                    self.env_interface.fail_once.discard(obj)
                    return {uid: "low"}, {uid: "Unexpected failure! - Failed to place"}
                parts = [x.strip() for x in a[1].split(",")]
                self.env_interface.place(obj, parts[2] if len(parts) > 2 else None)
            return {uid: "low"}, {uid: "Successful execution!"}
        return {uid: "low"}, {uid: ""}

    def _add_responses_to_prompt(self, responses):
        return " ".join(v for v in responses.values() if v)

    def get_last_agent_states(self): return {}
    def get_last_agent_positions(self): return {}
    def get_agent_collisions(self): return {}

    def get_next_action(self, instruction, observations, world_graph, verbose=False):
        uid = self.agents[0].uid
        if self.is_done:
            return {}, {"replanned": {uid: False}}, True
        if self.curr_prompt == "":
            self.curr_prompt, self.params = self.prepare_prompt(instruction, None)
        if self.replan_required:
            self.replanning_count += 1
            # partnr stops the agent once it has used up its planning calls (the count ends at threshold + 1)
            if self.replanning_count > self.planner_config.replanning_threshold:
                self.is_done = True
                return {}, {"replanned": {uid: False}}, True
            nxt = self.llm_script.pop(0) if self.llm_script else ("Done", None, None)
            self.curr_prompt += f" LLM:{nxt}"
            if nxt[0] == "Done":
                self.is_done = True
                return {}, {"replanned": {uid: True}, "high_level_actions": {uid: nxt}, "responses": {}}, True
            self.last_high_level_actions = {uid: nxt}
        hla = self.last_high_level_actions
        low, responses = self.process_high_level_actions(hla, observations)
        self.replan_required = any(responses.values())
        return low, {"replanned": {uid: True}, "high_level_actions": hla, "responses": responses}, False


sys.modules["habitat_llm.planner.llm_planner"].LLMPlanner = FakeBase
sys.modules["habitat_llm.planner.zero_shot_react_planner"].ZeroShotReactPlanner = FakeBase


# ---- PARTNR-like world --------------------------------------------------------------------------------------------
class Room:
    def __init__(self, n): self.name = n


class Furn:
    def __init__(self, n, room, handle): self.name = n; self.room = room; self.sim_handle = handle


class Obj:
    def __init__(self, n, handle, furn): self.name = n; self.sim_handle = handle; self.furn = furn


class Graph:
    def __init__(self, objs, rooms, furns): self._objs = objs; self._rooms = rooms; self._furns = furns
    def get_all_objects(self): return self._objs
    def get_all_furnitures(self): return self._furns
    def get_all_rooms(self): return self._rooms
    def find_furniture_for_object(self, o): return o.furn

    def get_room_for_entity(self, e):
        if isinstance(e, Furn):
            return e.room
        raise ValueError(f"No room found for entity {e}")


class Prop:
    def __init__(self, fn, **a): self.function_name = fn; self.args = a


class HistEl:
    def __init__(self, action, response=None): self.action = action; self.response = response


class World:
    """Four propositions 'put X on table_10'; by index parity the robot owns cup_0 and plate_1, the partner fork_0
    and bread_5 (allocation.py)."""

    def __init__(self, robot_sees=("fork_0", "bread_5", "cup_0", "plate_1"), fail_once=(), plate_room=None, bread_on=None):
        kitchen, living, bed = Room("kitchen_1"), Room("living_room_1"), Room("bedroom_1")
        self.counter = Furn("counter_2", kitchen, "c2_:0000"); self.table = Furn("table_10", living, "t1_:0000")
        self.bed = Furn("bed_5", bed, "b5_:0000"); self.door = Furn("door_7", bed, "d7_:0000")
        self.objs = [Obj("cup_0", "c_:0000", self.counter), Obj("fork_0", "a_:0000", self.counter),
                     Obj("plate_1", "d_:0000", self.counter), Obj("bread_5", "b_:0000", self.counter)]
        rooms, furns = [kitchen, living, bed], [self.table, self.counter, self.door, self.bed]
        if bread_on == "bed_5":              # bread_5 starts in the bedroom (found only by exploring it)
            self.objs[3].furn = self.bed
        self.full_world_graph = Graph(self.objs, rooms, furns)
        self.world_graph = {0: Graph([o for o in self.objs if o.name in robot_sees], rooms, furns), 1: Graph([], rooms, furns)}
        props = [Prop("is_on_top", object_handles=[o.sim_handle], receptacle_handles=["t1_:0000"]) for o in self.objs]
        if plate_room:      # plate_1 (a robot responsibility) becomes "bring plate_1 to the <plate_room>"
            props[2] = Prop("is_in_room", object_handles=["d_:0000"], room_ids=[plate_room])
        ep = types.SimpleNamespace(episode_id="0", evaluation_propositions=props)
        self.env = types.SimpleNamespace(env=types.SimpleNamespace(env=types.SimpleNamespace(_env=types.SimpleNamespace(current_episode=ep))))
        self.agent_action_history = {0: [], 1: []}; self.robot_agent_uid = 0
        self.placed = set(); self.fail_once = set(fail_once)

    def has_node(self, name):
        return any(x.name == name for x in self.objs + [self.table, self.counter, self.bed, self.door]) or \
            name in ("kitchen_1", "living_room_1", "bedroom_1")

    def explore(self, room):
        """Exploring a room reveals the objects on its furniture to the robot."""
        g = self.world_graph[0]
        for o in self.objs:
            if o.furn.room.name == room and o not in g._objs:
                g._objs.append(o)

    def place(self, name, furniture=None):
        self.placed.add(name)
        target = next((f for f in (self.table, self.counter, self.bed, self.door) if f.name == furniture), self.table)
        for o in self.objs:
            if o.name == name:
                o.furn = target


class AttrDict(dict):
    def __getattr__(self, k):
        v = self[k]
        return AttrDict(v) if isinstance(v, dict) else v


def run(mode, partner_script, robot_script, ccrp_cfg=None, steps=6000, **world_kw):
    """Step the robot planner like partnr's runner; partner actions are injected into the action history."""
    from habitat_llm_ext.planner.ccrp_robot_planner import CCRPRobotPlanner
    from habitat_llm_ext.tools.general.message_board import board
    w = World(**world_kw)
    import yaml
    with open(os.path.join(HERE, "..", "partnr", "conf", "planner", "ccrp_robot_planner.yaml")) as f:
        c = dict(yaml.safe_load(f)["plan_config"]["ccrp"])        # the deployed defaults
    c["mode"] = mode
    c.update(ccrp_cfg or {})
    cfg = AttrDict({"llm": {"user_tag": "<u>", "eot_tag": "<e>", "assistant_tag": "<a>"}, "replanning_threshold": 50, "ccrp": c})
    pl = CCRPRobotPlanner(cfg, w)
    pl.agents = [types.SimpleNamespace(uid=0, tools={})]
    pl.reset(); board.reset()
    pl.llm_script = list(robot_script)
    ph = sorted(partner_script)
    log = []
    for t in range(steps):
        while ph and ph[0][0] == t:
            _, a, r = ph.pop(0); w.agent_action_history[1].append(HistEl(a, r))
        low, info, done = pl.get_next_action("task", {}, w.world_graph)
        if isinstance(info, dict) and info.get("replanned", {}).get(0) and info.get("high_level_actions"):
            a = info["high_level_actions"][0]
            h = w.agent_action_history[0]
            if not h or h[-1].action != a or h[-1].response:
                h.append(HistEl(a)); log.append((t, a[0], str(a[1])))
        resp = (info.get("responses") or {}).get(0) if isinstance(info, dict) else None
        if resp and w.agent_action_history[0]:
            w.agent_action_history[0][-1].response = resp
        if done:
            log.append((t, "DONE", "")); break
    global LAST_PLANNER
    LAST_PLANNER = pl
    return log, w.placed, [m[1] for m in board.messages]
