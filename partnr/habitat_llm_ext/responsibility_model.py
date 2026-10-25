"""Responsibility model for PARTNR: partner hypotheses, evidence, and the ConditionalModel the planner needs.

Responsibilities are the episode's evaluation propositions allocated to the partner (allocation.py).  The task
value is the conjunction of all propositions (own and partner).  Conditional success rates are a rule table
(DESIGN.md); the consequence is the Birnbaum importance under the conjunction; takeover means the robot
executes the proposition itself with Rearrange.  One-step Lemma-1 quantities (base policy = continue),
scaled by the remaining step budget.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ccrp.core import Action, CONTINUE, ConditionalModel, Hypothesis, Posterior

from .allocation import allocate, describe_proposition, _names


# Failure messages whose stated reason is the agent's own positioning, grasp state or API use, not a broken skill.
BENIGN_FAILURE_REASONS = ("not close enough", "already held", "held by another agent", "not holding", "currently grasping",
                          "occluded", "wrong use of api", "node with name", "no valid", "too far")


@dataclass
class ModelConfig:
    base_rate: float = 0.85
    own_rate: float = 0.9
    goal_confused_rate: float = 0.05
    obj_wrong_factor: float = 0.3
    skill_fail_factor: float = 0.35
    eta_takeover: float = 0.75           # the robot must usually find the object first
    prior_bad: Dict[str, float] = field(default_factory=lambda: {"goal": 0.2, "obj": 0.15, "skill": 0.15, "room": 0.15})
    # shared wrong belief about where the partner's objects are (one cause behind all partner responsibilities)
    room_component: bool = False
    cost_takeover: float = 0.08          # in units of task success, per responsibility taken over
    cost_message: float = 0.03
    opportunity_cost_per_own: float = 0.03
    answer_truthful: float = 0.8
    lr_action_mentions_object: float = 3.0     # partner acts on a responsibility's object -> not dropped, object belief ok
    lr_failure_response: float = 4.0           # partner Pick/Place fails with no stated reason -> skill failing
    lr_success_response: float = 2.0           # partner Pick/Place succeeds -> skill working
    lr_untouched_half_budget: float = 3.0      # responsibility untouched by mid-budget -> dropped from the partner's goal
    lr_untouched_obj: float = 2.0              # ... or the partner cannot find its object (wrong belief)
    max_hypothesis_objects: int = 3
    quit_rate: float = 0.0                     # the partner has said Done: its unfinished responsibilities stay unfinished
    # inspect = keep watching until the mid-budget progress evidence arrives (diagnosis before treatment);
    # off in the v2 pilot (variant A), on in variant B
    inspect: bool = False
    cost_inspect: float = 0.01                 # delay cost of acting later instead of now
    p_untouched_drop: float = 0.9              # P(responsibility i untouched at mid-budget | partner dropped i)
    p_untouched_obj: float = 0.5               # ... | partner has a wrong belief about i's object
    p_untouched_ok: float = 0.25               # ... | partner fine (busy with its other responsibilities)
    joint_takeover: bool = True                # also offer taking over all remaining partner responsibilities at once
    # LLawCo's cooperation laws as model terms (law-structured Team-Plan). Avoid infinite loops: the robot's own ReAct
    # loop has a hidden state 'self' (good = working, bad = looping) inferred from its own progress per planning call.
    # Transport independently: executing its own open work directly (selfexec) is an action valued like a takeover.
    # Use clear and effective communication: a message derails the partner with probability derail_prob.
    # Parameters from the tuning episodes 20-49 (Team-Plan runs): robot out of calls in 58 of 150 episodes; own
    # responsibilities completed per planning call 0.084 when not out of calls and 0.010 when out of calls; partner out
    # of calls in 14 of 34 messaged goal episodes against 12 of 34 under ReAct+dialogue on the same episodes.
    self_model: bool = False
    prior_self: float = 0.39
    q_self_ok: float = 0.084
    q_self_loop: float = 0.010
    eta_self: float = 0.75
    cost_selfexec: float = 0.02
    derail_prob: float = 0.0
    eta_search: float = 0.6            # direct execution of own work whose object must first be found (assumption)


@dataclass
class Responsibility:
    index: int
    prop: Any
    objects: List[str]
    receptacles: List[str]
    rooms: List[str]
    function: str
    done: bool = False
    taken_over: bool = False          # the robot has taken this responsibility over (treated as its own from then on)

    env_interface: Any = None

    @property
    def text(self) -> str:
        return describe_proposition(self.prop, self.env_interface)


class PartnrResponsibilityModel(ConditionalModel):
    def __init__(self, env_interface, robot_uid: int, cfg: Optional[ModelConfig] = None):
        self.env = env_interface
        self.robot_uid = robot_uid
        self.human_uid = 1 - robot_uid
        self.cfg = cfg or ModelConfig()
        alloc = allocate(env_interface, robot_uid)
        self.partner = [self._mk(i, p, env_interface) for i, p in alloc[self.human_uid]]
        self.own = [self._mk(i, p, env_interface) for i, p in alloc[robot_uid]]
        objs: List[str] = []
        for r in self.partner:
            for o in r.objects:
                if o not in objs:
                    objs.append(o)
        self.hyp_objects = objs[: self.cfg.max_hypothesis_objects]
        self.posterior = self._prior()
        self.evidence_log: List[Dict[str, Any]] = []
        self._seen_actions = 0
        self._processed_responses: set = set()
        self._mention_kinds: set = set()      # (responsibility index, action name) pairs already counted
        self._touched: set = set()            # responsibilities the partner has acted on
        self._untouched_flagged = False
        self.partner_quit_step: Optional[int] = None   # simulator step at which the partner said Done
        self.room_known_fraction = 1.0       # share of open partner objects whose location the robot knows (set per decision)
        self.calls_used, self.calls_total = 0, 50          # the robot's planning calls (law-structured self model)
        self._self_seen_calls, self._self_seen_done = 0, 0

    # ---- construction --------------------------------------------------------------
    @staticmethod
    def _mk(i: int, p, env_interface=None) -> Responsibility:
        args = getattr(p, "args", None) or p.get("args", {})
        fn = getattr(p, "function_name", None) or p.get("function_name")
        return Responsibility(i, p, _names(args.get("object_handles"), env_interface),
                              _names(args.get("receptacle_handles"), env_interface), _names(args.get("room_ids"), env_interface), fn,
                              env_interface=env_interface)

    def component_names(self) -> List[str]:
        return (["goal"] + [f"obj:{o}" for o in self.hyp_objects] + ["skill"] + (["room"] if self.cfg.room_component else [])
                + (["self"] if self.cfg.self_model else []))

    def goal_values(self) -> List[str]:
        """'good' or 'drop:<i>' (the partner misread the goal and will not do responsibility i)."""
        return ["good"] + [f"drop:{r.index}" for r in self.partner]

    def _prior(self) -> Posterior:
        other = [n for n in self.component_names() if n != "goal"]
        gvals = self.goal_values(); pb_goal = self.cfg.prior_bad["goal"] if len(gvals) > 1 else 0.0
        hyps, probs = [], []
        for g in gvals:
            pg = (1 - pb_goal) if g == "good" else pb_goal / (len(gvals) - 1)
            for bits in itertools.product((0, 1), repeat=len(other)):
                d = {"goal": g}; pr = pg
                for name, b in zip(other, bits):
                    key = "obj" if name.startswith("obj:") else name
                    pb = self.cfg.prior_self if name == "self" else self.cfg.prior_bad.get(key, 0.15)
                    d[name] = "bad" if b else "good"; pr *= pb if b else 1 - pb
                hyps.append(Hypothesis.make(**d)); probs.append(pr)
        return Posterior(hyps, probs)

    def goal_bad_mass(self) -> float:
        return self.posterior.mass(lambda h: h.get("goal") != "good")

    def drop_mass(self, index: int) -> float:
        return self.posterior.mass(lambda h: h.get("goal") == f"drop:{index}")

    def components_of(self, r: Responsibility) -> List[str]:
        room = ["room"] if (self.cfg.room_component and r.objects) else []
        return ["goal"] + [f"obj:{o}" for o in r.objects if o in self.hyp_objects] + ["skill"] + room

    def wrong_belief(self, r: Responsibility, h: Hypothesis) -> bool:
        """Does h say the partner has a wrong belief about where r's object is (its own or the shared room belief)?"""
        if self.cfg.room_component and r.objects and h.get("room") == "bad":
            return True
        return any(h.get(f"obj:{o}") == "bad" for o in r.objects if o in self.hyp_objects)

    # ---- rates -----------------------------------------------------------------------
    def rate(self, r: Responsibility, h: Hypothesis) -> float:
        if r.done:
            return 1.0
        if r.taken_over:
            return self.cfg.eta_takeover
        if self.partner_quit_step is not None:
            return self.cfg.quit_rate
        p = self.cfg.base_rate
        if h.get("goal") == f"drop:{r.index}":
            p = self.cfg.goal_confused_rate
        if self.wrong_belief(r, h):
            p *= self.cfg.obj_wrong_factor
        if h.get("skill") == "bad":
            p *= self.cfg.skill_fail_factor
        return p

    def hyp_objects_names(self) -> List[str]:
        return [f"obj:{o}" for o in self.hyp_objects]

    def own_rate(self, r: Responsibility, h: Hypothesis) -> float:
        """Success of one of the robot's own responsibilities. With the self model: completed within the remaining
        planning calls, shared over the open own responsibilities, at q_self_ok per call when working and q_self_loop
        when looping; eta_self once it is being executed directly."""
        if r.done:
            return 1.0
        if not self.cfg.self_model:
            return self.cfg.own_rate
        if r.taken_over:
            return self.cfg.eta_self
        q = self.cfg.q_self_loop if h.get("self") == "bad" else self.cfg.q_self_ok
        k = max(1, sum(1 for o in self.own if not (o.done or o.taken_over)))
        m = max(0.0, float(self.calls_total - self.calls_used))
        return 1.0 - (1.0 - q) ** (m / k)

    def all_rates(self, h: Hypothesis) -> np.ndarray:
        own = [self.own_rate(r, h) for r in self.own]
        return np.array([self.rate(r, h) for r in self.partner] + own)

    def reliability(self, rates: np.ndarray) -> float:
        return float(np.prod(rates))                      # conjunction of propositions

    # ---- ConditionalModel ----------------------------------------------------------------
    def _resp(self, k: Any) -> Responsibility:
        return next(r for r in self.partner if r.index == k)

    @staticmethod
    def _targets(k: Any) -> Tuple[int, ...]:
        return tuple(k) if isinstance(k, tuple) else (k,)

    def _own(self, k: Any) -> Responsibility:
        return next(r for r in self.own if r.index == k)

    def uplift(self, action: Action, h: Hypothesis) -> float:
        """Uplift of the joint success of the taken-over set S: eta^|S| - prod_{j in S} p_j(h) (Lemma 1 with a set)."""
        S = self._targets(action.target)
        if action.kind == "selfexec":
            located = getattr(self, "located_own", None)
            eta = [self.cfg.eta_self if (located is None or k in located) else self.cfg.eta_search for k in S]
            return float(np.prod(eta)) - float(np.prod([self.own_rate(self._own(k), h) for k in S]))
        return self.cfg.eta_takeover ** len(S) - float(np.prod([self.rate(self._resp(k), h) for k in S]))

    def consequence(self, k: Any, h: Hypothesis) -> float:
        """Birnbaum importance of the set S under the conjunction: the product of all the other rates."""
        S = set(self._targets(k)); idx = [r.index for r in self.partner] + [r.index for r in self.own]
        rates = self.all_rates(h)
        keep = [j for j, i in enumerate(idx) if i not in S]       # own and partner indices are distinct propositions
        return float(np.prod(rates[keep])) if keep else 1.0

    def opportunity_cost(self, action: Action, h: Hypothesis) -> float:
        if action.kind == "selfexec":
            return 0.0
        n_own_left = sum(1 for r in self.own if not r.done)
        return self.cfg.opportunity_cost_per_own * n_own_left

    def continuation_value(self, h: Hypothesis) -> float:
        return self.reliability(self.all_rates(h))

    def repair(self, action: Action, h: Hypothesis) -> Tuple[Hypothesis, float]:
        comp = "goal" if action.kind == "clarify" else action.target
        if h.get(comp) != "good":
            r = 0.85 * (self.room_known_fraction if comp == "room" else 1.0)    # the message only fixes what it states
            return h.with_component(comp, "good"), r
        return h, 0.0

    def is_bad_for(self, r: Responsibility, h: Hypothesis) -> bool:
        """Does hypothesis h predict trouble on responsibility r (goal dropped for r, wrong object belief, broken skill)?"""
        if h.get("goal") == f"drop:{r.index}":
            return True
        return self.wrong_belief(r, h) or h.get("skill") == "bad"

    def pending_progress(self) -> List[Responsibility]:
        """Open partner responsibilities whose mid-budget progress evidence has not arrived yet."""
        if self._untouched_flagged:
            return []
        return [r for r in self.partner if not (r.done or r.taken_over) and r.index not in self._touched]

    def p_untouched(self, r: Responsibility, h: Hypothesis) -> float:
        if h.get("goal") == f"drop:{r.index}":
            return self.cfg.p_untouched_drop
        if self.wrong_belief(r, h):
            return self.cfg.p_untouched_obj
        return self.cfg.p_untouched_ok

    def answer_model(self, action: Action, h: Hypothesis) -> Dict[Any, float]:
        if action.kind == "inspect":
            pend = self.pending_progress(); out: Dict[Any, float] = {}
            for bits in itertools.product((0, 1), repeat=len(pend)):
                pr = 1.0
                for r, b in zip(pend, bits):
                    u = self.p_untouched(r, h); pr *= u if b else 1 - u
                out[frozenset(r.index for r, b in zip(pend, bits) if b)] = pr
            return out
        r = self._resp(action.target)
        bad = self.is_bad_for(r, h)
        t = self.cfg.answer_truthful
        return {"trouble": t if bad else 1 - t, "fine": 1 - t if bad else t}

    def logical_role(self, k: Any) -> Optional[str]:
        return "conjunctive"

    def sign_law_applies(self, k: Any) -> bool:
        return len(self.hyp_objects) == 0

    # ---- actions --------------------------------------------------------------------------
    def executable(self, r: Responsibility) -> bool:
        """Can the robot carry this proposition out itself (object on/inside a receptacle, or in a room)?"""
        if not r.objects:
            return False
        if r.function in ("is_on_top", "is_inside"):
            return bool(r.receptacles)
        return r.function == "is_in_room" and bool(r.rooms)

    def satisfied(self, r: Responsibility, world_graph) -> Optional[bool]:
        """Does world_graph show r as satisfied now?  None when it cannot tell (object not in the graph, unsupported
        proposition type).  Rooms are resolved through the object's furniture (get_room_for_entity takes furniture)."""
        if not self.executable(r):
            return None
        try:
            for o in world_graph.get_all_objects():
                if o.name != r.objects[0]:
                    continue
                furn = world_graph.find_furniture_for_object(o)
                if furn is None:
                    return False                         # held by an agent or on the floor
                if r.function == "is_in_room":
                    try:
                        rm = getattr(world_graph.get_room_for_entity(furn), "name", "") or ""
                    except Exception:
                        return False
                    return any(rm.startswith(x) for x in r.rooms)
                return any((getattr(furn, "name", "") or "").startswith(x) for x in r.receptacles)
        except Exception:
            return None
        return None

    def own_done(self) -> bool:
        """All of the robot's own *checkable* responsibilities (object on/inside a receptacle) are satisfied."""
        return all(r.done for r in self.own if self.executable(r))

    def actions(self, allow_queries: bool = True, used_repairs: Optional[set] = None,
                own_done_required: bool = True) -> List[Action]:
        """Candidate actions.  Responsibilities already taken over are excluded; each cause-level repair is offered
        at most once per episode (used_repairs holds the targets already sent)."""
        used = used_repairs or set()
        acts = [CONTINUE]
        for r in self.partner:
            if r.done or r.taken_over:
                continue
            if self.executable(r) and (not own_done_required or self.own_done()):
                acts.append(Action("takeover", r.index, self.cfg.cost_takeover))
            if allow_queries:
                acts.append(Action("query", r.index, self.cfg.cost_message))
        open_exec = [r.index for r in self.partner if not (r.done or r.taken_over) and self.executable(r)]
        if self.cfg.joint_takeover and len(open_exec) >= 2 and (not own_done_required or self.own_done()):
            acts.append(Action("takeover", tuple(open_exec), self.cfg.cost_takeover * len(open_exec)))
        if self.cfg.self_model:
            own_open = tuple(r.index for r in self.own if not (r.done or r.taken_over) and self.executable(r))
            if own_open:
                acts.append(Action("selfexec", own_open, self.cfg.cost_selfexec))
        if self.partner_quit_step is not None:      # nobody to talk to any more
            return [a for a in acts if a.kind in ("continue", "takeover", "selfexec")]
        if allow_queries and self.cfg.inspect and self.pending_progress():
            acts.append(Action("inspect", "progress", self.cfg.cost_inspect))
        if "goal" not in used and any(not (r.done or r.taken_over) for r in self.partner):
            acts.append(Action("clarify", "goal", self.cfg.cost_message))
        for o in self.hyp_objects:
            if f"obj:{o}" not in used:
                acts.append(Action("align", f"obj:{o}", self.cfg.cost_message))
        if self.cfg.room_component and "room" not in used and any(not (r.done or r.taken_over) and r.objects for r in self.partner):
            acts.append(Action("align", "room", self.cfg.cost_message))
        return acts

    def apply_repair(self, action: Action) -> None:
        """After sending a clarification/alignment: move posterior mass bad -> good with the repair success rate."""
        comp = "goal" if action.kind == "clarify" else action.target
        r = 0.85 * (self.room_known_fraction if comp == "room" else 1.0)
        new: Dict[Hypothesis, float] = {}
        for h, p in zip(self.posterior.hyps, self.posterior.probs):
            if h.get(comp) != "good":
                g = h.with_component(comp, "good")
                new[g] = new.get(g, 0.0) + p * r
                new[h] = new.get(h, 0.0) + p * (1 - r)
            else:
                new[h] = new.get(h, 0.0) + p
        hyps = list(new); self.posterior = Posterior(hyps, np.array([new[h] for h in hyps]))
        self.evidence_log.append({"type": "repair_sent", "component": comp})

    def object_known(self, r: Responsibility, world_graph) -> bool:
        """Is the responsibility's object already in the robot's (partial) world graph?"""
        if not r.objects:
            return False
        try:
            return any(o.name == r.objects[0] or o.name.startswith(r.objects[0]) for o in world_graph.get_all_objects())
        except Exception:
            return False

    def takeover_directive_many(self, rs: Sequence[Responsibility]) -> str:
        if len(rs) == 1:
            return self.takeover_directive(rs[0])
        items = "; ".join(f"'{r.text}'" for r in rs)
        return (f"Coordinator: the other agent may not complete these parts: {items}. Please take care of them yourself, one "
                f"at a time: for each object, if it is not in your object list, Explore rooms you have not explored to find "
                f"it; then Navigate to it, Pick it, Navigate to the target furniture and Place it there. If a Place fails "
                f"because you are not close enough, Navigate to the furniture first. Do not try the same failing action "
                f"more than twice.")

    def own_directive(self, rs: Sequence[Responsibility]) -> str:
        items = "; ".join(f"'{r.text}'" for r in rs)
        return (f"Coordinator: these parts of your own task are not done yet: {items}. Before saying Done, take care of "
                f"them one at a time: if the object is not in your object list, Explore rooms you have not explored to find "
                f"it; then Navigate to it, Pick it, Navigate to the target furniture and Place it there.")

    def announcement(self, rs: Sequence[Responsibility]) -> str:
        """Message to the partner when the robot takes responsibilities over (so that both do not do the same thing)."""
        items = "; ".join(r.text for r in rs)
        return f"I will take care of: {items}. You do not need to do these; please continue with the rest of your part."

    def set_room_known_fraction(self, world_graph) -> None:
        open_objs = [r.objects[0] for r in self.partner if not (r.done or r.taken_over) and r.objects]
        known = [o for o in open_objs if self._object_location_text(o, world_graph) is not None]
        self.room_known_fraction = len(known) / len(open_objs) if open_objs else 0.0

    def ground_truth_status(self) -> Dict[str, Any]:
        """Diagnostics only (never used for decisions): which responsibilities hold in the simulator's full world graph."""
        out: Dict[str, Any] = {}
        try:
            wg = self.env.full_world_graph
        except Exception:
            return out
        for who, rs in (("partner", self.partner), ("own", self.own)):
            for r in rs:
                out[f"{who}:{r.index}"] = self.satisfied(r, wg)
        return out

    def takeover_directive(self, r: Responsibility) -> str:
        rel = "inside" if r.function == "is_inside" else "on"
        where = f" {rel} {r.receptacles[0]}" if r.receptacles else (f" on a table or counter in the {r.rooms[0]}" if r.rooms else "")
        obj = r.objects[0] if r.objects else "the object"
        return (f"Coordinator: the other agent may not complete '{r.text}'. Please take care of it: if {obj} is not in your "
                f"object list, Explore rooms you have not explored to find it; then Navigate to it, Pick it, Navigate to "
                f"{r.receptacles[0] if r.receptacles else 'the target'} and Place it{where}. If a Place fails because you are "
                f"not close enough, Navigate to the furniture first. Do not try the same failing action more than twice.")

    def components_of_query(self, a: Action) -> Sequence[str]:
        if a.kind == "inspect":
            comps = ["goal"]
            for r in self.pending_progress():
                comps += [c for c in self.components_of(r) if (c.startswith("obj:") or c == "room") and c not in comps]
            return comps
        return self.components_of(self._resp(a.target)) if a.kind == "query" else []

    def mean_field_value(self, a: Action) -> float:
        """Partner-blind value: the team value computed from the per-responsibility *marginal* success rates only
        (product of marginals, i.e. the planner of Theorem 4 that does not see how failures are coupled)."""
        post = self.posterior; P = np.asarray(post.probs, dtype=float)
        R = np.array([self.all_rates(h) for h in post.hyps]); pbar = P @ R; base = float(np.prod(pbar))
        if a.kind == "continue":
            return 0.0
        if a.kind == "takeover":
            idx = [r.index for r in self.partner]; p2 = pbar.copy()
            for k in self._targets(a.target):
                p2[idx.index(k)] = self.cfg.eta_takeover
            oc = float(sum(p * self.opportunity_cost(a, h) for h, p in zip(post.hyps, P)))
            return float(np.prod(p2)) - base - a.cost - oc
        if a.kind in ("clarify", "align"):
            R2 = []
            for h in post.hyps:
                h2, rr = self.repair(a, h)
                R2.append(rr * self.all_rates(h2) + (1 - rr) * self.all_rates(h))
            return float(np.prod(P @ np.array(R2))) - base - a.cost
        return float("-inf")

    PLACEABLE = ("table", "counter", "shelves", "shelf", "bench", "cabinet", "chest_of_drawers", "stand", "desk",
                 "bed", "couch", "chair", "unit")

    def rearrange_hla(self, r: "Responsibility", world_graph=None) -> Optional[Tuple[str, str, None]]:
        """Oracle Rearrange that satisfies an on/inside proposition, or an in-room proposition (placed on a placeable
        piece of furniture in the target room, from the robot's world graph)."""
        if not r.objects:
            return None
        if r.function in ("is_on_top", "is_inside") and r.receptacles:
            rel = "on" if r.function == "is_on_top" else "within"
            return ("Rearrange", f"{r.objects[0]}, {rel}, {r.receptacles[0]}, None, None", None)
        if r.function == "is_in_room" and r.rooms and world_graph is not None:
            cands = []
            try:
                for f in world_graph.get_all_furnitures():
                    try:
                        rm = getattr(world_graph.get_room_for_entity(f), "name", "") or ""
                    except Exception:
                        continue
                    if any(rm.startswith(x) for x in r.rooms):
                        cands.append(f.name)
            except Exception:
                return None
            if cands:
                cands.sort(key=lambda n: (0 if any(k in n for k in self.PLACEABLE) else 1, n))
                return ("Rearrange", f"{r.objects[0]}, on, {cands[0]}, None, None", None)
        return None

    def open_own(self) -> List["Responsibility"]:
        """The robot's own checkable responsibilities, including partner responsibilities it has taken over, that its
        world graph does not show as satisfied."""
        mine = self.own + [r for r in self.partner if r.taken_over]
        return [r for r in mine if not r.done and self.executable(r)]

    def to_high_level_action(self, a: Action, world_graph) -> Optional[Tuple[str, str, None]]:
        """Map a planner action to a partnr high-level action (tool name, argument string, error)."""
        if a.kind == "takeover":
            if isinstance(a.target, tuple):
                return None
            return self.rearrange_hla(self._resp(a.target), world_graph)
        if a.kind == "query":
            r = self._resp(a.target)
            return ("SendMessage", f"Are you taking care of this: {r.text}? If you cannot find the object, tell me where you looked.", None)
        if a.kind == "clarify":
            parts = "; ".join(r.text for r in self.partner if not r.done)
            return ("SendMessage", f"Reminder of your part of the task: {parts}. Please do these.", None)
        if a.kind == "align" and a.target == "room":
            parts = []
            for r in self.partner:
                if r.done or r.taken_over or not r.objects:
                    continue
                loc = self._object_location_text(r.objects[0], world_graph)
                if loc is not None:
                    parts.append(f"the {r.objects[0]} is {loc}")
            if not parts:
                return None
            return ("SendMessage", "Here is where the objects you need are: " + "; ".join(parts) + ".", None)
        if a.kind == "align":
            obj = a.target.split(":", 1)[1]
            loc = self._object_location_text(obj, world_graph)
            if loc is None:
                return None
            return ("SendMessage", f"The {obj} is {loc}.", None)
        return None

    def _object_location_text(self, obj: str, world_graph) -> Optional[str]:
        """Where the robot last saw obj: furniture and its room.  WorldGraph.get_room_for_entity resolves furniture,
        not objects (it raises for an object), so the room is looked up through the furniture."""
        try:
            for o in world_graph.get_all_objects():
                if o.name == obj:
                    furn = world_graph.find_furniture_for_object(o)
                    if furn is None:
                        return None
                    try:
                        rm = getattr(world_graph.get_room_for_entity(furn), "name", None)
                    except Exception:
                        rm = None
                    return f"on the {furn.name} in the {rm}" if rm else f"on the {furn.name}"
        except Exception:
            return None
        return None

    def derail_prob(self, action: Action) -> float:
        """Law: use clear and effective communication. Every message risks derailing the partner."""
        return self.cfg.derail_prob if action.kind in ("clarify", "align", "query") else 0.0

    def derail_value(self, h: Hypothesis) -> float:
        """A derailed partner keeps issuing failing actions, as under the broken-skill cause."""
        return self.continuation_value(h.with_component("skill", "bad"))

    # ---- evidence ----------------------------------------------------------------------------
    def mark_done(self, world_graph) -> None:
        """done = satisfied now according to the robot's world graph (recomputed: partnr scores the terminal state, so
        an object moved away again makes its responsibility open again).  Unknown stays as it was."""
        for r in self.partner + self.own:
            s = self.satisfied(r, world_graph)
            if s is not None:
                r.done = s

    def update_from_history(self, step: int, step_budget: int) -> None:
        """Likelihood updates from the partner's action history (what it acts on, and how its skills end).
        `step` is the simulator step; `step_budget` the typical episode length in simulator steps."""
        hist = self.env.agent_action_history.get(self.human_uid, [])
        cfg = self.cfg
        # newly issued actions: acting on a responsibility's object argues against 'dropped' and 'wrong object belief'
        for el in hist[self._seen_actions:]:
            parts = [str(x) for x in (el.action or ()) if x is not None]
            kind = parts[0].lower() if parts else ""
            if kind == "done" and self.partner_quit_step is None:
                self.partner_quit_step = step
                self.evidence_log.append({"step": step, "type": "partner_done",
                                          "open": [r.index for r in self.partner if not (r.done or r.taken_over)]})
            text = " ".join(parts).lower()
            for r in self.partner:
                if not any(o.lower() in text for o in r.objects):
                    continue
                self._touched.add(r.index)
                if (r.index, kind) in self._mention_kinds:      # one update per (responsibility, action type)
                    continue
                self._mention_kinds.add((r.index, kind))
                ri = r.index; rr = r
                self.posterior.update(lambda h, ri=ri, rr=rr: -np.log(cfg.lr_action_mentions_object)
                                      if (h.get("goal") == f"drop:{ri}" or self.wrong_belief(rr, h)) else 0.0)
                self.evidence_log.append({"step": step, "type": "mentions", "resp": r.index, "action": kind})
        self._seen_actions = len(hist)
        # responses arrive when a skill ends: Pick/Place outcomes are the symptom of the partner's skill
        for idx, el in enumerate(hist):
            if idx in self._processed_responses or not el.response:
                continue
            self._processed_responses.add(idx)
            kind = str(el.action[0]).lower() if el.action else ""
            resp = str(el.response).lower()
            if kind not in ("pick", "place"):
                continue
            if "fail" in resp or "unexpected" in resp:
                if any(b in resp for b in BENIGN_FAILURE_REASONS):
                    continue          # the failure states its cause (position, grasp state, API use): not the skill
                self.posterior.update(lambda h: np.log(cfg.lr_failure_response) if h.get("skill") == "bad" else 0.0)
                self.evidence_log.append({"step": step, "type": "failure", "action": kind, "text": resp[:80]})
            elif "success" in resp:
                self.posterior.update(lambda h: -np.log(cfg.lr_success_response) if h.get("skill") == "bad" else 0.0)
                self.evidence_log.append({"step": step, "type": "success", "action": kind})
        # mid-budget: a responsibility the partner has not touched at all is probably missing from its goal
        if not self._untouched_flagged and step > step_budget / 2:
            self._untouched_flagged = True
            for r in self.partner:
                if r.index in self._touched or r.done or r.taken_over:
                    continue
                ri = r.index; rr = r
                self.posterior.update(lambda h, ri=ri, rr=rr:
                                      (np.log(cfg.lr_untouched_half_budget) if h.get("goal") == f"drop:{ri}" else 0.0)
                                      + (np.log(cfg.lr_untouched_obj) if self.wrong_belief(rr, h) else 0.0))
                self.evidence_log.append({"step": step, "type": "untouched", "resp": ri})

    def update_self(self, calls_used: int, calls_total: int) -> None:
        """Law: avoid infinite loops. Each robot planning call completes one of its own responsibilities with
        probability q_self_ok when its loop is working and q_self_loop when it is looping; the calls and completions
        since the last update are the evidence. Direct execution does not count as progress of the loop."""
        self.calls_total = int(calls_total)
        dn = max(0, int(calls_used) - self._self_seen_calls)
        done_now = sum(1 for r in self.own if r.done and not r.taken_over)
        dd = min(max(0, done_now - self._self_seen_done), dn)
        if self.cfg.self_model and dn > 0:
            qo, ql = self.cfg.q_self_ok, self.cfg.q_self_loop
            ll_ok = dd * np.log(qo) + (dn - dd) * np.log(1 - qo)
            ll_loop = dd * np.log(ql) + (dn - dd) * np.log(1 - ql)
            self.posterior.update(lambda h: ll_loop if h.get("self") == "bad" else ll_ok)
            self.evidence_log.append({"type": "self", "calls": int(calls_used), "own_done": done_now,
                                      "p_loop": round(self.posterior.mass(lambda h: h.get("self") == "bad"), 3)})
        self._self_seen_calls, self._self_seen_done, self.calls_used = int(calls_used), done_now, int(calls_used)

    def update_from_message(self, text: str, asked: Optional[int]) -> None:
        """A reply from the partner after a query about responsibility `asked` (keyword heuristic)."""
        t = text.lower()
        trouble = any(w in t for w in ("can't", "cannot", "could not", "couldn't", "not find", "no ", "don't", "do not", "where", "which"))
        y = "trouble" if trouble else "fine"
        if asked is None:
            return
        a = Action("query", asked, 0.0)
        self.posterior.update(lambda h: np.log(max(self.answer_model(a, h)[y], 1e-6)))
        self.evidence_log.append({"type": "answer", "resp": asked, "answer": y})
