"""CCRP robot planner for partnr-planner.

Runs the ReAct planner for the robot's own responsibilities and, at replanning points, one CCRP epoch over
the partner's responsibilities.  If the certified decision is an intervention it is carried out:
  takeover (one responsibility or the set of all remaining ones) -> announced to the partner (so that the two agents
      do not both do it) and handed to the ReAct planner as a directive;
  query / clarify / align -> sent as a message.
Otherwise the ReAct planner proceeds.  When the ReAct planner says Done while partner responsibilities are still
open, the robot keeps monitoring (Wait) instead of leaving, so that a late takeover remains possible.
Modes: 'ccrp' (default), 'ccrp_rl' (decisions from the value learned on the model, habitat_llm_ext/rl_value.py), 'partner_blind' (values from the marginal success rates only, Theorem 4's planner;
blind_scoring=product is the v2-pilot hybrid), 'never' (no interventions, no monitoring), 'always_takeover' (take
over all remaining partner responsibilities once half the budget is gone or the partner has stopped), 'protocol'
(the same intervention protocol with fixed rules instead of the decision model: a reminder at the first chance, one
location message, and the joint takeover once half the budget is gone or the partner has stopped).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ccrp.core import Action, CONTINUE, CCRPPlanner, PlannerConfig, Posterior, fill_table, certify

from habitat_llm_ext.tools.general.message_board import board
from habitat_llm_ext.responsibility_model import ModelConfig, PartnrResponsibilityModel
from .message_aware_planner import MessageAwareReactPlanner


def _log_dir() -> str:
    """Hydra's output directory for this run (hydra.job.chdir is off, so cwd is the repo root)."""
    try:
        from hydra.core.hydra_config import HydraConfig
        return HydraConfig.get().runtime.output_dir
    except Exception:
        return os.environ.get("CCRP_LOG_DIR", ".")


class CCRPRobotPlanner(MessageAwareReactPlanner):
    def __init__(self, plan_config, env_interface):
        # defaults first: the base constructor calls reset() before the rest of this constructor runs
        self.model = None; self.interventions = 0; self.last_intervention_step = -999; self.iter = 0
        self.pending_query = None; self.decision_log = []; self.used_repairs = set(); self.n_messages = 0; self.episode_id = None
        self.mode = "ccrp"; self.tier = "E"; self.monitoring = False; self.n_waits = 0; self.last_gt = {}; self.finished = False
        self.blind_scoring = "product"; self.silent_messages = False; self.own_check = False; self.own_checked = set()
        self.takeover_queue = []; self.queue_active = None; self.pending_directive = []; self.queue_log = []
        super().__init__(plan_config, env_interface)
        c = dict(plan_config.get("ccrp", {}))
        self.mode = c.get("mode", "ccrp")
        self.rl_value_path = c.get("rl_value_path", None)          # ccrp_rl: learned value (None: package default)
        self.tier = c.get("tier", "E")
        self.eps = float(c.get("eps", 0.0))
        self.alpha = float(c.get("alpha", 0.1))
        self.max_interventions = int(c.get("max_interventions", 4))      # takeovers per episode
        self.max_messages = int(c.get("max_messages", 2))                  # cause-level repairs + queries per episode
        # the clock is the simulator step (get_next_action is called every step; episodes run ~3k-14k steps, max 20k)
        self.step_budget = int(c.get("step_budget", 8000))       # typical episode length in simulator steps
        self.min_gap_steps = int(c.get("min_gap_steps", 400))    # simulator steps between interventions
        self.min_iter_before = int(c.get("min_iter_before_intervention", 600))   # let the partner start before judging
        self.directive_takeover = bool(c.get("directive_takeover", True))        # hand takeovers to the ReAct planner
        self.own_done_required = bool(c.get("takeover_only_when_own_done", True))
        self.monitor = bool(c.get("monitor", True)) and self.mode != "never"   # Done -> Wait while partner work is open
        self.max_waits = int(c.get("max_waits", 25))
        self.blind_scoring = str(c.get("blind_scoring", "product"))     # product (v2 pilot) | mean_field
        # v4: coordinator messages go straight to the partner (the robot's actions and prompt are not touched)
        self.silent_messages = bool(c.get("silent_messages", False))
        # v4: before the robot stops, its own unfinished checkable responsibilities are completed directly
        self.own_check = bool(c.get("own_check", False)) and self.mode != "never"
        # v5 budget guard: the 7B robot's usual failure is spending all its planning calls without finishing. Once it
        # has used budget_guard_frac of them, its own unfinished work is completed with direct skills (no planning
        # calls), and half of the call budget counts like half of the step budget for the takeover rules
        self.budget_guard = bool(c.get("budget_guard", False)) and self.mode != "never"
        self.budget_guard_frac = float(c.get("budget_guard_frac", 0.7))
        self.guard_fired = False
        # v7 search (law: find and transport independently): an object the robot has not located is searched for by the
        # coordinator, which explores the rooms nobody has explored yet with the Explore skill (no planning calls)
        self.search_unseen = bool(c.get("search_unseen", False)); self.searched_rooms = set()
        # v7: the waiting rules count simulator steps, but the robot's budget is planning calls, and a fast-failing loop
        # spends 50 calls in under 200 steps. The wait before judging the partner then applies only to actions aimed at
        # the partner, and decisions about the robot's own work may come earlier.
        self.self_first = bool(c.get("self_first", False))
        mc = {k: (dict(v) if hasattr(v, "items") else v) for k, v in c.get("model", {}).items()}
        self.model_cfg = ModelConfig(**mc) if mc else ModelConfig()

    def reset(self):
        if getattr(self, "model", None) is not None and not getattr(self, "_flushed", True):
            self._flush_log()
        super().reset()
        self.model = None; self.interventions = 0; self.last_intervention_step = -999; self.iter = 0
        self.pending_query = None; self.decision_log = []; self.used_repairs = set(); self.n_messages = 0
        self.monitoring = False; self.n_waits = 0; self.last_gt = {}; self.finished = False; self._flushed = False
        self.guard_fired = False; self.searched_rooms = set()
        self.takeover_queue = []; self.queue_active = None; self.pending_directive = []; self.queue_log = []
        self.own_checked = set()

    def _flush_log(self) -> None:
        """Append the previous episode's decisions (and the injected ground truth, for scoring only) to a JSONL
        file in the hydra run directory."""
        if self.model is None or getattr(self, "_flushed", False):
            return
        try:
            rec = {"episode_id": self.episode_id, "mode": self.mode, "tier": self.tier,
                   "interventions": self.interventions, "n_decisions": len(self.decision_log),
                   "injection": getattr(board, "extra", {}).get("injection"), "decisions": self.decision_log,
                   "evidence": self.model.evidence_log if self.model is not None else [],
                   "final_posterior_goal_bad": self.model.goal_bad_mass() if self.model else None,
                   "partner_done_step": self.model.partner_quit_step if self.model else None,
                   "waits": self.n_waits, "robot_calls": self.replanning_count, "budget_guard_fired": self.guard_fired,
                   "ground_truth_at_end": self.last_gt,
                   "direct_rearranges": getattr(self, "queue_log", [])}
            with open(os.path.join(_log_dir(), "ccrp_decisions.jsonl"), "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            self.decision_log = []; self._flushed = True
        except Exception as e:   # logging must never break an episode
            print(f"[CCRP] could not write decision log: {e}")

    # ---- helpers ---------------------------------------------------------------------------
    def _planner_info(self, high_level_actions, responses, print_str: str, thought: str) -> Dict[str, Any]:
        uid = self.agents[0].uid
        return {
            "replanned": {uid: True}, "replan_required": {uid: self.replan_required}, "responses": responses,
            "thought": {uid: thought}, "is_done": {uid: self.is_done}, "print": print_str,
            "high_level_actions": high_level_actions, "prompts": {uid: self.curr_prompt}, "traces": {uid: self.trace},
            "replanning_count": {uid: self.replanning_count}, "agent_states": self.get_last_agent_states(),
            "agent_positions": self.get_last_agent_positions(), "agent_collisions": self.get_agent_collisions(),
        }

    def _decide(self, world_graph) -> Tuple[Action, Dict[str, Any]]:
        m = self.model
        m.mark_done(world_graph)
        own_ok = (m.own_done() or self.iter >= self.step_budget // 2       # own checkable work done, or half the budget spent
                  or (self.budget_guard and self._calls_used() >= 0.5))
        acts = m.actions(allow_queries=(self.mode in ("ccrp", "ccrp_rl")), used_repairs=self.used_repairs,
                         own_done_required=(self.own_done_required and not own_ok))
        if self.n_messages >= self.max_messages:
            acts = [a for a in acts if a.kind not in ("clarify", "align", "query")]
        if self.interventions >= self.max_interventions and m.partner_quit_step is None:
            acts = [a for a in acts if a.kind != "takeover"]
        # a location message can only be sent for an object the robot has seen (otherwise nothing would be sent while
        # the message budget and the posterior were updated as if it had been)
        acts = [a for a in acts if a.kind != "align" or m.to_high_level_action(a, world_graph) is not None]
        if self.self_first and self.iter < self.min_iter_before:
            acts = [a for a in acts if a.kind in ("continue", "selfexec")]      # too early to judge the partner
        if any(a.kind == "selfexec" for a in acts):
            # transport independently: direct execution covers own work whose object the robot has located
            kept = []
            for a in acts:
                if a.kind != "selfexec":
                    kept.append(a); continue
                known = tuple(k for k in a.target if m.object_known(m._own(k), world_graph)
                              and m.rearrange_hla(m._own(k), world_graph) is not None)
                m.located_own = set(known)
                targets = tuple(k for k in a.target if k in known or (self.search_unseen and m.executable(m._own(k))))
                if targets:
                    kept.append(Action("selfexec", targets, a.cost))
            acts = kept
        if m.cfg.room_component:
            m.set_room_known_fraction(world_graph)
        if self.mode == "never":
            return CONTINUE, {"mode": "never"}
        if self.mode == "protocol":
            takes = sorted([a for a in acts if a.kind == "takeover"], key=lambda a: -len(m._targets(a.target)))
            if takes and (self._late() or m.partner_quit_step is not None):
                return takes[0], {"mode": "protocol", "rule": "joint takeover"}
            for kind in ("clarify", "align"):
                cand = [a for a in acts if a.kind == kind]
                if cand:
                    return cand[0], {"mode": "protocol", "rule": kind}
            return CONTINUE, {"mode": "protocol"}
        if self.mode == "always_takeover":
            cands = sorted([a for a in acts if a.kind == "takeover"], key=lambda a: -len(m._targets(a.target)))
            if cands and (self._late() or m.partner_quit_step is not None):
                return cands[0], {"mode": "always"}
            return CONTINUE, {"mode": "always"}
        if self.mode == "partner_blind" and self.blind_scoring == "mean_field":
            # values from the per-responsibility marginal success rates only (Theorem 4's partner-blind planner)
            vals = {a: m.mean_field_value(a) for a in acts if a.kind in ("continue", "takeover", "clarify", "align")}
            best = max(vals, key=lambda a: vals[a]) if vals else CONTINUE
            if vals.get(best, 0.0) <= 0.0:
                best = CONTINUE
            return best, {"mode": "partner_blind", "scoring": "mean_field", "value": vals.get(best, 0.0),
                          "mixture": {str(a): float(v) for a, v in vals.items()}}
        if self.mode == "partner_blind":
            # v2 pilot: marginal uplift x marginal consequence for takeovers (the myopic score E[u] E[L] of
            # Theorem 1); exact mixture for repairs; no queries
            post = m.posterior; best, best_v = CONTINUE, 0.0
            for a in acts:
                if a.kind == "takeover":
                    u = sum(p * m.uplift(a, h) for h, p in zip(post.hyps, post.probs))
                    L = sum(p * m.consequence(a.target, h) for h, p in zip(post.hyps, post.probs))
                    oc = sum(p * m.opportunity_cost(a, h) for h, p in zip(post.hyps, post.probs))
                    v = u * L - a.cost - oc
                elif a.kind in ("clarify", "align"):
                    v = sum(p * (lambda h2r: h2r[1] * (m.continuation_value(h2r[0]) - m.continuation_value(h)))(m.repair(a, h))
                            for h, p in zip(post.hyps, post.probs)) - a.cost
                else:
                    continue
                if v > best_v:
                    best, best_v = a, v
            return best, {"mode": "partner_blind", "scoring": "product", "value": best_v}
        if self.mode == "ccrp_rl":
            # Team-Plan + RL: the value learned on the planner's own world model (habitat_llm_ext/rl_value.py);
            # episodes outside the simulator's range fall through to the v4 one-step decision below
            from habitat_llm_ext.rl_value import rl_decide
            out = rl_decide(self, acts, self.rl_value_path)
            if out is not None:
                return out
        planner = CCRPPlanner(m, PlannerConfig(tier=self.tier, eps=self.eps, alpha=self.alpha), m.components_of_query)
        d = planner.decide(acts, m.posterior)
        info = {"mode": "ccrp", "certified": d.certified, "tier": d.tier,
                "mixture": {str(a): float(v) for a, v in zip(d.table.actions, d.table.mixture())},
                "lo": d.cert.lo.tolist(), "hi": d.cert.hi.tolist()}
        return d.action, info

    # ---- main --------------------------------------------------------------------------------
    def get_next_action(self, instruction, observations, world_graph, verbose: bool = False):
        uid = self.agents[0].uid
        if self.finished:
            return super().get_next_action(instruction, observations, world_graph, verbose=verbose)
        if self.model is None:
            self.model = PartnrResponsibilityModel(self.env_interface, uid, self.model_cfg)
            try:
                self.episode_id = getattr(self.env_interface.env.env.env._env.current_episode, "episode_id", None)
            except Exception:
                self.episode_id = None
        self.iter += 1
        # evidence: partner history and replies to a pending query
        self.model.update_from_history(self.iter, self.step_budget)
        at_replan = self.replan_required or self.curr_prompt == ""
        if at_replan and self.curr_prompt:
            for _, text, _ in board.unread(uid):      # read replies only at replanning points (skills keep running)
                self.model.update_from_message(text, self.pending_query)
                self.pending_query = None
                self.curr_prompt += f'{self.planner_config.llm.user_tag}\nMessage from the other agent: "{text}"\n{self.planner_config.llm.eot_tag}{self.planner_config.llm.assistant_tag}'
        if at_replan:
            self.last_gt = self.model.ground_truth_status()       # diagnostics only
        if at_replan and (self.takeover_queue or self.queue_active is not None or self.pending_directive):
            out = self._run_takeover_queue(uid, instruction, observations, world_graph, verbose)
            if out is not None:
                return out
        if (at_replan and self.budget_guard and self.own_check and self.curr_prompt and not self.guard_fired
                and self._calls_used() >= self.budget_guard_frac):
            self.guard_fired = True
            if self._queue_own_work(uid, world_graph):
                self.queue_log.append({"iter": self.iter, "budget_guard": self.replanning_count})
                out = self._run_takeover_queue(uid, instruction, observations, world_graph, verbose)
                if out is not None:
                    return out
        quit_ = self.model.partner_quit_step is not None
        self_work = False
        if self.model.cfg.self_model:
            # law-structured model: the robot's own planning calls are evidence about its own loop
            self.model.update_self(self.replanning_count, int(self.planner_config.replanning_threshold))
            self_work = any(not (r.done or r.taken_over) and self.model.executable(r) for r in self.model.own)
        early_self = self.self_first and self_work and self.iter < self.min_iter_before
        can_intervene = ((self.interventions < self.max_interventions or self.n_messages < self.max_messages or quit_ or self_work)
                         and (self.iter >= self.min_iter_before or early_self)
                         and (self.iter - self.last_intervention_step >= self.min_gap_steps or quit_ or self.monitoring)
                         and not self.is_done)
        if at_replan and can_intervene and self.model.partner:
            t0 = time.time()
            action, info = self._decide(world_graph[uid])
            info.update({"iter": self.iter, "action": str(action), "seconds": time.time() - t0,
                         "posterior_goal_bad": self.model.goal_bad_mass(), "drop_masses": {r.index: round(self.model.drop_mass(r.index), 3) for r in self.model.partner}})
            self.decision_log.append(info)
            try:
                with open(os.path.join(_log_dir(), "ccrp_decisions_stream.jsonl"), "a") as f:
                    f.write(json.dumps({"episode_id": self.episode_id, "injection": getattr(board, "extra", {}).get("injection"), **info}, default=str) + "\n")
            except Exception:
                pass
            if action.kind == "selfexec":
                # transport independently: the robot finishes its located own work with direct skills (no planning calls)
                queued = []
                for r in self.model.own:
                    if r.index in self._targets_of(action) and not r.done:
                        hla = self.model.rearrange_hla(r, world_graph[uid]) if self.model.object_known(r, world_graph[uid]) else None
                        if hla is not None:
                            self.takeover_queue.append({"r": r, "hla": hla, "tries": 0, "own": True})
                        elif self.search_unseen and self.model.executable(r):
                            self.takeover_queue.append({"r": r, "find": True, "explored": [], "own": True})
                        else:
                            continue
                        self.own_checked.add(r.index); r.taken_over = True; queued.append(r.index)
                if queued:
                    self.queue_log.append({"iter": self.iter, "selfexec": queued, "calls": self.replanning_count})
                    out = self._run_takeover_queue(uid, instruction, observations, world_graph, verbose)
                    if out is not None:
                        if isinstance(out[1], dict):
                            out[1]["ccrp"] = info
                        return out
            elif action.kind != "continue":
                if self.curr_prompt == "":
                    self.curr_prompt, self.params = self.prepare_prompt(instruction, world_graph[uid], observations=observations)
                    self.trace += f"Task: {instruction}\nThought: "
                # bookkeeping: repairs are sent once per target; a takeover makes the responsibility the robot's own
                if action.kind in ("clarify", "align", "query"):
                    self.n_messages += 1
                if action.kind in ("clarify", "align"):
                    self.used_repairs.add("goal" if action.kind == "clarify" else action.target)
                    self.model.apply_repair(action)
                if action.kind == "takeover":
                    rs = [self.model._resp(k) for k in self.model._targets(action.target)]
                    for r in rs:
                        r.taken_over = True
                    # announce, so that the partner does not do the same thing
                    if not quit_:
                        board.post(uid, self.model.announcement(rs))
                    for r in rs:
                        direct = (not self.directive_takeover) and self.model.object_known(r, world_graph[uid])
                        hla = self.model.to_high_level_action(Action("takeover", r.index), world_graph[uid]) if direct else None
                        if hla is not None:
                            self.takeover_queue.append({"r": r, "hla": hla, "tries": 0})
                        elif self.search_unseen and self.model.executable(r):
                            self.takeover_queue.append({"r": r, "find": True, "explored": []})
                        else:
                            self.pending_directive.append(r)
                    self.interventions += 1; self.last_intervention_step = self.iter; self.monitoring = False
                    out = self._run_takeover_queue(uid, instruction, observations, world_graph, verbose)
                    if out is not None:
                        if isinstance(out[1], dict):
                            out[1]["ccrp"] = info
                        return out
                hla = self.model.to_high_level_action(action, world_graph[uid])
                if hla is not None and hla[0] == "SendMessage" and self.silent_messages:
                    board.post(uid, hla[1])
                    self.last_intervention_step = self.iter
                    if action.kind == "query":
                        self.pending_query = action.target
                    hla = None          # nothing for the robot's body to do; the ReAct planner continues below
                elif hla is not None:
                    if self.curr_prompt == "":
                        self.curr_prompt, self.params = self.prepare_prompt(instruction, world_graph[uid], observations=observations)
                        self.trace += f"Task: {instruction}\nThought: "
                    high_level_actions = {uid: hla}
                    low_level_actions, responses = self.process_high_level_actions(high_level_actions, observations)
                    self.last_high_level_actions = high_level_actions
                    if hla[0] == "SendMessage":
                        note = f'You sent a message to the other agent: "{hla[1]}". Continue with your own part of the task.'
                    else:
                        note = f"Coordinator decision: {action}; executing {hla[0]}[{hla[1]}]."
                    self.curr_prompt += f"{self.planner_config.llm.user_tag}{note}\n{self.planner_config.llm.eot_tag}{self.planner_config.llm.assistant_tag}"
                    self.trace += note + "\n"
                    self.replan_required = any(responses.values())
                    print_str = note + "\n" + self._add_responses_to_prompt(responses)
                    if action.kind == "takeover":
                        self.interventions += 1
                    self.last_intervention_step = self.iter
                    if action.kind == "query":
                        self.pending_query = action.target
                    board.tick()
                    pinfo = self._planner_info(high_level_actions, responses, print_str, note)
                    pinfo["ccrp"] = info
                    return low_level_actions, pinfo, False
        if self.monitoring and at_replan:
            out = self._monitor_step(uid, observations)
            if out is not None:
                return out
            self.monitoring = False
            self.is_done = True; self.finished = True
            self._flush_log()
            return {}, self._planner_info({uid: ("Done", None, None)}, {}, "", "Done"), True
        out = super().get_next_action(instruction, observations, world_graph, verbose=verbose)
        if (out[2] and self.own_check and self.model is not None
                and self.replanning_count <= self.planner_config.replanning_threshold and self._queue_own_work(uid, world_graph)):
            # the ReAct planner wants to stop with its own part unfinished: finish it first
            self.is_done = False; self.monitoring = False
            res = self._run_takeover_queue(uid, instruction, observations, world_graph, verbose)
            if res is not None:
                return res
        if out[2] and self.monitor and self.model is not None and self.replanning_count <= self.planner_config.replanning_threshold:
            # the ReAct planner wants to stop: keep monitoring while partner responsibilities are still open
            self.is_done = False
            self.monitoring = True
            w = self._monitor_step(uid, observations)
            if w is not None:
                return w
            self.monitoring = False; self.is_done = True
        if out[2]:                       # episode done for this agent: persist the decision log now
            self.finished = True
            self.last_gt = self.model.ground_truth_status() if self.model is not None else {}
            self._flush_log()
        return out

    @staticmethod
    def _targets_of(action) -> tuple:
        return tuple(action.target) if isinstance(action.target, tuple) else (action.target,)

    def _calls_used(self) -> float:
        """Share of the robot's planning calls already used."""
        return self.replanning_count / max(1, int(self.planner_config.replanning_threshold))

    def _late(self) -> bool:
        """Half the step budget gone, or (budget guard) half the planning calls."""
        return self.iter > self.step_budget / 2 or (self.budget_guard and self._calls_used() >= 0.5)

    def _queue_own_work(self, uid, world_graph) -> bool:
        """Before the robot stops: queue its own unfinished checkable responsibilities (once each).  Seen objects are
        rearranged directly; unseen ones go to the ReAct planner as one directive."""
        m = self.model
        m.mark_done(world_graph[uid])
        queued = False
        for r in m.open_own():
            if r.index in self.own_checked:
                continue
            self.own_checked.add(r.index)
            hla = m.rearrange_hla(r, world_graph[uid]) if m.object_known(r, world_graph[uid]) else None
            if hla is not None:
                self.takeover_queue.append({"r": r, "hla": hla, "tries": 0, "own": True})
            elif self.search_unseen:
                self.takeover_queue.append({"r": r, "find": True, "explored": [], "own": True})
            else:
                self.pending_directive.append(r)
            queued = True
        if queued:
            self.queue_log.append({"iter": self.iter, "own_check": [r.index for r in m.open_own()]})
        return queued

    def _next_room(self, uid, world_graph, item):
        """First room that neither the robot nor this search has explored, in world-graph order."""
        done = set(self.searched_rooms) | set(item.get("explored", []))
        for el in self.env_interface.agent_action_history.get(uid, []):
            a = el.action or ()
            if a and a[0] == "Explore" and el.response and "success" in str(el.response).lower():
                done.add(str(a[1]))
        try:
            rooms = [rm.name for rm in world_graph.get_all_rooms()]
        except Exception:
            return None
        return next((rm for rm in rooms if rm not in done), None)

    def _robot_last_response(self, uid) -> str:
        hist = self.env_interface.agent_action_history.get(uid, [])
        return str(hist[-1].response or "") if hist else ""

    def _run_takeover_queue(self, uid, instruction, observations, world_graph, verbose=False):
        """Carry out taken-over responsibilities: objects the robot has seen are rearranged directly with the oracle
        Rearrange skill (one at a time, a failed one is retried once); unseen objects and repeated failures go to
        the ReAct planner as one directive.  Returns the planner output, or None when there is nothing left."""
        tag, eot, asst = self.planner_config.llm.user_tag, self.planner_config.llm.eot_tag, self.planner_config.llm.assistant_tag
        if self.queue_active is not None:                      # the previous queued action has just ended
            item = self.queue_active; self.queue_active = None
            if item.get("find"):
                self.takeover_queue.insert(0, item)            # an Explore ended: look for the object again
            else:
                resp = self._robot_last_response(uid).lower()
                if any(w in resp for w in ("fail", "unexpected", "wrong use", "not present", "not found")):
                    if item["tries"] < 1:
                        item["tries"] += 1; self.takeover_queue.insert(0, item)
                    elif self.search_unseen and not item.get("searched") and self.model.executable(item["r"]):
                        self.takeover_queue.insert(0, {"r": item["r"], "find": True, "explored": [], "own": item.get("own"),
                                                       "searched": True})
                    else:
                        self.pending_directive.append(item["r"])
                else:
                    self.queue_log.append({"iter": self.iter, "rearranged": item["r"].index, "own": item.get("own", False),
                                           "response": resp[:60]})
        while self.takeover_queue and self.takeover_queue[0].get("find"):
            item = self.takeover_queue[0]; r = item["r"]
            if self.model.object_known(r, world_graph[uid]):
                hla = self.model.rearrange_hla(r, world_graph[uid])
                if hla is None:
                    self.takeover_queue.pop(0); self.pending_directive.append(r); continue
                self.takeover_queue[0] = {"r": r, "hla": hla, "tries": 0, "own": item.get("own"), "searched": item.get("searched")}
                break
            room = self._next_room(uid, world_graph[uid], item)
            if room is None:                                   # every room explored and still not seen
                self.takeover_queue.pop(0); self.pending_directive.append(r); continue
            item["explored"].append(room); self.searched_rooms.add(room)
            self.takeover_queue.pop(0); self.queue_active = item
            self.queue_log.append({"iter": self.iter, "search": r.index, "explore": room})
            hla = {uid: ("Explore", room, None)}
            if self.curr_prompt == "":
                self.curr_prompt, self.params = self.prepare_prompt(instruction, world_graph[uid], observations=observations)
                self.trace += f"Task: {instruction}\nThought: "
            low_level_actions, responses = self.process_high_level_actions(hla, observations)
            self.last_high_level_actions = hla
            note = f"Coordinator: looking for {r.objects[0]} ('{r.text}'); exploring {room}."
            self.curr_prompt += f"{tag}{note}\n{eot}{asst}"; self.trace += note + "\n"
            self.replan_required = any(responses.values())
            board.tick()
            print_str = note + "\n" + self._add_responses_to_prompt(responses)
            return low_level_actions, self._planner_info(hla, responses, print_str, note), False
        if self.takeover_queue:
            item = self.takeover_queue.pop(0); self.queue_active = item
            hla = {uid: item["hla"]}
            if self.curr_prompt == "":
                self.curr_prompt, self.params = self.prepare_prompt(instruction, world_graph[uid], observations=observations)
                self.trace += f"Task: {instruction}\nThought: "
            low_level_actions, responses = self.process_high_level_actions(hla, observations)
            self.last_high_level_actions = hla
            r = item["r"]
            note = f"Coordinator: you are taking care of '{r.text}' yourself; executing {item['hla'][0]}[{item['hla'][1]}]."
            self.curr_prompt += f"{tag}{note}\n{eot}{asst}"; self.trace += note + "\n"
            self.replan_required = any(responses.values())
            board.tick()
            print_str = note + "\n" + self._add_responses_to_prompt(responses)
            return low_level_actions, self._planner_info(hla, responses, print_str, note), False
        if self.pending_directive:
            rs = self.pending_directive; self.pending_directive = []
            own = [r for r in rs if r in self.model.own]; theirs = [r for r in rs if r not in self.model.own]
            note = " ".join(([self.model.takeover_directive_many(theirs)] if theirs else []) +
                            ([self.model.own_directive(own)] if own else []))
            if self.curr_prompt == "":
                self.curr_prompt, self.params = self.prepare_prompt(instruction, world_graph[uid], observations=observations)
                self.trace += f"Task: {instruction}\nThought: "
            self.curr_prompt += f"{tag}{note}\n{eot}{asst}"; self.trace += note + "\n"
            self.replan_required = True
            board.tick()
            out = super().get_next_action(instruction, observations, world_graph, verbose=verbose)
            if out[2] and (self.budget_guard or self.model.cfg.self_model):
                # the ReAct planner answered the directive with Done: the robot stops here, since a Done it logs must
                # stay its last action (partnr rejects any later response)
                self.finished = True
                self.last_gt = self.model.ground_truth_status()
                self._flush_log()
            return out
        return None

    def _monitor_step(self, uid, observations):
        """Wait (one Wait skill, ~600 simulator steps) while some partner responsibility is open and the partner is
        still active; None when there is nothing left to watch."""
        m = self.model
        open_ = [r for r in m.partner if not (r.done or r.taken_over)]
        if not open_ or m.partner_quit_step is not None or self.n_waits >= self.max_waits:
            return None
        self.n_waits += 1
        hla = {uid: ("Wait", "1", None)}
        low, responses = self.process_high_level_actions(hla, observations)
        self.last_high_level_actions = hla
        self.replan_required = any(responses.values())
        note = "Waiting while the other agent finishes its part."
        return low, self._planner_info(hla, responses, note, note), False
