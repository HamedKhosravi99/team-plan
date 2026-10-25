"""End-to-end checks of the PARTNR robot planner on the offline harness (tests/partnr_sim.py)."""
import pytest

pytest.importorskip("numpy")
from partnr_sim import run  # noqa: E402

OK = "Successful execution!"
WORKING_PARTNER = [(50, ("Explore", "bedroom_1", None), OK), (900, ("Navigate", "fork_0", None), OK),
                   (1300, ("Pick", "fork_0", None), OK), (1600, ("Navigate", "bread_5", None), OK)]
QUITTING_PARTNER = [(50, ("Explore", "bedroom_1", None), OK), (1500, ("Done", None, None), None)]
# the robot's LLM places one of its two items, then says Done
FORGETFUL_ROBOT = [("Explore", "kitchen_1", None), ("Rearrange", "cup_0, on, table_10, None, None", None)] + [("Done", None, None)] * 4
DILIGENT_ROBOT = [("Explore", "kitchen_1", None), ("Rearrange", "cup_0, on, table_10, None, None", None),
                  ("Rearrange", "plate_1, on, table_10, None, None", None)] + [("Done", None, None)] * 4
V4 = {"silent_messages": True, "own_check": True}
V3 = {"silent_messages": False, "own_check": False}
V41 = {"silent_messages": True, "own_check": True, "search_unseen": True}


def test_own_check_finishes_forgotten_own_item():
    log, placed, _ = run("ccrp", WORKING_PARTNER, FORGETFUL_ROBOT, V4)
    assert {"cup_0", "plate_1"} <= placed
    assert any(a == "Rearrange" and arg.startswith("plate_1") for _, a, arg in log)


def test_without_own_check_the_forgotten_item_stays_undone():
    _, placed, _ = run("ccrp", WORKING_PARTNER, FORGETFUL_ROBOT, V3)
    assert "plate_1" not in placed


def test_silent_messages_reach_the_partner_but_not_the_robot_actions():
    log, _, msgs = run("ccrp", WORKING_PARTNER, DILIGENT_ROBOT, V4)
    assert not any(a == "SendMessage" for _, a, _ in log)
    assert any(m.startswith("Reminder of your part") for m in msgs)


def test_v3_messages_are_robot_actions():
    log, _, _ = run("ccrp", WORKING_PARTNER, DILIGENT_ROBOT, V3)
    assert any(a == "SendMessage" for _, a, _ in log)


def test_takeover_after_partner_quits_with_one_retry():
    log, placed, msgs = run("ccrp", QUITTING_PARTNER, DILIGENT_ROBOT, V4, fail_once={"bread_5"})
    assert {"fork_0", "bread_5", "cup_0", "plate_1"} <= placed
    assert sum(1 for _, a, arg in log if a == "Rearrange" and arg.startswith("bread_5")) == 2


def test_never_mode_does_not_intervene():
    log, placed, msgs = run("never", QUITTING_PARTNER, FORGETFUL_ROBOT, V4)
    assert placed == {"cup_0"} and not msgs


def test_protocol_shares_the_execution_layer():
    _, placed, _ = run("protocol", QUITTING_PARTNER, FORGETFUL_ROBOT, V4)
    assert {"cup_0", "plate_1", "fork_0", "bread_5"} <= placed


def test_own_check_handles_in_room_propositions():
    """'bring plate_1 to the bedroom' (is_in_room) is completed onto placeable bedroom furniture, not the door."""
    log, placed, _ = run("ccrp", WORKING_PARTNER, FORGETFUL_ROBOT, V4, plate_room="bedroom")
    assert "plate_1" in placed
    assert any(a == "Rearrange" and arg.startswith("plate_1, on, bed_5") for _, a, arg in log)


def test_done_is_recomputed_when_an_object_is_moved_away():
    from partnr_sim import World
    from habitat_llm_ext.responsibility_model import PartnrResponsibilityModel
    w = World()
    m = PartnrResponsibilityModel(w, 0)
    w.place("cup_0", "table_10"); m.mark_done(w.world_graph[0])
    own0 = next(r for r in m.own if r.objects == ["cup_0"])
    assert own0.done
    w.place("cup_0", "counter_2"); m.mark_done(w.world_graph[0])
    assert not own0.done


def test_search_finds_unseen_object_for_takeover():
    """The partner quits; bread_5 is in the bedroom, which nobody has explored: the coordinator explores the rooms the
    robot has not explored (living room, then bedroom) and rearranges bread_5 once it is seen."""
    log, placed, _ = run("ccrp", QUITTING_PARTNER, DILIGENT_ROBOT, V41,
                         robot_sees=("fork_0", "cup_0", "plate_1"), bread_on="bed_5")
    explores = [arg for _, a, arg in log if a == "Explore"]
    assert explores[:1] == ["kitchen_1"] and "bedroom_1" in explores
    assert {"fork_0", "bread_5"} <= placed


def test_stuck_robot_triggers_own_check():
    """The robot places cup_0 and then keeps navigating to a node that does not exist: after stuck_window failures the
    coordinator completes plate_1 itself."""
    stuck = [("Explore", "kitchen_1", None), ("Rearrange", "cup_0, on, table_10, None, None", None)] + \
        [("Navigate", "table_19 in kitchen_1", None)] * 12 + [("Done", None, None)] * 3
    log, placed, _ = run("ccrp", WORKING_PARTNER, stuck, V41)
    assert "plate_1" in placed
    first_plate = next(i for i, (_, a, arg) in enumerate(log) if a == "Rearrange" and arg.startswith("plate_1"))
    assert sum(1 for _, a, _ in log[:first_plate] if a == "Navigate") >= 6


def test_unseen_own_item_is_searched_for():
    log, placed, _ = run("ccrp", WORKING_PARTNER, FORGETFUL_ROBOT, V41, robot_sees=("cup_0",))
    assert "plate_1" in placed


# ---- Team-Plan + RL (mode ccrp_rl): decisions from the value learned on the planner's world model -----------------
def _rl_decisions():
    import glob, json, os
    rows = []
    for f in glob.glob(os.path.join(os.environ["CCRP_LOG_DIR"], "ccrp_decisions_stream.jsonl")):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return [r for r in rows if r.get("mode") == "ccrp_rl"]


def test_rl_mode_takes_over_after_partner_quits():
    n0 = len(_rl_decisions())
    log, placed, msgs = run("ccrp_rl", QUITTING_PARTNER, DILIGENT_ROBOT, V4)
    assert {"fork_0", "bread_5", "cup_0", "plate_1"} <= placed
    new = _rl_decisions()[n0:]
    assert new and all("mixture" in r for r in new)                 # the learned value made the decisions
    assert any(r["action"].startswith("takeover") for r in new)


def test_rl_view_maps_the_planner_state():
    import numpy as np
    from habitat_llm_ext.rl_value import sim_view, to_sim_action
    from habitat_llm_ext.responsibility_model import ModelConfig, PartnrResponsibilityModel
    from ccrp.core import Action
    import partnr_sim as ps
    w = ps.World()
    pl = type("P", (), {})()
    pl.model = PartnrResponsibilityModel(w, 0, ModelConfig(room_component=True))
    pl.iter, pl.min_iter_before, pl.min_gap_steps, pl.interventions, pl.n_messages = 2600, 600, 400, 0, 1
    pl.used_repairs = {"goal"}
    sim, post, maps = sim_view(pl)
    assert sim.sp.n_partner == len(pl.model.partner) and sim.sp.n_own == len(pl.model.own)
    assert abs(post.sum() - 1) < 1e-9 and sim.s.t == 5 and "goal" in sim.s.used and sim.s.msgs == 1
    # prior mass on 'goal misread' is preserved by the mapping
    gb = pl.model.goal_bad_mass(); assert abs(post[sim.hy.goal >= 0].sum() - gb) < 1e-6
    r0 = pl.model.partner[0].index
    assert to_sim_action(Action("takeover", r0, 0.08), maps).target == (0,)
    assert to_sim_action(Action("clarify", "goal", 0.03), maps).kind == "clarify"


def test_rl_v2_value_loads_by_relative_path_and_decides():
    """The value learned across the calibrated worlds (data/value_partnr_v2.pkl, mode ccrp_rl2) runs through the same
    deployed path as v1 and carries the calibrated repair probability its table uses."""
    from habitat_llm_ext.rl_value import rl_decide, _value
    from habitat_llm_ext.responsibility_model import ModelConfig, PartnrResponsibilityModel
    import os
    import partnr_sim as ps
    import habitat_llm_ext.rl_value as rv
    w = ps.World()
    pl = type("P", (), {})()
    pl.model = PartnrResponsibilityModel(w, 0, ModelConfig(room_component=True))
    pl.iter, pl.min_iter_before, pl.min_gap_steps, pl.interventions, pl.n_messages = 1400, 600, 400, 0, 0
    pl.used_repairs = set()
    acts = pl.model.actions(allow_queries=True, used_repairs=set(), own_done_required=False)
    out = rl_decide(pl, acts, "data/value_partnr_v2.pkl")
    assert out is not None and out[0] in acts
    V = _value(os.path.join(os.path.dirname(rv.__file__), "data", "value_partnr_v2.pkl"))
    assert 0.2 <= V.repair <= 0.85 and isinstance(V.world, list) and len(V.world) > 1


def test_always_takeover_mode_takes_over_after_partner_quits():
    """Ablation 'take over everything': once the partner stops, every open partner responsibility is taken over."""
    log, placed, _ = run("always_takeover", QUITTING_PARTNER, DILIGENT_ROBOT, V4)
    assert {"fork_0", "bread_5", "cup_0", "plate_1"} <= placed


def test_partner_blind_mode_runs_on_the_final_planner():
    """Ablation 'partner-blind scoring': decisions come from marginal rates and the episode completes."""
    log, placed, msgs = run("partner_blind", QUITTING_PARTNER, DILIGENT_ROBOT, V4)
    assert {"cup_0", "plate_1"} <= placed


# ---- v5 budget guard: the robot's ReAct loop spends its 50 planning calls without finishing its own part ----------
LOOPING_ROBOT = [("Explore", "kitchen_1", None), ("Rearrange", "cup_0, on, table_10, None, None", None)] + \
    [("Explore", "kitchen_1", None)] * 80          # every Explore succeeds, nothing new gets done


def test_looping_robot_runs_out_of_calls_without_the_guard():
    """The failure seen in most lost PARTNR episodes: the robot stops at 51 calls and the own check never runs."""
    log, placed, _ = run("ccrp", WORKING_PARTNER, LOOPING_ROBOT, V4, steps=30000)
    assert "cup_0" in placed and "plate_1" not in placed
    assert log[-1][1] == "DONE"


def test_budget_guard_finishes_own_work_before_the_calls_run_out():
    log, placed, _ = run("ccrp", WORKING_PARTNER, LOOPING_ROBOT, {**V4, "budget_guard": True}, steps=30000)
    assert {"cup_0", "plate_1"} <= placed
    first_plate = next(i for i, (_, a, arg) in enumerate(log) if a == "Rearrange" and arg.startswith("plate_1"))
    explores_before = sum(1 for _, a, _ in log[:first_plate] if a == "Explore")
    assert 30 <= explores_before <= 40              # at 70% of the 50 calls, not at the start and not too late


def test_budget_guard_is_off_in_never_mode():
    _, placed, _ = run("never", WORKING_PARTNER, LOOPING_ROBOT, {**V4, "budget_guard": True}, steps=30000)
    assert "plate_1" not in placed


def test_budget_guard_runs_with_the_rl_value_table():
    """ccrp_rl2_bg: the learned table decides, and the v5 guard still finishes the looping robot's own work."""
    _, placed, _ = run("ccrp_rl", WORKING_PARTNER, LOOPING_ROBOT,
                       {**V4, "budget_guard": True, "rl_value_path": "data/value_partnr_v2.pkl"}, steps=30000)
    assert {"cup_0", "plate_1"} <= placed


def test_budget_guard_unlocks_takeover_at_half_the_calls():
    """Take over everything waits for half the step budget; with the guard, half the robot's calls is enough. The
    looping robot never finishes its own part, so before the guard the own-work rule keeps the takeover locked."""
    stalled_partner = [(50, ("Explore", "bedroom_1", None), OK)]          # works once, then idles without quitting
    fast_loop = LOOPING_ROBOT[:2] + [("Navigate", "table_10", None)] * 80  # 50-step moves: many calls, little time
    log_v4, placed_v4, _ = run("always_takeover", stalled_partner, fast_loop, V4, steps=3500)
    log_bg, placed_bg, _ = run("always_takeover", stalled_partner, fast_loop, {**V4, "budget_guard": True}, steps=3500)
    assert not ({"fork_0", "bread_5"} & placed_v4)
    assert {"fork_0", "bread_5"} & placed_bg


def test_no_message_variant_sends_nothing_and_still_takes_over():
    """ccrp_nomsg (max_messages=0): no reminder, location message or query reaches the partner; takeovers remain."""
    log, placed, msgs = run("ccrp", QUITTING_PARTNER, DILIGENT_ROBOT, {**V4, "max_messages": 0, "budget_guard": True})
    assert msgs == [] or all("taking care" in m.lower() or "i will" in m.lower() or "i'll" in m.lower() for m in msgs)
    assert {"fork_0", "bread_5"} & placed



# ---- law-structured Team-Plan (ccrp_laws): LLawCo's laws as terms of the decision model ------------------------
def _laws_cfg():
    import os
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "..", "partnr", "conf", "planner", "ccrp_robot_planner.yaml")) as f:
        model = dict(yaml.safe_load(f)["plan_config"]["ccrp"]["model"])
    model.update({"self_model": True, "derail_prob": 0.06})
    return {**V4, "model": model}


def test_law_model_executes_own_work_when_its_loop_stalls():
    """Avoid infinite loops: calls without own progress raise the posterior that the robot's loop is stuck, and the
    model then chooses to execute the robot's own work directly, well before the 51-call stop."""
    import partnr_sim as ps
    log, placed, _ = run("ccrp", WORKING_PARTNER, LOOPING_ROBOT, _laws_cfg(), steps=30000)
    assert {"cup_0", "plate_1"} <= placed
    q = [x for x in ps.LAST_PLANNER.queue_log if "selfexec" in x]
    assert q and 5 <= q[0]["calls"] <= 40


def test_law_model_leaves_a_progressing_robot_alone():
    import partnr_sim as ps
    run("ccrp", WORKING_PARTNER, DILIGENT_ROBOT, _laws_cfg())
    assert not [x for x in ps.LAST_PLANNER.queue_log if "selfexec" in x]


def test_self_loop_posterior_rises_with_stalled_calls_and_falls_with_progress():
    import partnr_sim as ps
    from habitat_llm_ext.responsibility_model import ModelConfig, PartnrResponsibilityModel
    m = PartnrResponsibilityModel(ps.World(), 0, ModelConfig(room_component=True, self_model=True))
    loop = lambda: m.posterior.mass(lambda h: h.get("self") == "bad")
    p0 = loop(); m.update_self(10, 50); p1 = loop()
    assert abs(p0 - 0.39) < 1e-6 and p1 > p0
    m.own[0].done = True; m.update_self(12, 50)
    assert loop() < p1


def test_derailment_lowers_the_value_of_a_message():
    """Use clear and effective communication: with derail_prob > 0 every message is worth less."""
    import partnr_sim as ps
    from ccrp.core import fill_table
    from habitat_llm_ext.responsibility_model import ModelConfig, PartnrResponsibilityModel
    vals = []
    for d in (0.0, 0.06):
        m = PartnrResponsibilityModel(ps.World(), 0, ModelConfig(room_component=True, derail_prob=d))
        acts = m.actions(allow_queries=False, used_repairs=set(), own_done_required=False)
        t = fill_table(acts, m.posterior, m)
        i = next(k for k, a in enumerate(t.actions) if a.kind == "clarify")
        vals.append(float(t.mixture()[i]))
    assert vals[1] < vals[0]


def test_law_model_does_not_execute_unlocated_own_work():
    """Transport independently: direct execution needs a located object. With plate_1 never seen, it is never part of
    a direct execution (the first live run executed unseen work at the first call)."""
    import partnr_sim as ps
    never_explores = [("Rearrange", "cup_0, on, table_10, None, None", None)] + [("Navigate", "table_19 in kitchen_1", None)] * 80
    run("ccrp", WORKING_PARTNER, never_explores, _laws_cfg(), steps=30000, robot_sees=("cup_0",))
    plate = next(r.index for r in ps.LAST_PLANNER.model.own if r.objects and r.objects[0] == "plate_1")
    assert all(plate not in x["selfexec"] for x in ps.LAST_PLANNER.queue_log if "selfexec" in x)


def test_law_model_without_the_takeover_gate_rescues_a_failing_skill_early():
    """v6 (ccrp_lawv6): with the self model, takeovers are valued against the robot's own open work (opportunity
    cost) instead of being locked until that work is done. A partner whose picks keep failing is taken over while the
    looping robot's own part is still open; with the lock, the takeover waits."""
    import partnr_sim as ps
    fail = "Unexpected failure! - Failed to pick!"
    failing_partner = [(50, ("Explore", "bedroom_1", None), OK)] + \
        [(400 + 120 * k, ("Pick", "fork_0" if k % 2 else "bread_5", None), fail) for k in range(12)]
    first_take = {}
    for gate in (True, False):
        cfg = _laws_cfg(); cfg["takeover_only_when_own_done"] = gate
        log, placed, _ = run("ccrp", failing_partner, LOOPING_ROBOT, cfg, steps=3000)
        takes = [x.get("iter") for x in ps.LAST_PLANNER.decision_log if str(x.get("action", "")).startswith("takeover")]
        first_take[gate] = takes[0] if takes else None
    assert first_take[False] is not None
    assert first_take[True] is None or first_take[False] < first_take[True]


def test_law_model_with_search_finds_and_places_unlocated_own_work():
    """v7 (find and transport independently): the robot never explores and never sees plate_1. With search, the
    coordinator explores the unexplored rooms itself, finds plate_1 and places it directly. Without search it stays
    undone."""
    import partnr_sim as ps
    never_explores = [("Rearrange", "cup_0, on, table_10, None, None", None)] + [("Navigate", "table_19 in kitchen_1", None)] * 80
    placed_by = {}
    for search in (False, True):
        cfg = _laws_cfg(); cfg["search_unseen"] = search; cfg["self_first"] = search
        _, placed, _ = run("ccrp", WORKING_PARTNER, never_explores, cfg, steps=30000, robot_sees=("cup_0",))
        placed_by[search] = "plate_1" in placed
        if search:
            assert any("search" in x for x in ps.LAST_PLANNER.queue_log)
    assert placed_by == {False: False, True: True}
