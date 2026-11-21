# Live TUM simulator tests for the callable policy interface, about 1.5s per race
import pytest

from f1_ai.sim_policy import (IllegalAction, fixed_laps_policy, historical_policy,
                              other_strategies_unchanged, run_policy)


@pytest.fixture(scope="module")
def historical():
    return run_policy(historical_policy)


def test_historical_policy_reproduces_replay(historical):
    o = historical.outcome
    assert o["stops"] == [13, 20] and o["compounds"] == ["A3", "A3", "A2"]
    assert o["finished"] and o["laps"] == 52 and o["position"] == 2
    assert o["seconds"] == pytest.approx(4890.732, abs=1e-3)
    assert o["actions_executed"] and o["used_two_compounds"]
    assert len(historical.trace) == 52  # one decision per lap, VER never retires


def test_repeatable(historical):
    again = run_policy(historical_policy)
    assert again.outcome == historical.outcome
    assert again.field.equals(historical.field)


def test_intervention_executes_and_isolates_other_drivers(historical):
    r = run_policy(fixed_laps_policy({13: "A3", 24: "A2"}))
    assert r.outcome["stops"] == [13, 24] and r.outcome["actions_executed"]
    assert r.outcome["position"] == 8
    assert r.outcome["seconds"] == pytest.approx(4920.468, abs=1e-3)  # matches PR #6
    assert other_strategies_unchanged(historical, r)
    # identical decisions before the first changed lap
    before = historical.trace["lap"] < 20
    assert r.trace[before].equals(historical.trace[before])


def test_decision_state_contents(historical):
    t = historical.trace.set_index("lap")
    assert t.loc[1, "compound"] == "A3" and t.loc[1, "stops_so_far"] == 0
    assert t.loc[13, "action"] == "A3" and t.loc[14, "stops_so_far"] == 1
    assert t.loc[21, "used_two_compounds"]
    assert t.loc[21, "fcy"] == "SC"  # safety car laps ~20-23 in the parameter file
    assert set(t["available_compounds"].iloc[0]) >= {"A2", "A3", "A4"}


def test_illegal_compound_rejected():
    with pytest.raises(IllegalAction):
        run_policy(fixed_laps_policy({10: "A9"}))


def test_final_lap_pit_rejected():
    with pytest.raises(IllegalAction):
        run_policy(fixed_laps_policy({52: "A2"}))
