""" Callable pit-policy interface for the TUM simulator.

A policy maps DecisionState -> None (stay out) or a compound like "A2" (pit this lap).
Other drivers follow their historical schedules.
"""
from __future__ import annotations

import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "race-simulation"
CONFIG_DIR = SIM / "racesim" / "input" / "parameters"


@dataclass(frozen=True)
class DecisionState:
    lap: int                          # current lap
    total_laps: int
    compound: str                     # TUM compound, e.g. "A3"
    tyre_age: float                   # degradation age
    stops_so_far: int
    position_prev_lap: int
    gap_ahead_s: float                # nan if leading
    gap_behind_s: float               # nan if last or car behind retired
    fcy: Optional[str]                # None, "SC" or "VSC" this lap
    used_two_compounds: bool
    available_compounds: tuple        # dry compounds set up for this race
    historical_action: Optional[str]  # what the real schedule does this lap

Policy = Callable[[DecisionState], Optional[str]]


@dataclass
class SimResult:
    outcome: dict
    trace: pd.DataFrame
    field: pd.DataFrame


class IllegalAction(ValueError):
    pass


def historical_policy(state: DecisionState) -> Optional[str]:
    return state.historical_action


def fixed_laps_policy(stops: dict) -> Policy:
    stops = {int(k): v for k, v in stops.items()}

    def policy(state: DecisionState) -> Optional[str]:
        return stops.get(state.lap)
    return policy


def _gaps(i, positions_prevlap, racetimes_prevlap, bool_driving_prevlap):
    pos = positions_prevlap[i]

    def time_at(p):
        idx = np.flatnonzero(positions_prevlap == p)
        if idx.size == 0 or not bool_driving_prevlap[idx[0]]:
            return np.nan
        return float(racetimes_prevlap[idx[0]])

    me = float(racetimes_prevlap[i])
    ahead = me - time_at(pos - 1) if pos > 1 else np.nan
    behind = time_at(pos + 1) - me if pos < len(positions_prevlap) else np.nan
    return ahead, behind


def _realstrategy_config(race: str) -> str:
    text = (CONFIG_DIR / f"pars_{race}.ini").read_text()
    head, tail = text.split('"vse_type"', 1)
    return head + '"vse_type"' + re.sub(r'"(supervised|reinforcement|basestrategy)"', '"realstrategy"', tail)


def run_policy(policy: Policy, driver: str = "VER", race: str = "Silverstone_2019") -> SimResult:
    if str(SIM) not in sys.path:
        sys.path.insert(0, str(SIM))
    from main_racesim import main
    from racesim.src.vse import VSE

    original = VSE.decide_pitstop
    trace = []

    def decide(self, **s):
        actions = list(original(self, **s))
        ids = s["driver_initials"]
        if len(ids) <= 1 or driver not in ids:
            return actions  # single-driver pre-simulation: do not interfere
        i = ids.index(driver)
        if not s["bool_driving"][i]:
            return actions
        lap, total = s["cur_lap"], s["tot_no_laps"]
        ahead, behind = _gaps(i, s["positions_prevlap"], s["racetimes_prevlap"], s["bool_driving_prevlap"])
        state = DecisionState(
            lap=lap, total_laps=total, compound=s["cur_compounds"][i],
            tyre_age=float(s["tire_ages"][i]), stops_so_far=int(s["no_past_tirechanges"][i]),
            position_prev_lap=int(s["positions_prevlap"][i]), gap_ahead_s=ahead, gap_behind_s=behind,
            fcy=s["fcy_types"][i], used_two_compounds=bool(s["used_2compounds"][i]),
            available_compounds=tuple(self.param_dry_compounds), historical_action=actions[i])
        action = policy(state)
        if action is not None:
            if action not in self.param_dry_compounds:
                raise IllegalAction(f"lap {lap}: compound {action!r} not in {self.param_dry_compounds}")
            if lap >= total:
                raise IllegalAction(f"lap {lap}: cannot pit on the final lap")
        actions[i] = action
        # TUM's VSE cache depends on the returned action, keep consistent.
        self.cache_tireageprogress_corr_prevlap[i] = 0.0 if action is not None else state.tyre_age / total
        trace.append({**asdict(state), "action": action})
        return actions

    opts = dict(use_prob_infl=False, create_rand_events=False, use_vse=True, no_sim_runs=1,
                no_workers=1, use_print=False, use_print_result=False, use_plot=False)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", dir=CONFIG_DIR, delete=False) as f:
        f.write(_realstrategy_config(race))
        tmp = Path(f.name)
    try:
        with patch.object(VSE, "decide_pitstop", decide):
            sim = main(sim_opts=opts, race_pars_file=tmp.name, mcs_pars_file="pars_mcs.ini")[0]
    finally:
        tmp.unlink(missing_ok=True)

    rows = []
    for k, d in enumerate(sim.drivers_list):
        last = sim.get_last_compl_lap(idx=k)
        rows.append(dict(driver=d.initials,
                         stops=[int(x[0]) for x in d.strategy_info if x[0] > 0],
                         compounds=[x[1] for x in d.strategy_info],
                         laps=int(last), position=int(sim.positions[-1, k]),
                         seconds=float(sim.racetimes[last, k])))
    field = pd.DataFrame(rows).set_index("driver")
    trace = pd.DataFrame(trace)

    me = field.loc[driver]
    executed = me["stops"] == [int(r["lap"]) for _, r in trace.iterrows() if r["action"] is not None]
    outcome = dict(driver=driver, race=race, stops=me["stops"], compounds=me["compounds"],
                   laps=me["laps"], finished=me["laps"] == sim.race_pars["tot_no_laps"],
                   position=me["position"], seconds=me["seconds"],
                   used_two_compounds=len(set(me["compounds"])) > 1,  # F1 dry-race rule
                   actions_executed=bool(executed))
    return SimResult(outcome=outcome, trace=trace, field=field)


def other_strategies_unchanged(a: SimResult, b: SimResult, driver: str = "VER") -> bool:
    cols = ["stops", "compounds"]
    return a.field.drop(index=driver)[cols].equals(b.field.drop(index=driver)[cols])


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compare pit policies for one driver in the TUM simulator")
    p.add_argument("--driver", default="VER")
    p.add_argument("--out", type=Path, default=ROOT / "local_results" / "sim_policy")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    policies = {"historical": historical_policy,
                "fixed_lap_24": fixed_laps_policy({24: "A2"}),
                "delayed_second_stop_13_24": fixed_laps_policy({13: "A3", 24: "A2"})}
    results = {name: run_policy(fn, driver=args.driver) for name, fn in policies.items()}
    base = results["historical"]
    rows = []
    for name, r in results.items():
        r.trace.to_csv(args.out / f"{name}_trace.csv", index=False)
        rows.append({"policy": name, **r.outcome,
                     "others_unchanged": other_strategies_unchanged(base, r, args.driver)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "policy_comparison.csv", index=False)
    print(df[["policy", "stops", "compounds", "laps", "position", "seconds",
              "used_two_compounds", "actions_executed", "others_unchanged"]].to_string(index=False))
