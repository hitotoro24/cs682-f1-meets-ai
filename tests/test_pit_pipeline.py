"""Isolated contract tests: no real FastF1 or TUM race execution."""
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from f1_ai.pit_pipeline import historical_examples, run_race, smoke_test


def test_synthetic_training_and_timing():
    smoke_test()


def test_reject_missing_lap():
    laps = pd.DataFrame({'Driver': ['VER', 'VER'], 'LapNumber': [1, 3],
                         'Position': [1, 1], 'PitInTime': [pd.NaT, pd.NaT]})
    try:
        historical_examples(laps, 'VER', 52, 20)
    except ValueError:
        return
    raise AssertionError('Missing lap silently accepted')


def test_mock_vse_intervention(tmp_path):
    """Stand-in for documented VSE/race interface, not a simulator validation."""
    config_dir = tmp_path / 'race-simulation/racesim/input/parameters'
    config_dir.mkdir(parents=True)
    source = config_dir / 'pars_Silverstone_2019.ini'
    source.write_text('"vse_type": {"VER": "supervised", "HAM": "supervised"}')
    original_config = source.read_text()
    other_stops = []

    class VSE:
        param_dry_compounds = ['A2', 'A3', 'A4']
        def decide_pitstop(self, **state):
            lap = state['cur_lap']
            result = [{13: 'A3', 20: 'A2'}.get(lap), {20: 'A2'}.get(lap)]
            self.cache_tireageprogress_corr_prevlap = [0.0 if a else 0.2 for a in result]
            return result

    def main(sim_opts, race_pars_file, mcs_pars_file):
        assert 'realstrategy' in (config_dir / race_pars_file).read_text()
        vse = VSE()
        drivers = [SimpleNamespace(initials=d, strategy_info=[[0, 'A3', 0, 0.]])
                   for d in ('VER', 'HAM')]
        for lap in range(1, 53):
            state = dict(driver_initials=['VER', 'HAM'], bool_driving=[True, True],
                         cur_lap=lap, tot_no_laps=52, positions_prevlap=[1, 2],
                         no_past_tirechanges=[len(d.strategy_info)-1 for d in drivers],
                         cur_compounds=[d.strategy_info[-1][1] for d in drivers],
                         tire_ages=[lap, lap])
            actions = vse.decide_pitstop(**state)
            for d, action in zip(drivers, actions):
                if action is not None:
                    d.strategy_info.append([lap, action, 0, 0.])
            if actions[0] is not None:
                assert vse.cache_tireageprogress_corr_prevlap[0] == 0.
        other_stops.append([s[0] for s in drivers[1].strategy_info[1:]])
        return [SimpleNamespace(drivers_list=drivers, positions=np.tile([1, 2], (53, 1)),
                                racetimes=np.zeros((53, 2)), get_last_compl_lap=lambda idx: 52)]

    module_main = ModuleType('main_racesim')
    module_main.main = main
    module_vse = ModuleType('racesim.src.vse')
    module_vse.VSE = VSE
    original_method = VSE.decide_pitstop
    with patch.dict(sys.modules, {'main_racesim': module_main, 'racesim': ModuleType('racesim'),
                                  'racesim.src': ModuleType('racesim.src'), 'racesim.src.vse': module_vse}):
        baseline_trace, baseline = run_race(tmp_path)
        alternative_trace, alternative = run_race(tmp_path, True)
    assert baseline['stops'] == [13, 20]
    assert alternative['stops'] == [24]
    assert other_stops == [[20], [20]]
    assert alternative_trace.loc[alternative_trace.lap == 25, 'previous_pits'].item() == 1
    assert baseline_trace.loc[baseline_trace.lap == 14, 'previous_pits'].item() == 1
    assert VSE.decide_pitstop is original_method
    assert source.read_text() == original_config
    assert list(config_dir.iterdir()) == [source]
