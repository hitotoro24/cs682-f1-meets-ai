"""Minimal CS682 pit-decision experiment. Synthetic tests are not race evidence."""
import argparse
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

FEATURES = ('race_progress', 'position_fraction', 'previous_pits')


def encode(lap, total_laps, previous_position, previous_pits, field_size):
    """Information available at start of lap t; target is pit during lap t."""
    if not (1 <= lap <= total_laps and 1 <= previous_position <= field_size
            and previous_pits >= 0 and field_size >= 1):
        raise ValueError('Invalid decision state')
    return np.array([(lap - 1) / total_laps, previous_position / field_size,
                     previous_pits], dtype=np.float32)


def historical_examples(laps, driver, total_laps, field_size):
    """Use previous-lap position. Current-lap PitInTime is a label, never a feature."""
    rows = laps.loc[laps.Driver == driver].sort_values('LapNumber')
    numbers = rows.LapNumber.to_numpy()
    if len(numbers) == 0 or not np.array_equal(numbers, np.arange(1, len(numbers) + 1)):
        raise ValueError('Require a contiguous observed lap sequence starting at lap 1')
    records, pits, previous = [], 0, None
    for row in rows.itertuples():
        lap = int(row.LapNumber)
        label = int(pd.notna(row.PitInTime))
        if previous is not None and pd.notna(previous.Position):
            x = encode(lap, total_laps, previous.Position, pits, field_size)
            records.append(dict(lap=lap, **dict(zip(FEATURES, x)), pit=label))
        pits += label
        previous = row
    return pd.DataFrame(records, columns=['lap', *FEATURES, 'pit'])


def load_silverstone(cache):
    import fastf1
    Path(cache).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    session = fastf1.get_session(2019, 'British Grand Prix', 'R')
    session.load(telemetry=False, weather=False, messages=False)
    return historical_examples(session.laps, 'VER', 52, len(session.results))


class PitMLP(nn.Module):
    """Input (N, D), output (N,) logits. BCEWithLogitsLoss applies sigmoid internally."""
    def __init__(self, input_dim=len(FEATURES), hidden_dim=8):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def smoke_test():
    """Entirely synthetic, checks feature timing and a tiny training loop."""
    laps = pd.DataFrame({'Driver': ['VER'] * 4, 'LapNumber': [1, 2, 3, 4],
                         'Position': [4, 8, 5, 3],
                         'PitInTime': [pd.NaT, pd.Timedelta(seconds=1), pd.NaT, pd.NaT]})
    data = historical_examples(laps, 'VER', 52, 20)
    assert data.pit.tolist() == [1, 0, 0]
    assert data.previous_pits.tolist() == [0, 1, 1]
    np.testing.assert_allclose(data.iloc[0][list(FEATURES)].to_numpy(dtype=float), encode(2, 52, 4, 0, 20))
    changed = laps.copy()
    changed.loc[1, 'Position'] = 19
    changed.loc[1, 'PitInTime'] = pd.NaT
    np.testing.assert_allclose(historical_examples(changed, 'VER', 52, 20).iloc[0][list(FEATURES)].to_numpy(dtype=float),
                               data.iloc[0][list(FEATURES)].to_numpy(dtype=float))
    torch.manual_seed(7)
    torch.set_num_threads(1)
    x = torch.tensor(data[list(FEATURES)].to_numpy(), dtype=torch.float32)
    y = torch.tensor(data.pit.to_numpy(), dtype=torch.float32)
    model = PitMLP()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    initial = criterion(model(x), y).item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    final = criterion(model(x), y).item()
    assert model(x).shape == (3,) and final < initial
    print('SYNTHETIC ONLY: feature and no-current-lap-leakage checks passed')
    print(data.to_string(index=False))
    print(f'PyTorch toy overfit: BCE {initial:.6f} -> {final:.6f}; shape {tuple(model(x).shape)}')


def run_race(root, alternative=False):
    """Trace real simulator state and optionally force a single VER stop at lap 24.

    No upstream files are edited; temporary config is deleted. No learned policy yet.
    All other drivers use historical strategies. Requires the legacy TUM environment.
    """
    sim = root / 'race-simulation'
    sys.path.insert(0, str(sim))
    from main_racesim import main
    from racesim.src.vse import VSE
    config_dir = sim / 'racesim' / 'input' / 'parameters'
    text = (config_dir / 'pars_Silverstone_2019.ini').read_text()
    head, tail = text.split('"vse_type"', 1)
    config = head + '"vse_type"' + re.sub(r'"(supervised|reinforcement|basestrategy)"', '"realstrategy"', tail)
    original = VSE.decide_pitstop
    trace = []

    def decide(self, **state):
        actions = list(original(self, **state))
        ids = state['driver_initials']
        if len(ids) <= 1 or 'VER' not in ids:
            return actions  # do not interfere with single-driver pre-simulation
        i = ids.index('VER')
        if not state['bool_driving'][i]:
            return actions
        lap = state['cur_lap']
        features = encode(lap, state['tot_no_laps'], state['positions_prevlap'][i],
                          state['no_past_tirechanges'][i], len(ids))
        old = actions[i]
        if alternative:
            if lap == 24:
                if 'A2' not in self.param_dry_compounds or state['cur_compounds'][i] == 'A2':
                    raise ValueError('Toy strategy requires a valid distinct A2 compound')
                actions[i] = 'A2'
            else:
                actions[i] = None
            # Original VSE caches depend on the returned action; maintain consistency.
            self.cache_tireageprogress_corr_prevlap[i] = (0.0 if actions[i] is not None
                else state['tire_ages'][i] / state['tot_no_laps'])
        trace.append(dict(lap=lap, **dict(zip(FEATURES, features)), original_action=old, action=actions[i]))
        return actions

    opts = dict(use_prob_infl=False, create_rand_events=False, use_vse=True, no_sim_runs=1,
                no_workers=1, use_print=False, use_print_result=False, use_plot=False)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', dir=config_dir, delete=False) as f:
        f.write(config)
        name = Path(f.name)
    try:
        with patch.object(VSE, 'decide_pitstop', decide):
            race = main(sim_opts=opts, race_pars_file=name.name, mcs_pars_file='pars_mcs.ini')[0]
    finally:
        name.unlink(missing_ok=True)
    i = next(i for i, d in enumerate(race.drivers_list) if d.initials == 'VER')
    driver = race.drivers_list[i]
    stops = [int(s[0]) for s in driver.strategy_info if s[0] > 0]
    expected = [int(r['lap']) for r in trace if r['action'] is not None]
    assert trace and stops == expected, (stops, expected)
    if alternative:
        assert stops == [24], stops
    last = race.get_last_compl_lap(idx=i)
    return pd.DataFrame(trace), dict(policy='fixed_lap_24' if alternative else 'historical',
        stops=stops, laps=int(last), position=int(race.positions[-1, i]), seconds=float(race.racetimes[last, i]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke', 'data', 'race'])
    parser.add_argument('--out', type=Path, default=Path('local_results'))
    args = parser.parse_args()
    if args.mode == 'smoke':
        smoke_test()
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'data':
        data = load_silverstone(args.out / 'fastf1_cache')
        data.to_csv(args.out / 'historical_states.csv', index=False)
        print(data.head().to_string(index=False))
        print('Pit-labelled laps:', data.loc[data.pit == 1, 'lap'].tolist())
    else:
        root = Path(__file__).resolve().parents[1]
        results = []
        for alternative in (False, True):
            trace, result = run_race(root, alternative)
            trace.to_csv(args.out / (result['policy'] + '_trace.csv'), index=False)
            results.append(result)
        df = pd.DataFrame(results)
        df.to_csv(args.out / 'policy_comparison.csv', index=False)
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
