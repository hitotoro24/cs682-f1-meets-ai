# Silverstone 2019 local verification

## Environment

The Apple Silicon Mac exposed Python 3.9.6 at `/usr/bin/python3` and Python 3.11.9 at `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11`. A pre-existing sibling `../.sim-venv` used Python 3.11.18 and could run the simulator but had no pip. The commands below use a separate repository-local `.venv` created from 3.11.9. Homebrew was unnecessary. The install used the simulator's existing dependency pins, including NumPy 1.26.4 and TensorFlow 2.16.2. Local verification used FastF1 3.9.2, PyTorch 2.15.0, pytest 9.2.0, nbconvert 7.17.2, and ipykernel 7.4.0.

Run from the repository root. Use an available Python 3.11 executable in place of the macOS Framework path on other systems.

```sh
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r race-simulation/requirements.txt torch==2.15.0 pytest==9.2.0 nbconvert==7.17.2 ipykernel==7.4.0
.venv/bin/python -m pip check
mkdir -p local_results
.venv/bin/python race-simulation/main_racesim.py > local_results/tum_baseline.log
.venv/bin/python -m pytest -q
.venv/bin/python -m f1_ai.pit_pipeline smoke
.venv/bin/python -m f1_ai.pit_pipeline race
.venv/bin/jupyter-nbconvert --to notebook --execute --inplace 02_silverstone_simulator_walkthrough.ipynb
```

## Observed simulator results

The original entry point in the team's existing Python 3.11 port ran one valid 52-lap race and reported zero invalid runs. With the bundled strategies and fixed event inputs, Hamilton won in 4878.232 s. Verstappen was second in 4890.732 s, 12.500 s behind. Räikkönen was third in 4891.957 s. Three cars retired. These are replay outputs from TUM's timing model.

[The executed notebook](../02_silverstone_simulator_walkthrough.ipynb) explains the lap update order, array dimensions, and model assumptions. It moves only VER's second stop, on A2, from lap 20 to 24. His lap-13 A3 stop and the other drivers' schedules stay fixed. All 20 drivers' lap times and positions match through lap 19. VER completed 52 laps in each run and finished P8 in 4920.468 s with the later stop, 29.736 s slower. The notebook writes the per-lap schedule and observed states to `local_results/ver_stop_timing_trace.csv`. This measures sensitivity to the stop timing under the bundled inputs.

The existing team VSE pipeline also ran successfully. Its historical replay logged VER stops on laps 13 and 20, P2, and 4890.731840 s. Its alternative scheduled just one A2 stop on lap 24 and returned P8 in 4920.466865 s. This is a separate policy comparison because it also removes the lap-13 stop. The pipeline wrote `historical_trace.csv`, `fixed_lap_24_trace.csv`, and `policy_comparison.csv` to `local_results/`.

## Verstappen result summary

| Run | Executed stops | Laps | Finish | Race time (s) |
| --- | --- | ---: | ---: | ---: |
| Historical race via Jolpica | 13 and 20 | 52 | P5 | 4907.910 |
| TUM scheduled replay | 13 (A3), 20 (A2) | 52 | P2 | 4890.732 |
| Notebook timing intervention | 13 (A3), 24 (A2) | 52 | P8 | 4920.468 |
| Team VSE historical replay | 13 (A3), 20 (A2) | 52 | P2 | 4890.731840 |
| Team VSE one-stop alternative | 24 (A2) | 52 | P8 | 4920.466865 |

The historical pit-stop records report durations of 27.827 s on lap 13 and 27.950 s on lap 20. The [checked-in controlled-run trace](experiments/ver_stop_timing_trace.csv) is the CSV produced by the executed notebook. Its lap-time columns are simulation outcomes.

Review the executed `02` notebook and controlled-run trace with these notes. The existing `01_silverstone_proof_of_concept.ipynb` and `f1_ai/pit_pipeline.py` provide context for the separate VSE run. The TUM reference entry points are `race-simulation/main_racesim.py` and `race-simulation/racesim/src/race.py`. All are already in the repository.

## Historical records and pipeline tests

The public Jolpica endpoints through FastF1 returned all 20 classification rows, all 52 VER lap-time rows, and both VER pit stops on laps 13 and 20. The actual classification puts HAM P1 in 4868.452 s and VER P5 in 4907.910 s. The replay is 9.780 s slower for HAM and 17.178 s faster for VER than the recorded times. The downloaded records are in `local_results/jolpica_classification.csv`, `jolpica_ver_laps.csv`, and `jolpica_ver_stops.csv`. Retrieve them again with the following commands from the repository root.

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
from fastf1 import Cache
from fastf1.ergast import Ergast

out = Path("local_results")
cache = out / "fastf1_cache"
cache.mkdir(parents=True, exist_ok=True)
Cache.enable_cache(str(cache))
ergast = Ergast()
response = ergast.get_race_results(season=2019, circuit="silverstone")
round_number = int(response.description.iloc[0]["round"])
classification = response.content[0]
selection = dict(season=2019, round=round_number, driver="max_verstappen", limit=100)
laps = ergast.get_lap_times(**selection).content[0]
stops = ergast.get_pit_stops(**selection).content[0]
assert len(classification) == 20 and len(laps) == 52
assert stops["lap"].tolist() == [13, 20]
classification.to_csv(out / "jolpica_classification.csv", index=False)
laps.to_csv(out / "jolpica_ver_laps.csv", index=False)
stops.to_csv(out / "jolpica_ver_stops.csv", index=False)
PY
```

The existing `.venv/bin/python -m f1_ai.pit_pipeline data` exited 1. FastF1 reported unavailable F1 account credentials and raised `DataNotLoadedError` when the pipeline accessed `session.laps`. The public lap-time and pit-stop tables above verify historical records. The pipeline's richer session-lap extraction remains unverified, and no historical feature CSV was generated.

`pip check` found no broken requirements. All three existing tests passed in 16.87 s. The synthetic smoke test passed its decision-time feature checks and reduced toy BCE from 0.714567 to 0.000583 for three outputs. Those synthetic results test the pipeline wiring. The actual simulations and historical retrieval are reported above.

## Attribution

TUM's vendored source and parameter files have no changes from this work. The walkthrough, controlled stop-timing comparison, and this verification record are team work. The public CS 682 notes provide evaluation background. The simulator details follow the checked-in source and the original publications. [CS 682 course notes](https://compsci682-fa21.github.io/notes/) · [TUM implementation](https://github.com/TUMFTM/race-simulation) · [Race simulation paper](https://doi.org/10.1109/ITSC.2018.8570012) · [VSE paper](https://doi.org/10.3390/app10217805)
