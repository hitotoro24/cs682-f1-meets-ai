# replay the real 2019 strategies in the simulator and compare with the real result (FastF1)
# install FastF1 dataset: pip install fastf1 or conda install -c conda-forge fastf1

# recommend using conda env or venv
# python experiments/sanity_check.py

import os
import re
import sys

import numpy as np
import pandas as pd
import fastf1

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from main_racesim import main 

RACE = "Silverstone"
YEAR = 2019
PARS_DIR = os.path.join(REPO, "racesim", "input", "parameters")
CACHE_DIR = os.path.join(REPO, "fastf1_cache")


def make_realstrategy_pars() -> str:
    src = os.path.join(PARS_DIR, f"pars_{RACE}_{YEAR}.ini")
    dst_name = f"pars_{RACE}_{YEAR}_realstrat.ini"
    with open(src) as fh:
        txt = fh.read()
    head, tail = txt.split('"vse_type"', 1)
    tail = re.sub(r'"(supervised|reinforcement|basestrategy)"', '"realstrategy"', tail)
    with open(os.path.join(PARS_DIR, dst_name), "w") as fh:
        fh.write(head + '"vse_type"' + tail)
    return dst_name


def simulate(pars_file: str) -> pd.DataFrame:
    sim_opts = {"use_prob_infl": False, "create_rand_events": False, "use_vse": True,
                "no_sim_runs": 1, "no_workers": 1, "use_print": False,
                "use_print_result": False, "use_plot": False}
    race = main(sim_opts=sim_opts, race_pars_file=pars_file, mcs_pars_file="pars_mcs.ini")[0]
    rows = []
    for idx, drv in enumerate(race.drivers_list):
        last = race.get_last_compl_lap(idx=idx)
        rows.append({"driver": drv.initials,
                     "sim_pos": int(race.positions[-1, idx]),
                     "sim_laps": int(last),
                     "sim_time": float(race.racetimes[last, idx])})
    return pd.DataFrame(rows).set_index("driver")


# simulator track name -> Ergast circuitId, only where they differ from track name in lowercase
CIRCUIT_IDS = {"YasMarina": "yas_marina", "Catalunya": "catalunya", "MonteCarlo": "monaco",
               "Montreal": "villeneuve", "Spielberg": "red_bull_ring", "Budapest": "hungaroring",
               "Melbourne": "albert_park", "Sakhir": "bahrain", "Shanghai": "shanghai", "Sochi": "sochi",
               "Spa": "spa", "Monza": "monza", "Suzuka": "suzuka", "Austin": "americas",
               "MexicoCity": "rodriguez", "SaoPaulo": "interlagos", "Singapore": "marina_bay",
               "Baku": "baku", "LeCastellet": "ricard", "Hockenheim": "hockenheimring",
               "KualaLumpur": "sepang", "Silverstone": "silverstone"}


def real_result() -> pd.DataFrame:
    # results come from Ergast/Jolpica, which needs no F1 account (the livetiming API does)
    from fastf1.ergast import Ergast
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    ergast = Ergast()
    circuit = CIRCUIT_IDS.get(RACE, RACE.lower())
    sched = ergast.get_race_schedule(season=YEAR)
    rnd = int(sched.loc[sched["circuitId"] == circuit, "round"].iloc[0])
    res = ergast.get_race_results(season=YEAR, round=rnd).content[0]
    df = pd.DataFrame({"driver": res["driverCode"],
                       "real_pos": res["position"].astype(int),
                       "real_status": res["status"],
                       "real_time": res["totalRaceTimeMillis"] / 1000.0})
    df["real_gap"] = df["real_time"] - df["real_time"].min()
    return df.set_index("driver")


if __name__ == "__main__":
    sim = simulate(make_realstrategy_pars())
    real = real_result()

    cmp = sim.join(real, how="outer")
    finished = cmp["sim_laps"] == cmp["sim_laps"].max()
    winner_sim = cmp.loc[cmp["sim_pos"] == 1, "sim_time"].iloc[0]
    cmp["sim_gap"] = np.where(finished, cmp["sim_time"] - winner_sim, np.nan)
    cmp["pos_diff"] = cmp["sim_pos"] - cmp["real_pos"]
    cmp = cmp.sort_values("real_pos")

    pd.set_option("display.width", 200)
    print(cmp[["real_pos", "sim_pos", "pos_diff", "real_gap", "sim_gap", "real_status"]].round(2))

    both = cmp.dropna(subset=["real_gap", "sim_gap"])
    print(f"\nWinner race time  real {cmp['real_time'].min():.1f}s | sim {winner_sim:.1f}s "
          f"| diff {winner_sim - cmp['real_time'].min():+.1f}s")
    print(f"Mean |position error|: {cmp['pos_diff'].abs().mean():.2f}")
    print(f"Spearman rank corr (all): {cmp['sim_pos'].corr(cmp['real_pos'], method='spearman'):.3f}")
    print(f"Mean |gap error| (finishers on lead lap in both): {(both['sim_gap'] - both['real_gap']).abs().mean():.1f}s")

    out = os.path.join(REPO, "experiments", f"sanity_{RACE}_{YEAR}.csv")
    cmp.to_csv(out)
    print(f"\nSaved {out}")
