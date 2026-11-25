"""Compare Verstappen's historical and simulated decision inputs on laps 12–21."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from f1_ai.pit_pipeline import FEATURES


def audit(historical_csv: Path, replay_csv: Path) -> pd.DataFrame:
    """Return the aligned rows after checking progress, pit laps and stop counts."""
    laps = pd.Index(range(12, 22), name="lap")
    history = pd.read_csv(historical_csv)
    real = history.loc[history["driver"].eq("VER")].set_index(
        "lap", verify_integrity=True
    ).loc[laps, [*FEATURES, "pit"]]
    replay = pd.read_csv(replay_csv).set_index(
        "lap", verify_integrity=True
    ).loc[laps, [*FEATURES, "action"]]

    expected_pits = laps.isin([13, 20]).astype(int)
    expected_prior = expected_pits.cumsum() - expected_pits
    comparison = pd.DataFrame(index=laps)
    for source, rows, pit in (
        ("real", real, real["pit"]),
        ("sim", replay, replay["action"].notna().astype(int)),
    ):
        for field, expected in (
            ("race_progress", (laps - 1) / 52),
            ("pit", expected_pits),
            ("previous_pits", expected_prior),
        ):
            observed = pit if field == "pit" else rows[field]
            matches = np.isclose(observed, expected, rtol=0, atol=1e-7)
            if not matches.all():
                raise ValueError(f"{source}: unexpected {field} on laps {laps[~matches].tolist()}")

        ranks = rows["position_fraction"].to_numpy() * 20
        valid_ranks = (
            np.isclose(ranks, ranks.round(), rtol=0, atol=1e-5)
            & (ranks.round() >= 1)
            & (ranks.round() <= 20)
        )
        if not valid_ranks.all():
            raise ValueError(f"{source}: invalid position on laps {laps[~valid_ranks].tolist()}")
        comparison[f"{source}_position"] = ranks.round().astype(int)
        comparison[f"{source}_prior_stops"] = rows["previous_pits"].astype(int)
        comparison[f"{source}_pit"] = pit.astype(int)

    comparison.insert(0, "race_progress", real["race_progress"])
    comparison["sim_compound"] = replay["action"].fillna("")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical_csv", type=Path)
    parser.add_argument("replay_csv", type=Path)
    args = parser.parse_args()
    print(audit(args.historical_csv, args.replay_csv).to_string())


if __name__ == "__main__":
    main()
