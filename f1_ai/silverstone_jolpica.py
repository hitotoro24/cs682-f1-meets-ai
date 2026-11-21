"""Preprocess 2019 Silverstone lap positions and pit records via FastF1's Jolpica client."""

from pathlib import Path

import pandas as pd
from fastf1 import Cache
from fastf1.ergast import Ergast

from f1_ai.pit_pipeline import historical_examples


def read_pages(endpoint, **filters):
    """Collect all records, including lap groups split across API pages."""
    pages = []
    offset = 0
    while True:
        response = endpoint(limit=100, offset=offset, **filters)
        rows = sum(len(table) for table in response.content)
        if not rows:
            raise ValueError(f"{endpoint.__name__} returned an empty page at {offset}")
        pages.extend(response.content)
        offset += rows
        if offset >= response.total_results:
            return pd.concat(pages, ignore_index=True)


def preprocess_silverstone(cache):
    """Return one feature row per observed driver-lap with a prior lap."""
    cache.mkdir(parents=True, exist_ok=True)
    Cache.enable_cache(str(cache))
    source = Ergast()
    filters = dict(season=2019, round=10)
    results = read_pages(source.get_race_results, **filters).set_index("driverId")
    laps = read_pages(source.get_lap_times, **filters).rename(
        columns={"number": "LapNumber", "position": "Position"}
    )
    stops = read_pages(source.get_pit_stops, **filters)
    keys = ["driverId", "LapNumber"]
    stop_keys = stops.rename(columns={"lap": "LapNumber"})[keys].drop_duplicates()
    laps = laps.merge(stop_keys, on=keys, how="outer", validate="one_to_one", indicator=True)
    unmatched_stops = laps.loc[laps["_merge"].eq("right_only"), keys].copy()
    laps = laps.loc[laps["_merge"].ne("right_only")].copy()
    counts = laps.groupby("driverId").size()
    if set(counts.index) != set(results.index) or not counts.eq(results["laps"]).all():
        raise ValueError("Lap-record counts do not match the race classification")
    laps["LapNumber"] = laps["LapNumber"].astype(int)
    laps["Position"] = laps["Position"].astype(int)
    laps["Driver"] = laps["driverId"].map(results["driverCode"])
    # The shared encoder uses only presence in this field. No timestamp is inferred.
    laps["PitInTime"] = laps["_merge"].eq("both").map({True: True, False: None})
    features = pd.concat(
        [
            historical_examples(laps, driver, int(results["laps"].max()), len(results))
            .assign(driver=driver)
            for driver in sorted(laps["Driver"].unique())
        ],
        ignore_index=True,
    )
    return features[["driver", *features.columns.drop("driver")]], stops, unmatched_stops


def main():
    out = Path("local_results")
    features, stops, unmatched_stops = preprocess_silverstone(out / "fastf1_cache")
    path = out / "silverstone_2019_jolpica_features.csv"
    features.to_csv(path, index=False)
    ver = features.loc[features["driver"].eq("VER")]
    print(f"{path}: {len(features)} examples across {features['driver'].nunique()} drivers")
    print(f"Pit-positive examples: {features['pit'].sum()} of {len(stops)} source stop records")
    print(f"VER: {len(ver)} examples, pit laps {ver.loc[ver['pit'].eq(1), 'lap'].tolist()}")
    print(f"Source stops on lap 1 (no feature row): {stops['lap'].eq(1).sum()}")
    if not unmatched_stops.empty:
        print("Stop records without an observed lap:")
        print(unmatched_stops.to_string(index=False))


if __name__ == "__main__":
    main()
