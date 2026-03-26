#!/usr/bin/env python3
"""Step 1 — Load ALL raw player KPIs for every goalkeeper and aggregate.

Processes ~32,000 match files to extract every KPI value for each GK,
then computes per-keeper mean aggregates.

Output: KPI_Weights/output/keeper_all_kpis.parquet

Run: python3 KPI_Weights/build_kpi_dataset.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
GK_DATA = PROJECT / "GK_Data"
COMPETITIONS = GK_DATA / "competitions"
OUTPUT = Path(__file__).resolve().parent / "output"
OUTPUT.mkdir(exist_ok=True)

CACHE_PATH = OUTPUT / "keeper_all_kpis.parquet"


def build_match_index():
    """Pre-build a dict mapping match_dir_name -> full path.

    This avoids scanning all competition dirs for every match lookup.
    """
    print("  Building match directory index...")
    index = {}
    for cs_dir in COMPETITIONS.iterdir():
        if not cs_dir.is_dir() or cs_dir.name.startswith("."):
            continue
        for match_dir in cs_dir.iterdir():
            if match_dir.is_dir() and not match_dir.name.startswith("."):
                index[match_dir.name] = match_dir
    print(f"  Indexed {len(index)} match directories")
    return index


def load_kpi_definitions():
    """Load KPI ID -> name mapping."""
    with open(GK_DATA / "player_kpi_definitions.json") as f:
        raw = json.load(f).get("data", [])
    return {k["id"]: k["name"] for k in raw}


def load_keeper_kpis(player_id, match_dirs_str, match_index):
    """Load raw player_kpis for a GK from all origin matches."""
    all_kpis = []
    if not isinstance(match_dirs_str, str) or not match_dirs_str:
        return all_kpis

    for match_dir_name in match_dirs_str.split("|"):
        match_dir_name = match_dir_name.strip()
        if not match_dir_name:
            continue

        match_path = match_index.get(match_dir_name)
        if match_path is None:
            continue

        pkpi_path = match_path / "player_kpis.json"
        if not pkpi_path.exists():
            continue

        with open(pkpi_path) as f:
            data = json.load(f).get("data", {})

        for side in ["squadHome", "squadAway"]:
            for player in data.get(side, {}).get("players", []):
                if player["id"] == player_id:
                    if (player.get("position") == "GOALKEEPER"
                            and player.get("matchShare", 0) >= 0.5):
                        kpis = {
                            k["kpiId"]: k["value"]
                            for k in player.get("kpis", [])
                        }
                        all_kpis.append(kpis)
                    break

    return all_kpis


def build_dataset():
    """Load all KPIs for all keepers and aggregate."""
    if CACHE_PATH.exists():
        print(f"Cache found: {CACHE_PATH}")
        df = pd.read_parquet(CACHE_PATH)
        mean_cols = [c for c in df.columns if c.startswith("mean_")]
        print(f"Loaded {len(df)} keepers x {len(mean_cols)} KPI features")
        return df

    dataset = pd.read_csv(GK_DATA / "gk_dataset_final.csv")
    kpi_defs = load_kpi_definitions()
    match_index = build_match_index()
    total = len(dataset)

    print(f"Processing {total} goalkeepers...")
    t0 = time.time()

    all_rows = []
    for idx, row in dataset.iterrows():
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / rate if rate > 0 else 0
            print(f"  {idx + 1}/{total}  ({rate:.1f}/s, ETA {eta:.0f}s)")

        matches = load_keeper_kpis(row["playerId"], row["origin_match_dirs"], match_index)
        if not matches:
            continue

        # Collect all KPI IDs
        kpi_ids = set()
        for m in matches:
            kpi_ids.update(m.keys())

        features = {
            "playerId": row["playerId"],
            "name": row["name"],
            "status": row["status"],
            "direction": row["direction"],
            "age": row["age"],
            "origin_team": row.get("origin_team", ""),
            "origin_comp": row.get("origin_comp", ""),
            "origin_median": row["origin_median"],
            "origin_matches": row["origin_matches"],
            "n_matches_loaded": len(matches),
        }

        for kid in kpi_ids:
            values = [m[kid] for m in matches if kid in m]
            if values:
                kname = kpi_defs.get(kid, f"KPI_{kid}")
                features[f"mean_{kname}"] = np.mean(values)

        all_rows.append(features)

    df = pd.DataFrame(all_rows)
    elapsed = time.time() - t0
    mean_cols = [c for c in df.columns if c.startswith("mean_")]
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Keepers: {len(df)}")
    print(f"KPI features: {len(mean_cols)}")
    print(f"Status: {df['status'].value_counts().to_dict()}")

    df.to_parquet(CACHE_PATH, index=False)
    print(f"Saved: {CACHE_PATH}")
    return df


if __name__ == "__main__":
    build_dataset()
