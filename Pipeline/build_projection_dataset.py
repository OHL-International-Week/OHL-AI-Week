#!/usr/bin/env python3
"""Build post-transfer KPI dataset for keepers who moved up.

For PLAYS and BENCH keepers, loads their current_match_dirs (higher-league)
KPIs, so we can train: lower-league KPIs → higher-league KPIs.

Output: Pipeline/output/keeper_current_kpis.parquet
"""

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GK_DATA = PROJECT_ROOT / "GK_Data"
COMPETITIONS = GK_DATA / "competitions"
OUTPUT = Path(__file__).resolve().parent / "output"
CACHE = OUTPUT / "keeper_current_kpis.parquet"


def load_kpi_definitions():
    with open(GK_DATA / "player_kpi_definitions.json") as f:
        return {d["id"]: d["name"] for d in json.load(f).get("data", [])}


def load_keeper_kpis(player_id, match_dirs_str):
    """Load player_kpis for a goalkeeper from match dirs."""
    all_kpis = []
    if not isinstance(match_dirs_str, str) or not match_dirs_str:
        return all_kpis

    for match_dir in match_dirs_str.split("|"):
        match_dir = match_dir.strip()
        if not match_dir:
            continue
        for cs_dir in COMPETITIONS.iterdir():
            if not cs_dir.is_dir() or cs_dir.name.startswith("."):
                continue
            pkpi_path = cs_dir / match_dir / "player_kpis.json"
            if not pkpi_path.exists():
                continue
            with open(pkpi_path) as f:
                data = json.load(f).get("data", {})
            found = False
            for side in ["squadHome", "squadAway"]:
                for player in data.get(side, {}).get("players", []):
                    if player["id"] == player_id:
                        if player.get("position") == "GOALKEEPER" and player.get("matchShare", 0) >= 0.5:
                            kpis = {k["kpiId"]: k["value"] for k in player.get("kpis", [])}
                            all_kpis.append(kpis)
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return all_kpis


def build_dataset():
    if CACHE.exists():
        print(f"Loading cached: {CACHE}")
        return pd.read_parquet(CACHE)

    dataset = pd.read_csv(GK_DATA / "gk_dataset_final.csv")
    kpi_defs = load_kpi_definitions()

    # Only keepers who moved up (have current_match_dirs)
    moved = dataset[
        dataset["current_match_dirs"].notna() &
        (dataset["current_match_dirs"] != "") &
        dataset["status"].isin(["PLAYS", "BENCH"])
    ].copy()

    print(f"Building current KPI dataset for {len(moved)} keepers...")
    rows = []
    for idx, row in moved.iterrows():
        matches = load_keeper_kpis(row["playerId"], row["current_match_dirs"])
        if not matches:
            continue

        kpi_ids = set()
        for m in matches:
            kpi_ids.update(k for k in m.keys() if isinstance(k, int))

        features = {
            "playerId": row["playerId"],
            "name": row["name"],
            "status": row["status"],
            "current_comp": row["current_comp"],
            "current_median": row["current_median"],
            "n_current_matches": len(matches),
        }
        for kpi_id in kpi_ids:
            values = [m[kpi_id] for m in matches if kpi_id in m]
            if values:
                features[f"cur_{kpi_defs.get(kpi_id, f'KPI_{kpi_id}')}"] = np.mean(values)

        rows.append(features)
        if len(rows) % 20 == 0:
            print(f"  {len(rows)} keepers loaded...")

    df = pd.DataFrame(rows)
    print(f"Dataset: {len(df)} keepers × {len(df.columns)} columns")
    df.to_parquet(CACHE, index=False)
    return df


if __name__ == "__main__":
    df = build_dataset()
    print(f"\nFinal: {df.shape}")
    print(f"Status: {df['status'].value_counts().to_dict()}")
    cur_cols = [c for c in df.columns if c.startswith("cur_")]
    print(f"Current KPI features: {len(cur_cols)}")
