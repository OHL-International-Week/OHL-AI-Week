#!/usr/bin/env python3
"""Build comprehensive KPI dataset from ALL player_kpis.json match files.

Extracts every available KPI for every goalkeeper across all origin matches,
producing a wide DataFrame with ~300+ KPI features per keeper.

This replaces the score-level aggregation (30 features) with the full
granular KPI-level data (300+ features), enabling thorough feature selection.

Output: pipeline/output/keeper_all_kpis.parquet  (fast reload)
        pipeline/output/keeper_all_kpis.csv      (human-readable)
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

GK_DATA = PROJECT_ROOT / "GK_Data"
COMPETITIONS = GK_DATA / "competitions"
OUTPUT = Path(__file__).resolve().parent / "output"
OUTPUT.mkdir(exist_ok=True)

CACHE_PARQUET = OUTPUT / "keeper_all_kpis.parquet"
CACHE_CSV = OUTPUT / "keeper_all_kpis.csv"


def load_kpi_definitions():
    """Load KPI ID -> name mapping."""
    with open(GK_DATA / "player_kpi_definitions.json") as f:
        return {d["id"]: d["name"] for d in json.load(f).get("data", [])}


def load_keeper_kpis(player_id, match_dirs_str):
    """Load player_kpis for a goalkeeper from all origin matches.

    Returns list of dicts, one per match, mapping kpiId -> value.
    """
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
                        if (player.get("position") == "GOALKEEPER"
                                and player.get("matchShare", 0) >= 0.5):
                            kpis = {
                                k["kpiId"]: k["value"]
                                for k in player.get("kpis", [])
                            }
                            kpis["_matchShare"] = player.get("matchShare", 0)
                            kpis["_playDuration"] = player.get("playDuration", 0)
                            all_kpis.append(kpis)
                        found = True
                        break
                if found:
                    break
            if found:
                break  # found the match, move to next

    return all_kpis


def build_dataset():
    """Build the full KPI dataset for all 693 goalkeepers."""
    if CACHE_PARQUET.exists():
        print(f"Loading cached dataset from {CACHE_PARQUET}")
        return pd.read_parquet(CACHE_PARQUET)

    dataset = pd.read_csv(GK_DATA / "gk_dataset_final.csv")
    kpi_defs = load_kpi_definitions()

    print(f"Building full KPI dataset for {len(dataset)} goalkeepers...")
    print(f"KPI definitions available: {len(kpi_defs)}")

    all_keeper_features = []
    total = len(dataset)

    for idx, row in dataset.iterrows():
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processing {idx + 1}/{total}...")

        matches = load_keeper_kpis(row["playerId"], row["origin_match_dirs"])
        if not matches:
            continue

        # Collect all KPI IDs seen across matches
        kpi_ids = set()
        for m in matches:
            kpi_ids.update(k for k in m.keys() if isinstance(k, int))

        # Build feature vector: mean and std per KPI
        features = {
            "playerId": row["playerId"],
            "name": row["name"],
            "status": row["status"],
            "direction": row.get("direction", "NONE"),
            "age": row["age"],
            "origin_team": row.get("origin_team", ""),
            "origin_comp": row.get("origin_comp", ""),
            "origin_median": row["origin_median"],
            "origin_matches": row["origin_matches"],
            "n_matches_loaded": len(matches),
        }

        for kpi_id in kpi_ids:
            values = [m[kpi_id] for m in matches if kpi_id in m]
            if values:
                kpi_name = kpi_defs.get(kpi_id, f"KPI_{kpi_id}")
                features[f"mean_{kpi_name}"] = np.mean(values)
                if len(values) > 1:
                    features[f"std_{kpi_name}"] = np.std(values)

        all_keeper_features.append(features)

    df = pd.DataFrame(all_keeper_features)
    print(f"\nDataset: {len(df)} keepers × {len(df.columns)} columns")
    print(f"Mean columns: {sum(1 for c in df.columns if c.startswith('mean_'))}")

    # Save
    df.to_parquet(CACHE_PARQUET, index=False)
    print(f"Saved: {CACHE_PARQUET}")

    # Also save CSV (without std columns to keep it manageable)
    mean_cols = [c for c in df.columns if not c.startswith("std_")]
    df[mean_cols].to_csv(CACHE_CSV, index=False)
    print(f"Saved: {CACHE_CSV}")

    return df


if __name__ == "__main__":
    df = build_dataset()
    print(f"\nFinal dataset: {df.shape}")
    print(f"Status distribution: {df['status'].value_counts().to_dict()}")
    mean_cols = [c for c in df.columns if c.startswith("mean_")]
    print(f"KPI features (mean_*): {len(mean_cols)}")
