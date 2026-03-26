"""Shared data loading, definitions, and constants for all analysis phases.

Loads goalkeeper labels from gk_dataset_final.csv, aggregates player scores
from per-match JSON files, and provides common constants/helpers.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="colorblind")

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GK_DATA = PROJECT_ROOT / "GK_Data"
COMPETITIONS = GK_DATA / "competitions"

# ── Constants ──────────────────────────────────────────────────────────
STATUS_ORDER = ["PLAYS", "BENCH", "STAYED", "DROPPED"]

# GK-specific player score IDs (from GUIDE.md)
GK_SCORE_IDS = [164, 166, 167, 168, 169, 170, 171, 184, 186, 189, 190, 191, 192]

# General player score IDs relevant for goalkeepers
GENERAL_SCORE_IDS = [0, 1, 2, 9, 10, 17, 52, 55, 81, 101, 163, 228, 229, 232]

# Meta / context features
META_COLS = ["age", "origin_median", "n_matches_loaded"]


def load_definitions():
    """Load score and KPI definition mappings from JSON files.

    Returns
    -------
    score_defs : dict[int, str]  — score ID to name
    score_labels : dict[int, str]  — score ID to label
    kpi_defs : dict[int, str]  — KPI ID to name
    """
    with open(GK_DATA / "player_score_definitions.json") as f:
        score_defs_raw = json.load(f).get("data", [])
    score_defs = {d["id"]: d["name"] for d in score_defs_raw}
    score_labels = {
        d["id"]: d.get("details", {}).get("label", d["name"])
        for d in score_defs_raw
    }
    with open(GK_DATA / "player_kpi_definitions.json") as f:
        kpi_defs = {d["id"]: d["name"] for d in json.load(f).get("data", [])}
    return score_defs, score_labels, kpi_defs


def _load_keeper_scores(player_id, match_dirs_str):
    """Load player_scores for a goalkeeper from all origin matches.

    Cross-references player_kpis.json to verify position == GOALKEEPER
    and matchShare >= 0.5 before loading scores.
    """
    all_scores = []
    if not isinstance(match_dirs_str, str) or not match_dirs_str:
        return all_scores

    for match_dir in match_dirs_str.split("|"):
        match_dir = match_dir.strip()
        if not match_dir:
            continue
        for cs_dir in COMPETITIONS.iterdir():
            if not cs_dir.is_dir() or cs_dir.name.startswith("."):
                continue
            match_path = cs_dir / match_dir
            pkpi_path = match_path / "player_kpis.json"
            pscore_path = match_path / "player_scores.json"
            if not pkpi_path.exists():
                continue

            # Verify position and match share from player_kpis
            with open(pkpi_path) as f:
                kpi_data = json.load(f).get("data", {})
            is_gk = False
            match_share = 0
            for side in ["squadHome", "squadAway"]:
                for player in kpi_data.get(side, {}).get("players", []):
                    if player["id"] == player_id:
                        if (player.get("position") == "GOALKEEPER"
                                and player.get("matchShare", 0) >= 0.5):
                            is_gk = True
                            match_share = player.get("matchShare", 0)
                        break
            if not is_gk:
                break

            # Load scores
            if pscore_path.exists():
                with open(pscore_path) as f:
                    score_data = json.load(f).get("data", {})
                for side in ["squadHome", "squadAway"]:
                    for player in score_data.get(side, {}).get("players", []):
                        if player["id"] == player_id:
                            scores = {
                                s["playerScoreId"]: s["value"]
                                for s in player.get("playerScores", [])
                            }
                            scores["matchShare"] = match_share
                            scores["playDuration"] = player.get("playDuration", 0)
                            all_scores.append(scores)
                            break
            break
    return all_scores


def load_and_aggregate_data(score_defs, cache_path=None):
    """Load labels and aggregate player scores per keeper.

    If cache_path exists, loads from cache. Otherwise processes all match files
    and saves cache.

    Returns
    -------
    dataset : DataFrame  — raw labels (693 rows)
    df : DataFrame       — one row per keeper with mean/std score features
    """
    dataset = pd.read_csv(GK_DATA / "gk_dataset_final.csv")

    # Check for cached features
    if cache_path and Path(cache_path).exists():
        df = pd.read_csv(cache_path)
        print(f"Loaded cached features: {len(df)} keepers from {cache_path}")
        return dataset, df

    print(f"Processing {len(dataset)} goalkeepers from match files...")
    all_keeper_features = []
    total = len(dataset)

    for idx, row in dataset.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{total}")

        matches = _load_keeper_scores(row["playerId"], row["origin_match_dirs"])
        if not matches:
            continue

        score_ids = set()
        for m in matches:
            score_ids.update(k for k in m.keys() if isinstance(k, int))

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

        for sid in score_ids:
            values = [m[sid] for m in matches if sid in m]
            if values:
                sname = score_defs.get(sid, f"SCORE_{sid}")
                features[f"mean_{sname}"] = np.mean(values)
                features[f"std_{sname}"] = np.std(values) if len(values) > 1 else 0

        all_keeper_features.append(features)

    df = pd.DataFrame(all_keeper_features)
    print(f"Loaded {len(df)} keepers with score data")

    if cache_path:
        df.to_csv(cache_path, index=False)
        print(f"Saved cache: {cache_path}")

    return dataset, df


def select_features(df, score_defs):
    """Select features, handle missingness, return modelling-ready DataFrame.

    Returns
    -------
    df_model : DataFrame
    feature_cols : list[str]   — clean feature column names
    gk_cols : list[str]        — GK-specific score columns
    general_cols : list[str]   — general score columns
    """
    gk_cols = []
    general_cols = []

    for sid in GK_SCORE_IDS:
        col = f"mean_{score_defs.get(sid, f'SCORE_{sid}')}"
        if col in df.columns:
            gk_cols.append(col)
    for sid in GENERAL_SCORE_IDS:
        col = f"mean_{score_defs.get(sid, f'SCORE_{sid}')}"
        if col in df.columns:
            general_cols.append(col)

    all_cols = gk_cols + general_cols + META_COLS
    # Drop features with >50% missing
    missing_pct = df[all_cols].isnull().mean()
    high_missing = missing_pct[missing_pct > 0.5].index.tolist()
    feature_cols = [c for c in all_cols if c not in high_missing]

    df_model = df.copy()
    for col in feature_cols:
        if df_model[col].isnull().any():
            df_model[col] = df_model[col].fillna(df_model[col].median())

    score_cols = [c for c in feature_cols if c.startswith("mean_")]
    df_model = df_model.dropna(subset=score_cols, how="all")

    print(f"Features: {len(feature_cols)} ({len(gk_cols)} GK + {len(general_cols)} general + {len(META_COLS)} meta)")
    print(f"Keepers: {len(df_model)} (after cleaning)")
    print(f"Status: {df_model['status'].value_counts().to_dict()}")

    return df_model, feature_cols, gk_cols, general_cols


def get_cache_path():
    """Return the default cache path for keeper features."""
    return PROJECT_ROOT / "output" / "keeper_features.csv"
