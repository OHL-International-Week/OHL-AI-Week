"""Load labels and player scores for all goalkeepers."""

import json

import numpy as np
import pandas as pd

from .config import GK_DATA, OUTPUT


def _load_keeper_scores(player_id, match_dirs_str):
    """Load player_scores and identify position from player_kpis for a keeper."""
    all_scores = []
    if not isinstance(match_dirs_str, str) or not match_dirs_str:
        return all_scores

    for match_dir in match_dirs_str.split("|"):
        match_dir = match_dir.strip()
        if not match_dir:
            continue
        for cs_dir in (GK_DATA / "competitions").iterdir():
            if not cs_dir.is_dir() or cs_dir.name.startswith("."):
                continue

            match_path = cs_dir / match_dir
            pkpi_path = match_path / "player_kpis.json"
            pscore_path = match_path / "player_scores.json"

            if not pkpi_path.exists():
                continue

            # Check position and matchShare from player_kpis
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

            break  # found match folder, move to next
    return all_scores


def load_data(score_defs):
    """Load labels CSV and player scores for all keepers.

    Returns
    -------
    dataset : DataFrame  — raw labels (693 rows)
    df : DataFrame       — one row per keeper with aggregated score features
    """
    print("=" * 70)
    print("1. LOADING DATA")
    print("=" * 70)

    dataset = pd.read_csv(GK_DATA / "gk_dataset_final.csv")
    print(f"Total goalkeepers: {len(dataset)}")
    print(f"\nStatus distribution:\n{dataset['status'].value_counts().to_string()}")
    print(f"\nDirection distribution:\n{dataset['direction'].value_counts().to_string()}")

    # Load scores for all keepers
    print("\n" + "=" * 70)
    print("2. LOADING PLAYER SCORES FOR ALL KEEPERS")
    print("=" * 70)

    all_keeper_features = []
    total = len(dataset)
    for idx, row in dataset.iterrows():
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processing keeper {idx + 1}/{total}: {row['name']}")

        matches = _load_keeper_scores(row["playerId"], row["origin_match_dirs"])
        if not matches:
            continue

        # Aggregate: mean of all score IDs across matches
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
    print(f"\nLoaded data for {len(df)} keepers (out of {total})")
    print(f"Status distribution in loaded data:")
    print(df["status"].value_counts().to_string())

    df.to_csv(OUTPUT / "keeper_features.csv", index=False)

    return dataset, df
