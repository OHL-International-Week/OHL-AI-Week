import json
from pathlib import Path

import numpy as np
import pandas as pd

GK_DATA = Path("GK_Data")

# 1. Load labels
dataset = pd.read_csv(GK_DATA / "gk_dataset_final.csv")
print(f"Total: {len(dataset)} keepers")
print(dataset["status"].value_counts())

# 2. Load definition files
with open(GK_DATA / "player_kpi_definitions.json") as f:
    kpi_defs = {d["id"]: d["name"] for d in json.load(f).get("data", [])}

with open(GK_DATA / "player_score_definitions.json") as f:
    score_defs = {d["id"]: d["name"] for d in json.load(f).get("data", [])}


# 3. Load KPIs for a single keeper
def load_keeper_match_kpis(player_id, match_dirs_str):
    """Load player_kpis for a keeper from all his matches."""
    all_kpis = []
    if not isinstance(match_dirs_str, str) or not match_dirs_str:
        return all_kpis

    for match_dir in match_dirs_str.split("|"):
        match_dir = match_dir.strip()
        if not match_dir:
            continue
        # Search in all competition-season folders
        for cs_dir in (GK_DATA / "competitions").iterdir():
            if not cs_dir.is_dir():
                continue
            pkpi_path = cs_dir / match_dir / "player_kpis.json"
            if pkpi_path.exists():
                with open(pkpi_path) as f:
                    data = json.load(f).get("data", {})
                for side in ["squadHome", "squadAway"]:
                    for player in data.get(side, {}).get("players", []):
                        if (
                            player["id"] == player_id
                            and player.get("position") == "GOALKEEPER"
                            and player.get("matchShare", 0) >= 0.5
                        ):
                            kpis = {
                                k["kpiId"]: k["value"] for k in player.get("kpis", [])
                            }
                            kpis["matchShare"] = player["matchShare"]
                            kpis["playDuration"] = player.get("playDuration", 0)
                            all_kpis.append(kpis)
                break  # found, move to next match
    return all_kpis


# 4. Example: load data for first keeper
row = dataset.iloc[0]
kpis = load_keeper_match_kpis(row["playerId"], row["origin_match_dirs"])
print(
    f"{row['name']}: {len(kpis)} matches, {len(kpis[0]) if kpis else 0} KPIs per match"
)

# 5. Aggregate into feature vector
if kpis:
    kpi_ids = set()
    for m in kpis:
        kpi_ids.update(k for k in m.keys() if isinstance(k, int))

    features = {}
    for kpi_id in kpi_ids:
        values = [m[kpi_id] for m in kpis if kpi_id in m]
        kpi_name = kpi_defs.get(kpi_id, f"KPI_{kpi_id}")
        features[f"mean_{kpi_name}"] = np.mean(values)

    print(f"Feature vector: {len(features)} features")
