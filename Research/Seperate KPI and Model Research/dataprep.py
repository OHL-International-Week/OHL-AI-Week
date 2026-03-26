import pandas as pd
import numpy as np
import json
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
GK_DATA = Path("GK_Data")
INPUT_CSV = GK_DATA / "gk_dataset_final.csv"
KPI_DEFS_JSON = GK_DATA / "player_kpi_definitions.json"
OUTPUT_CSV = "baseline_keeper_dataset.csv"

# =========================
# HELPERS
# =========================
def load_keeper_match_kpis(player_id, match_dirs_str, kpi_defs):
    all_kpis = []
    if not isinstance(match_dirs_str, str) or not match_dirs_str:
        return all_kpis

    for match_dir in map(str.strip, match_dirs_str.split("|")):
        if not match_dir:
            continue
        for comp_dir in (GK_DATA / "competitions").iterdir():
            if not comp_dir.is_dir():
                continue
            pkpi_path = comp_dir / match_dir / "player_kpis.json"
            if not pkpi_path.exists():
                continue
            with open(pkpi_path) as f:
                data = json.load(f).get("data", {})
            for side in ["squadHome", "squadAway"]:
                players = data.get(side, {}).get("players", [])
                for player in players:
                    if (
                        player["id"] == player_id
                        and player.get("position") == "GOALKEEPER"
                        and player.get("matchShare", 0) >= 0.5
                    ):
                        kpis = {k["kpiId"]: k["value"] for k in player.get("kpis", [])}
                        kpis["matchShare"] = player["matchShare"]
                        kpis["playDuration"] = player.get("playDuration", 0)
                        all_kpis.append(kpis)
            break
    return all_kpis

# =========================
# MAIN PIPELINE
# =========================
def main():
    print("Step 1: Loading initial CSV and KPI definitions...")
    dataset = pd.read_csv(INPUT_CSV)
    with open(KPI_DEFS_JSON) as f:
        kpi_defs = {d["id"]: d["name"] for d in json.load(f).get("data", [])}

    # -------------------------
    # Extract match-level rows
    # -------------------------
    print("Step 2: Extracting match performance from JSONs...")
    all_match_rows = []
    for _, row in dataset.iterrows():
        match_kpis = load_keeper_match_kpis(row["playerId"], row["origin_match_dirs"], kpi_defs)
        
        if not match_kpis:
            continue

        base_entry = row.drop(["origin_match_dirs", "current_match_dirs"]).to_dict()
        for match_data in match_kpis:
            entry = base_entry.copy()
            for kpi_id, value in match_data.items():
                kpi_name = kpi_defs.get(kpi_id, f"KPI_{kpi_id}")
                entry[kpi_name] = value
            all_match_rows.append(entry)

    df = pd.DataFrame(all_match_rows)
    print(f"Result: Created raw dataset with {len(df)} rows.")

    # -------------------------
    # Adjust step direction
    # -------------------------
    if {"direction", "step"}.issubset(df.columns):
        df.loc[df["direction"] == "DOWN", "step"] = -df["step"].abs()

    # -------------------------
    # Correlation Filtering (Selecting the most relevant KPIs)
    # -------------------------
        print("Step 3: Filtering for significant KPIs (Corr > 0.1)...")
    metadata_cols = [
        "playerId",
        "name",
        "age",
        "birthdate",
        "status",
        "direction",
        "origin_team",
        "origin_comp",
        "origin_season",
        "origin_median",
        "origin_matches",
        "current_team",
        "current_comp",
        "current_season",
        "current_median",
        "current_matches",
        "step",
        "origin_match_dirs",
        "current_match_dirs"
    ]
    numeric_df = df.select_dtypes(include=["number"])
    correlations = numeric_df.corr()["step"].abs().sort_values(ascending=False)
    significant_kpis = correlations[correlations > 0.1].index.tolist()
    
    final_cols = list(set(metadata_cols + significant_kpis))
    final_cols = [col for col in final_cols if col in df.columns]
    
    # We keep the raw rows here - NO MERGING yet
    useful_df = df[final_cols].copy()

    # -------------------------
    # Simple Deduplication
    # -------------------------
    # Only removes rows that are EXACTLY identical across all selected columns
    useful_df = useful_df.drop_duplicates()

    # -------------------------
    # Final Sort
    # -------------------------
    useful_df["current_matches"] = pd.to_numeric(useful_df["current_matches"], errors="coerce")
    useful_df = useful_df.sort_values(by=["playerId", "current_matches"])

    # -------------------------
    # Diagnostics (Unique Player Counts)
    # -------------------------
    print("\n" + "="*30)
    print("DIAGNOSTICS")
    print("="*30)
    print(f"Original unique Keepers in CSV: {dataset['playerId'].nunique()}")
    print(f"Keepers with found match data: {useful_df['playerId'].nunique()}")
    print(f"Total Rows (Match Performances): {len(useful_df)}")
    
    # Check for specific sparse data issues
    print("\nTop 5 Keepers by match count in this raw data:")
    print(useful_df['playerId'].value_counts().head(5))

    useful_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nBaseline saved to: {OUTPUT_CSV}")

        # =========================
    # GROUPING (PLAYER x LEAGUE)
    # =========================
    print("\nStep 4: Aggregating per player per competition...")

    # 🔥 FIX: Create unified competition column
    useful_df["comp"] = useful_df["current_comp"].fillna(useful_df["origin_comp"])

    # Optional: debug remaining missing comps
    print("Rows with STILL NaN comp:", useful_df["comp"].isna().sum())

    group_keys = ["playerId", "comp"]

    # Identify numeric KPI columns
    numeric_cols = useful_df.select_dtypes(include=[np.number]).columns

    kpis_to_sum = [
        col for col in numeric_cols
        if col not in group_keys and col not in ["age", "step"]
    ]

    # Aggregation logic
    agg_logic = {col: "sum" for col in kpis_to_sum}

    # Special handling
    agg_logic["step"] = "sum"   # ✅ aggregate, don't group
    agg_logic["age"] = "max"

    if "name" in useful_df.columns:
        agg_logic["name"] = "first"

    if "status" in useful_df.columns:
        agg_logic["status"] = "first"

    if "current_team" in useful_df.columns:
        agg_logic["current_team"] = "first"

    # Count number of matches per group
    counts = (
        useful_df
        .groupby(group_keys)
        .size()
        .reset_index(name="rows_merged")
    )

    # Perform aggregation
    df_grouped = (
        useful_df
        .groupby(group_keys, as_index=False)
        .agg(agg_logic)
    )

    # Merge counts
    df_final = df_grouped.merge(counts, on=group_keys)

    # -------------------------
    # Final Output
    # -------------------------
    print(f"\nFinal grouped dataset rows: {len(df_final)}")

    print("\nSample:")
    print(df_final[["playerId", "comp", "rows_merged", "step"]].head(10))

    df_final.to_csv("aggregated_" + OUTPUT_CSV, index=False)
    print(f"\nAggregated dataset saved to: aggregated_{OUTPUT_CSV}")

if __name__ == "__main__":
    main()