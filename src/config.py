"""Shared paths, constants, and definition loaders."""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="colorblind")

# ── Paths ──────────────────────────────────────────────────────────────
GK_DATA = Path("Data")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
STATUS_ORDER = ["PLAYS", "BENCH", "STAYED", "DROPPED"]

# GK-specific score IDs
GK_SCORE_IDS = [164, 166, 167, 168, 169, 170, 171, 184, 186, 189, 190, 191, 192]
# General scores relevant for GK
GENERAL_SCORE_IDS = [0, 1, 2, 9, 10, 17, 52, 55, 81, 101, 163, 228, 229, 232]
# Meta features
META_COLS = ["age", "origin_median", "n_matches_loaded"]


# ── Definition loaders ─────────────────────────────────────────────────
def load_definitions():
    """Load score and KPI definition mappings from JSON files."""
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
