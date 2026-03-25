# GK Progression Dataset — Guide

## Objective

This dataset has been built to develop a **prediction model** that forecasts whether a goalkeeper will progress to a higher competition level. The central question:

> Given a goalkeeper's performance in a lower league, can you predict whether he will make it at a higher level?

---

## Dataset Overview

### Four categories of goalkeepers

| Category | Count | Under 27 | Description |
|----------|-------|----------|-------------|
| **PLAYS** | 99 | 57 | Transferred to a stronger league and plays regularly (>=5 matches) |
| **BENCH** | 31 | 15 | Transferred to a stronger league but does not play (<5 matches) |
| **DROPPED** | 128 | 43 | Previously played at a higher level, but has since dropped back down |
| **STAYED** | 435 | 235 | Remained at the same competition level (>=15 matches, never transferred up) |
| **Total** | **693** | **350** | |

### Label file

**`gk_dataset_final.csv`** — A single unified file containing all 693 goalkeepers.

| Column | Description |
|--------|-------------|
| `playerId` | Unique player ID (links to all match data) |
| `name` | Player name |
| `age` | Age (as of March 2026) |
| `birthdate` | Date of birth (YYYY-MM-DD) |
| `status` | **PLAYS**, **BENCH**, **DROPPED**, or **STAYED** |
| `direction` | **UP** (transferred up), **DOWN** (dropped down), **NONE** (stayed) |
| `origin_team` | Club where the keeper previously played |
| `origin_comp` | Competition of origin |
| `origin_season` | Season at origin club |
| `origin_median` | Strength of origin competition (0-1 scale) |
| `origin_matches` | Number of matches at origin club |
| `current_team` | Club where the keeper currently plays (empty for STAYED) |
| `current_comp` | Current competition (empty for STAYED) |
| `current_season` | Season at current club |
| `current_median` | Strength of current competition |
| `current_matches` | Number of matches at current club |
| `step` | Difference in competition strength (positive = upward, 0 = stayed) |
| `origin_match_dirs` | Pipe-separated match folders at origin club |
| `current_match_dirs` | Pipe-separated match folders at current club |

**Guideline for modeling:** Use `origin_match_dirs` to load the match data for training. This is the data from BEFORE the keeper transferred.

---

## File Structure

```
GK_Data/
├── gk_dataset_final.csv             # Unified labels (693 keepers, 4 categories)
├── impect_keepers.json              # Player info (name, date of birth, nationality)
├── player_kpi_definitions.json      # 1,458 KPI definitions (for player_kpis)
├── player_score_definitions.json    # 134 player score definitions (for player_scores)
├── event_kpi_definitions.json       # 103 event KPI definitions (for event_kpis)
├── GUIDE.md                         # This guide
├── competitions/                    # Match data per competition-season
│   ├── belgian_pro_league_2024-2025/
│   │   ├── 20240802 KRC Genk - Standard Lüttich/
│   │   │   ├── player_kpis.json     # Player KPIs (aggregated per match)
│   │   │   ├── player_scores.json   # Impect scores (aggregated per match)
│   │   │   ├── event_kpis.json      # KPIs per event/action (optional, advanced)
│   │   │   ├── match_details.json   # Match metadata, formations
│   │   │   └── match_meta.json      # Match ID, date, teams
│   │   └── ...
│   ├── eredivisie_2024-2025/
│   └── ...
```

**Note:** Definition files are complete and up-to-date from the Impect API. `impect_keepers.json` only contains the 693 goalkeepers in the dataset.

---

## Match Data Files

### 1. player_kpis.json — Player KPIs per match

Contains aggregated KPIs per player per match. Includes ALL players (not just goalkeepers), so you must filter by position.

**Structure:**
```json
{
  "data": {
    "squadHome": {
      "id": 1234,
      "players": [
        {
          "id": 56789,
          "position": "GOALKEEPER",
          "playDuration": 5400,
          "matchShare": 1.0,
          "kpis": [
            {"kpiId": 0, "value": 26.6},
            {"kpiId": 1, "value": 15.0}
          ]
        }
      ]
    },
    "squadAway": { ... }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Player ID (links to gk_dataset_final.csv) |
| `position` | string | Position: GOALKEEPER, CENTRAL_DEFENDER, etc. |
| `playDuration` | int | Playing time in seconds |
| `matchShare` | float | Proportion of match played (0.0 - 1.0) |
| `kpis` | array | List of `{kpiId, value}` pairs |

**Number of KPIs:** Up to 1,458 per player per match. Use `player_kpi_definitions.json` for the kpiId-to-name mapping.

**Filter:** Look for `position == "GOALKEEPER"` and `matchShare >= 0.5`.

**Key KPI categories for goalkeepers:**

| Category | Example KPI IDs | Description |
|----------|-----------------|-------------|
| BYPASSED_OPPONENTS | 0-9, 29+ | Opponents bypassed through actions (packing) |
| BALL_LOSS / BALL_WIN | 20-27 | Ball losses and recoveries |
| SUCCESSFUL/UNSUCCESSFUL_PASSES | 90-91+ | Pass statistics (with sub-categories by type/zone) |
| SHOT_XG | 82 | Expected goals from shots |
| REVERSE_PLAY | 16-17 | Backward play (slowing down the game) |
| PXT | 1404-1421 | Packing Expected Threat per action type |
| OFFENSIVE/DEFENSIVE_TOUCHES | 92-93 | Ball touches in attack/defense |
| NUMBER_OF_PRESSES | 1536 | Pressing actions |
| EXPECTED_PASSES | 1783 | Pass difficulty (how difficult were the passes?) |

---

### 2. player_scores.json — Impect Scores per match

**Composite scores** that combine multiple KPIs into a single number. Higher abstraction than player_kpis.

**Structure:**
```json
{
  "data": {
    "squadHome": {
      "id": 1234,
      "players": [
        {
          "id": 56789,
          "playDuration": 5400,
          "matchShare": 1.0,
          "playerScores": [
            {"playerScoreId": 0, "value": 0.55}
          ]
        }
      ]
    }
  }
}
```

**Important:** player_scores does **NOT** have a `position` field. Cross-reference with `player_kpis.json` (same match) to determine the player's position.

**GK-specific Player Scores (most important for this project):**

| ID | Name | Description |
|----|------|-------------|
| 164 | GK_PREVENTED_GOALS_TOTAL_POSTSHOT_XG | Goals prevented based on post-shot xG |
| 166 | GK_PREVENTED_GOALS_TOTAL_POSTSHOT_XG_PERCENT | Same, as a percentage |
| 167-171 | GK_PREVENTED_GOALS_..._RATIO | Shot stopping by shot type (long range, mid range, close range, 1v1, headers) |
| 184 | GK_PREVENTED_GOALS_TOTAL_SHOT_XG | Goals prevented based on pre-shot xG |
| 186 | GK_PREVENTED_GOALS_TOTAL_SHOT_XG_PERCENT | Same, as a percentage |
| 189 | GK_DEFENSIVE_TOUCHES_OUTSIDE_OWN_BOX | Sweeper activity: touches outside the box |
| 190 | GK_CAUGHT_HIGH_BALLS_PERCENT | Percentage of high balls caught |
| 191 | GK_CAUGHT_AND_PUNCHED_HIGH_BALLS_PERCENT | Percentage of high balls caught + punched |
| 192 | GK_SUCCESSFUL_LAUNCHES_PERCENT | Success rate of long balls |

**General Player Scores (also relevant for goalkeepers):**

| ID | Name | Description |
|----|------|-------------|
| 0 | IMPECT_SCORE_PACKING | Overall Impect score |
| 10 | OFFENSIVE_IMPECT_SCORE_PACKING | Offensive Impect score |
| 17 | DEFENSIVE_IMPECT_SCORE_PACKING | Defensive Impect score |
| 9 | PROGRESSION_SCORE_PACKING | Progression (forward actions) |
| 52 | LOW_PASS_SCORE | Low pass quality |
| 55 | DIAGONAL_PASS_SCORE | Diagonal pass quality |
| 81 | GOAL_KICK_SCORE | Goal kick quality |
| 101 | TOTAL_TOUCHES | Total ball touches |
| 163 | RATIO_PASSING_ACCURACY | Passing accuracy (%) |
| 228-229 | FOOT_USAGE_RATIO_RIGHT/LEFT | Foot preference (%) |
| 232 | PASS_COMPLETION_OVER_EXPECTED | Pass completion above expectation |

Full list: see `player_score_definitions.json`.

**Limitation:** GK_PREVENTED_GOALS_* scores are only available from the 2025-2026 season onwards for some competitions. Keepers with data from older seasons will be missing these metrics.

---

### 3. event_kpis.json — KPIs per individual action (OPTIONAL)

Event-level data. Each individual action (pass, dribble, duel) has KPI values. **Most granular data source, but requires more processing.**

```json
[
  {
    "position": "GOALKEEPER",
    "playerId": 29528,
    "eventId": 5861721232,
    "kpiId": 1524,
    "value": -0.001796
  }
]
```

103 event KPIs available. Definitions in `event_kpi_definitions.json`. Key ones: BYPASSED_OPPONENTS (0), BALL_LOSS_NUMBER (22), BALL_WIN_NUMBER (27), SUCCESSFUL_PASSES (90), SHOT_XG (82).

---

### 4. match_details.json — Match metadata

Formation, starting positions, substitutions. Useful for context (e.g., did the team play with 3 or 4 defenders?).

---

### 5. match_meta.json — Match identification

Match ID, date, home/away team.

---

## Competition Strength

Strength is determined by the **median of all squad ratings** per competition (scale 0-1).

| Competition | Median | Level |
|-------------|--------|-------|
| Premier League | 0.850 | Top |
| LaLiga | 0.830 | Top |
| Bundesliga 1 | 0.826 | Top |
| Serie A | 0.820 | Top |
| Ligue 1 | 0.757 | High |
| Eredivisie | 0.692 | High |
| Primeira Liga | 0.682 | High |
| J1 League | 0.580 | Mid |
| Belgian Pro League | 0.570 | Mid |
| Bundesliga 2 | 0.559 | Mid |
| K League 1 | 0.550 | Mid |
| Superligaen (DK) | 0.510 | Mid |
| Allsvenskan | 0.420 | Low |
| Bundesliga 3 | 0.394 | Low |
| Challenger Pro League | 0.349 | Low |
| Keuken Kampioen Divisie | 0.321 | Low |
| Regionalliga | ~0.12-0.15 | Very low |

---

## Recommended Approach

### Data Sources: Three Levels of Depth

The dataset provides three levels of data, from high-level to granular:

| Level | File | Features | Best for |
|-------|------|----------|----------|
| **1. Player Scores** | `player_scores.json` | 134 composite scores | Starting point — pre-aggregated, interpretable |
| **2. Player KPIs** | `player_kpis.json` | 1,458 KPIs | Deeper analysis — raw metrics per match |
| **3. Event KPIs** | `event_kpis.json` | 103 KPIs per action | Advanced — individual actions (passes, saves, etc.) |

**Recommendation:** Start with **player_scores** (Level 1) to get a working model quickly. Then add **player_kpis** (Level 2) for more granularity. Only go to **event_kpis** (Level 3) if you want to build custom features from individual actions (e.g., distribution of pass difficulty, timing of actions).

### Step 1: Load Data
1. Read `gk_dataset_final.csv` for labels and match references
2. Use `origin_match_dirs` to locate match folders
3. Load `player_scores.json` and/or `player_kpis.json` per match
4. Filter on `position == "GOALKEEPER"` and `matchShare >= 0.5` (position field is in `player_kpis.json` — cross-reference with `player_scores.json`)

### Step 2: Feature Engineering
- **Aggregate per player:** Mean (and optionally median, std) of KPIs across all matches
- **Use definition files** to map IDs to names (`player_kpi_definitions.json`, `player_score_definitions.json`, `event_kpi_definitions.json`)
- **Basic features:** Age, competition strength (`origin_median`), number of matches
- **Tip:** Start with player_scores (134 features) before adding player_kpis (1,458 features)

### Step 3: Feature Selection
- 1,458 KPIs is too many — use feature selection (PCA, mutual information, or domain knowledge)
- Start with the **GK-specific scores** (ID 164-192)
- Add **distribution metrics** (passes, bypassed opponents) — these measure whether a keeper can play with his feet
- **Not all KPIs are reliable** — some have high match-to-match variance

### Step 4: Modeling

Possible approaches:
- **Binary:** PLAYS vs (BENCH + STAYED + DROPPED) — "Will this keeper progress?"
- **Binary:** PLAYS vs BENCH — "Given a transfer, will he make it?"
- **Multi-class:** STAYED / DROPPED / BENCH / PLAYS as increasing levels of success
- **Recommended models:** Logistic Regression, Random Forest, XGBoost, LightGBM
- **Class imbalance:** BENCH (31) is small — consider SMOTE, class weights, or stratified sampling

### Step 5: Validation
- Stratified k-fold cross-validation (k=5)
- Report precision, recall, F1-score, and AUC-ROC
- Watch out for overfitting on the smaller classes (BENCH)

---

## Technical Details

### Player Identification
- `playerId` is the unique identifier across all files
- `impect_keepers.json` contains name, date of birth, and nationality for all 693 goalkeepers

### Data Coverage
- **Seasons:** 2019-2020 through 2025-2026
- **Competitions:** 40+ competitions (from Regionalliga to top-5 leagues)
- **Matches:** ~21,000 matches with goalkeeper data
- **Dataset size:** ~24 GB

### Limitations
1. `player_scores.json` with GK_PREVENTED_GOALS_* is only available for some seasons (mainly 2025-2026). Older keepers will be missing shot stopping data.
2. Competition strengths are snapshots and change across seasons.
3. Match data contains ALL players (not just goalkeepers). Filter on `position == "GOALKEEPER"` in `player_kpis.json`.
4. Age is calculated as of March 2026, not at the time of transfer.

### Python Example: Loading Data

```python
import json
import pandas as pd
import numpy as np
from pathlib import Path

GK_DATA = Path('/path/to/GK_Data')

# 1. Load labels
dataset = pd.read_csv(GK_DATA / 'gk_dataset_final.csv')
print(f"Total: {len(dataset)} keepers")
print(dataset['status'].value_counts())

# 2. Load definition files
with open(GK_DATA / 'player_kpi_definitions.json') as f:
    kpi_defs = {d['id']: d['name'] for d in json.load(f).get('data', [])}

with open(GK_DATA / 'player_score_definitions.json') as f:
    score_defs = {d['id']: d['name'] for d in json.load(f).get('data', [])}

# 3. Load KPIs for a single keeper
def load_keeper_match_kpis(player_id, match_dirs_str):
    """Load player_kpis for a keeper from all his matches."""
    all_kpis = []
    if not isinstance(match_dirs_str, str) or not match_dirs_str:
        return all_kpis

    for match_dir in match_dirs_str.split('|'):
        match_dir = match_dir.strip()
        if not match_dir:
            continue
        # Search in all competition-season folders
        for cs_dir in (GK_DATA / 'competitions').iterdir():
            if not cs_dir.is_dir():
                continue
            pkpi_path = cs_dir / match_dir / 'player_kpis.json'
            if pkpi_path.exists():
                with open(pkpi_path) as f:
                    data = json.load(f).get('data', {})
                for side in ['squadHome', 'squadAway']:
                    for player in data.get(side, {}).get('players', []):
                        if (player['id'] == player_id
                            and player.get('position') == 'GOALKEEPER'
                            and player.get('matchShare', 0) >= 0.5):
                            kpis = {k['kpiId']: k['value']
                                    for k in player.get('kpis', [])}
                            kpis['matchShare'] = player['matchShare']
                            kpis['playDuration'] = player.get('playDuration', 0)
                            all_kpis.append(kpis)
                break  # found, move to next match
    return all_kpis

# 4. Example: load data for first keeper
row = dataset.iloc[0]
kpis = load_keeper_match_kpis(row['playerId'], row['origin_match_dirs'])
print(f"{row['name']}: {len(kpis)} matches, {len(kpis[0]) if kpis else 0} KPIs per match")

# 5. Aggregate into feature vector
if kpis:
    kpi_ids = set()
    for m in kpis:
        kpi_ids.update(k for k in m.keys() if isinstance(k, int))

    features = {}
    for kpi_id in kpi_ids:
        values = [m[kpi_id] for m in kpis if kpi_id in m]
        kpi_name = kpi_defs.get(kpi_id, f'KPI_{kpi_id}')
        features[f'mean_{kpi_name}'] = np.mean(values)

    print(f"Feature vector: {len(features)} features")
```
