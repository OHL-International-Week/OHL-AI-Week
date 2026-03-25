"""Build feature columns, handle missingness, produce modelling-ready DataFrame."""

import pandas as pd

from .config import GK_SCORE_IDS, GENERAL_SCORE_IDS, META_COLS


def select_features(df, score_defs):
    """Select and clean features for modelling.

    Returns
    -------
    df_model : DataFrame
    feature_cols_clean : list[str]
    gk_score_cols : list[str]
    general_score_cols : list[str]
    """
    print("\n" + "=" * 70)
    print("3. FEATURE SELECTION")
    print("=" * 70)

    gk_score_cols = []
    general_score_cols = []

    for sid in GK_SCORE_IDS:
        col = f"mean_{score_defs.get(sid, f'SCORE_{sid}')}"
        if col in df.columns:
            gk_score_cols.append(col)

    for sid in GENERAL_SCORE_IDS:
        col = f"mean_{score_defs.get(sid, f'SCORE_{sid}')}"
        if col in df.columns:
            general_score_cols.append(col)

    all_feature_cols = gk_score_cols + general_score_cols + META_COLS

    print(f"GK-specific score features: {len(gk_score_cols)}")
    for c in gk_score_cols:
        print(f"  - {c}")
    print(f"\nGeneral score features: {len(general_score_cols)}")
    for c in general_score_cols:
        print(f"  - {c}")
    print(f"\nMeta features: {META_COLS}")
    print(f"\nTotal features: {len(all_feature_cols)}")

    # Check missingness
    missing_pct = df[all_feature_cols].isnull().mean().sort_values(ascending=False)
    print("\nMissingness per feature:")
    for col in all_feature_cols:
        pct = missing_pct.get(col, 0)
        if pct > 0:
            print(f"  {col}: {pct:.1%}")

    high_missing = missing_pct[missing_pct > 0.5]
    if len(high_missing) > 0:
        print(f"\nFeatures with >50% missing (will be dropped):")
        for col, pct in high_missing.items():
            print(f"  {col}: {pct:.1%}")

    # Drop features with >50% missing
    feature_cols_clean = [c for c in all_feature_cols if c not in high_missing.index]

    # Fill remaining NaN with median
    df_model = df.copy()
    for col in feature_cols_clean:
        if df_model[col].isnull().any():
            df_model[col] = df_model[col].fillna(df_model[col].median())

    # Drop rows where all score features are NaN
    score_cols_in_clean = [c for c in feature_cols_clean if c.startswith("mean_")]
    df_model = df_model.dropna(subset=score_cols_in_clean, how="all")

    print(f"\nAfter cleaning: {len(df_model)} keepers, {len(feature_cols_clean)} features")
    print(f"Status distribution after cleaning:")
    print(df_model["status"].value_counts().to_string())

    return df_model, feature_cols_clean, gk_score_cols, general_score_cols
