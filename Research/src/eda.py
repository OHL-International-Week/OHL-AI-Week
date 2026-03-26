"""Exploratory data analysis — generates plots 01–05."""

import matplotlib.pyplot as plt
import seaborn as sns

from .config import OUTPUT, STATUS_ORDER, META_COLS


def run_eda(df_model, feature_cols_clean, gk_score_cols, general_score_cols):
    """Generate all EDA visualisations."""
    print("\n" + "=" * 70)
    print("4. EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # Binary target for later use
    df_model["progressed"] = (df_model["status"] == "PLAYS").astype(int)

    # 4a. Box plots for GK-specific scores
    available_gk_cols = [c for c in gk_score_cols if c in feature_cols_clean]
    if available_gk_cols:
        n_plots = len(available_gk_cols)
        ncols = 3
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, col in enumerate(available_gk_cols):
            ax = axes[i]
            sns.boxplot(data=df_model, x="status", y=col, order=STATUS_ORDER, ax=ax)
            short_name = col.replace("mean_", "").replace("GK_", "")
            ax.set_title(short_name, fontsize=10)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("GK-Specific Scores by Career Status", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT / "01_gk_scores_boxplots.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: 01_gk_scores_boxplots.png")

    # 4b. Box plots for general scores
    available_gen_cols = [c for c in general_score_cols if c in feature_cols_clean]
    if available_gen_cols:
        n_plots = len(available_gen_cols)
        ncols = 3
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, col in enumerate(available_gen_cols):
            ax = axes[i]
            sns.boxplot(data=df_model, x="status", y=col, order=STATUS_ORDER, ax=ax)
            short_name = col.replace("mean_", "")
            ax.set_title(short_name, fontsize=10)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("General Scores by Career Status", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT / "02_general_scores_boxplots.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: 02_general_scores_boxplots.png")

    # 4c. Correlation heatmap
    corr_cols = [c for c in feature_cols_clean if c.startswith("mean_")]
    if len(corr_cols) > 2:
        fig, ax = plt.subplots(figsize=(16, 14))
        corr_matrix = df_model[corr_cols].corr()
        short_labels = [c.replace("mean_", "").replace("GK_", "")[:30] for c in corr_cols]
        sns.heatmap(
            corr_matrix,
            xticklabels=short_labels,
            yticklabels=short_labels,
            cmap="RdBu_r",
            center=0,
            annot=False,
            ax=ax,
        )
        ax.set_title("Feature Correlation Matrix", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT / "03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: 03_correlation_heatmap.png")

    # 4d. Distribution of meta features
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, META_COLS):
        sns.boxplot(data=df_model, x="status", y=col, order=STATUS_ORDER, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    plt.suptitle("Meta Features by Career Status", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT / "04_meta_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 04_meta_features.png")

    # 4e. Category distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df_model["status"].value_counts().reindex(STATUS_ORDER)
    bars = ax.bar(STATUS_ORDER, counts.values, color=sns.color_palette("colorblind", 4))
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(count), ha="center", va="bottom", fontsize=12)
    ax.set_title("Goalkeeper Distribution by Career Status")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT / "05_category_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 05_category_distribution.png")
