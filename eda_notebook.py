"""
eda_notebook.py
===============
Comprehensive Exploratory Data Analysis (EDA) script for a movie ratings dataset.
Generates summary statistics and three high-resolution visualisation plots.
"""

import os
import sys
import warnings

# Fix Windows console Unicode output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


import matplotlib
matplotlib.use('Agg')  # Non-GUI backend — must be set before importing pyplot

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Resolve paths relative to the script's own location
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset.csv")

# Colour palettes
BAR_COLOR = "#3b82f6"
KDE_COLOR = "#ffffff"
MEAN_LINE_COLOR = "#ef4444"

GENRE_PALETTE = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
    "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
]

LANGUAGE_PALETTE = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
    "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
    "#06b6d4", "#e11d48",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _section(title: str) -> None:
    """Print a formatted section header."""
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _apply_dark_style(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply a consistent dark-background style to the given axes/figure."""
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="white", which="both")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#334155")


# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """Load the CSV dataset and perform basic type coercion."""
    df = pd.read_csv(path)
    # Ensure correct types
    df["user_id"] = df["user_id"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
    df["is_anomaly"] = pd.to_numeric(df["is_anomaly"], errors="coerce").astype("Int64")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 2. Summary statistics
# ---------------------------------------------------------------------------
def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print comprehensive summary statistics to stdout."""

    # --- Shape, dtypes, memory ---
    _section("BASIC INFO")
    print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print("\nColumn dtypes:")
    for col in df.columns:
        print(f"  {col:20s}  {str(df[col].dtype):>15s}")

    # --- Missing values ---
    _section("MISSING VALUES")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values detected — dataset is complete.")
    else:
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            print(f"  {col:20s}  {cnt:>6,}  ({pct:5.2f}%)")

    # --- Unique counts ---
    _section("UNIQUE COUNTS PER COLUMN")
    for col in df.columns:
        print(f"  {col:20s}  {df[col].nunique():>6,}")

    # --- Rating distribution statistics ---
    _section("RATING DISTRIBUTION STATISTICS")
    rating = df["rating"].dropna()
    skewness = float(stats.skew(rating, nan_policy="omit"))
    print(f"  Count    : {len(rating):,}")
    print(f"  Mean     : {rating.mean():.4f}")
    print(f"  Median   : {rating.median():.4f}")
    print(f"  Std Dev  : {rating.std():.4f}")
    print(f"  Min      : {rating.min():.1f}")
    print(f"  Max      : {rating.max():.1f}")
    print(f"  Skewness : {skewness:+.4f}")
    print(f"  25th pct : {rating.quantile(0.25):.2f}")
    print(f"  75th pct : {rating.quantile(0.75):.2f}")

    # --- Top 10 most rated movies ---
    _section("TOP 10 MOST RATED MOVIES")
    top_movies = (
        df.groupby("movie_title")["rating"]
        .agg(num_ratings="count", avg_rating="mean")
        .sort_values("num_ratings", ascending=False)
        .head(10)
    )
    for rank, (title, row) in enumerate(top_movies.iterrows(), 1):
        print(f"  {rank:>2}. {title:40s}  ratings={int(row['num_ratings']):>4}  avg={row['avg_rating']:.2f}")

    # --- Top 10 most active users ---
    _section("TOP 10 MOST ACTIVE USERS")
    top_users = (
        df.groupby("user_id")["rating"]
        .agg(num_ratings="count", avg_rating="mean")
        .sort_values("num_ratings", ascending=False)
        .head(10)
    )
    for rank, (uid, row) in enumerate(top_users.iterrows(), 1):
        print(f"  {rank:>2}. {uid:10s}  ratings={int(row['num_ratings']):>4}  avg={row['avg_rating']:.2f}")

    # --- Anomaly rate ---
    _section("ANOMALY RATE")
    total = len(df)
    anomalies = int(df["is_anomaly"].sum())
    pct = anomalies / total * 100
    print(f"  Total rows   : {total:,}")
    print(f"  Anomalies    : {anomalies:,}")
    print(f"  Anomaly rate : {pct:.2f}%")

    # --- Average rating per genre ---
    _section("AVERAGE RATING PER GENRE")
    genre_stats = (
        df.groupby("genre")["rating"]
        .agg(["count", "mean", "std"])
        .sort_values("mean", ascending=False)
    )
    for genre, row in genre_stats.iterrows():
        print(f"  {genre:15s}  n={int(row['count']):>5}  mean={row['mean']:.3f}  std={row['std']:.3f}")

    # --- Average rating per language ---
    _section("AVERAGE RATING PER LANGUAGE")
    lang_stats = (
        df.groupby("language")["rating"]
        .agg(["count", "mean", "std"])
        .sort_values("mean", ascending=False)
    )
    for lang, row in lang_stats.iterrows():
        print(f"  {lang:15s}  n={int(row['count']):>5}  mean={row['mean']:.3f}  std={row['std']:.3f}")

    # --- Temporal distribution ---
    _section("TEMPORAL DISTRIBUTION OF RATINGS")
    if df["timestamp"].notna().any():
        ts = df["timestamp"].dropna()
        print(f"  Earliest rating : {ts.min()}")
        print(f"  Latest rating   : {ts.max()}")
        print(f"  Time span       : {ts.max() - ts.min()}")
        print("\n  Ratings per year-month (top 12):")
        monthly = ts.dt.to_period("M").value_counts().sort_index()
        for period, cnt in monthly.tail(12).items():
            print(f"    {period}  →  {cnt:>5,}")
    else:
        print("  No valid timestamps found.")


# ---------------------------------------------------------------------------
# 3a. Plot 1 — Rating distribution histogram with KDE
# ---------------------------------------------------------------------------
def plot_rating_distribution(df: pd.DataFrame, save_path: str) -> None:
    """Histogram of rating values with KDE overlay, mean line, and stat box."""
    rating = df["rating"].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_style(ax, fig)

    # Histogram
    counts, bins, patches = ax.hist(
        rating, bins=np.arange(0.75, 5.75, 0.5), color=BAR_COLOR,
        edgecolor="#1e293b", alpha=0.85, zorder=3, label="Count",
    )

    # KDE overlay
    kde_x = np.linspace(rating.min() - 0.3, rating.max() + 0.3, 300)
    kde = stats.gaussian_kde(rating)
    kde_y = kde(kde_x) * len(rating) * 0.5  # scale to histogram
    ax.plot(kde_x, kde_y, color=KDE_COLOR, linewidth=2, label="KDE", zorder=4)

    # Mean line
    mean_val = rating.mean()
    ax.axvline(mean_val, color=MEAN_LINE_COLOR, linestyle="--", linewidth=2, zorder=5, label=f"Mean ({mean_val:.2f})")
    ax.annotate(
        f"Mean = {mean_val:.2f}",
        xy=(mean_val, ax.get_ylim()[1] * 0.92),
        xytext=(mean_val + 0.35, ax.get_ylim()[1] * 0.92),
        fontsize=10, color=MEAN_LINE_COLOR, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=MEAN_LINE_COLOR, lw=1.5),
    )

    # Stat text box
    median_val = rating.median()
    std_val = rating.std()
    stats_text = f"Mean   : {mean_val:.2f}\nMedian : {median_val:.2f}\nStd    : {std_val:.2f}"
    props = dict(boxstyle="round,pad=0.5", facecolor="#334155", edgecolor="#64748b", alpha=0.9)
    ax.text(
        0.97, 0.75, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=props, color="white", family="monospace",
    )

    ax.set_xlabel("Rating (1.0 – 5.0)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Movie Ratings", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlim(0.5, 5.5)
    ax.legend(loc="upper left", fontsize=10, facecolor="#1e293b", edgecolor="#64748b", labelcolor="white")
    ax.grid(axis="y", color="#334155", linewidth=0.5, zorder=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Saved: {os.path.basename(save_path)}")


# ---------------------------------------------------------------------------
# 3b. Plot 2 — Genre distribution horizontal bar chart
# ---------------------------------------------------------------------------
def plot_genre_distribution(df: pd.DataFrame, save_path: str) -> None:
    """Horizontal bar chart of ratings volume per genre, sorted descending."""
    genre_counts = df["genre"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    _apply_dark_style(ax, fig)

    n_genres = len(genre_counts)
    colors = (GENRE_PALETTE * ((n_genres // len(GENRE_PALETTE)) + 1))[:n_genres]

    bars = ax.barh(
        genre_counts.index, genre_counts.values,
        color=colors, edgecolor="#1e293b", zorder=3, height=0.65,
    )

    # Value labels
    max_val = genre_counts.max()
    for bar, val in zip(bars, genre_counts.values):
        ax.text(
            bar.get_width() + max_val * 0.015, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", ha="left", fontsize=11,
            color="white", fontweight="bold",
        )

    ax.set_xlabel("Number of Ratings", fontsize=12)
    ax.set_title("Volume of Ratings by Genre", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlim(0, max_val * 1.12)
    ax.tick_params(axis="y", labelsize=11, colors="white")
    ax.grid(axis="x", color="#334155", linewidth=0.5, zorder=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Saved: {os.path.basename(save_path)}")


# ---------------------------------------------------------------------------
# 3c. Plot 3 — Language breakdown donut chart
# ---------------------------------------------------------------------------
def plot_language_breakdown(df: pd.DataFrame, save_path: str) -> None:
    """Donut / pie chart showing rating distribution by language."""
    lang_counts = df["language"].value_counts().sort_values(ascending=False)

    n_langs = len(lang_counts)
    colors = (LANGUAGE_PALETTE * ((n_langs // len(LANGUAGE_PALETTE)) + 1))[:n_langs]

    # Explode the largest slice
    explode = [0.06 if i == 0 else 0.0 for i in range(n_langs)]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0f172a")

    wedges, texts, autotexts = ax.pie(
        lang_counts.values,
        labels=None,  # we'll use a legend instead
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=140,
        colors=colors,
        explode=explode,
        pctdistance=0.80,
        wedgeprops=dict(width=0.45, edgecolor="#0f172a", linewidth=2),
    )

    # Style percentage labels
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(9)
        t.set_fontweight("bold")

    # Centre label
    ax.text(0, 0, "Languages", ha="center", va="center", fontsize=13,
            color="white", fontweight="bold")

    ax.set_title(
        "Linguistic Diversity: Rating Distribution by Language",
        fontsize=15, fontweight="bold", color="white", pad=20,
    )

    # Legend outside
    legend = ax.legend(
        wedges, [f"{lang}  ({cnt:,})" for lang, cnt in lang_counts.items()],
        title="Language", loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize=10, frameon=True, facecolor="#1e293b", edgecolor="#64748b",
        labelcolor="white",
    )
    legend.get_title().set_color("white")
    legend.get_title().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {os.path.basename(save_path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point: load data, print statistics, generate plots."""
    print("=" * 70)
    print("  MOVIE RATINGS — EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # Load
    if not os.path.isfile(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    df = load_dataset(DATASET_PATH)
    print(f"\nDataset loaded: {DATASET_PATH}")

    # Summary statistics
    print_summary_statistics(df)

    # Plots
    _section("GENERATING PLOTS")
    plot_files = [
        ("eda_rating_distribution.png", plot_rating_distribution),
        ("eda_genre_distribution.png", plot_genre_distribution),
        ("eda_language_breakdown.png", plot_language_breakdown),
    ]

    saved_files: list[str] = []
    for filename, plot_fn in plot_files:
        save_path = os.path.join(SCRIPT_DIR, filename)
        plot_fn(df, save_path)
        saved_files.append(save_path)

    # Completion message
    _section("COMPLETE")
    print("All EDA outputs have been generated successfully.\n")
    print("Saved files:")
    for f in saved_files:
        print(f"  • {f}")
    print()


if __name__ == "__main__":
    main()
