"""
generate_dataset.py
====================
Generates a 5,000-row movie-ratings dataset by reading the existing 500-row
dataset.csv, converting / enriching it, generating 4,500 new rows, and saving
the combined result back to dataset.csv.  Also produces data_dictionary.csv.

Usage:
    python generate_dataset.py
"""

import os
import random
import math
import csv
import sys
from datetime import datetime, timedelta

# Fix Windows console Unicode output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
TOTAL_ROWS = 5000
ANOMALY_RATE_LOW = 0.02   # 2 %
ANOMALY_RATE_HIGH = 0.05  # 5 %

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset.csv")
DATA_DICT_PATH = os.path.join(SCRIPT_DIR, "data_dictionary.csv")

# User IDs in U0001..U0200 format
USER_IDS = [f"U{i:04d}" for i in range(1, 201)]

# Genres (original + new)
GENRES = [
    "Action", "Drama", "Comedy", "Thriller", "Romance", "Sci-Fi",
    "Crime", "Animation",
]

# Languages (original + new)
LANGUAGES = [
    "Hindi", "Tamil", "Telugu", "Kannada",
    "Malayalam", "Bengali", "Marathi", "English", "Korean", "Japanese",
]

# ──────────────────────────────────────────────────────────────────────────────
# Movie-title pool  (≥ 80 titles, organised by primary language)
# ──────────────────────────────────────────────────────────────────────────────
MOVIE_POOL = {
    "Hindi": [
        "Dangal", "3 Idiots", "PK", "Lagaan", "Sholay", "Dil Chahta Hai",
        "Queen", "Zindagi Na Milegi Dobara", "Barfi", "Rang De Basanti",
        "Gangs of Wasseypur", "Tumbbad", "Article 15", "Andhadhun",
        "Gully Boy", "War", "Pathaan", "Jawan", "Animal", "Stree",
    ],
    "Tamil": [
        "Vikram", "Ponniyin Selvan", "Master", "Kaithi", "Asuran",
        "Super Deluxe", "Soorarai Pottru", "Jailer", "Leo", "Rocketry",
    ],
    "Telugu": [
        "Baahubali", "RRR", "Pushpa", "Arjun Reddy", "Eega",
        "Ala Vaikunthapurramuloo", "Bheemla Nayak", "Salaar",
        "Kalki 2898 AD", "HIT",
    ],
    "Malayalam": [
        "Drishyam", "Premam", "Bangalore Days", "Kumbalangi Nights",
        "Jallikattu", "Minnal Murali", "2018", "Manjummel Boys",
        "Aavesham", "Bramayugam",
    ],
    "Bengali": [
        "Pather Panchali", "Charulata", "Feluda series", "Haami",
        "Belaseshe",
    ],
    "Kannada": [
        "KGF", "Kantara", "777 Charlie", "Ulidavaru Kandanthe",
        "Garuda Gamana Vrishabha Vahana",
    ],
    "English": [
        "Inception", "The Dark Knight", "Interstellar", "Parasite",
        "The Matrix", "Fight Club", "Pulp Fiction", "Forrest Gump",
        "The Shawshank Redemption", "Gladiator",
    ],
    "Korean": [
        "Oldboy", "Train to Busan", "Memories of Murder",
    ],
    "Japanese": [
        "Spirited Away", "Your Name", "Akira",
    ],
    "Marathi": [
        "Sairat", "Court", "Natsamrat", "Harishchandrachi Factory",
    ],
}

# Flat list of all movie titles (for random selection)
ALL_MOVIES = []
for titles in MOVIE_POOL.values():
    ALL_MOVIES.extend(titles)

# Mapping: movie → its primary language (for language-aware generation)
MOVIE_TO_LANGUAGE = {}
for lang, titles in MOVIE_POOL.items():
    for title in titles:
        MOVIE_TO_LANGUAGE[title] = lang

# Rating options: 1.0 to 5.0 in 0.5 increments
RATING_OPTIONS = [r / 2 for r in range(2, 11)]  # [1.0, 1.5, ..., 5.0]

# Bell-curve weights centred around 3.5 (index 5 in RATING_OPTIONS)
# Using a Gaussian pdf with mean=3.5 and sigma=0.9
RATING_WEIGHTS = []
for r in RATING_OPTIONS:
    weight = math.exp(-0.5 * ((r - 3.5) / 0.9) ** 2)
    RATING_WEIGHTS.append(weight)

# Release-year range
RELEASE_YEAR_MIN = 1955
RELEASE_YEAR_MAX = 2023

# Timestamp window
TS_START = datetime(2023, 1, 1, 0, 0, 0)
TS_END = datetime(2025, 12, 31, 23, 59, 59)
TS_SPAN_SECONDS = int((TS_END - TS_START).total_seconds())


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _convert_user_id(old_id) -> str:
    """Convert an old integer user_id (e.g. 496) to the new U-prefixed format."""
    try:
        num = int(old_id)
    except (ValueError, TypeError):
        # Already a string like "U0042" – return as-is
        return str(old_id)
    # Clamp to 1..200
    num = max(1, min(num, 200))
    return f"U{num:04d}"


def _random_rating(rng: random.Random) -> float:
    """Return a rating from 1.0-5.0 in 0.5 steps with bell-curve distribution."""
    return rng.choices(RATING_OPTIONS, weights=RATING_WEIGHTS, k=1)[0]


def _random_timestamp(rng: random.Random) -> str:
    """Return a random YYYY-MM-DD HH:MM:SS timestamp in [2023-01-01, 2025-12-31]."""
    offset = rng.randint(0, TS_SPAN_SECONDS)
    dt = TS_START + timedelta(seconds=offset)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _generate_sequential_timestamps(rng: random.Random, n: int) -> list:
    """Generate *n* timestamps in sorted order spanning the timestamp window."""
    offsets = sorted(rng.randint(0, TS_SPAN_SECONDS) for _ in range(n))
    return [(TS_START + timedelta(seconds=o)).strftime("%Y-%m-%d %H:%M:%S") for o in offsets]


def _pick_language_for_movie(movie: str, rng: random.Random) -> str:
    """
    80 % of the time return the movie's primary language; 20 % pick any language
    (simulates dubbed / multi-language releases).
    """
    primary = MOVIE_TO_LANGUAGE.get(movie)
    if primary and rng.random() < 0.80:
        return primary
    return rng.choice(LANGUAGES)


# ──────────────────────────────────────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────────────────────────────────────

def read_existing_dataset(path: str) -> list:
    """
    Read the existing 500-row dataset.csv and return rows as list of dicts.
    Converts old integer user_ids to the new U-prefixed string format.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "user_id": _convert_user_id(row["user_id"]),
                "movie_title": row["movie_title"].strip(),
                "genre": row["genre"].strip(),
                "language": row["language"].strip(),
                "release_year": int(row["release_year"]),
                "rating": float(row["rating"]),
                "timestamp": row["timestamp"].strip(),
                "is_anomaly": 0,
            })
    return rows


def generate_new_rows(n: int, rng: random.Random) -> list:
    """Generate *n* new rating rows with realistic distributions."""
    print(f"  → Generating {n} new rows …")

    # Pre-generate sorted timestamps for the new rows
    timestamps = _generate_sequential_timestamps(rng, n)

    rows = []
    for i in range(n):
        movie = rng.choice(ALL_MOVIES)
        language = _pick_language_for_movie(movie, rng)
        genre = rng.choice(GENRES)
        release_year = rng.randint(RELEASE_YEAR_MIN, RELEASE_YEAR_MAX)
        rating = _random_rating(rng)

        rows.append({
            "user_id": rng.choice(USER_IDS),
            "movie_title": movie,
            "genre": genre,
            "language": language,
            "release_year": release_year,
            "rating": rating,
            "timestamp": timestamps[i],
            "is_anomaly": 0,
        })

    return rows


def inject_anomalies(rows: list, rng: random.Random) -> list:
    """
    Mark 2-5 % of all rows as anomalies: set is_anomaly=1 and force
    the rating to an extreme value (1.0 or 5.0).
    """
    n = len(rows)
    anomaly_count = rng.randint(
        int(n * ANOMALY_RATE_LOW),
        int(n * ANOMALY_RATE_HIGH),
    )
    indices = rng.sample(range(n), anomaly_count)
    for idx in indices:
        rows[idx]["is_anomaly"] = 1
        rows[idx]["rating"] = rng.choice([1.0, 5.0])
    print(f"  → Injected {anomaly_count} anomalies ({anomaly_count / n * 100:.1f} % of {n} rows)")
    return rows


def save_dataset(rows: list, path: str) -> None:
    """Write the final dataset to CSV."""
    fieldnames = [
        "user_id", "movie_title", "genre", "language",
        "release_year", "rating", "timestamp", "is_anomaly",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  → Saved {len(rows)} rows to {path}")


def generate_data_dictionary(path: str) -> None:
    """Auto-generate data_dictionary.csv describing every column."""
    entries = [
        {
            "column_name": "user_id",
            "data_type": "string",
            "description": "Unique user identifier in U-prefixed zero-padded format (U0001 to U0200).",
            "example_value": "U0042",
            "nullable": "No",
        },
        {
            "column_name": "movie_title",
            "data_type": "string",
            "description": "Title of the movie being rated, drawn from a curated pool of 80+ Indian and international films.",
            "example_value": "Dangal",
            "nullable": "No",
        },
        {
            "column_name": "genre",
            "data_type": "string",
            "description": "Genre category of the movie. One of: Action, Drama, Comedy, Thriller, Romance, Sci-Fi, Crime, Animation.",
            "example_value": "Thriller",
            "nullable": "No",
        },
        {
            "column_name": "language",
            "data_type": "string",
            "description": "Language of the movie version rated. One of: Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, English, Korean, Japanese.",
            "example_value": "Hindi",
            "nullable": "No",
        },
        {
            "column_name": "release_year",
            "data_type": "integer",
            "description": "Year the movie was released, ranging from 1955 to 2023.",
            "example_value": "2018",
            "nullable": "No",
        },
        {
            "column_name": "rating",
            "data_type": "float",
            "description": "User rating on a 1.0 to 5.0 scale in 0.5 increments. Follows a bell-curve distribution centred around 3.5; anomaly rows have forced extreme values (1.0 or 5.0).",
            "example_value": "3.5",
            "nullable": "No",
        },
        {
            "column_name": "timestamp",
            "data_type": "string (datetime)",
            "description": "Date and time when the rating was submitted, in YYYY-MM-DD HH:MM:SS format, between 2023-01-01 and 2025-12-31.",
            "example_value": "2024-06-15 14:23:07",
            "nullable": "No",
        },
        {
            "column_name": "is_anomaly",
            "data_type": "integer (binary)",
            "description": "Binary flag indicating whether the rating is an anomaly (simulated bot/noise). 1 = anomaly, 0 = normal. Approximately 2-5 % of rows are anomalies.",
            "example_value": "0",
            "nullable": "No",
        },
    ]

    fieldnames = ["column_name", "data_type", "description", "example_value", "nullable"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)
    print(f"  → Saved data dictionary to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: read → generate → merge → inject anomalies → save."""
    rng = random.Random(SEED)
    random.seed(SEED)  # Also set module-level seed for random.choices compatibility

    print("[1/5] Reading existing dataset …")
    existing_rows = read_existing_dataset(DATASET_PATH)
    print(f"  → Loaded {len(existing_rows)} existing rows")

    rows_to_generate = TOTAL_ROWS - len(existing_rows)
    if rows_to_generate <= 0:
        print("  ⚠  Dataset already has ≥ 5,000 rows. No new rows needed.")
        rows_to_generate = 0

    print(f"[2/5] Generating {rows_to_generate} new rows …")
    new_rows = generate_new_rows(rows_to_generate, rng)

    print("[3/5] Merging existing + new rows …")
    all_rows = existing_rows + new_rows
    # Sort by timestamp for realistic chronological ordering
    all_rows.sort(key=lambda r: r["timestamp"])
    print(f"  → Total rows: {len(all_rows)}")

    print("[4/5] Injecting anomalies …")
    all_rows = inject_anomalies(all_rows, rng)

    print("[5/5] Saving outputs …")
    save_dataset(all_rows, DATASET_PATH)
    generate_data_dictionary(DATA_DICT_PATH)

    # Quick summary
    anomaly_count = sum(1 for r in all_rows if r["is_anomaly"] == 1)
    unique_users = len(set(r["user_id"] for r in all_rows))
    unique_movies = len(set(r["movie_title"] for r in all_rows))
    unique_genres = len(set(r["genre"] for r in all_rows))
    unique_langs = len(set(r["language"] for r in all_rows))
    print("\n✅  Done!")
    print(f"   Rows          : {len(all_rows)}")
    print(f"   Unique users  : {unique_users}")
    print(f"   Unique movies : {unique_movies}")
    print(f"   Unique genres : {unique_genres}")
    print(f"   Languages     : {unique_langs}")
    print(f"   Anomalies     : {anomaly_count} ({anomaly_count / len(all_rows) * 100:.1f} %)")


if __name__ == "__main__":
    main()
