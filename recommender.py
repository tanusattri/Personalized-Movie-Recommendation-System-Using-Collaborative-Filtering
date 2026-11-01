"""
recommender.py
==============
Collaborative-filtering movie recommendation engine.

Supports three algorithms:
  • User-based CF  (cosine similarity)
  • Item-based CF  (cosine similarity)
  • SVD latent-factor model

Author: auto-generated
"""

import os
import sys
import subprocess
import warnings
from collections import defaultdict

# Fix Windows console Unicode output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class RecommenderEngine:
    """End-to-end collaborative filtering recommender for an Indian-movie
    rating dataset.

    Parameters
    ----------
    dataset_path : str
        Path to ``dataset.csv``.  If the file does not exist the engine
        attempts to create it by running ``generate_dataset.py``.
    """

    # ------------------------------------------------------------------
    # Construction & data loading
    # ------------------------------------------------------------------
    def __init__(self, dataset_path: str = "dataset.csv"):
        """Load or generate the dataset, clean it, build the user-item
        rating matrix and fit all similarity / SVD models.

        Parameters
        ----------
        dataset_path : str
            Relative or absolute path to the CSV dataset.
        """
        self.dataset_path = dataset_path

        # --- load (or generate then load) --------------------------------
        if not os.path.exists(self.dataset_path):
            gen_script = os.path.join(
                os.path.dirname(os.path.abspath(self.dataset_path)),
                "generate_dataset.py",
            )
            if os.path.exists(gen_script):
                subprocess.run(
                    ["python", gen_script],
                    cwd=os.path.dirname(os.path.abspath(self.dataset_path)),
                    check=True,
                )
            else:
                raise FileNotFoundError(
                    f"Dataset not found at '{self.dataset_path}' and "
                    f"generator script '{gen_script}' is also missing."
                )

        self.raw_df = pd.read_csv(self.dataset_path)

        # --- clean -------------------------------------------------------
        self._clean_data()

        # --- user-item matrix --------------------------------------------
        self._build_user_item_matrix()

        # --- movie metadata lookup (title -> genre, language) ------------
        self._build_movie_metadata()

        # --- fit models --------------------------------------------------
        self.fit_user_similarity()
        self.fit_item_similarity()
        self.fit_svd()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _clean_data(self):
        """Drop anomalous rows, handle missing values, and normalise
        types for the training dataframe ``self.df``.
        """
        df = self.raw_df.copy()

        # Ensure is_anomaly column exists; if not, assume all rows clean
        if "is_anomaly" in df.columns:
            df = df[df["is_anomaly"] != 1].copy()

        # Drop rows with missing key fields
        df.dropna(subset=["user_id", "movie_title", "rating"], inplace=True)

        # Fill remaining NaNs in auxiliary columns
        df["genre"] = df["genre"].fillna("Unknown")
        df["language"] = df["language"].fillna("Unknown")
        df["release_year"] = df["release_year"].fillna(0).astype(int)

        # Clamp ratings to [1, 5]
        df["rating"] = df["rating"].clip(1.0, 5.0)

        # Keep only the latest rating per (user, movie) pair
        if "timestamp" in df.columns:
            df.sort_values("timestamp", inplace=True)
        df.drop_duplicates(subset=["user_id", "movie_title"], keep="last", inplace=True)

        self.df = df.reset_index(drop=True)

    def _build_user_item_matrix(self):
        """Pivot ``self.df`` into a user × movie rating matrix, filling
        unobserved entries with 0.
        """
        self.user_item_matrix = self.df.pivot_table(
            index="user_id",
            columns="movie_title",
            values="rating",
            aggfunc="mean",
        ).fillna(0)

    def _build_movie_metadata(self):
        """Create a fast lookup dict: movie_title -> {genre, language}."""
        meta = (
            self.df.drop_duplicates(subset=["movie_title"])
            .set_index("movie_title")[["genre", "language"]]
        )
        self.movie_meta = meta.to_dict(orient="index")

    def _movie_info(self, title: str) -> dict:
        """Return genre and language for *title*, defaulting to 'Unknown'."""
        info = self.movie_meta.get(title, {})
        return {
            "genre": info.get("genre", "Unknown"),
            "language": info.get("language", "Unknown"),
        }

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------
    def fit_user_similarity(self):
        """Compute user–user cosine similarity from the user-item matrix
        and store as ``self.user_similarity`` (DataFrame).
        """
        sim = cosine_similarity(self.user_item_matrix.values)
        self.user_similarity = pd.DataFrame(
            sim,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

    def fit_item_similarity(self):
        """Compute item–item cosine similarity from the transposed
        user-item matrix and store as ``self.item_similarity`` (DataFrame).
        """
        sim = cosine_similarity(self.user_item_matrix.values.T)
        self.item_similarity = pd.DataFrame(
            sim,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns,
        )

    def fit_svd(self, n_factors: int = 50):
        """Fit a latent-factor SVD model.

        Tries the *surprise* library first (full SVD with cross-validation
        on trainset).  Falls back to scikit-learn ``TruncatedSVD`` when
        *surprise* is not installed.

        Parameters
        ----------
        n_factors : int
            Number of latent factors.
        """
        try:
            from surprise import SVD, Dataset, Reader, accuracy

            reader = Reader(rating_scale=(1.0, 5.0))
            surprise_df = self.df[["user_id", "movie_title", "rating"]]
            data = Dataset.load_from_df(surprise_df, reader)
            trainset = data.build_full_trainset()

            algo = SVD(n_factors=min(n_factors, trainset.n_items - 1), random_state=42)
            algo.fit(trainset)

            # Evaluate on the training set (to mirror the spec)
            testset = trainset.build_testset()
            predictions = algo.test(testset)
            self._surprise_rmse = accuracy.rmse(predictions, verbose=False)

            # Build full predicted-rating matrix
            users = self.user_item_matrix.index
            movies = self.user_item_matrix.columns
            pred = np.zeros((len(users), len(movies)))
            for i, u in enumerate(users):
                for j, m in enumerate(movies):
                    pred[i, j] = algo.predict(u, m).est
            self.svd_predictions = pd.DataFrame(pred, index=users, columns=movies)
            self._svd_source = "surprise"

        except ImportError:
            # Fallback: TruncatedSVD from sklearn
            n_components = min(n_factors, min(self.user_item_matrix.shape) - 1)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            user_factors = svd.fit_transform(self.user_item_matrix.values)
            sigma = svd.singular_values_
            item_factors = svd.components_  # (n_components, n_movies)

            reconstructed = user_factors @ np.diag(sigma) @ item_factors
            # Clip to valid rating range
            reconstructed = np.clip(reconstructed, 1.0, 5.0)

            self.svd_predictions = pd.DataFrame(
                reconstructed,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.columns,
            )
            self._svd_source = "sklearn"

    # ------------------------------------------------------------------
    # Recommendation algorithms
    # ------------------------------------------------------------------
    def recommend_user_cf(self, user_id: str, k: int = 5) -> list[dict]:
        """User-based collaborative filtering.

        Finds the most similar users and predicts ratings for movies the
        target user has not yet rated via a similarity-weighted average.

        Parameters
        ----------
        user_id : str
            Target user identifier (e.g. ``"U0001"``).
        k : int
            Number of recommendations to return.

        Returns
        -------
        list[dict]
            Each dict: ``{movie_title, genre, language, predicted_rating}``.
        """
        if user_id not in self.user_item_matrix.index:
            return []

        # Similarities for target user (exclude self)
        sim_scores = self.user_similarity.loc[user_id].drop(user_id)

        # Top-N similar users (use at most 20 neighbours)
        n_neighbours = min(20, len(sim_scores))
        top_users = sim_scores.nlargest(n_neighbours)
        top_user_ids = top_users.index

        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index

        if len(unrated_movies) == 0:
            return []

        predictions = {}
        for movie in unrated_movies:
            neighbour_ratings = self.user_item_matrix.loc[top_user_ids, movie]
            # Only consider neighbours who actually rated this movie
            mask = neighbour_ratings > 0
            if mask.sum() == 0:
                continue
            sims = top_users[mask]
            rats = neighbour_ratings[mask]
            denom = sims.abs().sum()
            if denom == 0:
                continue
            predictions[movie] = (sims.values * rats.values).sum() / denom

        # Sort and take top-k
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for title, pred_rating in sorted_preds:
            info = self._movie_info(title)
            results.append(
                {
                    "movie_title": title,
                    "genre": info["genre"],
                    "language": info["language"],
                    "predicted_rating": round(float(pred_rating), 2),
                }
            )
        return results

    def recommend_item_cf(self, user_id: str, k: int = 5) -> list[dict]:
        """Item-based collaborative filtering.

        For each unrated movie, predict the rating using its similarity
        with movies the user has already rated.

        Parameters
        ----------
        user_id : str
            Target user identifier.
        k : int
            Number of recommendations to return.

        Returns
        -------
        list[dict]
            Each dict: ``{movie_title, genre, language, predicted_rating}``.
        """
        if user_id not in self.user_item_matrix.index:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index
        unrated_movies = user_ratings[user_ratings == 0].index

        if len(rated_movies) == 0 or len(unrated_movies) == 0:
            return []

        predictions = {}
        for movie in unrated_movies:
            if movie not in self.item_similarity.index:
                continue
            # Similarities between this unrated movie and all rated movies
            sims = self.item_similarity.loc[movie, rated_movies]
            # Use top-N most similar rated movies
            n_top = min(10, len(sims))
            top_sims = sims.nlargest(n_top)
            top_rated = user_ratings[top_sims.index]
            denom = top_sims.abs().sum()
            if denom == 0:
                continue
            predictions[movie] = (top_sims.values * top_rated.values).sum() / denom

        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for title, pred_rating in sorted_preds:
            info = self._movie_info(title)
            results.append(
                {
                    "movie_title": title,
                    "genre": info["genre"],
                    "language": info["language"],
                    "predicted_rating": round(float(pred_rating), 2),
                }
            )
        return results

    def recommend_svd(self, user_id: str, k: int = 5) -> list[dict]:
        """SVD latent-factor recommendations.

        Looks up predicted ratings from the pre-computed SVD
        reconstruction matrix for movies the user has not rated.

        Parameters
        ----------
        user_id : str
            Target user identifier.
        k : int
            Number of recommendations to return.

        Returns
        -------
        list[dict]
            Each dict: ``{movie_title, genre, language, predicted_rating}``.
        """
        if user_id not in self.svd_predictions.index:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index

        if len(unrated_movies) == 0:
            return []

        preds = self.svd_predictions.loc[user_id, unrated_movies].sort_values(
            ascending=False
        )[:k]

        results = []
        for title, pred_rating in preds.items():
            info = self._movie_info(title)
            results.append(
                {
                    "movie_title": title,
                    "genre": info["genre"],
                    "language": info["language"],
                    "predicted_rating": round(float(pred_rating), 2),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Unified recommend interface
    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: str,
        algo: str = "svd",
        k: int = 5,
        diversity: bool = True,
    ) -> list[dict]:
        """Generate top-*k* recommendations with optional diversity
        re-ranking.

        Parameters
        ----------
        user_id : str
            Target user identifier.
        algo : str
            Algorithm to use: ``'svd'``, ``'user_cf'``, or ``'item_cf'``.
        k : int
            Number of recommendations.
        diversity : bool
            If ``True``, apply genre-diversity re-ranking.

        Returns
        -------
        list[dict]
            Each dict: ``{movie_title, genre, language, predicted_rating,
            diversity_applied}``.
        """
        dispatch = {
            "svd": self.recommend_svd,
            "user_cf": self.recommend_user_cf,
            "item_cf": self.recommend_item_cf,
        }
        if algo not in dispatch:
            raise ValueError(
                f"Unknown algorithm '{algo}'. Choose from {list(dispatch.keys())}."
            )

        # Fetch a larger candidate pool when diversity is enabled so that
        # re-ranking has room to shuffle.
        pool_size = k * 3 if diversity else k
        candidates = dispatch[algo](user_id, k=pool_size)

        if not candidates:
            return []

        if diversity:
            candidates = self._diversity_rerank(candidates, k)

        # Trim to requested size and attach flag
        results = []
        for item in candidates[:k]:
            results.append(
                {
                    "movie_title": item["movie_title"],
                    "genre": item["genre"],
                    "language": item["language"],
                    "predicted_rating": round(item["predicted_rating"], 2),
                    "diversity_applied": diversity,
                }
            )
        return results

    @staticmethod
    def _diversity_rerank(candidates: list[dict], k: int) -> list[dict]:
        """Re-rank *candidates* using a genre-penalised score.

        ``adjusted_score = score / (1.0 + 0.3 * genre_count)``

        where *genre_count* is the number of times that genre already
        appears in the recommendation list above the current item.

        Parameters
        ----------
        candidates : list[dict]
            Candidate list (must contain ``predicted_rating`` and ``genre``).
        k : int
            Number of items to select.

        Returns
        -------
        list[dict]
            Re-ranked list of length min(k, len(candidates)).
        """
        remaining = list(candidates)
        selected: list[dict] = []
        genre_count: dict[str, int] = defaultdict(int)

        while remaining and len(selected) < k:
            best_idx = -1
            best_adj = -np.inf
            for idx, item in enumerate(remaining):
                g = item["genre"]
                adj = item["predicted_rating"] / (1.0 + 0.3 * genre_count[g])
                if adj > best_adj:
                    best_adj = adj
                    best_idx = idx
            chosen = remaining.pop(best_idx)
            genre_count[chosen["genre"]] += 1
            # Store the diversity-adjusted rating as the predicted_rating
            chosen = dict(chosen)
            chosen["predicted_rating"] = round(best_adj, 2)
            selected.append(chosen)

        return selected

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self) -> dict[str, dict]:
        """Evaluate all three algorithms on an 80/20 train/test split.

        Metrics per algorithm:
          • RMSE, MAE
          • Precision@5, Recall@5
          • Catalog Coverage (fraction of unique movies recommended /
            total movies)
          • Diversity Score (1 − mean intra-list genre similarity)

        Returns
        -------
        dict[str, dict]
            ``{algo_name: {rmse, mae, precision_at_5, recall_at_5,
            coverage, diversity_score}}``.
        """
        # 80/20 split
        train_df, test_df = train_test_split(
            self.df, test_size=0.2, random_state=42
        )

        # Build a temporary engine on the training fold
        tmp = _LightEngine(train_df, self.user_item_matrix.columns)

        # Ground truth grouped by user: {user_id: {movie: rating}}
        test_lookup: dict[str, dict[str, float]] = defaultdict(dict)
        for _, row in test_df.iterrows():
            test_lookup[row["user_id"]][row["movie_title"]] = row["rating"]

        all_movies = set(self.user_item_matrix.columns)
        algo_names = ["svd", "user_cf", "item_cf"]
        results: dict[str, dict] = {}

        for algo in algo_names:
            errors: list[float] = []
            abs_errors: list[float] = []
            precisions: list[float] = []
            recalls: list[float] = []
            recommended_movies: set[str] = set()
            intra_list_sims: list[float] = []

            for user_id in tmp.user_item_matrix.index:
                # Get recommendations from the training-fold model
                recs = tmp.recommend(user_id, algo=algo, k=5, diversity=False)
                rec_titles = [r["movie_title"] for r in recs]
                recommended_movies.update(rec_titles)

                # --- Precision / Recall ----------------------------------
                ground_truth_movies = set(test_lookup.get(user_id, {}).keys())
                # Consider a "relevant" movie as one the user rated >= 3.5
                relevant = {
                    m
                    for m, r in test_lookup.get(user_id, {}).items()
                    if r >= 3.5
                }
                rec_set = set(rec_titles)
                hits = rec_set & relevant
                if len(rec_set) > 0:
                    precisions.append(len(hits) / len(rec_set))
                if len(relevant) > 0:
                    recalls.append(len(hits) / len(relevant))

                # --- RMSE / MAE for test ratings we can predict ----------
                for movie, true_rating in test_lookup.get(user_id, {}).items():
                    pred_rating = tmp.predict_single(user_id, movie, algo)
                    if pred_rating is not None:
                        errors.append((pred_rating - true_rating) ** 2)
                        abs_errors.append(abs(pred_rating - true_rating))

                # --- Intra-list genre similarity -------------------------
                genres = [r["genre"] for r in recs]
                if len(genres) > 1:
                    unique_g = list(set(genres))
                    match_count = 0
                    pair_count = 0
                    for i in range(len(genres)):
                        for j in range(i + 1, len(genres)):
                            pair_count += 1
                            if genres[i] == genres[j]:
                                match_count += 1
                    intra_list_sims.append(
                        match_count / pair_count if pair_count else 0
                    )

            rmse = float(np.sqrt(np.mean(errors))) if errors else 0.0
            mae = float(np.mean(abs_errors)) if abs_errors else 0.0
            precision_at_5 = float(np.mean(precisions)) if precisions else 0.0
            recall_at_5 = float(np.mean(recalls)) if recalls else 0.0
            coverage = len(recommended_movies) / len(all_movies) if all_movies else 0.0
            diversity_score = (
                1.0 - float(np.mean(intra_list_sims)) if intra_list_sims else 0.0
            )

            results[algo] = {
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "precision_at_5": round(precision_at_5, 4),
                "recall_at_5": round(recall_at_5, 4),
                "coverage": round(coverage, 4),
                "diversity_score": round(diversity_score, 4),
            }

        return results

    # ------------------------------------------------------------------
    # Live update
    # ------------------------------------------------------------------
    def add_user_rating(
        self, user_id: str, movie_title: str, rating: float
    ) -> dict:
        """Append a new rating, persist to CSV, and refit all models.

        Parameters
        ----------
        user_id : str
            User identifier (existing or new).
        movie_title : str
            Movie title (must already exist in the catalog for metadata).
        rating : float
            Rating value in [1.0, 5.0].

        Returns
        -------
        dict
            Confirmation with status and details.
        """
        rating = float(np.clip(rating, 1.0, 5.0))
        info = self._movie_info(movie_title)

        new_row = {
            "user_id": user_id,
            "movie_title": movie_title,
            "genre": info["genre"],
            "language": info["language"],
            "release_year": 0,
            "rating": rating,
            "timestamp": pd.Timestamp.now().isoformat(),
            "is_anomaly": 0,
        }

        # Append to internal dataframe
        self.df = pd.concat(
            [self.df, pd.DataFrame([new_row])], ignore_index=True
        )
        # Keep latest per (user, movie)
        self.df.drop_duplicates(
            subset=["user_id", "movie_title"], keep="last", inplace=True
        )

        # Persist to CSV
        self.raw_df = pd.concat(
            [self.raw_df, pd.DataFrame([new_row])], ignore_index=True
        )
        self.raw_df.to_csv(self.dataset_path, index=False)

        # Rebuild everything
        self._build_user_item_matrix()
        self._build_movie_metadata()
        self.fit_user_similarity()
        self.fit_item_similarity()
        self.fit_svd()

        return {
            "status": "success",
            "user_id": user_id,
            "movie_title": movie_title,
            "rating": rating,
            "message": "Rating added and models re-fitted.",
        }

    # ------------------------------------------------------------------
    # Catalog queries
    # ------------------------------------------------------------------
    def get_all_users(self) -> list[str]:
        """Return a sorted list of all user IDs in the dataset."""
        return sorted(self.user_item_matrix.index.tolist())

    def get_all_movies(self) -> list[str]:
        """Return a sorted list of all movie titles in the dataset."""
        return sorted(self.user_item_matrix.columns.tolist())


# ======================================================================
# Lightweight internal engine used only inside evaluate()
# ======================================================================
class _LightEngine:
    """A stripped-down recommender built on an arbitrary training
    DataFrame — used internally by ``RecommenderEngine.evaluate()`` to
    avoid mutating the main engine's state.
    """

    def __init__(self, train_df: pd.DataFrame, all_movie_cols: pd.Index):
        self.df = train_df.copy()

        # Build user-item matrix aligned to the full movie catalog
        self.user_item_matrix = (
            self.df.pivot_table(
                index="user_id",
                columns="movie_title",
                values="rating",
                aggfunc="mean",
            )
            .reindex(columns=all_movie_cols, fill_value=0)
            .fillna(0)
        )

        # Movie metadata lookup
        meta = (
            self.df.drop_duplicates(subset=["movie_title"])
            .set_index("movie_title")[["genre", "language"]]
        )
        self.movie_meta = meta.to_dict(orient="index")

        # Similarities
        sim_u = cosine_similarity(self.user_item_matrix.values)
        self.user_similarity = pd.DataFrame(
            sim_u,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

        sim_i = cosine_similarity(self.user_item_matrix.values.T)
        self.item_similarity = pd.DataFrame(
            sim_i,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns,
        )

        # SVD
        n_components = min(50, min(self.user_item_matrix.shape) - 1)
        if n_components < 1:
            self.svd_predictions = self.user_item_matrix.copy()
        else:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            U = svd.fit_transform(self.user_item_matrix.values)
            sigma = svd.singular_values_
            Vt = svd.components_
            recon = np.clip(U @ np.diag(sigma) @ Vt, 1.0, 5.0)
            self.svd_predictions = pd.DataFrame(
                recon,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.columns,
            )

    def _movie_info(self, title: str) -> dict:
        info = self.movie_meta.get(title, {})
        return {
            "genre": info.get("genre", "Unknown"),
            "language": info.get("language", "Unknown"),
        }

    # --------------- prediction helpers --------------------------------
    def predict_single(
        self, user_id: str, movie: str, algo: str
    ) -> float | None:
        """Predict a single (user, movie) rating for a given algorithm."""
        if user_id not in self.user_item_matrix.index:
            return None
        if movie not in self.user_item_matrix.columns:
            return None

        if algo == "svd":
            return float(self.svd_predictions.loc[user_id, movie])

        if algo == "user_cf":
            if user_id not in self.user_similarity.index:
                return None
            sim_scores = self.user_similarity.loc[user_id].drop(user_id, errors="ignore")
            n_neighbours = min(20, len(sim_scores))
            top_users = sim_scores.nlargest(n_neighbours)
            neighbour_ratings = self.user_item_matrix.loc[top_users.index, movie]
            mask = neighbour_ratings > 0
            if mask.sum() == 0:
                return None
            sims = top_users[mask]
            rats = neighbour_ratings[mask]
            denom = sims.abs().sum()
            if denom == 0:
                return None
            return float((sims.values * rats.values).sum() / denom)

        if algo == "item_cf":
            user_ratings = self.user_item_matrix.loc[user_id]
            rated_movies = user_ratings[user_ratings > 0].index
            if len(rated_movies) == 0 or movie not in self.item_similarity.index:
                return None
            sims = self.item_similarity.loc[movie, rated_movies]
            n_top = min(10, len(sims))
            top_sims = sims.nlargest(n_top)
            top_rated = user_ratings[top_sims.index]
            denom = top_sims.abs().sum()
            if denom == 0:
                return None
            return float((top_sims.values * top_rated.values).sum() / denom)

        return None

    def recommend(
        self, user_id: str, algo: str = "svd", k: int = 5, diversity: bool = False
    ) -> list[dict]:
        """Simplified recommend for evaluation purposes."""
        if user_id not in self.user_item_matrix.index:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]
        unrated = user_ratings[user_ratings == 0].index

        preds = {}
        for movie in unrated:
            p = self.predict_single(user_id, movie, algo)
            if p is not None:
                preds[movie] = p

        sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for title, pred_rating in sorted_preds:
            info = self._movie_info(title)
            results.append(
                {
                    "movie_title": title,
                    "genre": info["genre"],
                    "language": info["language"],
                    "predicted_rating": round(float(pred_rating), 2),
                }
            )
        return results


# ======================================================================
# Demo / CLI
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Movie Recommender Engine — Demo")
    print("=" * 60)

    engine = RecommenderEngine("dataset.csv")

    users = engine.get_all_users()
    movies = engine.get_all_movies()
    print(f"\nDataset loaded: {len(users)} users, {len(movies)} movies")
    print(f"Training rows : {len(engine.df)}")

    sample_user = users[0]
    print(f"\n--- Recommendations for {sample_user} ---\n")

    for algo in ("svd", "user_cf", "item_cf"):
        recs = engine.recommend(sample_user, algo=algo, k=5, diversity=True)
        print(f"[{algo.upper()}]")
        for i, r in enumerate(recs, 1):
            print(
                f"  {i}. {r['movie_title']:30s} | {r['genre']:12s} | "
                f"{r['language']:10s} | ★ {r['predicted_rating']:.2f}"
            )
        print()

    print("--- Evaluation Metrics ---\n")
    metrics = engine.evaluate()
    header = f"{'Algorithm':>10s}  {'RMSE':>7s}  {'MAE':>7s}  {'P@5':>7s}  {'R@5':>7s}  {'Cov':>7s}  {'Div':>7s}"
    print(header)
    print("-" * len(header))
    for algo, m in metrics.items():
        print(
            f"{algo:>10s}  {m['rmse']:7.4f}  {m['mae']:7.4f}  "
            f"{m['precision_at_5']:7.4f}  {m['recall_at_5']:7.4f}  "
            f"{m['coverage']:7.4f}  {m['diversity_score']:7.4f}"
        )
    print()
