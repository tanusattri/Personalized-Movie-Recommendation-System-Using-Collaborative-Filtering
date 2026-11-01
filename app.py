"""
Flask Web Application for Movie Recommendation System.
HCLTech Career Shaper Programme | Project #28 | Academic Year 2025-26

Provides REST API endpoints for movie recommendations using collaborative
filtering (SVD, User-Based CF, Item-Based CF) and a responsive Bootstrap 5 web UI.
"""

import sys
import os
import logging
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# Ensure the project directory is on sys.path so we can import recommender.py
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from recommender import RecommenderEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Application Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Instantiate the recommender engine once at startup
app.logger.info("Initializing RecommenderEngine...")
engine = RecommenderEngine()

# Pre-compute evaluation metrics for all algorithms
app.logger.info("Pre-computing evaluation metrics...")
try:
    _precomputed_metrics = engine.evaluate()
except Exception as exc:
    app.logger.warning("Could not pre-compute metrics: %s", exc)
    _precomputed_metrics = {}

app.logger.info("Startup complete – engine and metrics ready.")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_metrics() -> dict:
    """Return the pre-computed metrics dict (recompute if empty)."""
    global _precomputed_metrics
    if not _precomputed_metrics:
        try:
            _precomputed_metrics = engine.evaluate()
        except Exception:
            _precomputed_metrics = {}
    return _precomputed_metrics


def _default_recommendations(user_id, k: int = 5):
    """Get default SVD recommendations for the landing page."""
    try:
        return engine.recommend(str(user_id), algo="svd", k=k, diversity=True)
    except Exception as e:
        app.logger.warning(f"Error generating default recs: {e}")
        return []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the main single-page application."""
    users = engine.get_all_users()
    movies = engine.get_all_movies()
    metrics = _safe_metrics()

    # Default recommendations for the first user
    default_user = users[0] if users else "U0001"
    default_recs = _default_recommendations(default_user)

    return render_template(
        "index.html",
        users=users,
        movies=movies,
        metrics=metrics,
        default_recs=default_recs,
        default_user=default_user,
    )


@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    """
    Return movie recommendations as JSON.

    Query params:
        user_id   - required (e.g. U0001)
        algo      - svd | user | item (default: svd)
        k         - 3-10 (default: 5)
        diversity - true | false (default: true)
    """
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing required parameter: user_id"}), 400
    
    user_id = str(user_id).strip()
    if user_id.isdigit():
        user_id = f"U{int(user_id):04d}"

    # Normalize algorithm identifier
    algo_key = request.args.get("algo", "svd").lower().strip()
    if algo_key in ("user", "user_cf", "user-cf"):
        algo = "user_cf"
    elif algo_key in ("item", "item_cf", "item-cf"):
        algo = "item_cf"
    elif algo_key == "svd":
        algo = "svd"
    else:
        return jsonify({"error": f"Invalid algo: {algo_key}. Choose from svd, user, item."}), 400

    # Number of recommendations K
    try:
        k = int(request.args.get("k", 5))
    except (ValueError, TypeError):
        k = 5
    k = max(3, min(10, k))

    # Diversity toggle
    diversity_str = request.args.get("diversity", "true").lower()
    diversity = diversity_str in ("true", "1", "yes")

    # Generate recommendations
    try:
        recs = engine.recommend(user_id, algo=algo, k=k, diversity=diversity)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "recommendations": recs,
        "user_id": user_id,
        "algorithm": algo,
        "k": k,
        "diversity_applied": diversity
    })


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return evaluation metrics for all three algorithms."""
    metrics = _safe_metrics()
    return jsonify(metrics)


@app.route("/api/rate", methods=["POST"])
def api_rate():
    """
    Accept a new user rating dynamically.

    JSON body: {user_id: str, movie_title: str, rating: float}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    user_id = data.get("user_id")
    movie_title = data.get("movie_title")
    rating = data.get("rating")

    if user_id is None or movie_title is None or rating is None:
        return jsonify({"error": "Missing required fields: user_id, movie_title, rating."}), 400

    user_id = str(user_id).strip()
    if user_id.isdigit():
        user_id = f"U{int(user_id):04d}"

    try:
        rating = float(rating)
    except (ValueError, TypeError):
        return jsonify({"error": "rating must be a number."}), 400

    if not (1.0 <= rating <= 5.0):
        return jsonify({"error": "rating must be between 1.0 and 5.0."}), 400

    try:
        res = engine.add_user_rating(user_id, str(movie_title).strip(), rating)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "status": "success",
        "message": f"Rating of {rating} for '{movie_title}' by user {user_id} recorded successfully.",
        "user_id": user_id,
        "movie_title": movie_title,
        "rating": rating,
        "details": res
    })


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
