
from pathlib import Path
import json
import joblib
import pandas as pd


class MovieRecommender:
    """Load saved model artifacts and provide recommendation operations."""

    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)

        required_files = [
            "best_model.joblib",
            "movies_clean.pkl",
            "ratings_model.pkl",
            "model_comparison.csv",
            "ranking_metrics.csv",
            "metadata.json",
        ]

        missing = [
            name for name in required_files
            if not (self.model_dir / name).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing files in {self.model_dir.resolve()}: "
                + ", ".join(missing)
            )

        # Load the trained Surprise SVD model.
        self.model = joblib.load(
            self.model_dir / "best_model.joblib"
        )

        # Load movie metadata, ratings, and evaluation results.
        self.movies = pd.read_pickle(
            self.model_dir / "movies_clean.pkl"
        )
        self.ratings = pd.read_pickle(
            self.model_dir / "ratings_model.pkl"
        )
        self.comparison = pd.read_csv(
            self.model_dir / "model_comparison.csv"
        )
        self.ranking_metrics = pd.read_csv(
            self.model_dir / "ranking_metrics.csv"
        )

        with open(
            self.model_dir / "metadata.json",
            encoding="utf-8"
        ) as file:
            self.metadata = json.load(file)

        # Ensure IDs have consistent integer types.
        self.movies["movieId"] = (
            pd.to_numeric(self.movies["movieId"])
            .astype(int)
        )
        self.ratings["movieId"] = (
            pd.to_numeric(self.ratings["movieId"])
            .astype(int)
        )
        self.ratings["userId"] = (
            pd.to_numeric(self.ratings["userId"])
            .astype(int)
        )

        self.movies = (
            self.movies.drop_duplicates("movieId")
            .reset_index(drop=True)
        )

        self.user_ids = set(
            self.ratings["userId"].unique()
        )
        self.movie_ids = set(
            self.movies["movieId"].unique()
        )

        self.movie_lookup = (
            self.movies.set_index("movieId")["title"]
            .to_dict()
        )

    def user_history(self, user_id):
        """Return the movies rated by a user."""
        user_id = int(user_id)

        history = self.ratings[
            self.ratings["userId"] == user_id
        ].copy()

        if history.empty:
            history["title"] = pd.Series(dtype=str)
            return history

        history["title"] = history["movieId"].map(
            self.movie_lookup
        )

        return history.sort_values(
            "rating", ascending=False
        )

    def recommend_movies(self, user_id, n=10, genre=None):
        """Rank movies the user has not rated in the model sample."""
        user_id = int(user_id)

        if user_id not in self.user_ids:
            raise ValueError(
                "This user ID is not present in the "
                "training sample."
            )

        seen_movies = set(
            self.ratings.loc[
                self.ratings["userId"] == user_id,
                "movieId"
            ]
        )

        candidates = self.movies[
            ~self.movies["movieId"].isin(seen_movies)
        ].copy()

        if genre:
            candidates = candidates[
                candidates["genres"]
                .fillna("")
                .str.contains(
                    str(genre),
                    case=False,
                    regex=False
                )
            ]

        if candidates.empty:
            return candidates

        # Use the trained model to predict candidate ratings.
        candidates["predicted_rating"] = (
            candidates["movieId"].map(
                lambda movie_id: float(
                    self.model.predict(
                        user_id, int(movie_id)
                    ).est
                )
            )
        )

        return (
            candidates.sort_values(
                "predicted_rating",
                ascending=False
            )
            .head(int(n))
            .reset_index(drop=True)
        )

    def predict_rating(self, user_id, movie_id):
        """Estimate a user's rating for a movie."""
        user_id = int(user_id)
        movie_id = int(movie_id)

        if user_id not in self.user_ids:
            raise ValueError(
                "User ID is not present in the training sample."
            )

        if movie_id not in self.movie_ids:
            raise ValueError(
                "Movie ID was not found in the catalogue."
            )

        prediction = self.model.predict(
            user_id, movie_id
        )

        return {
            "title": self.movie_lookup.get(
                movie_id, "Unknown movie"
            ),
            "estimated_rating": float(prediction.est),
        }

    def similar_movies_by_genre(self, movie_id, n=10):
        """Find movies with similar genre sets using Jaccard similarity."""
        movie_id = int(movie_id)

        selected = self.movies[
            self.movies["movieId"] == movie_id
        ]

        if selected.empty:
            raise ValueError("Movie ID was not found.")

        target_genres = (
            set(str(selected.iloc[0]["genres"]).split("|"))
            - {"", "(no genres listed)"}
        )

        if not target_genres:
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "title",
                    "genres",
                    "genre_similarity",
                ]
            )

        candidates = self.movies[
            self.movies["movieId"] != movie_id
        ].copy()

        def jaccard_similarity(genres):
            other_genres = (
                set(str(genres).split("|"))
                - {"", "(no genres listed)"}
            )

            union = target_genres | other_genres

            if not union:
                return 0.0

            return (
                len(target_genres & other_genres)
                / len(union)
            )

        candidates["genre_similarity"] = (
            candidates["genres"]
            .fillna("")
            .apply(jaccard_similarity)
        )

        return (
            candidates.sort_values(
                ["genre_similarity", "title"],
                ascending=[False, True]
            )
            .head(int(n))
            .reset_index(drop=True)
        )

    def get_model_comparison(self):
        """Return saved prediction-model comparison metrics."""
        return self.comparison.copy()

    def get_ranking_metrics(self):
        """Return saved recommendation-ranking metrics."""
        return self.ranking_metrics.copy()
