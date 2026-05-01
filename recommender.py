"""
recommender.py
==============
Core recommendation engine for the Bollywood Movie Recommendation System.

Techniques used:
  - TF-IDF Vectorization  (scikit-learn)
  - Cosine Similarity      (scikit-learn)
  - Content-based filtering on a combined "soup" of text features
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "data.csv")


# ─────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────
def load_data(filepath: str = DATA_FILE) -> pd.DataFrame:
    """
    Load the Bollywood dataset from CSV.
    Applies cleaning, handles missing values, and engineers the feature soup.
    """
    df = pd.read_csv(filepath)

    # ── Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── Fill missing values with empty strings for text fields
    text_cols = ["genre", "cast", "director", "keywords", "overview"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # ── Fill numeric missing values
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(df["rating"].median())
    df["year"]   = pd.to_numeric(df["year"],   errors="coerce").fillna(2000).astype(int)

    # ── Clean text: lowercase, strip whitespace
    for col in text_cols:
        df[col] = df[col].str.lower().str.strip()

    # ── Feature Engineering: build a combined "soup" for TF-IDF
    df["soup"] = df.apply(_build_soup, axis=1)

    return df.reset_index(drop=True)


def _build_soup(row: pd.Series) -> str:
    """
    Combine relevant text columns into a single string for TF-IDF vectorisation.
    Director and keywords are repeated to boost their signal weight.
    """
    parts = [
        row.get("genre", "").replace(",", " "),
        row.get("director", "").replace(" ", "_"),   # treat as single token
        row.get("director", "").replace(" ", "_"),   # boosted
        row.get("cast", "").replace(",", " "),
        row.get("keywords", "").replace(",", " "),
        row.get("keywords", "").replace(",", " "),   # boosted
        row.get("overview", ""),
    ]
    return " ".join(p for p in parts if p)


# ─────────────────────────────────────────────
# 2. BUILD RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def build_recommendation_engine(df: pd.DataFrame):
    """
    Fit a TF-IDF vectoriser on the soup column and compute the cosine
    similarity matrix.

    Returns:
        cosine_sim  – (n_movies × n_movies) cosine-similarity array
        indices     – dict mapping movie title → DataFrame index
    """
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams for richer matching
        max_features=5_000,
        min_df=1,
    )

    tfidf_matrix = tfidf.fit_transform(df["soup"])
    cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Build title → index lookup (handles duplicate titles gracefully)
    indices = pd.Series(df.index, index=df["title"]).to_dict()

    return cosine_sim, indices


# ─────────────────────────────────────────────
# 3. RECOMMENDATION FUNCTIONS
# ─────────────────────────────────────────────

def get_content_recommendations(
    title: str,
    df: pd.DataFrame,
    cosine_sim: np.ndarray,
    indices: dict,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the top-n movies most similar to the given title
    using cosine similarity on the TF-IDF soup matrix.
    """
    idx = indices.get(title)
    if idx is None:
        return pd.DataFrame()   # title not found – return empty

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : n + 1]   # exclude the movie itself

    movie_indices = [i[0] for i in sim_scores]
    result = df.iloc[movie_indices].copy()
    result["similarity"] = [round(s[1] * 100, 1) for s in sim_scores]
    return result.reset_index(drop=True)


def get_genre_recommendations(
    df: pd.DataFrame,
    genres: list,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return top-n highest-rated movies that match ANY of the selected genres.
    """
    if not genres:
        return pd.DataFrame()

    genre_str = "|".join(g.lower() for g in genres)
    mask = df["genre"].str.contains(genre_str, case=False, na=False)
    filtered = df[mask].copy()

    if filtered.empty:
        return pd.DataFrame()

    return filtered.sort_values("rating", ascending=False).head(n).reset_index(drop=True)


def get_actor_director_recommendations(
    df: pd.DataFrame,
    actor: str | None = None,
    director: str | None = None,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return top-n highest-rated movies featuring the specified actor and/or director.
    If both are provided both filters are applied (AND logic).
    """
    filtered = df.copy()

    if actor:
        filtered = filtered[
            filtered["cast"].str.contains(actor.lower(), case=False, na=False)
        ]
    if director:
        filtered = filtered[
            filtered["director"].str.contains(director.lower(), case=False, na=False)
        ]

    if filtered.empty:
        return pd.DataFrame()

    return filtered.sort_values("rating", ascending=False).head(n).reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. TOP-RATED & TRENDING
# ─────────────────────────────────────────────

def get_top_rated_movies(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the n highest-rated movies (all-time)."""
    return df.sort_values("rating", ascending=False).head(n).reset_index(drop=True)


def get_trending_movies(df: pd.DataFrame, n: int = 10, recent_years: int = 5) -> pd.DataFrame:
    """
    Return n 'trending' movies: recent releases (within last `recent_years` years)
    sorted by rating descending.
    """
    max_year = df["year"].max()
    recent = df[df["year"] >= max_year - recent_years].copy()
    if recent.empty:
        recent = df.copy()
    return recent.sort_values("rating", ascending=False).head(n).reset_index(drop=True)
