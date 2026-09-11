"""
recommender.py
----------------
Content-based movie recommendation engine.

How it works
============
1. For every movie, combine its genres, keywords, director, cast, and
   overview into one text "soup".
2. Convert every movie's soup into a TF-IDF vector (a numeric fingerprint
   of which words matter most for that movie, downweighting common words).
3. Compare every pair of movies with cosine similarity — a score from 0
   (nothing in common) to 1 (identical) based on the angle between their
   TF-IDF vectors.
4. To recommend movies similar to "X", look up X's row in the similarity
   matrix and return the highest-scoring movies other than X itself.

This is a self-contained, offline demo: no external API or account is
required, and the bundled dataset (data/movies.csv) has 40 well-known
films so the recommendations are easy to sanity-check.
"""

import difflib
import os
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "movies.csv")


def load_movies(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the movie dataset and make sure there are no missing fields."""
    df = pd.read_csv(path)
    text_columns = ["genres", "director", "cast", "keywords", "overview"]
    for col in text_columns:
        df[col] = df[col].fillna("")
    return df


def _build_soup(row: pd.Series) -> str:
    """Combine a movie's metadata into a single text blob for TF-IDF."""
    # Repeating genres/keywords gives them a bit more weight than the
    # free-text overview, since they're a stronger, cleaner similarity signal.
    return " ".join(
        [
            row["genres"],
            row["genres"],
            row["keywords"],
            row["keywords"],
            row["director"],
            row["cast"],
            row["overview"],
        ]
    )


def build_similarity_matrix(df: pd.DataFrame):
    """Return the movie x movie cosine similarity matrix for the dataset."""
    soup = df.apply(_build_soup, axis=1)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(soup)
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def _closest_title(title: str, titles: list) -> Optional[str]:
    """Case-insensitive exact match, falling back to fuzzy matching so
    small typos ('interstelar') still resolve to the right movie."""
    lower_map = {t.lower(): t for t in titles}
    if title.lower() in lower_map:
        return lower_map[title.lower()]

    close = difflib.get_close_matches(title.lower(), lower_map.keys(), n=1, cutoff=0.5)
    return lower_map[close[0]] if close else None


def recommend(title: str, df: pd.DataFrame = None, sim_matrix=None, top_n: int = 10):
    """
    Recommend up to `top_n` movies similar to `title`.

    Returns a tuple: (matched_title_or_None, list_of_recommendation_dicts)
    """
    if df is None:
        df = load_movies()
    if sim_matrix is None:
        sim_matrix = build_similarity_matrix(df)

    titles = df["title"].tolist()
    matched = _closest_title(title, titles)
    if matched is None:
        return None, []

    idx = df.index[df["title"] == matched][0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda pair: pair[1], reverse=True)
    scores = [pair for pair in scores if pair[0] != idx][:top_n]

    results = []
    for i, score in scores:
        movie = df.iloc[i]
        results.append(
            {
                "title": movie["title"],
                "year": int(movie["year"]),
                "genres": movie["genres"],
                "director": movie["director"],
                "score": round(float(score), 3),
            }
        )
    return matched, results


def _print_recommendations(title: str, top_n: int) -> None:
    df = load_movies()
    sim_matrix = build_similarity_matrix(df)
    matched, results = recommend(title, df, sim_matrix, top_n=top_n)

    if matched is None:
        print(f'No close match found for "{title}". Try a different title.')
        return

    if matched.lower() != title.lower():
        print(f'Showing results for "{matched}" (closest match to "{title}")\n')
    else:
        print(f"Because you watched: {matched}\n")

    for rank, movie in enumerate(results, start=1):
        print(
            f"{rank:>2}. {movie['title']} ({movie['year']}) — {movie['genres']}"
            f"  [similarity: {movie['score']}]"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Get movie recommendations using TF-IDF + cosine similarity."
    )
    parser.add_argument("title", help="A movie title from the dataset, e.g. 'Inception'")
    parser.add_argument(
        "-n", "--top_n", type=int, default=10, help="Number of recommendations to show"
    )
    args = parser.parse_args()
    _print_recommendations(args.title, args.top_n)
