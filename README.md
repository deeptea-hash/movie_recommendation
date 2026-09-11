# CineMatch — Movie Recommender (Cosine Similarity)

A content-based movie recommendation engine built with **TF-IDF vectorization** and **cosine similarity**, wrapped in a small Flask web app and a command-line tool. No external API, account, or internet connection needed — everything runs on a bundled sample dataset.

## How the recommendations work

1. **Combine features** — for every movie, its genres, keywords, director, cast, and a short overview are combined into one text "soup."
2. **Vectorize with TF-IDF** — `TfidfVectorizer` (scikit-learn) turns each movie's soup into a numeric vector, weighting distinctive words more heavily than common ones.
3. **Score similarity** — `cosine_similarity` compares every pair of movie vectors and returns a score from `0` (nothing in common) to `1` (identical), based on the angle between the vectors rather than their magnitude.
4. **Recommend** — for a chosen movie, the engine returns the highest-scoring movies other than the one you picked.

This is the same technique used in the "Because you watched..." style of recommendation, just applied to metadata instead of user-viewing history (that would be *collaborative* filtering, a different approach).

## Project structure

```
movie-recommender/
├── app.py                 Flask web app
├── recommender.py          Core TF-IDF + cosine similarity engine (also runnable as a CLI)
├── data/
│   └── movies.csv          Sample dataset — 40 movies with genre/cast/keyword metadata
├── templates/
│   └── index.html          Web UI template
├── static/
│   └── style.css           Web UI styling
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# 1. Clone or download this project, then move into it
cd movie-recommender

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Type a movie title (an autocomplete dropdown suggests matches from the dataset) and submit to see the top 8 most similar movies, each with a similarity percentage.

## Running the command-line version

```bash
python recommender.py "Inception"
python recommender.py "interstelar" -n 5    # typos are fuzzy-matched, and -n controls result count
```

Example output:

```
Because you watched: Inception

 1. The Matrix (1999) — Sci-Fi Action                [similarity: 0.171]
 2. Avatar (2009) — Sci-Fi Adventure Action           [similarity: 0.092]
 3. A Quiet Place (2018) — Horror Sci-Fi Thriller     [similarity: 0.09]
 4. Get Out (2017) — Horror Mystery Thriller          [similarity: 0.087]
 5. Black Panther (2018) — Action Sci-Fi Adventure    [similarity: 0.085]
```

## Using your own dataset

Replace `data/movies.csv` with a larger dataset — for example [MovieLens](https://grouplens.org/datasets/movielens/) or [TMDB 5000](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) — as long as it keeps (or you remap) these columns:

| Column | Description |
|---|---|
| `title` | Movie title |
| `year` | Release year |
| `genres` | Space-separated genres |
| `director` | Director name(s) |
| `cast` | Space-separated lead cast names |
| `keywords` | Space-separated theme/plot keywords |
| `overview` | A short plot description |

No code changes are required — `recommender.py` rebuilds the TF-IDF matrix from whatever is in `data/movies.csv` at startup. For datasets with thousands of rows, consider caching the similarity matrix (e.g. with `pickle` or `joblib`) instead of recomputing it every time the app starts.

## Running it in VS Code

1. Open the `movie-recommender` folder in VS Code.
2. Select the Python interpreter for your virtual environment (`Ctrl/Cmd+Shift+P` → "Python: Select Interpreter").
3. Open a terminal in VS Code and run `python app.py` or `python recommender.py "<title>"` as above.

## Pushing this project to GitHub

```bash
git init
git add .
git commit -m "Initial commit: CineMatch movie recommender"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

`.gitignore` already excludes `venv/`, `__pycache__/`, and other local-only files, so your repo stays clean.

## Notes and possible extensions

- This is **content-based filtering** (similar items by metadata), not collaborative filtering (similar users' behavior) — a good next step would be adding user ratings and blending both approaches.
- The bundled dataset is intentionally small (40 movies) so results are easy to verify by eye; swap in a larger dataset for production use.
- The Flask app rebuilds the TF-IDF matrix at startup, which is instant for 40 rows but would need caching for a dataset of, say, 50,000+ movies.
