"""
app.py
------
A small Flask front-end for the cosine-similarity movie recommender.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request

from recommender import build_similarity_matrix, load_movies, recommend

app = Flask(__name__)

# Loaded once at startup — with only 40 movies this is instant, but for a
# larger dataset you'd cache this matrix to disk instead of rebuilding it
# on every restart.
MOVIES_DF = load_movies()
SIMILARITY_MATRIX = build_similarity_matrix(MOVIES_DF)
ALL_TITLES = sorted(MOVIES_DF["title"].tolist())


@app.route("/", methods=["GET"])
def index():
    query = request.args.get("title", "").strip()
    matched, results, error = None, [], None

    if query:
        matched, results = recommend(
            query, df=MOVIES_DF, sim_matrix=SIMILARITY_MATRIX, top_n=8
        )
        if matched is None:
            error = f'No close match found for "{query}". Try another title from the list.'

    return render_template(
        "index.html",
        titles=ALL_TITLES,
        query=query,
        matched=matched,
        results=results,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
