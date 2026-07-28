from flask import Flask, render_template, request
from recommender import recommend, get_movie_titles

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    selected_movie = ""

    if request.method == "POST":
        selected_movie = request.form["movie"]
        recommendations = recommend(selected_movie)

    movies = get_movie_titles()

    return render_template(
        "index.html",
        movies=movies,
        recommendations=recommendations,
        selected_movie=selected_movie
    )


if __name__ == "__main__":
    if __name__ == "__main__":
        app.run()