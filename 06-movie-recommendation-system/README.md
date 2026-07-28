# 🎬 Movie Recommendation System

A web-based Movie Recommendation System built using **Python**, **Flask**, **Pandas**, and **Scikit-learn**. The application recommends similar movies based on genre similarity using **TF-IDF Vectorization** and **Cosine Similarity**.

## 📌 Project Overview

The Movie Recommendation System helps users discover movies similar to their selected movie. It uses a content-based recommendation approach by analyzing movie genres and calculating similarity scores between movies.

The project features a clean web interface built with Flask and can be deployed online using Render.

---

## 🚀 Features

* Movie recommendation based on content similarity
* User-friendly web interface
* Fast recommendation generation
* Content-Based Filtering
* Flask web application
* Ready for deployment on Render

---

## 🛠️ Tech Stack

* Python 3
* Flask
* Pandas
* NumPy
* Scikit-learn
* HTML5
* CSS3
* Gunicorn
* Render

---

## 📂 Project Structure

```text
Movie-Recommendation-System/
│── app.py
│── recommender.py
│── movies.csv
│── requirements.txt
│── Procfile
│── README.md
│── templates/
│     └── index.html
│── static/
│     └── style.css
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/AkshatGarg2005/Movie-Recommendation-System
```

Navigate to the project folder:

```bash
cd Movie-Recommendation-System
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Flask server:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 🧠 Recommendation Algorithm

This project uses **Content-Based Filtering**.

### Workflow

1. Load the movie dataset.
2. Extract movie genres.
3. Convert genres into TF-IDF vectors.
4. Compute cosine similarity between movies.
5. Recommend the top five most similar movies.

---

## 📊 Dataset

The project uses the **MovieLens Latest Small Dataset** provided by GroupLens Research.

Dataset includes:

* Movie titles
* Genres
* Ratings

---

## 🌐 Deployment

This application can be deployed on **Render** using:

* **Build Command**

```text
pip install -r requirements.txt
```

* **Start Command**

```text
gunicorn app:app
```

---

## 📈 Future Improvements

* Movie posters
* Search autocomplete
* TMDb API integration
* User authentication
* Personalized recommendations
* Rating prediction
* Responsive UI
* Dark mode

---

## 📄 License

This project is developed for educational and learning purposes.
