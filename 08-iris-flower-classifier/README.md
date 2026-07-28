# Iris Flower Classifier API & Web UI

[cite_start]A streamlined MLOps pipeline that trains a Random Forest model using scikit-learn, wraps it in a Flask web API, and containerizes the entire environment with Docker[cite: 1, 2]. It includes an elegant front-end interface built with Tailwind CSS for making real-time interactive predictions.

## Developer Info
* **Name:** Akshat Garg
* **Registration Number:** 23BCE10641

---

## Project Structure
[cite_start]The repository contains the following core files and directories [cite: 3, 11-16]:
```text
iris-api/
├── templates/
│   └── index.html      # Frontend user interface
├── app.py              # Flask API server & inference endpoints
├── train.py            # Model training and serialization script
├── iris_model.pkl      # Serialized random forest model weights
├── Dockerfile          # Container environment configuration
├── requirements.txt    # Application dependencies
└── .gitignore          # Version control file exclusions

```

---

## Local Setup & Installation

### 1. Environment Configuration

Create and activate an isolated development environment using Conda:

```bash
conda create -n iris-mlops python=3.12 -y
conda activate iris-mlops

```

### 2. Dependency Installation

Install the necessary project dependencies listed in the requirements file:

```bash
pip install -r requirements.txt

```

* Dependencies include `flask`, `gunicorn`, `scikit-learn`, `joblib`, and `numpy` .

---

## Execution Guide

### 1. Train the Model

Generate a freshly trained model binary by executing the training script:

```bash
python train.py

```

* This loads the native scikit-learn Iris dataset, trains a classification model, and dumps it to `iris_model.pkl` .



### 2. Run the App Locally

Fire up the Flask server in your terminal environment:

```bash
PORT=5001 python app.py

```

* Open `http://127.0.0.1:5001` in your web browser to access the interactive web dashboard.

---

## Running with Docker

### 1. Build the Docker Image

Compile the isolated application runtime using the Dockerfile configuration:

```bash
docker build -t iris-api .

```

### 2. Start the Container

Spin up the container instance by mapping the internal environment port to your local machine:

```bash
docker run -e PORT=5001 -p 5001:5001 iris-api

```

* Access the web platform natively at `http://localhost:5001`.

---

## API Documentation

### Health Check Route

* **Endpoint:** `GET /`
* **Description:** Renders the frontend UI form dashboard.

### Prediction Route

* 
**Endpoint:** `POST /predict` 


* 
**Headers:** `Content-Type: application/json` 


* 
**Payload Format:** 


```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

```


* 
**Success Response (200 OK):** 


```json
{
  "class": "Setosa"
}

```
