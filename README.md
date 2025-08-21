
# 🎬 Movie Recommender System

A movie recommendation system combining **Collaborative Filtering** and **AI-powered Content-Based Search**.

---

## Features

- **Collaborative Filtering:** Personalized recommendations based on user ratings and similarity between movies.
- **Content-Based Search:** Search movies by title or natural language descriptions using Sentence-Transformers (`all-roberta-large-v1`).
- **Responsive Frontend:** Simple UI with search modes and rating functionality.
- **Backend:** Flask REST API serving search and recommendation endpoints.

---

## Project Structure

```
/
├── model/                  # Pretrained models and data (not included)
├── app.py                  # Flask backend
├── index.html              # Frontend UI
└── README.md
````

---

## Getting Started

### Prerequisites

- Python 3.8+
- Node or any HTTP server to serve `index.html` (optional, can open directly in browser)
- Required Python packages:


```bash
  pip install flask flask-cors numpy pandas scipy scikit-learn sentence-transformers
```

### Running the Backend

```bash
python app.py
```

The API will be available at `http://localhost:5000`.

### Running the Frontend

Open `index.html` in your browser. Ensure backend is running on `localhost:5000`.

---

## Important Notes

* *Model files and datasets are NOT included* due to size and licensing.
* You need to provide your own trained models and data files under the `model/` directory as expected by the backend.

---

## Technologies Used

* Flask, Flask-CORS
* Sentence-Transformers (`all-roberta-large-v1`)
* Numpy, Pandas, Scipy, scikit-learn
* Vanilla JavaScript, HTML, CSS

---


😊 THANK YOU 😊
