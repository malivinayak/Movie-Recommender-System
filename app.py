from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from scipy.sparse import load_npz
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Libraries loaded")

app = Flask(__name__)
CORS(app)

# --- Collaborative Filtering ---
print("🔄 Loading Collaborative Filtering Data...")
similarity_matrix = load_npz("./model/collaborative/similarity_matrix.npz")
with open("./model/collaborative/movie_mapper.pkl", "rb") as f: movie_mapper = pickle.load(f)
with open("./model/collaborative/movie_inverse_mapper.pkl", "rb") as f: movie_inverse_mapper = pickle.load(f)
with open("./model/collaborative/movie_metadata.pkl", "rb") as f: movie_metadata = pickle.load(f)
movie_metadata = movie_metadata.set_index('imdbId')
print("✅ Collaborative model loaded.")

def normalize(r):
    return ((r - 0.5) / (5.0 - 0.5)) * 10 - 5

def recommend_collaborative(input_movies, top_n=10):
    print(f"🎯 Generating collaborative recommendations for: {input_movies}")
    scored_rows, weights = [], []

    for imdb_id, rating in input_movies.items():
        if imdb_id in movie_mapper:
            idx = movie_mapper[imdb_id]
            scored_rows.append(similarity_matrix[idx])
            weights.append(normalize(rating))
            print(f"  ✔️ Found mapping for {imdb_id} with normalized rating {normalize(rating)}")
        else:
            print(f"  ⚠️ IMDb ID {imdb_id} not found in movie mapper.")

    if not scored_rows:
        print("❌ No valid movies provided. Returning empty list.")
        return []

    # Weighted sum
    sim_matrix_subset = scored_rows[0].copy().multiply(weights[0])
    for i in range(1, len(scored_rows)):
        sim_matrix_subset += scored_rows[i].multiply(weights[i])

    total_scores = sim_matrix_subset.toarray().flatten()
    rated_indices = [movie_mapper[imdb_id] for imdb_id in input_movies if imdb_id in movie_mapper]
    total_scores[rated_indices] = -np.inf  # prevent already rated

    top_indices = np.argpartition(-total_scores, top_n)[:top_n]
    top_indices = top_indices[np.argsort(-total_scores[top_indices])]
    top_imdb_ids = [movie_inverse_mapper[i] for i in top_indices]

    results = []
    for imdb_id in top_imdb_ids:
        title = movie_metadata.loc[imdb_id]['title'] if imdb_id in movie_metadata.index else 'Unknown'
        results.append({'imdbId': int(imdb_id), 'title': title})
        print(f"  🎬 Recommended: {title} (IMDb ID: {imdb_id})")

    return results

# --- Content-Based (Search) ---
print("🔄 Loading Content-Based Search Model...")
save_dir = 'model/content_based/'
title_embeddings = np.load(f'{save_dir}title/title_embeddings.npy')
title_df = pd.read_csv(f'{save_dir}title/title_lookup.csv')
movie_embeddings = np.load(f'{save_dir}search/movie_embeddings.npy')
movie_df = pd.read_csv(f'{save_dir}search/movie_info.csv')
model_roberta = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
print("✅ Content-based model ready.")

@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query", "")
    print(f"🔍 Received search query: '{query}'")

    if not query:
        print("⚠️ Empty query received. Returning empty result.")
        return jsonify([])
 
    query_embedding = model_roberta.encode(query, convert_to_numpy=True)
    similarities = cosine_similarity([query_embedding], title_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:5]
    results = title_df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]

    response = [
        { 
            'imdbId': int(row['imdb_id']),
            'title': row['title'],
            'score': round(row['similarity'], 3)
        }
        for _, row in results.iterrows()
    ]

    for res in response:
        print(f"  ✅ {res['title']} (IMDb ID: {res['imdbId']} | Score: {res['score']})")

    return jsonify(response)

@app.route("/smartsearch", methods=["POST"])
def smart_search():
    query = request.json.get("query", "")
    print(f"🧠 Received smart search query: '{query}'")

    if not query:
        print("⚠️ Empty smart search query received. Returning empty result.")
        return jsonify([])

    query_embedding = model_roberta.encode(query, convert_to_numpy=True)
    similarities = cosine_similarity([query_embedding], movie_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:5]
    results = movie_df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]

    response = [
        {
            'imdbId': int(row['imdb_id']),
            'title': row['title'],
            'score': round(row['similarity'], 3)
        }
        for _, row in results.iterrows()
    ]

    for res in response:
        print(f"  ✅ {res['title']} (IMDb ID: {res['imdbId']} | Score: {res['score']})")

    return jsonify(response)


@app.route("/recommend", methods=["POST"])
def recommend():
    ratings = request.json.get("ratings", {})
    print(f"📩 Received ratings for recommendation: {ratings}")
    parsed_ratings = {int(k): float(v) for k, v in ratings.items()}
    recommendations = recommend_collaborative(parsed_ratings)
    print(f"✅ Returning {len(recommendations)} recommendations.")
    return jsonify(recommendations)

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True)
