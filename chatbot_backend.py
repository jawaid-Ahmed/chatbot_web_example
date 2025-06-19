from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)
CORS(app)

# 🧪 Test route to confirm server is working
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "alive"})

# 🔄 Load dataset with error handling
try:
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    print("✅ Loaded", len(qa_data), "Q&A entries")
except Exception as e:
    print("❌ Failed to load qa_dataset.json:", e)
    qa_data = []

if qa_data:
    questions = [q["question"].lower() for q in qa_data]
    answers = [q["answer"] for q in qa_data]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)
else:
    questions = []
    answers = []
    X = None

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "").lower()
        print("👉 Received:", user_input)

        if not user_input or not questions:
            return jsonify({"reply": "Chatbot is not ready or no data available."})

        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, X)
        idx = similarity.argmax()
        confidence = similarity[0][idx]

        print("🧠 Best match:", questions[idx])
        print("📊 Confidence:", confidence)

        if confidence > 0.2:
            return jsonify({"reply": answers[idx]})
        else:
            return jsonify({"reply": "Sorry, I don't have an answer for that."})
    except Exception as e:
        print("❌ Error during chat:", e)
        return jsonify({"reply": "Internal server error."})

if __name__ == "__main__":
    app.run(port=5000)
