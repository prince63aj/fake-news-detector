from flask import Flask, request, jsonify
import pickle
import re
from flask_cors import CORS
import string
import os  # <-- Needed for getting PORT

# Load model and vectorizer
with open("models/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("models/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)
CORS(app)

# Cleaning function (same as train_model.py)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Label decoder
def output_label(n):
    return "Fake News" if n == 0 else "Your News is True"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({"error": "Missing 'text' in request body"}), 400

        raw_text = data["text"]
        cleaned = clean_text(raw_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        return jsonify({
            "prediction": output_label(prediction)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Root test route
@app.route("/", methods=["GET"])
def home():
    return "✅ Fake News Detection Backend is running."

# Start the server (Render-compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT or 5000 locally
    app.run(host="0.0.0.0", port=port)
