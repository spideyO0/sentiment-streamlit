from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
import torch
from accelerate import Accelerator
import newspaper
from newspaper import Article

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Accelerator
accelerator = Accelerator()

# Load sentiment analysis model
hf_token = os.getenv("HUGGINGFACE_TOKEN")
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment', use_auth_token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment', use_auth_token=hf_token)

# Load Sentence Transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create FAISS index
def create_faiss_index(documents):
    try:
        embeddings = embedding_model.encode(documents, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()  # Convert to numpy array
        index = faiss.IndexFlatL2(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index, embeddings
    except Exception as e:
        return None, None

# Sentiment analysis function with star ratings
def analyze_sentiment_with_stars(text):
    try:
        device = accelerator.device
        model.to(device)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        scores = outputs[0][0].detach().cpu().numpy()
        probabilities = (np.exp(scores) / np.sum(np.exp(scores))).tolist()  # Softmax

        # Map to star ratings (1 to 5)
        star_rating = np.argmax(probabilities) + 1

        return star_rating, probabilities
    except Exception as e:
        return None, []

# Function to extract text from a URL
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return None

# API route to analyze sentiment of text from URL
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()

    # Extract URL from the request
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Extract text from the URL
    text = extract_text_from_url(url)
    if not text:
        return jsonify({"error": "Failed to extract text from URL"}), 400

    # Perform sentiment analysis on the extracted text
    star_rating, probabilities = analyze_sentiment_with_stars(text)
    if star_rating:
        return jsonify({
            "star_rating": star_rating,
            "probabilities": probabilities
        })
    else:
        return jsonify({"error": "Failed to analyze sentiment"}), 500

# API route to analyze sentiment of provided text directly
@app.route('/analyze_sentiment_text', methods=['POST'])
def analyze_sentiment_text():
    data = request.get_json()

    # Extract text from the request
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Perform sentiment analysis on the provided text
    star_rating, probabilities = analyze_sentiment_with_stars(text)
    if star_rating:
        return jsonify({
            "star_rating": star_rating,
            "probabilities": probabilities
        })
    else:
        return jsonify({"error": "Failed to analyze sentiment"}), 500

# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True)
