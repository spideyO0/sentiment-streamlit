import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
from sentence_transformers import util
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load JSON documents
def load_documents(file_path):
    try:
        dataset = load_dataset('json', data_files=file_path)
        documents = dataset['train']['text']
        return documents
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return []

# Create FAISS index
def create_faiss_index(documents, embedding_model):
    try:
        embeddings = embedding_model.encode(documents, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()  # Convert to numpy array
        index = faiss.IndexFlatL2(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index, embeddings
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return None, None

# Enhanced severity score logic for negative sentiment
def calculate_severity_score(scores):
    negative_score = scores[0]
    total_score = sum(scores)

    # Calculate a more nuanced severity score
    severity_score = (negative_score / total_score) * 5  # Scale to 0-5
    severity_score = round(severity_score, 2)  # Round to 2 decimal places
    severity_score = min(max(severity_score, 0), 5)  # Ensure it remains within 0-5

    # Further differentiate priorities
    if severity_score < 1:
        priority = "Very Low"
    elif severity_score < 2:
        priority = "Low"
    elif severity_score < 3:
        priority = "Moderate"
    elif severity_score < 4:
        priority = "High"
    else:
        priority = "Very High"

    return severity_score, priority

# Sentiment analysis function with star rating
def analyze_sentiment_with_stars(text, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        scores = outputs[0][0].detach().numpy()
        probabilities = (np.exp(scores) / np.sum(np.exp(scores))).tolist()  # Softmax

        # Map to star ratings (1 to 5)
        star_rating = np.argmax(probabilities) + 1

        return star_rating, probabilities
    except Exception as e:
        st.error(f"Error analyzing sentiment with stars: {e}")
        return None, []

# Main function
def main():
    # Set page title
    st.title("Sentiment Analysis with RAG and Star Ratings")

    # Load documents
    json_file = 'sample_documents.json'  # Replace with your JSON file name
    documents = load_documents(json_file)

    if not documents:
        st.warning("No documents loaded. Please check the JSON file.")
        return

    # Create embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    index, embeddings = create_faiss_index(documents, embedding_model)

    if index is None:
        st.warning("Failed to create FAISS index.")
        return

    # Load sentiment analysis model
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment', use_auth_token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment', use_auth_token=hf_token)

    # User input
    user_input = st.text_area("Enter text for sentiment analysis:", "")

    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            star_rating, probabilities = analyze_sentiment_with_stars(user_input, tokenizer, model)
            if star_rating:
                st.success(f"Star Rating: {star_rating} Stars")
                st.write(f"Probabilities: {probabilities}")

    # Display sentiment analysis results for documents in JSON file
    st.subheader("Sentiment Analysis of JSON Documents")
    for doc in documents:
        star_rating, probabilities = analyze_sentiment_with_stars(doc, tokenizer, model)
        st.write(f"Text: {doc}")
        if star_rating:
            st.write(f"Star Rating: {star_rating} Stars")
            st.write(f"Probabilities: {probabilities}")

# Run the app
if __name__ == "__main__":
    main()
