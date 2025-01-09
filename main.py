import streamlit as st
import requests
import time
import json
import pandas as pd
from flask import Flask, request, jsonify, Response, stream_with_context
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from accelerate import Accelerator
from bs4 import BeautifulSoup
import threading
import urllib.parse
import random
from readability.readability import Document
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Accelerator
accelerator = Accelerator()

# Load sentiment analysis model
hf_token = os.getenv("HUGGINGFACE_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
headers = {"Authorization": f"Bearer {hf_token}"}

# Function to query the Hugging Face Inference API
def query_huggingface_api(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Sentiment analysis function with star ratings using Hugging Face Inference API
def analyze_sentiment_with_stars(text):
    try:
        payload = {"inputs": text}
        response = query_huggingface_api(payload)
        scores = response[0]['scores']
        probabilities = (np.exp(scores) / np.sum(np.exp(scores))).tolist()  # Softmax

        # Map to star ratings (1 to 5)
        star_rating = int(np.argmax(probabilities) + 1)  # Convert to standard Python int

        # Determine sentiment label
        if star_rating in [1, 2]:
            sentiment_label = "Negative"
        elif star_rating == 3:
            sentiment_label = "Neutral"
        else:
            sentiment_label = "Positive"

        return star_rating, sentiment_label, probabilities
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return None, None, []

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
        logging.error(f"Error creating FAISS index: {e}")
        return None, None

# Function to extract text from a URL
def extract_text_from_url(url):
    try:
        response = httpx.get(url)
        response.raise_for_status()
        html = response.text
        doc = Document(html)
        content = doc.summary()
        soup = BeautifulSoup(content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        logging.error(f"Error extracting text from URL {url}: {e}")
        return None

# API route to analyze sentiment of text from search query
@app.route('/analyze_sentiment', methods=['GET'])
def analyze_sentiment():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No search query provided"}), 400

    # Scrape Google search results and analyze sentiment
    results = scrape_and_analyze(query, num_pages=1)
    if results:
        return jsonify(results)
    else:
        return jsonify({"error": "Failed to analyze sentiment"}), 500

# Function to scrape Google search results using httpx and analyze sentiment
def scrape_and_analyze(query, num_pages=1):
    """
    Scrape Google search results using httpx, analyze sentiment, and save results to a JSON file.
    
    Args:
        query (str): The search query.
        num_pages (int): Number of pages to scrape.
    """
    results = []
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
    ]

    for page in range(num_pages):
        try:
            # Build the URL
            params = {"q": query, "start": page * 10, "hl": "en"}
            url = f"https://www.google.com/search?{urllib.parse.urlencode(params)}"
            logging.info(f"Fetching: {url}")

            # Rotate user agent
            headers = {"User-Agent": random.choice(user_agents)}

            # Fetch the page
            response = httpx.get(url, headers=headers)
            response.raise_for_status()
            html = response.text

            if response.status_code == 429:
                logging.warning("Received 429 Too Many Requests. Sleeping for a while...")
                time.sleep(60)  # Sleep for 60 seconds before retrying
                response = httpx.get(url, headers=headers)
                response.raise_for_status()
                html = response.text

            # Parse the HTML content
            soup = BeautifulSoup(html, 'html.parser')

            # Extract search result links
            for result in soup.select("div.yuRUbf a"):
                link = result.get('href')
                if link:
                    logging.info(f"Found link: {link}")
                    # Extract text from the URL
                    text = extract_text_from_url(link)
                    if text:
                        logging.info(f"Extracted text from {link}")
                        # Perform sentiment analysis
                        star_rating, sentiment_label, _ = analyze_sentiment_with_stars(text)
                        if star_rating:
                            logging.info(f"Analyzed sentiment for {link}: {star_rating} stars, {sentiment_label}")
                            results.append({
                                "search_result": text,
                                "sentiment": f"{star_rating} ({sentiment_label})",
                                "search_result_link": link
                            })

            # Add delay between page requests
            time.sleep(random.randint(5, 10))
        except Exception as e:
            logging.error(f"Error scraping page {page}: {e}")

    # Create a safe filename from the query
    safe_query = "".join([c if c.isalnum() else "_" for c in query])
    output_dir = "/mount/src/sentiment-streamlit/JSON-output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{safe_query}.json")

    # Save results to JSON file
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=4)
        logging.info(f"Results saved to {output_file}")
        # Update Streamlit status
        st.session_state.scraping_done = True
        st.session_state.output_file = output_file
    except Exception as e:
        logging.error(f"Error saving results to file: {e}")
        st.session_state.scraping_done = False  # Ensure scraping_done is set to False on error

    return results

# POST endpoint to accept search query and start scraping process
@app.route('/start_scraping', methods=['POST'])
def start_scraping():
    data = request.get_json()
    query = data.get("query")
    num_pages = data.get("num_pages", 1)
    
    if not query:
        return jsonify({"error": "No search query provided"}), 400

    # Start the scraping process in a separate thread
    threading.Thread(target=scrape_and_analyze, args=(query, num_pages)).start()
    return jsonify({"message": "Scraping started", "query": query, "num_pages": num_pages}), 200

# GET endpoint to stream the results from the JSON file
@app.route('/stream_results', methods=['GET'])
def stream_results():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No search query provided"}), 400

    # Create a safe filename from the query
    safe_query = "".join([c if c.isalnum() else "_" for c in query])
    output_file = os.path.join("/mount/src/sentiment-streamlit/JSON-output", f"{safe_query}.json")

    try:
        def generate():
            with open(output_file, "r", encoding="utf-8") as file:
                for line in file:
                    yield line

        return Response(stream_with_context(generate()), mimetype='application/json')
    except FileNotFoundError:
        return jsonify({"error": "Results file not found"}), 404
    except Exception as e:
        logging.error(f"Error streaming results: {e}")
        return jsonify({"error": "Failed to stream results"}), 500

# Flag to indicate if the Flask server is running
flask_server_running = False

# Function to run the Flask app
def run_flask():
    global flask_server_running
    if not flask_server_running:
        flask_server_running = True
        app.run(debug=True, use_reloader=False, port=8503)  # Use port 8503

# Start the Flask server in a separate thread if not already running
if not flask_server_running:
    threading.Thread(target=run_flask).start()

# Streamlit app title
st.title("Web Results Sentiment Analysis")

# Sample search queries
sample_queries = [
    "Maha Kumbh Prayagraj 2025 scam fraud threat theft attack mis-management",
    "Climate change impact on agriculture",
    "Latest technology trends 2023",
    "COVID-19 vaccine effectiveness",
    "Artificial intelligence in healthcare"
]

# Dropdown for sample search queries
query = st.selectbox("Select a sample search query or enter your own:", [""] + sample_queries)

# Input box for user input search query
user_query = st.text_input("Or enter your own search query:")

# Use the user input if provided
if user_query:
    query = user_query

# Input box for number of pages to scrape
num_pages = st.number_input("Enter the number of pages to scrape:", min_value=1, max_value=100, value=1)

# Button to start scraping
if st.button("Start Scraping"):
    if not query:
        st.error("Please enter a search query.")
    else:
        st.info("Scraping data in progress...")
        st.session_state.scraping_done = False
        st.session_state.scraping_started = True
        response = requests.post("http://localhost:8503/start_scraping", json={"query": query, "num_pages": num_pages})
        if response.status_code == 200:
            st.success("Scraping started. Please wait for the process to complete.")
        else:
            st.error("Failed to start scraping. Please try again.")

# Dropdown to view or download the result
if st.button("View/Download Results"):
    if not query:
        st.error("Please enter a search query.")
    else:
        safe_query = "".join([c if c.isalnum() else "_" for c in query])
        output_file = os.path.join("/mount/src/sentiment-streamlit/JSON-output", f"{safe_query}.json")
        
        # Fetch the results
        response = requests.get(f"http://localhost:8503/stream_results?query={query}")
        if response.status_code == 200:
            try:
                results = response.json()
                # Convert results to a DataFrame for better display
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Download link for the JSON file
                st.download_button(
                    label="Download Results as JSON",
                    data=json.dumps(results, indent=4),
                    file_name=f"{safe_query}.json",
                    mime="application/json"
                )
            except json.JSONDecodeError:
                st.error("Failed to decode JSON response. Please try again.")
        else:
            st.error("Failed to fetch results. Please try again.")

# Button to view all stored results
if st.button("View All Stored Results"):
    output_dir = "/mount/src/sentiment-streamlit/JSON-output"
    if os.path.exists(output_dir) and os.listdir(output_dir):
        all_results = []
        for filename in os.listdir(output_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(output_dir, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    results = json.load(file)
                    all_results.extend(results)
        
        if all_results:
            df = pd.DataFrame(all_results)
            st.dataframe(df)
        else:
            st.info("No results found in the JSON files.")
    else:
        st.info("No results found in the /mount/src/sentiment-streamlit/JSON-output folder.")

# Periodically check the scraping status and update the UI
if st.session_state.get("scraping_started", False) and not st.session_state.get("scraping_done", False):
    st.info("Scraping is still in progress. Please wait...")
    time.sleep(5)

