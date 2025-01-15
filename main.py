import streamlit as st
import requests
import time
import json
import pandas as pd
from flask import Flask, request, jsonify, Response, stream_with_context
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from accelerate import Accelerator
from bs4 import BeautifulSoup
import threading
from readability.readability import Document
import httpx
import logging
from duckduckgo_search import DDGS

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Accelerator
accelerator = Accelerator()

# Hugging Face Inference API details
hf_token = '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'
# hf_token = os.getenv('HUGGINGFACE_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
headers = {"Authorization": f"Bearer {hf_token}"}

# Function to query the Hugging Face Inference API
def query_huggingface_api(payload):
    logging.info("Querying Hugging Face Inference API...")
    response = requests.post(API_URL, headers=headers, json=payload)
    logging.info(f"API response status code: {response.status_code}")
    if response.status_code != 200:
        logging.error(f"Error querying Hugging Face Inference API: {response.text}")
        return None
    return response.json()

# Sentiment analysis function with star ratings using Hugging Face Inference API
def analyze_sentiment_with_stars(text):
    try:
        # Truncate the text to fit within the model's maximum input length
        max_length = 512
        truncated_text = text[:max_length]
        
        payload = {"inputs": truncated_text}
        response = query_huggingface_api(payload)
        if response is None:
            return None, None, []
        logging.info(f"API response: {response}")
        
        # Extract scores from the response
        labels = [item['label'] for item in response[0]]
        scores = [item['score'] for item in response[0]]
        probabilities = np.exp(scores) / np.sum(np.exp(scores))  # Softmax

        # Extract the star rating from the label
        star_rating = int(labels[np.argmax(scores)].split()[0])  # Extract the numeric part of the label

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

    # Scrape DuckDuckGo search results and analyze sentiment
    results = list(scrape_and_analyze(query, num_pages=1))
    if results:
        return jsonify(results)
    else:
        return jsonify({"error": "Failed to analyze sentiment"}), 500

# Function to scrape DuckDuckGo search results using DDGS and analyze sentiment
def scrape_and_analyze(query, num_pages=1):
    """
    Scrape DuckDuckGo search results using DDGS, analyze sentiment, and yield results.
    
    Args:
        query (str): The search query.
        num_pages (int): Number of pages to scrape.
    """
    ddgs = DDGS()
    results = ddgs.text(query, region='in-en', max_results=num_pages * 10)

    for result in results:
        title = result.get("title", "No Title")
        link = result.get("href")
        source = result.get("source", "Unknown Source")
        snippet = result.get("body", "No Snippet")

        if link:
            logging.info(f"Found news article: {title} from {source}")
            # Extract text from the URL
            text = extract_text_from_url(link)
            if text:
                # Perform sentiment analysis
                star_rating, sentiment_label, _ = analyze_sentiment_with_stars(text)
                if star_rating:
                    result_data = {
                        "title": title,
                        "source": source,
                        "snippet": snippet,
                        "sentiment": f"{star_rating} ({sentiment_label})",
                        "link": link
                    }
                    yield result_data  # Yield results for streaming

# Commented out Google search functionality
# def scrape_and_analyze_google(query, num_pages=1):
#     """
#     Scrape Google search results using httpx, analyze sentiment, and yield results.
    
#     Args:
#         query (str): The search query.
#         num_pages (int): Number of pages to scrape.
#     """
#     user_agents = [
#         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
#         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
#         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
#         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
#     ]

#     for page in range(num_pages):
#         try:
#             # Build the URL
#             params = {"q": query, "start": page * 10, "hl": "en"}
#             url = f"https://www.google.com/search?{urllib.parse.urlencode(params)}"
#             logging.info(f"Fetching: {url}")

#             # Rotate user agent
#             headers = {"User-Agent": random.choice(user_agents)}

#             # Fetch the page
#             response = httpx.get(url, headers=headers)
#             response.raise_for_status()
#             html = response.text

#             if response.status_code == 429:
#                 logging.warning("Received 429 Too Many Requests. Sleeping for a while...")
#                 time.sleep(60)  # Sleep for 60 seconds before retrying
#                 response = httpx.get(url, headers=headers)
#                 response.raise_for_status()
#                 html = response.text

#             # Parse the HTML content
#             soup = BeautifulSoup(html, 'html.parser')

#             # Extract news results
#             for result in soup.select("div#search a"):
#                 title = result.select_one("h3").get_text(strip=True) if result.select_one('h3') else "No Title"
#                 link = result.get("href") or None
#                 source = result.select_one("span.VuuXrf").get_text(strip=True) if result.select_one("span.VuuXrf") else "Unknown Source"
#                 snippet = result.select_one("div.VwiC3b").get_text(strip=True) if result.select_one("div.VwiC3b") else "No Snippet"

#                 if link:
#                     logging.info(f"Found news article: {title} from {source}")
#                     # Extract text from the URL
#                     text = extract_text_from_url(link)
#                     if text:
#                         # Perform sentiment analysis
#                         star_rating, sentiment_label, _ = analyze_sentiment_with_stars(text)
#                         if star_rating:
#                             result_data = {
#                                 "title": title,
#                                 "source": source,
#                                 "snippet": snippet,
#                                 "sentiment": f"{star_rating} ({sentiment_label})",
#                                 "link": link
#                             }
#                             yield result_data  # Yield results for streaming

#             # Add delay between page requests
#             time.sleep(random.randint(5, 10))
#         except Exception as e:
#             logging.error(f"Error scraping page {page}: {e}")

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
    # output_file = os.path.join("D:/Python-Projects/sentiment-app/JSON-output", f"{safe_query}.json")
    output_file = os.path.join("/mount/src/sentiment-streamlit/JSON-output", f"{safe_query}.json")

    # Wait until the file is created
    while not os.path.exists(output_file):
        time.sleep(1)

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
        logging.info("Starting Flask server...")
        app.run(debug=True, use_reloader=False, port=8503)  # Use port 8503

# Start the Flask server in a separate thread if not already running
if 'flask_server_running' not in st.session_state:
    st.session_state.flask_server_running = False

if not st.session_state.flask_server_running:
    st.session_state.flask_server_running = True
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

# Placeholder for results table
results_placeholder = st.empty()

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = []

# Load stored results from JSON files
def load_stored_results():
    # output_dir = r"D:\Python-Projects\sentiment-app\JSON-output"
    output_dir = r"/mount/src/sentiment-streamlit/JSON-output"
    all_results = []
    if os.path.exists(output_dir) and os.listdir(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(output_dir, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    results = json.load(file)
                    all_results.extend(results)
    return all_results

# Load and display stored results on app start
stored_results = load_stored_results()
if stored_results:
    st.session_state.results = stored_results
    results_placeholder.dataframe(pd.DataFrame(st.session_state.results))

# Callback function to update results in the main thread
def update_results(result_data):
    if 'results' not in st.session_state:
        st.session_state.results = []
    st.session_state.results.append(result_data)
    results_placeholder.dataframe(pd.DataFrame(st.session_state.results))

# Button to start scraping
if st.button("Start Scraping"):
    if not query:
        st.error("Please enter a search query.")
    else:
        with st.spinner('Scraping in progress...'):
            try:
                # Initialize session state for results
                if 'results' not in st.session_state:
                    st.session_state.results = []

                # Start the scraping process
                response = requests.post("http://localhost:8503/start_scraping", json={"query": query, "num_pages": num_pages})
                response.raise_for_status()
                st.success("Scraping started. Streaming results...")

                # Initialize an empty DataFrame to store results
                results_df = pd.DataFrame(columns=["Title", "Snippet", "Sentiment", "Link"])

                # Create a placeholder for the table
                table_placeholder = st.empty()

                # Stream results in real-time
                for result in scrape_and_analyze(query, num_pages):
                    # Append the result to the DataFrame
                    new_row = {
                        "Title": result.get("title", "No Title"),
                        "Snippet": result.get("snippet", "No Snippet")[:100] + "..." if result.get("snippet") else "No Snippet",  # Truncate snippet
                        "Sentiment": result.get("sentiment", "No Sentiment"),  # Use the correct sentiment
                        "Link": result.get("link", "No Link")[:50] + "..." if result.get("link") else "No Link"  # Truncate link
                    }
                    results_df = results_df._append(new_row, ignore_index=True)
                    # Update the table in place
                    table_placeholder.dataframe(results_df, width=1000)  # Adjust width as needed

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")

# Button to clear all stored results
if st.button("Clear All Stored Results"):
    # output_dir = r"D:\Python-Projects\sentiment-app\JSON-output"
    output_dir = r"/mount/src/sentiment-streamlit/JSON-output"
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        st.success("All stored results have been cleared.")
        st.session_state.results = []
        results_placeholder.empty()
    else:
        st.info("No results found in the /mount/src/sentiment-streamlit/JSON-output folder.")

# Periodically check the scraping status and update the UI
if st.session_state.get("scraping_started", False) and not st.session_state.get("scraping_done", False):
    # st.info("Scraping is still in progress. Please wait...")
    time.sleep(5)
