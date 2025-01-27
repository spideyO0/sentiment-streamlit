import streamlit as st
import requests
import time
import json
import pandas as pd
from flask import Flask, request, jsonify, Response, stream_with_context
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
from accelerate import Accelerator
import threading
import httpx
import logging
from duckduckgo_search import DDGS
from googlesearch import search as google_search
from newspaper import Article
from readability.readability import Document
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Accelerator
accelerator = Accelerator()

# Hugging Face Inference API details
hf_token = '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'
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

# Function to extract text from a URL using newspaper3k and readability-lxml
def extract_text_from_url(url):
    """
    Extracts readable text content from a given URL, ignoring PDFs.
    
    Args:
        url (str): The URL of the web page to extract text from.

    Returns:
        str: Extracted text content as a single string, or None if the URL is a PDF or extraction fails.
    """
    try:
        # Fetch the web page
        response = httpx.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Check if the content type is PDF
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' in content_type:
            logger.info(f"Skipping PDF URL: {url}")
            return None

        # Use readability-lxml to extract readable content
        doc = Document(response.text)
        readable_html = doc.summary()

        # Use BeautifulSoup to extract text from the readable HTML
        soup = BeautifulSoup(readable_html, 'html.parser')
        readable_text = soup.get_text(separator=' ')

        # If readability-lxml fails to extract content, fall back to newspaper3k
        if not readable_text.strip():
            article = Article(url)
            article.download()
            article.parse()
            readable_text = article.text

        return readable_text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from URL {url}: {e}")
        return None

# Function to perform Google search without retries
def google_search_results(query, num_results):
    results = []
    delay = 10  # Delay between requests in seconds

    try:
        # Perform the search using the google-search library
        search_results = google_search(
            term=query,
            num_results=num_results,
            safe="off",
            advanced=True,
            region="in",
            sleep_interval=delay  # Add delay between requests
        )

        # Process the results
        for result in search_results:
            results.append({
                "title": result.title if hasattr(result, 'title') else "No Title",
                "url": result.url if hasattr(result, 'url') else result,
                "description": result.description if hasattr(result, 'description') else "No Description"
            })

    except Exception as e:
        logger.error(f"Google search error: {e}")
        # Return an empty list to trigger fallback to DuckDuckGo
        return []

    return results

# Function to perform DuckDuckGo search
def duckduckgo_search_results(query, num_results):
    results = []
    ddg = DDGS()
    ddg_results = ddg.text(keywords=query, max_results=num_results, region='wt-wt', safesearch='off', backend='html')
    for result in ddg_results:
        results.append({
            "title": result.get('title', "No Title"),
            "url": result.get('href', "No URL"),
            "description": result.get('body', "No Description")
        })
    return results

def is_social_media_url(url):
    """
    Check if the URL belongs to a social media domain.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL is a social media link, False otherwise.
    """
    # List of social media domains to ignore
    SOCIAL_MEDIA_DOMAINS = [
        "instagram.com", "facebook.com", "twitter.com", "x.com", "linkedin.com",
        "youtube.com", "tiktok.com", "pinterest.com", "reddit.com", "snapchat.com",
        "tumblr.com", "weibo.com", "whatsapp.com", "telegram.org", "discord.com",
        "medium.com", "quora.com",
    ]

    if not url:
        return False
    return any(domain in url.lower() for domain in SOCIAL_MEDIA_DOMAINS)

# Function to scrape DuckDuckGo and Google search results and analyze sentiment
def scrape_and_analyze(base_query, extra_keywords):
    seen_urls = set()

    # Construct search queries based on presence of extra_keywords
    if extra_keywords:
        search_queries = [f"{base_query} {keyword.strip()}" for keyword in extra_keywords]
    else:
        search_queries = [base_query]

    for search_query in search_queries:
        logger.info(f"Scraping query: {search_query}")
        try:
            # Try Google search first
            google_results = google_search_results(search_query, 150)  
            if not google_results:  # Fallback to DuckDuckGo if Google fails
                logger.warning("Google search failed. Falling back to DuckDuckGo.")
                ddg_results = duckduckgo_search_results(search_query, 150)  
                results = ddg_results
            else:
                results = google_results

            # Process results
            for result in results:
                if result.get('url') not in seen_urls:
                    seen_urls.add(result.get('url'))
                    title = result.get("title", "No Title")
                    link = result.get("url")
                    snippet = result.get("description", "No Snippet")

                    # Skip PDF URLs
                    if link and link.lower().endswith('.pdf'):
                        logger.info(f"Skipping PDF URL: {link}")
                        continue

                    # Skip social media URLs
                    if is_social_media_url(link):
                        logger.info(f"Skipping social media URL: {link}")
                        continue

                    if link:
                        logger.info(f"Found news article: {title}")
                        # Extract text from the URL
                        text = extract_text_from_url(link)
                        if text:
                            # Perform sentiment analysis
                            star_rating, sentiment_label, _ = analyze_sentiment_with_stars(text)
                            if star_rating:
                                yield {
                                    "title": title,
                                    "snippet": snippet,
                                    "sentiment": f"{star_rating} ({sentiment_label})",
                                    "link": link
                                }
        except Exception as e:
            logger.error(f"Error scraping query {search_query}: {e}")

# API route to analyze sentiment of text from search query
@app.route('/analyze_sentiment', methods=['GET'])
def analyze_sentiment():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No search query provided"}), 400

    # Scrape DuckDuckGo and Google search results and analyze sentiment
    results = list(scrape_and_analyze(query, []))
    if results:
        return jsonify(results)
    else:
        return jsonify({"error": "Failed to analyze sentiment"}), 500

# POST endpoint to accept search query and start scraping process
@app.route('/start_scraping', methods=['POST'])
def start_scraping():
    data = request.get_json()
    base_query = data.get("base_query", "")
    extra_keywords = data.get("extra_keywords", [])
    
    if not base_query and not extra_keywords:
        return jsonify({"error": "No base query or keywords provided"}), 400
    
    # Log the search queries
    logger.info(f"Starting scraping for base query: {base_query} with extra keywords: {extra_keywords}")

    # Start the scraping process in a separate thread
    try:
        threading.Thread(
            target=scrape_and_analyze,
            args=(base_query, extra_keywords)
        ).start()
        return jsonify({
            "message": "Scraping started",
            "base_query": base_query,
            "extra_keywords": extra_keywords,
        }), 200
    except Exception as e:
        logger.error(f"Error starting scraping: {e}")
        return jsonify({"error": f"Failed to start scraping: {e}"}), 500

# GET endpoint to stream the results from the JSON file
@app.route('/stream_results', methods=['GET'])
def stream_results():
    base_query = request.args.get('base_query', "")
    extra_keywords = request.args.getlist('extra_keywords')
    
    if not base_query and not extra_keywords:
        return jsonify({"error": "No base query or keywords provided"}), 400

    # Create search queries
    search_queries = [base_query] + [f"{base_query} {keyword}" for keyword in extra_keywords]

    logger.info(f"Streaming results for queries: {search_queries}")

    def generate():
        seen_urls = set()  # Track seen URLs to avoid duplicates
        try:
            for query in search_queries:
                logger.info(f"Starting scraping for query: {query}")
                google_results = google_search_results(query, 150)
                ddg_results = duckduckgo_search_results(query, 150)
                results = google_results + ddg_results
                for result in results:
                    # Skip if we've already seen this URL
                    if result.get('url') in seen_urls:
                        continue
                    
                    # Add the URL to our seen set
                    seen_urls.add(result.get('url'))
                    
                    # Add metadata about the search
                    keyword = query.replace(base_query, "").strip()
                    result['base_query'] = base_query
                    result['keyword'] = keyword
                    
                    yield json.dumps(result) + "\n"
        except Exception as e:
            logger.error(f"Error streaming results: {e}")
            yield json.dumps({"error": f"Failed to stream results: {e}"}) + "\n"

    return Response(stream_with_context(generate()), mimetype='application/json')

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

# Initialize session state for extra keywords
if 'extra_keywords' not in st.session_state:
    st.session_state.extra_keywords = []

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

# Extra keywords section
st.subheader("Additional Keywords")

# Input for new keyword
new_keyword = st.text_input("Enter an additional keyword:", key="new_keyword_input")

# Add keyword button
if st.button("Add Keyword"):
    if new_keyword.strip():
        st.session_state.extra_keywords.append(new_keyword.strip())
    else:
        st.warning("Please enter a keyword before adding.")

# Display current keywords
if st.session_state.extra_keywords:
    st.write("**Added Keywords:**")
    for i, keyword in enumerate(st.session_state.extra_keywords):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"- {keyword}")
        with col2:
            if st.button(f"Remove", key=f"remove_{i}"):
                del st.session_state.extra_keywords[i]
                st.rerun()

    if st.button("Clear All Keywords"):
        st.session_state.extra_keywords = []
        st.rerun()

# Placeholder for results table
results_placeholder = st.empty()

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = []

# Button to start scraping
if st.button("Start Scraping"):
    if not query and not st.session_state.extra_keywords:
        st.error("Please enter a search query or add keywords.")
    else:
        with st.spinner('Scraping in progress...'):
            try:
                # Start the scraping process
                response = requests.post(
                    "http://localhost:8503/start_scraping",
                    json={
                        "base_query": query,
                        "extra_keywords": st.session_state.extra_keywords,
                        "num_pages": 30  # Default number of pages to scrape
                    }
                )
                response.raise_for_status()
                st.success("Scraping started. Streaming results...")

                # Initialize an empty DataFrame to store results
                results_df = pd.DataFrame(columns=["Title", "Snippet", "Sentiment", "Link"])

                # Create a placeholder for the table
                table_placeholder = st.empty()

                # Stream results in real-time
                for result in scrape_and_analyze(query, st.session_state.extra_keywords):
                    # Append the result to the DataFrame
                    new_row = {
                        "Title": result.get("title", "No Title"),
                        "Snippet": result.get("snippet", "No Snippet"),
                        "Sentiment": result.get("sentiment", "No Sentiment"),
                        "Link": result.get("link", "No Link")
                    }
                    results_df = results_df._append(new_row, ignore_index=True)
                    # Update the table in place
                    table_placeholder.dataframe(results_df, width=1000)  # Adjust width as needed

                # Export the DataFrame with full links
                st.download_button(
                    label="Export Results",
                    data=results_df.to_csv(index=False).encode('utf-8'),
                    file_name='web_results.csv',
                    mime='text/csv',
                )

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")

# Periodically check the scraping status and update the UI
if st.session_state.get("scraping_started", False) and not st.session_state.get("scraping_done", False):
    time.sleep(5)