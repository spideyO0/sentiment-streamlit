import streamlit as st
import requests
import time
import json
import pandas as pd

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
        response = requests.post("http://localhost:5000/start_scraping", json={"query": query, "num_pages": num_pages})
        if response.status_code == 200:
            st.success("Scraping started. Please wait for the process to complete.")
            time.sleep(10)  # Wait for scraping to complete (adjust as needed)
            st.info(f"Scraping done. Storing results in {query.replace(' ', '_')}.json")
        else:
            st.error("Failed to start scraping. Please try again.")

# Dropdown to view or download the result
if st.button("View/Download Results"):
    if not query:
        st.error("Please enter a search query.")
    else:
        safe_query = "".join([c if c.isalnum() else "_" for c in query])
        output_file = f"{safe_query}.json"
        
        # Fetch the results
        response = requests.get(f"http://localhost:5000/stream_results?query={query}")
        if response.status_code == 200:
            results = response.json()
            
            # Convert results to a DataFrame for better display
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            # Download link for the JSON file
            st.download_button(
                label="Download Results as JSON",
                data=json.dumps(results, indent=4),
                file_name=output_file,
                mime="application/json"
            )
        else:
            st.error("Failed to fetch results. Please try again.")
