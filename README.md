

# CSV Mapper, Cleaner, and Enrichment Tool

This is a **Streamlit-based application** that helps you:

✅ Upload a CSV  
✅ Map and clean the columns  
✅ Filter and deduplicate data  
✅ Enrich company records using **Jina AI** (web scraping) and **OpenAI GPT-4o** (summarization)  
✅ Download the cleaned and enriched CSV files  

---

## 🚀 Features

- **CSV Upload & Mapping**  
  Upload your own CSV and flexibly map your existing columns to the desired target fields.

- **Data Cleaning**  
  - Select countries to include.
  - Remove rows with missing emails or websites.
  - Deduplicate entries by email.
  - Clean up company names by removing trailing parentheses or extra info.

- **Automated Enrichment**  
  - Scrape website content using Jina AI.
  - Generate a concise one-liner business summary using GPT-4o.
  - Classify the company's vertical/industry sector.

- **Error Handling & Retry**  
  - Track rows where scraping failed.
  - Rerun only the failed rows with a longer timeout.

- **Download Outputs**  
  Export cleaned data or fully enriched data as CSV files.

---

## 🏗 How It Works

## Install dependencies
pip install -r requirements.txt

##  Launch the app
streamlit run app_v02.py
