import streamlit as st
import pandas as pd
import requests
import json
import re
import time
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

# --- Configurations ---
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
JINA_RATE_LIMIT = 10
OPENAI_THREAD_LIMIT = 5
NUM_WORKERS = 10
SAVE_INTERVAL = 120

# --- Functions ---
def clean_markdown(text):
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def one_liner_prompt(company_name, url, web_content):
    return f"""Generate a concise company description in this structure:
**[verb-ing] [product/solution] for [target market/application]**

Requirements:
- Return **only one** answer.
- Use a business-focused verb like: commercializing, deploying, scaling, delivering
- Keep it brief and clear (around 5–8 words)
- Focus on the commercial product/solution
- End with the target market or application
- No company name in the output
- No metrics or technical jargon

Examples:
✓ commercializing low-carbon concrete for infrastructure projects
✓ scaling sustainable batteries for electric vehicles
✓ delivering plant-based proteins for food manufacturers

Input:
Company name: {company_name}
Website: {url}
Content: {web_content[:4000]}
"""

def vertical_prompt(web_content, one_liner):
    return f"""You are a domain expert tasked with identifying and generating a company’s **core vertical/industry space**.

Requirements:
- Focus on the company’s core product offering, business model, and target market.
- Prioritize the "One-liner Summary" as a concise representation of the company’s focus.
- Use the Website Content for additional context, especially if the one-liner is vague.
- Do NOT overemphasize investor or team bios unless they clearly describe the company’s product.
- Use **standard industry terms**, 2-3 words maximum, all lowercase with spaces.
- Avoid generic terms like “technology” or “software.”

Examples:
✓ ai infrastructure
✓ cleantech robotics
✓ consumer fintech
✓ digital identity

Output:
- Return **only** the vertical as a 1-line answer.
- Do **not** include reasoning, explanation, or formatting.

Input:
One-liner Summary: {one_liner}

Website Content: {web_content[:4000]}
"""

def fetch_content_from_jina(url, jina_key):
    headers_jina = {"Authorization": f"Bearer {jina_key}"}
    try:
        res = requests.get(f"https://r.jina.ai/{url}", headers=headers_jina, timeout=15)
        if res.status_code == 429:
            return "rate limit hit"
        return res.text.strip() if res.status_code == 200 else "error in scraping"
    except:
        return "error in scraping"

def query_openai(prompt, openai_key, semaphore):
    headers_openai = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 512
    }
    with semaphore:
        try:
            res = requests.post(OPENAI_URL, headers=headers_openai, data=json.dumps(payload))
            return res.json()["choices"][0]["message"]["content"].strip() if res.status_code == 200 else "Input not found"
        except:
            return "Input not found"

def process_row(idx, row, openai_key, jina_key, semaphore):
    company = str(row["company name"])
    url = str(row["website"])
    raw = fetch_content_from_jina(url, jina_key)
    if raw.strip() in ["", "error in scraping", "rate limit hit"]:
        return idx, raw, "Input not found", "Input not found"
    cleaned = clean_markdown(raw)
    summary = query_openai(one_liner_prompt(company, url, cleaned), openai_key, semaphore)
    vertical = query_openai(vertical_prompt(cleaned, summary), openai_key, semaphore)
    return idx, cleaned, summary, vertical

# --- Streamlit UI ---
st.title("🔍 Enrich CSV with Company Summary & Vertical")
openai_key = st.text_input("Enter OpenAI API Key", type="password")
jina_key = st.text_input("Enter Jina API Key", type="password")
file = st.file_uploader("Upload CSV", type="csv")
start_process = st.button("🚀 Start Processing")

if file and openai_key and jina_key and start_process:
    df = pd.read_csv(file).head(100)
    df.columns = df.columns.str.replace('\xa0', '', regex=False).str.strip()

    drop_cols = ["id", "quality", "result", "free", "role", "pitchbook id", "pitchbook ID firm", "ellipsis href 3", "linkedin"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    df = df[['website', 'descriptions', 'investors', 'first name', 'last name', 'phone', 'name', 'website.1', 'country', 'city']]
    df = df.rename(columns={
        "website": "title", "descriptions": "bio", "investors": "email",
        "city": "linkedin", "country": "city", "website.1": "country",
        "phone": "website", "name": "company name"
    })
    df = df[df['country'].isin(['United States', 'USA', 'United States of America'])]
    df = df[df['website'].notna() & (df['website'].str.strip() != '')]
    df['company name'] = df['company name'].str.replace(r'\s*\([^)]*\)$', '', regex=True)
    df = df.drop_duplicates(subset=['email'])
    df["webcontent"] = ""
    df["summary_oneliner"] = ""
    df["vertical"] = ""

    st.write("📊 Processing records...")
    failed = 0
    rate_limited = 0
    semaphore = Semaphore(OPENAI_THREAD_LIMIT)
    progress_text = st.empty()
    progress_bar = st.progress(0)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_row, idx, row, openai_key, jina_key, semaphore): idx
            for idx, row in df.iterrows()
        }
        total = len(futures)
        done = 0
        for future in as_completed(futures):
            idx, webcontent, summary, vertical = future.result()
            df.at[idx, "webcontent"] = webcontent
            df.at[idx, "summary_oneliner"] = summary
            df.at[idx, "vertical"] = vertical
            done += 1
            progress_bar.progress(done / total)
            progress_text.text(f"Completed {done}/{total} rows")
            if webcontent == "error in scraping":
                failed += 1
            elif webcontent == "rate limit hit":
                rate_limited += 1

    st.success(f"✅ Done. Failed scrapes: {failed}, Rate limited: {rate_limited}")

    st.dataframe(df[df['webcontent'].isin(["error in scraping", "rate limit hit"])]
                 [["company name", "website", "webcontent"]].reset_index(drop=True))

    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Enriched CSV", data=csv_data, file_name="enriched_companies.csv", mime="text/csv")
