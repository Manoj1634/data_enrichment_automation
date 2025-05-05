import pandas as pd
import streamlit as st
import requests
import re
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

st.set_page_config(page_title="CSV Mapper + Enrichment", layout="wide")
st.title("CSV Mapper, Cleaner, and Enrichment Tool")

openai_semaphore = Semaphore(5)

def clean_markdown(text):
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def fetch_content_from_jina(url, headers_jina, timeout=15):
    try:
        res = requests.get(f"https://r.jina.ai/{url}", headers=headers_jina, timeout=timeout)
        return res.text.strip() if res.status_code == 200 else "error in scraping"
    except:
        return "error in scraping"

def scrape_plain_text(url, follow_links=False):
    try:
        if not url.startswith('http'):
            url = 'https://' + url
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style']):
            element.decompose()
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines()]
        clean_text = '\n'.join(line for line in lines if line)
        word_count = len(clean_text.split())

        if follow_links and word_count < 200:
            links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
            extra_texts = []
            for link in links[:5]:
                try:
                    extra = scrape_plain_text(link, follow_links=False)
                    if extra:
                        extra_texts.append(extra)
                except:
                    continue
            clean_text += '\n' + '\n'.join(extra_texts)
        return clean_text
    except:
        return "error in scraping"

def query_openai(prompt, headers_openai):
    with openai_semaphore:
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 512
        }
        try:
            res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers_openai, data=json.dumps(payload))
            return res.json()["choices"][0]["message"]["content"].strip() if res.status_code == 200 else "Input not found"
        except:
            return "Input not found"
def one_liner_prompt(company_name, url, web_content):
    return f"""Generate a concise company description in this structure:
**[verb-ing] [product/solution] for [target market/application]**

Focus:
- Identify the **specific business problem or need** the company is solving, as described in the web content.
- Emphasize what the product/solution enables or improves (not just what it is).

Requirements:
- Return **only one** answer.
- Use a precise, business-focused verb like: commercializing, deploying, scaling, delivering.
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
Content: {web_content[:3000]}
"""

def vertical_prompt(web_content, one_liner):
    return f"""You are a domain expert tasked with identifying and generating a company’s **core vertical/industry space**.

Requirements:
- Focus on the company’s core product offering, business model, and target market.
- Prioritize the "One-liner Summary" as a concise representation of the company’s focus.
- Use the Website Content for additional context, especially if the one-liner is vague.
- Use **standard industry terms**, 2-3 words max, all lowercase.
- Avoid vague categories; be as precise as the content allows.

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

Website Content: {web_content[:3000]}
"""

def process_row(idx, row, headers_jina, headers_openai, timeout=15, method="Jina first, fallback to MJ"):
    company = str(row["company name"])
    url = str(row["website"])
    if method == "MJ only":
        raw = scrape_plain_text(url)
    elif method == "Jina only":
        raw = fetch_content_from_jina(url, headers_jina, timeout=timeout)
    else:
        raw = fetch_content_from_jina(url, headers_jina, timeout=timeout)
        if raw.strip() in ["", "error in scraping"]:
            raw = scrape_plain_text(url)
    if raw.strip() in ["", "error in scraping", None]:
        return idx, "error in scraping", "Input not found", "Input not found"
    cleaned = clean_markdown(raw)
    summary = query_openai(one_liner_prompt(company, url, cleaned), headers_openai)
    vertical = query_openai(vertical_prompt(cleaned, summary), headers_openai)
    return idx, cleaned, summary, vertical

# ---------------------------
# Upload CSV
# ---------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.replace('\xa0', '', regex=False).str.strip()
    st.write("### Preview of Uploaded Data", df.head())

    desired_columns = [
        'title', 'bio', 'email', 'first name', 'last name',
        'website', 'company name', 'country', 'city', 'linkedin'
    ]

    default_mapping = {
        'title': 'id',
        'bio': 'descriptions',
        'email': 'investors',
        'first name': 'first name',
        'last name': 'last name',
        'website': 'phone',
        'company name': 'name',
        'country': 'website.1',
        'city': 'country',
        'linkedin': 'ellipsis href 3'
    }

    st.write("### Step 1: Map Your Columns")
    column_mapping = {}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Desired Columns (Final Names)**")
        st.write(desired_columns)
    with col2:
        for target_col in desired_columns:
            options = [None] + list(df.columns)
            default_value = default_mapping.get(target_col)
            default_index = options.index(default_value) if default_value in options else 0
            selected = st.selectbox(f"Select CSV column for '{target_col}'", options, index=default_index)
            if selected:
                column_mapping[target_col] = selected

    if st.button("Apply Mapping"):
        unmapped = [col for col in desired_columns if col not in column_mapping]
        if unmapped:
            st.error(f"❌ Unmapped columns: {', '.join(unmapped)}.")
        else:
            mapped_df = pd.DataFrame()
            for target_col, source_col in column_mapping.items():
                if source_col in df.columns:
                    mapped_df[target_col] = df[source_col]
            st.session_state['mapped_df'] = mapped_df
            st.success("✅ Mapping applied!")

# ---------------------------
# Cleaning Section
# ---------------------------
if 'mapped_df' in st.session_state:
    mapped_df = st.session_state['mapped_df']
    st.write("### Mapped Data Preview", mapped_df.head())
    unique_countries = sorted(mapped_df['country'].dropna().unique())
    selected_countries = st.multiselect("Select countries to include", options=unique_countries, default=unique_countries)

    if st.button("Apply Cleaning"):
        cleaned_df = mapped_df.copy()
        cleaned_df = cleaned_df[cleaned_df['country'].isin(selected_countries)]
        cleaned_df = cleaned_df.dropna(subset=['email', 'website'])
        cleaned_df = cleaned_df.drop_duplicates(subset=['email'])
        cleaned_df['company name'] = cleaned_df['company name'].str.replace(r'\s*\([^)]*\)$', '', regex=True)
        if not cleaned_df.empty:
            st.session_state['cleaned_df'] = cleaned_df
            st.success("✅ Cleaning applied!")
        else:
            st.warning("⚠️ No data left after cleaning.")

# ---------------------------
# Enrichment Section
# ---------------------------
if 'cleaned_df' in st.session_state:
    df = st.session_state['cleaned_df']
    st.write("### Final Cleaned Data", df.head())
    st.download_button("Download Cleaned CSV", df.to_csv(index=False).encode('utf-8'), "cleaned_output.csv", "text/csv")

    st.write("---")
    st.write("### Step 3: Data Enrichment")

    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    jina_api_key = st.text_input("Enter Jina API Key (if using Jina)", type="password")
    scraping_method = st.selectbox("Choose scraping method", ["Jina first, fallback to MJ", "MJ only"])

    if openai_api_key:
        headers_openai = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
        headers_jina = {"Authorization": f"Bearer {jina_api_key}"} if scraping_method != "MJ only" else None

        if st.button("Start Enrichment"):
            enriched_df = df.copy()
            enriched_df["webcontent"] = ""
            enriched_df["summary_oneliner"] = ""
            enriched_df["vertical"] = ""

            progress_bar = st.progress(0)
            status_text = st.empty()
            total = len(enriched_df)
            completed = 0

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(process_row, idx, row, headers_jina, headers_openai, method=scraping_method): idx
                    for idx, row in enriched_df.iterrows()
                }
                for future in as_completed(futures):
                    idx, webcontent, summary, vertical = future.result()
                    enriched_df.at[idx, "webcontent"] = webcontent
                    enriched_df.at[idx, "summary_oneliner"] = summary
                    enriched_df.at[idx, "vertical"] = vertical
                    completed += 1
                    percent_complete = int((completed / total) * 100)
                    progress_bar.progress(percent_complete)
                    status_text.text(f"Processing {percent_complete}% complete")

            st.session_state['enriched_df'] = enriched_df
            error_count = (enriched_df['webcontent'] == "error in scraping").sum()
            st.success(f"🎉 Enrichment done! {error_count} rows had scraping errors.")

# ---------------------------
# Rerun Failed Rows
# ---------------------------
if 'enriched_df' in st.session_state:
    enriched_df = st.session_state['enriched_df']
    failed_rows = enriched_df[enriched_df['webcontent'] == "error in scraping"]

    if failed_rows.empty:
        st.info("✅ No failed rows to rerun.")
    else:
        st.write(f"⚠️ {len(failed_rows)} rows still have scraping errors.")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Rerun Failed Rows (Jina Only)"):
                if not jina_api_key:
                    st.error("❌ Provide Jina API Key to rerun with Jina.")
                    st.stop()
                total_failed = len(failed_rows)
                completed = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                st.write("🔄 Starting Jina-only rerun...")
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {
                        executor.submit(process_row, idx, row, {"Authorization": f"Bearer {jina_api_key}"}, headers_openai, timeout=60, method="Jina only"): idx
                        for idx, row in failed_rows.iterrows()
                    }
                    for future in as_completed(futures):
                        idx, webcontent, summary, vertical = future.result()
                        enriched_df.at[idx, "webcontent"] = webcontent
                        enriched_df.at[idx, "summary_oneliner"] = summary
                        enriched_df.at[idx, "vertical"] = vertical
                        completed += 1
                        percent_complete = int((completed / total_failed) * 100)
                        progress_bar.progress(percent_complete)
                        status_text.text(f"Processing {percent_complete}% complete")
                st.success("✅ Jina rerun complete!")
                remaining_errors = (enriched_df['webcontent'] == "error in scraping").sum()
                st.write(f"Remaining errors: {remaining_errors}")

        with col2:
            if st.button("Rerun Failed Rows (MJ Only)"):
                total_failed = len(failed_rows)
                completed = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                st.write("🔄 Starting MJ-only rerun...")
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {
                        executor.submit(process_row, idx, row, None, headers_openai, timeout=60, method="MJ only"): idx
                        for idx, row in failed_rows.iterrows()
                    }
                    for future in as_completed(futures):
                        idx, webcontent, summary, vertical = future.result()
                        enriched_df.at[idx, "webcontent"] = webcontent
                        enriched_df.at[idx, "summary_oneliner"] = summary
                        enriched_df.at[idx, "vertical"] = vertical
                        completed += 1
                        percent_complete = int((completed / total_failed) * 100)
                        progress_bar.progress(percent_complete)
                        status_text.text(f"Processing {percent_complete}% complete")
                st.success("✅ MJ rerun complete!")
                remaining_errors = (enriched_df['webcontent'] == "error in scraping").sum()
                st.write(f"Remaining errors: {remaining_errors}")

    st.download_button("Download Final Enriched CSV", enriched_df.to_csv(index=False).encode('utf-8'), "final_enriched_output.csv", "text/csv")
