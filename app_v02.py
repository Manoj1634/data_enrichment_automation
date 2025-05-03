import pandas as pd
import streamlit as st
import requests
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

st.set_page_config(page_title="CSV Mapper + Enrichment", layout="wide")
st.title("CSV Mapper, Cleaner, and Enrichment Tool")

# ---------------------------
# Global Enrichment Functions
# ---------------------------
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
- Use a precise, business-focused verb like: commercializing, deploying, scaling, delivering, anything else which is more suitable.
- Keep it brief and clear (around 5–8 words)
- Focus on the commercial product/solution
- End with the target market or application
- No company name in the output
- No metrics or technical jargon
- Dont Hallucinate

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
-  Dont Hallucinate- Do **not** hallucinate or overgeneralize.
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

def process_row(idx, row, headers_jina, headers_openai, timeout=15):
    company = str(row["company name"])
    url = str(row["website"])

    raw = fetch_content_from_jina(url, headers_jina, timeout=timeout)
    if raw.strip() in ["", "error in scraping"]:
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
        st.markdown("**Select CSV Columns to Map To**")
        for target_col in desired_columns:
            options = [None] + list(df.columns)
            default_value = default_mapping.get(target_col)
            default_index = options.index(default_value) if default_value in options else 0
            selected = st.selectbox(
                f"Select CSV column for '{target_col}'",
                options,
                index=default_index
            )
            if selected:
                column_mapping[target_col] = selected

    if st.button("Apply Mapping"):
        unmapped = [col for col in desired_columns if col not in column_mapping]
        if unmapped:
            st.error(f"❌ The following columns are not mapped: {', '.join(unmapped)}. Please map them before proceeding.")
        else:
            mapped_df = pd.DataFrame()
            for target_col, source_col in column_mapping.items():
                if source_col in df.columns:
                    mapped_df[target_col] = df[source_col]
            st.session_state['mapped_df'] = mapped_df
            st.success("✅ Mapping applied! Review below and proceed to cleaning.")

# ---------------------------
# Cleaning Section
# ---------------------------
if 'mapped_df' in st.session_state:
    mapped_df = st.session_state['mapped_df']
    st.write("### Step 2: Select Cleaning Options")
    st.write("### Mapped Data Preview", mapped_df.head())

    unique_countries = sorted(mapped_df['country'].dropna().unique())
    selected_countries = st.multiselect(
        "Select countries to include",
        options=unique_countries,
        default=unique_countries
    )

    if st.button("Apply Cleaning"):
        cleaned_df = mapped_df.copy()
        cleaned_df = cleaned_df[cleaned_df['country'].isin(selected_countries)]
        cleaned_df = cleaned_df.dropna(subset=['email', 'website'])
        cleaned_df = cleaned_df.drop_duplicates(subset=['email'])
        cleaned_df['occurences'] = cleaned_df.groupby('website').cumcount() + 1
        cleaned_df['company name'] = (cleaned_df['company name'].str.replace(r'\s*\([^)]*\)', '', regex=True).str.replace(r'\s*LLC', '', regex=True).str.replace('✓', '', regex=False).str.replace('"', '', regex=False).str.strip())

        if not cleaned_df.empty:
            st.session_state['cleaned_df'] = cleaned_df
            st.success("✅ Cleaning applied! See the cleaned data below.")
        else:
            st.warning("⚠️ No data left after cleaning. Please adjust your selections.")

# ---------------------------
# Enrichment Section
# ---------------------------
if 'cleaned_df' in st.session_state:
    df = st.session_state['cleaned_df']
    st.write("### Final Cleaned Data", df.head())
    cleaned_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Final Cleaned CSV",
        data=cleaned_csv,
        file_name="final_cleaned_output.csv",
        mime="text/csv"
    )

    st.write("---")
    st.write("### Step 3: Data Enrichment")

    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    jina_api_key = st.text_input("Enter your Jina API Key", type="password")

    if openai_api_key and jina_api_key:
        headers_jina = {"Authorization": f"Bearer {jina_api_key}"}
        headers_openai = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }

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
                futures = {executor.submit(process_row, idx, row, headers_jina, headers_openai): idx for idx, row in enriched_df.iterrows()}
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
    if st.button("Rerun Failed Rows (with longer timeout)"):
        failed_rows = enriched_df[enriched_df['webcontent'] == "error in scraping"]

        if failed_rows.empty:
            st.info("✅ No failed rows to rerun — all clean!")
        else:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(process_row, idx, row, headers_jina, headers_openai, timeout=60): idx for idx, row in failed_rows.iterrows()}
                for future in as_completed(futures):
                    idx, webcontent, summary, vertical = future.result()
                    enriched_df.at[idx, "webcontent"] = webcontent
                    enriched_df.at[idx, "summary_oneliner"] = summary
                    enriched_df.at[idx, "vertical"] = vertical

            remaining_errors = (enriched_df['webcontent'] == "error in scraping").sum()
            st.success(f"✅ Re-run complete! {remaining_errors} rows still have scraping errors.")

    enriched_csv = enriched_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Final Enriched CSV",
        data=enriched_csv,
        file_name="final_enriched_output.csv",
        mime="text/csv"
    )
