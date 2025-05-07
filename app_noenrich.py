import streamlit as st
import pandas as pd

st.set_page_config(page_title="Column Mapper & Cleaner", layout="wide")
st.title("🔄 CSV Column Mapper and Cleaner")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.replace('\xa0', '', regex=False).str.strip()
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # Define desired target columns
    desired_columns = [
        'email', 'first_name', 'last_name', 'city', 'country',
        'company_name', 'website', 'linkedin', 'one_line', 'vertical', 'title', 'bio'
    ]

    # Best-guess default mapping (based on your provided column names)
    default_mapping = {
        'email': 'email',
        'first_name': 'first_name',
        'last_name': 'last_name',
        'city': 'city',
        'country': 'country',
        'company_name': 'Company Name',
        'website': 'website',
        'linkedin': 'linkedin',
        'one_line': 'one line',
        'vertical': 'vertical',
        'title': 'title',
        'bio': 'bio'
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
    st.write("### Mapped Data Preview")
    st.dataframe(mapped_df.head())

    # Filter countries
    unique_countries = sorted(mapped_df['country'].dropna().unique())
    selected_countries = st.multiselect("Select countries to include", options=unique_countries, default=unique_countries)

    if st.button("Apply Cleaning"):
        cleaned_df = mapped_df.copy()
        cleaned_df = cleaned_df[cleaned_df['country'].isin(selected_countries)]
        cleaned_df = cleaned_df.dropna(subset=['email', 'website'])
        cleaned_df = cleaned_df.drop_duplicates(subset=['email'])

        # Add 'occurences' column
        cleaned_df['occurences'] = cleaned_df.groupby('website').cumcount() + 1

        # Clean company names
        cleaned_df['company_name'] = cleaned_df['company_name'].str.replace(r'\s*\([^)]*\)', '', regex=True)\
                                                               .str.replace(r'\s*LLC', '', regex=True)\
                                                               .str.replace('✓', '', regex=False)\
                                                               .str.replace('"', '', regex=False)\
                                                               .str.strip()

        if not cleaned_df.empty:
            st.session_state['cleaned_df'] = cleaned_df
            st.success("✅ Cleaning applied!")
            st.write("### Cleaned Data Preview")
            st.dataframe(cleaned_df.head())

            # Download button
            csv = cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Cleaned CSV", data=csv, file_name='cleaned_output.csv', mime='text/csv')
        else:
            st.warning("⚠️ No data left after cleaning.")
