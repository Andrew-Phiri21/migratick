import streamlit as st
import pandas as pd
import io
import re

# Set page layout to wide for dashboard-style display
st.set_page_config(page_title="Migratick | Precision Radar Engine", layout="wide")


# --- HELPER & SANITIZATION PIPELINE ---

def detect_header_row(file_bytes, is_csv, max_scan_rows=10):
    """
    Scans the first N rows of a file to dynamically identify the header row index based on string density.
    """
    try:
        if is_csv:
            df_raw = pd.read_csv(io.BytesIO(file_bytes), nrows=max_scan_rows, header=None)
        else:
            df_raw = pd.read_excel(io.BytesIO(file_bytes), nrows=max_scan_rows, header=None, engine="openpyxl")
        
        string_counts = df_raw.apply(
            lambda row: row.map(lambda x: isinstance(x, str) and not x.replace('.', '', 1).isdigit()).sum(), 
            axis=1
        )
        return int(string_counts.idxmax())
    except Exception:
        return 0


def clean_phone_number(val):
    """
    Strips phone number prefixes such as '26', '+26', or non-numeric characters and removes spaces.
    """
    if pd.isna(val) or val == "":
        return ""
    
    # Strip whitespace and non-breaking spaces
    val_str = str(val).replace('\xa0', ' ').split('.')[0].strip()
    digits = re.sub(r'\D', '', val_str)
    
    if digits.startswith('26') and len(digits) > 8:
        digits = digits[2:]
        
    return digits


def normalize_value(val, is_phone=False):
    """
    Applies Mark VI Data Hygiene: Removes non-breaking spaces, trims leading/trailing spaces,
    collapses multiple spaces, and normalizes float-integers (e.g. '1359.0' -> '1359').
    """
    if pd.isna(val) or val is None:
        return ""
    
    if is_phone:
        return clean_phone_number(val)
        
    # Replace non-breaking spaces and convert to string
    val_str = str(val).replace('\xa0', ' ')
    
    # Collapse multiple internal spaces down to a single space
    val_str = re.sub(r'\s+', ' ', val_str).strip()
    
    # Handle float representation of integers (e.g., '1359.0' -> '1359')
    if val_str.endswith('.0'):
        val_str = val_str[:-2]
        
    return val_str


def sanitize_dataframe(df):
    """
    Applies rigorous column and cell-level whitespace trimming and normalization across all fields.
    """
    df_clean = df.copy()
    
    # Clean Column Headers: strip and collapse whitespace
    df_clean.columns = [re.sub(r'\s+', ' ', str(col).replace('\xa0', ' ')).strip() for col in df_clean.columns]
    
    for col in df_clean.columns:
        col_lower = col.lower()
        is_phone_col = any(keyword in col_lower for keyword in ['phone', 'mobile', 'cell', 'contact', 'tel'])
        df_clean[col] = df_clean[col].apply(lambda x: normalize_value(x, is_phone=is_phone_col))
            
    return df_clean


def auto_detect_primary_key(df_a, df_b):
    """
    Evaluates column uniqueness ratios across datasets to identify the optimal primary alignment key.
    """
    common_cols = [col for col in df_a.columns if col in df_b.columns]
    best_key = None
    highest_score = -1.0
    
    for col in common_cols:
        unique_ratio_a = df_a[col].nunique() / len(df_a) if len(df_a) > 0 else 0
        unique_ratio_b = df_b[col].nunique() / len(df_b) if len(df_b) > 0 else 0
        avg_score = (unique_ratio_a + unique_ratio_b) / 2.0
        
        col_lower = col.lower()
        if any(k in col_lower for k in ['id', 'number', 'code', 'card', 'phone', 'account', 'email', 'identifier', 'name']):
            avg_score += 0.25
            
        if avg_score > highest_score:
            highest_score = avg_score
            best_key = col
            
    return best_key


# --- CORE COMPARISON ENGINE ---

def run_precision_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, primary_key: str):
    """
    Executes precise record alignment via direct Key Matching.
    Calculates per-column match counts, mismatches, and confidence percentages.
    Safely guarantees all return variables are explicitly initialized.
    """
    summary_df = pd.DataFrame()
    mismatched_records = []
    missing_b = []
    missing_a = []
    total_matched_keys = 0

    common_cols = [col for col in df_a.columns if col in df_b.columns]
    if not common_cols or primary_key not in common_cols:
        return summary_df, mismatched_records, missing_b, missing_a, total_matched_keys

    df_a_aligned = df_a[common_cols].copy()
    df_b_aligned = df_b[common_cols].copy()
    
    df_a_aligned[primary_key] = df_a_aligned[primary_key].astype(str).str.strip()
    df_b_aligned[primary_key] = df_b_aligned[primary_key].astype(str).str.strip()

    df_a_unique = df_a_aligned.drop_duplicates(subset=[primary_key]).copy()
    df_b_unique = df_b_aligned.drop_duplicates(subset=[primary_key]).copy()

    keys_a = set(df_a_unique[primary_key])
    keys_b = set(df_b_unique[primary_key])
    
    missing_b = sorted(list(keys_a - keys_b))
    missing_a = sorted(list(keys_b - keys_a))

    merged = pd.merge(
        df_a_unique, 
        df_b_unique, 
        on=primary_key, 
        suffixes=('_A', '_B'), 
        how='inner'
    )
    
    total_matched_keys = len(merged)
    if total_matched_keys == 0:
        return summary_df, mismatched_records, missing_b, missing_a, total_matched_keys

    compare_cols = [c for c in common_cols if c != primary_key]

    for _, row in merged.iterrows():
        key_val = row[primary_key]
        row_diffs = {}
        
        for col in compare_cols:
            val_a = str(row[f"{col}_A"]) if pd.notna(row[f"{col}_A"]) else ""
            val_b = str(row[f"{col}_B"]) if pd.notna(row[f"{col}_B"]) else ""
            
            if val_a != val_b:
                row_diffs[col] = {"File_A": val_a, "File_B": val_b}
                
        if row_diffs:
            rec_a = {col: row[f"{col}_A"] for col in compare_cols}
            rec_a[primary_key] = key_val
            rec_b = {col: row[f"{col}_B"] for col in compare_cols}
            rec_b[primary_key] = key_val
            
            mismatched_records.append({
                "Primary_Key": key_val,
                "Mismatched_Columns": ", ".join(row_diffs.keys()),
                "Deltas": row_diffs,
                "Full_A": rec_a,
                "Full_B": rec_b
            })

    col_summary = []
    for col in compare_cols:
        col_a = merged[f"{col}_A"].fillna("").astype(str)
        col_b = merged[f"{col}_B"].fillna("").astype(str)
        
        matches = int((col_a == col_b).sum())
        mismatches_count = total_matched_keys - matches
        confidence = (matches / total_matched_keys * 100) if total_matched_keys > 0 else 0.0
        
        col_summary.append({
            "Column Name": col,
            "Matched Records": matches,
            "Mismatched Records": mismatches_count,
            "Confidence Rating": f"{confidence:.2f}%"
        })
        
    summary_df = pd.DataFrame(col_summary)

    return summary_df, mismatched_records, missing_b, missing_a, total_matched_keys


# --- DASHBOARD UI ---

st.title("⚡ Migratick | Precision Radar Engine")
st.caption("Enterprise Dataset Parity, Auto-Normalization & Excel-Grade Analytics")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Dataset A (Base)", type=["csv", "xlsx"])
with col2:
    file_b = st.file_uploader("Upload Dataset B (Comparison)", type=["csv", "xlsx"])

if file_a and file_b:
    st.divider()
    
    is_csv_a = file_a.name.endswith(".csv")
    is_csv_b = file_b.name.endswith(".csv")
    bytes_a, bytes_b = file_a.getvalue(), file_b.getvalue()
    
    header_idx_a = detect_header_row(bytes_a, is_csv_a)
    header_idx_b = detect_header_row(bytes_b, is_csv_b)
    
    df_a_raw = pd.read_csv(io.BytesIO(bytes_a), header=header_idx_a) if is_csv_a else pd.read_excel(io.BytesIO(bytes_a), header=header_idx_a, engine="openpyxl")
    df_b_raw = pd.read_csv(io.BytesIO(bytes_b), header=header_idx_b) if is_csv_b else pd.read_excel(io.BytesIO(bytes_b), header=header_idx_b, engine="openpyxl")
    
    # Apply Mark VI Data Hygiene Pipeline
    df_a = sanitize_dataframe(df_a_raw)
    df_b = sanitize_dataframe(df_b_raw)
    
    detected_key = auto_detect_primary_key(df_a, df_b)
    common_columns = [col for col in df_a.columns if col in df_b.columns]
    
    st.info(f"🔍 **Auto-Detection Active:** Headers set at Row {header_idx_a + 1} (File A) & Row {header_idx_b + 1} (File B).")
    
    selected_key = st.selectbox(
        "📌 Select Primary Key for Sorting & Row Alignment:", 
        common_columns, 
        index=common_columns.index(detected_key) if detected_key in common_columns else 0
    )
    
    summary_df, mismatches, missing_b, missing_a, total_keys = run_precision_comparison(df_a, df_b, selected_key)
    
    # TOP-LEVEL METRICS
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("File A Rows", len(df_a))
    m2.metric("File B Rows", len(df_b))
    m3.metric("Aligned Key Pairs", total_keys)
    m4.metric("Row-Level Mismatches", len(mismatches), delta_color="inverse")
    m5.metric("Missing Keys", len(missing_b) + len(missing_a), delta_color="inverse")
    
    st.divider()
    
    # EXCEL-STYLE SUMMARY MATRIX
    st.subheader("📊 Column-Level Confidence & Parity Summary")
    
    def highlight_mismatches(val):
        try:
            val_int = int(val)
            if val_int > 0:
                return 'background-color: #ff4b4b33; color: #ff4b4b; font-weight: bold;'
        except (ValueError, TypeError):
            pass
        return ''

    if not summary_df.empty:
        st.dataframe(
            summary_df.style.map(highlight_mismatches, subset=["Mismatched Records"]),
            use_container_width=True
        )
    else:
        st.warning("No overlapping columns or matched keys found between the datasets.")
    
    st.divider()
    
    tab_mismatches, tab_missing, tab_sanitized = st.tabs([
        "🔴 Detailed Cell Mismatches", 
        "⚠️ Missing Record Keys", 
        "🧹 Sanitized Data Preview"
    ])
    
    with tab_mismatches:
        if mismatches:
            st.subheader(f"Identified {len(mismatches)} Rows with Value Deviations")
            
            preview_table = []
            for item in mismatches:
                preview_table.append({
                    "Primary Key": item["Primary_Key"],
                    "Mismatched Fields": item["Mismatched_Columns"],
                    "Field Deltas": str(item["Deltas"])
                })
            
            st.dataframe(pd.DataFrame(preview_table), use_container_width=True)
            
            selected_mismatch_key = st.selectbox("Inspect Record Key:", [m["Primary_Key"] for m in mismatches])
            if selected_mismatch_key:
                record = next(m for m in mismatches if m["Primary_Key"] == selected_mismatch_key)
                
                c_left, c_right = st.columns(2)
                with c_left:
                    st.write("**File A Data:**")
                    st.json(record["Full_A"])
                with c_right:
                    st.write("**File B Data:**")
                    st.json(record["Full_B"])
        else:
            st.success("🎉 100% Data Parity Achieved across all aligned records!")

    with tab_missing:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.subheader(f"Keys in A, Missing in B ({len(missing_b)})")
            st.dataframe(pd.DataFrame({selected_key: missing_b}), use_container_width=True)
        with col_m2:
            st.subheader(f"Keys in B, Missing in A ({len(missing_a)})")
            st.dataframe(pd.DataFrame({selected_key: missing_a}), use_container_width=True)

    with tab_sanitized:
        c1, c2 = st.columns(2)
        with c1:
            st.write("Sanitized File A (Top 5):", df_a.head())
        with c2:
            st.write("Sanitized File B (Top 5):", df_b.head())
