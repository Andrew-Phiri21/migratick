import streamlit as st
import pandas as pd
import io
import re

# Initialize Streamlit Dashboard Layout
st.set_page_config(page_title="Migratick | Precision Data Engine", layout="wide")

def detect_header_row(file_bytes, is_csv, max_scan_rows=10):
    """
    Scans the first N rows of a file to dynamically identify the header row index based on string density.
    """
    if is_csv:
        df_raw = pd.read_csv(io.BytesIO(file_bytes), nrows=max_scan_rows, header=None)
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes), nrows=max_scan_rows, header=None)
    
    # Calculate non-numeric string counts per row to detect the header candidate
    string_counts = df_raw.apply(
        lambda row: row.map(lambda x: isinstance(x, str) and not x.replace('.', '', 1).isdigit()).sum(), 
        axis=1
    )
    return int(string_counts.idxmax())

def clean_phone_number(val):
    """
    Strips phone number prefixes such as '26', '+26', or non-numeric characters.
    """
    if pd.isna(val):
        return ""
    # Convert float/int representations to clean string
    val_str = str(val).split('.')[0].strip()
    
    # Extract only digit characters
    digits = re.sub(r'\D', '', val_str)
    
    # Strip leading country code '26' if present
    if digits.startswith('26') and len(digits) > 8:
        digits = digits[2:]
        
    return digits

def sanitize_dataframe(df):
    """
    Applies data trimming, normalization, and phone number prefix stripping.
    """
    df_clean = df.copy()
    
    # Strip column headers
    df_clean.columns = [str(col).strip() for col in df_clean.columns]
    
    for col in df_clean.columns:
        # Evaluate if column represents phone numbers
        col_lower = col.lower()
        is_phone_col = any(keyword in col_lower for keyword in ['phone', 'mobile', 'cell', 'contact', 'tel'])
        
        if is_phone_col:
            df_clean[col] = df_clean[col].apply(clean_phone_number)
        else:
            # General string trimming and whitespace cleanup
            df_clean[col] = df_clean[col].apply(
                lambda x: str(x).strip() if pd.notna(x) and str(x).strip() != "" else ""
            )
            
    return df_clean

def auto_detect_primary_key(df_a, df_b):
    """
    Evaluates column candidates across both datasets to find the optimal primary key based on uniqueness.
    """
    common_cols = [col for col in df_a.columns if col in df_b.columns]
    
    best_key = None
    highest_score = -1.0
    
    for col in common_cols:
        # Check uniqueness ratio in both datasets
        unique_ratio_a = df_a[col].nunique() / len(df_a) if len(df_a) > 0 else 0
        unique_ratio_b = df_b[col].nunique() / len(df_b) if len(df_b) > 0 else 0
        
        # Combined score prioritizing higher uniqueness
        avg_score = (unique_ratio_a + unique_ratio_b) / 2.0
        
        # Favor common unique identifiers like ID, Card Number, Phone Number
        col_lower = col.lower()
        if any(k in col_lower for k in ['id', 'number', 'code', 'card', 'phone', 'account', 'email']):
            avg_score += 0.2
            
        if avg_score > highest_score:
            highest_score = avg_score
            best_key = col
            
    return best_key

def run_migratick_engine(df_a, df_b, primary_key):
    """
    Executes aligned record matching, sorting, line-by-line comparison, and missing record identification.
    """
    # Align structural schemas
    common_cols = [col for col in df_a.columns if col in df_b.columns]
    df_a = df_a[common_cols]
    df_b = df_b[common_cols]
    
    # Sort datasets deterministically by Primary Key
    df_a = df_a.sort_values(by=primary_key).reset_index(drop=True)
    df_b = df_b.sort_values(by=primary_key).reset_index(drop=True)
    
    # Indexed comparison map
    dict_a = df_a.set_index(primary_key).to_dict(orient='index')
    dict_b = df_b.set_index(primary_key).to_dict(orient='index')
    
    keys_a = set(dict_a.keys())
    keys_b = set(dict_b.keys())
    
    common_keys = keys_a.intersection(keys_b)
    missing_in_b = keys_a - keys_b
    missing_in_a = keys_b - keys_a
    
    mismatches = []
    matched_records = []
    
    for key in common_keys:
        row_a = dict_a[key]
        row_b = dict_b[key]
        
        diffs = {}
        for col in row_a:
            val_a = row_a[col]
            val_b = row_b[col]
            if val_a != val_b:
                diffs[col] = {"Dataset_A": val_a, "Dataset_B": val_b}
                
        if diffs:
            mismatches.append({
                "Primary_Key": key,
                "Mismatched_Columns": ", ".join(diffs.keys()),
                "Deltas": diffs,
                "Full_Record_A": row_a,
                "Full_Record_B": row_b
            })
        else:
            matched_records.append({primary_key: key, **row_a})
            
    df_matched = pd.DataFrame(matched_records)
    
    return df_matched, mismatches, list(missing_in_b), list(missing_in_a)

# --- UI DASHBOARD ---
st.title("⚡ Migratick | Precision Radar Engine")
st.caption("Automated Natural Key Alignment, Sanitization & Mismatch Analysis")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Dataset A (Base)", type=["csv", "xlsx"])
with col2:
    file_b = st.file_uploader("Upload Dataset B (Comparison)", type=["csv", "xlsx"])

if file_a and file_b:
    st.divider()
    
    is_csv_a = file_a.name.endswith(".csv")
    is_csv_b = file_b.name.endswith(".csv")
    
    bytes_a = file_a.getvalue()
    bytes_b = file_b.getvalue()
    
    # 1. Dynamic Header Detection
    header_idx_a = detect_header_row(bytes_a, is_csv_a)
    header_idx_b = detect_header_row(bytes_b, is_csv_b)
    
    # Load raw data with detected headers
    df_a_raw = pd.read_csv(io.BytesIO(bytes_a), header=header_idx_a) if is_csv_a else pd.read_excel(io.BytesIO(bytes_a), header=header_idx_a)
    df_b_raw = pd.read_csv(io.BytesIO(bytes_b), header=header_idx_b) if is_csv_b else pd.read_excel(io.BytesIO(bytes_b), header=header_idx_b)
    
    # 2. Sanitization Pipeline (Trimming + Phone Prefix Stripping)
    df_a = sanitize_dataframe(df_a_raw)
    df_b = sanitize_dataframe(df_b_raw)
    
    # 3. Auto Natural Primary Key Detection
    detected_key = auto_detect_primary_key(df_a, df_b)
    
    st.info(f"🔍 **Auto-Configuration Complete:** Headers detected (A: Row {header_idx_a + 1}, B: Row {header_idx_b + 1}). Selected Natural Key: **`{detected_key}`**")
    
    # Allow manual override if needed
    common_columns = [col for col in df_a.columns if col in df_b.columns]
    selected_key = st.selectbox("Primary Key Alignment Column:", common_columns, index=common_columns.index(detected_key) if detected_key in common_columns else 0)
    
    # Run Comparison Engine
    df_matched, mismatches, missing_b, missing_a = run_migratick_engine(df_a, df_b, selected_key)
    
    # Metrics Overview
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Dataset A Total", len(df_a))
    m2.metric("Dataset B Total", len(df_b))
    m3.metric("Perfect Matches", len(df_matched))
    m4.metric("Value Mismatches", len(mismatches), delta_color="inverse")
    m5.metric("Unmatched Keys", len(missing_b) + len(missing_a), delta_color="inverse")
    
    st.divider()
    
    # Tab Views
    tab_mismatches, tab_missing, tab_matched, tab_clean = st.tabs([
        "🔴 Cell Mismatches", 
        "⚠️ Missing Records", 
        "🟢 Perfect Matches", 
        "🧹 Sanitized Preview"
    ])
    
    with tab_mismatches:
        if mismatches:
            st.subheader(f"Identified {len(mismatches)} Value Deviations")
            
            # Formatted Preview Table
            preview_table = []
            for item in mismatches:
                preview_table.append({
                    "Key Value": item["Primary_Key"],
                    "Mismatched Fields": item["Mismatched_Columns"],
                    "Field Deltas": str(item["Deltas"])
                })
            
            st.dataframe(pd.DataFrame(preview_table), use_container_width=True)
            
            # Drilldown Panel
            selected_mismatch_key = st.selectbox("Inspect Record Key:", [m["Primary_Key"] for m in mismatches])
            if selected_mismatch_key:
                record = next(m for m in mismatches if m["Primary_Key"] == selected_mismatch_key)
                
                c_left, c_right = st.columns(2)
                with c_left:
                    st.write("**Dataset A Record:**")
                    st.json(record["Full_Record_A"])
                with c_right:
                    st.write("**Dataset B Record:**")
                    st.json(record["Full_Record_B"])
        else:
            st.success("🎉 All matching keys contain identical cell values.")

    with tab_missing:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.subheader(f"Missing in Dataset B ({len(missing_b)})")
            st.write(pd.DataFrame({selected_key: missing_b}))
        with col_m2:
            st.subheader(f"Missing in Dataset A ({len(missing_a)})")
            st.write(pd.DataFrame({selected_key: missing_a}))

    with tab_matched:
        st.dataframe(df_matched, use_container_width=True)

    with tab_clean:
        st.write("Sanitized Dataset A (First 5 Rows):", df_a.head())
        st.write("Sanitized Dataset B (First 5 Rows):", df_b.head())
