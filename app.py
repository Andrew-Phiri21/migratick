import streamlit as st
import pandas as pd
import io
import re

st.set_page_config(page_title="Migratick | Precision Radar Engine", layout="wide")

def detect_header_row(file_bytes, is_csv, max_scan_rows=10):
    if is_csv:
        df_raw = pd.read_csv(io.BytesIO(file_bytes), nrows=max_scan_rows, header=None)
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes), nrows=max_scan_rows, header=None)
    
    string_counts = df_raw.apply(
        lambda row: row.map(lambda x: isinstance(x, str) and not x.replace('.', '', 1).isdigit()).sum(), 
        axis=1
    )
    return int(string_counts.idxmax())

def clean_phone_number(val):
    if pd.isna(val):
        return ""
    val_str = str(val).split('.')[0].strip()
    digits = re.sub(r'\D', '', val_str)
    
    if digits.startswith('26') and len(digits) > 8:
        digits = digits[2:]
        
    return digits

def sanitize_dataframe(df):
    df_clean = df.copy()
    df_clean.columns = [str(col).strip() for col in df_clean.columns]
    
    for col in df_clean.columns:
        col_lower = col.lower()
        is_phone_col = any(keyword in col_lower for keyword in ['phone', 'mobile', 'cell', 'contact', 'tel'])
        
        if is_phone_col:
            df_clean[col] = df_clean[col].apply(clean_phone_number)
        else:
            df_clean[col] = df_clean[col].apply(
                lambda x: str(x).strip() if pd.notna(x) and str(x).strip() != "" else ""
            )
            
    return df_clean

def auto_detect_primary_key(df_a, df_b):
    common_cols = [col for col in df_a.columns if col in df_b.columns]
    
    best_key = None
    highest_score = -1.0
    
    for col in common_cols:
        unique_ratio_a = df_a[col].nunique() / len(df_a) if len(df_a) > 0 else 0
        unique_ratio_b = df_b[col].nunique() / len(df_b) if len(df_b) > 0 else 0
        
        avg_score = (unique_ratio_a + unique_ratio_b) / 2.0
        
        col_lower = col.lower()
        if any(k in col_lower for k in ['id', 'number', 'code', 'card', 'phone', 'account', 'email']):
            avg_score += 0.2
            
        if avg_score > highest_score:
            highest_score = avg_score
            best_key = col
            
    return best_key

def build_unique_indexed_dict(df, primary_key):
    """
    Constructs a dictionary indexing records while appending occurrence markers 
    if duplicate keys exist in the dataset to prevent index collision errors.
    """
    indexed_dict = {}
    occurrence_counts = {}
    
    # Sort deterministically
    df_sorted = df.sort_values(by=primary_key).reset_index(drop=True)
    
    for _, row in df_sorted.iterrows():
        raw_key = str(row[primary_key])
        
        # Track duplicate occurrences
        count = occurrence_counts.get(raw_key, 0) + 1
        occurrence_counts[raw_key] = count
        
        # Construct composite unique key if duplicate key found
        unique_composite_key = raw_key if count == 1 else f"{raw_key} (Dup #{count})"
        
        indexed_dict[unique_composite_key] = {
            "raw_key": raw_key,
            "data": row.to_dict()
        }
        
    return indexed_dict

def run_migratick_engine(df_a, df_b, primary_key):
    """
    Executes aligned record matching using duplicate-safe indexing.
    """
    common_cols = [col for col in df_a.columns if col in df_b.columns]
    df_a = df_a[common_cols]
    df_b = df_b[common_cols]
    
    dict_a = build_unique_indexed_dict(df_a, primary_key)
    dict_b = build_unique_indexed_dict(df_b, primary_key)
    
    keys_a = set(dict_a.keys())
    keys_b = set(dict_b.keys())
    
    common_keys = keys_a.intersection(keys_b)
    missing_in_b = [dict_a[k]["raw_key"] for k in (keys_a - keys_b)]
    missing_in_a = [dict_b[k]["raw_key"] for k in (keys_b - keys_a)]
    
    mismatches = []
    matched_records = []
    
    for key in common_keys:
        row_a = dict_a[key]["data"]
        row_b = dict_b[key]["data"]
        raw_key = dict_a[key]["raw_key"]
        
        diffs = {}
        for col in row_a:
            val_a = row_a[col]
            val_b = row_b[col]
            if val_a != val_b:
                diffs[col] = {"Dataset_A": val_a, "Dataset_B": val_b}
                
        if diffs:
            mismatches.append({
                "Primary_Key": raw_key,
                "Composite_Key": key,
                "Mismatched_Columns": ", ".join(diffs.keys()),
                "Deltas": diffs,
                "Full_Record_A": row_a,
                "Full_Record_B": row_b
            })
        else:
            matched_records.append(row_a)
            
    df_matched = pd.DataFrame(matched_records)
    
    return df_matched, mismatches, missing_b, missing_a

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
    
    header_idx_a = detect_header_row(bytes_a, is_csv_a)
    header_idx_b = detect_header_row(bytes_b, is_csv_b)
    
    df_a_raw = pd.read_csv(io.BytesIO(bytes_a), header=header_idx_a) if is_csv_a else pd.read_excel(io.BytesIO(bytes_a), header=header_idx_a, engine="openpyxl")
    df_b_raw = pd.read_csv(io.BytesIO(bytes_b), header=header_idx_b) if is_csv_b else pd.read_excel(io.BytesIO(bytes_b), header=header_idx_b, engine="openpyxl")
    
    df_a = sanitize_dataframe(df_a_raw)
    df_b = sanitize_dataframe(df_b_raw)
    
    detected_key = auto_detect_primary_key(df_a, df_b)
    
    st.info(f"🔍 **Auto-Configuration Complete:** Headers detected (A: Row {header_idx_a + 1}, B: Row {header_idx_b + 1}). Selected Natural Key: **`{detected_key}`**")
    
    common_columns = [col for col in df_a.columns if col in df_b.columns]
    selected_key = st.selectbox("Primary Key Alignment Column:", common_columns, index=common_columns.index(detected_key) if detected_key in common_columns else 0)
    
    df_matched, mismatches, missing_b, missing_a = run_migratick_engine(df_a, df_b, selected_key)
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Dataset A Total", len(df_a))
    m2.metric("Dataset B Total", len(df_b))
    m3.metric("Perfect Matches", len(df_matched))
    m4.metric("Value Mismatches", len(mismatches), delta_color="inverse")
    m5.metric("Unmatched Keys", len(missing_b) + len(missing_a), delta_color="inverse")
    
    st.divider()
    
    tab_mismatches, tab_missing, tab_matched, tab_clean = st.tabs([
        "🔴 Cell Mismatches", 
        "⚠️ Missing Records", 
        "🟢 Perfect Matches", 
        "🧹 Sanitized Preview"
    ])
    
    with tab_mismatches:
        if mismatches:
            st.subheader(f"Identified {len(mismatches)} Value Deviations")
            
            preview_table = []
            for item in mismatches:
                preview_table.append({
                    "Key Value": item["Primary_Key"],
                    "Record Marker": item["Composite_Key"],
                    "Mismatched Fields": item["Mismatched_Columns"],
                    "Field Deltas": str(item["Deltas"])
                })
            
            st.dataframe(pd.DataFrame(preview_table), use_container_width=True)
            
            selected_mismatch_key = st.selectbox("Inspect Record Marker:", [m["Composite_Key"] for m in mismatches])
            if selected_mismatch_key:
                record = next(m for m in mismatches if m["Composite_Key"] == selected_mismatch_key)
                
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
