import io
import re
from difflib import SequenceMatcher
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import streamlit as st

# High-performance C-accelerated Levenshtein distance fallback
try:
  from rapidfuzz.distance import Levenshtein

  HAS_RAPIDFUZZ = True
except ImportError:
  HAS_RAPIDFUZZ = False

# --- STAGE 0: STARK PAGE CONFIG & DASHBOARD STYLING ---
st.set_page_config(
    page_title="Migratick | Mark XII Universal Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background: #1e222d;
        padding: 14px;
        border-radius: 8px;
        border: 1px solid #2d313e;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #2d313e;
        border-radius: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- STAGE 1: JUNK GUARD & CANONICAL SANITIZATION ---


def is_valid_record_key(val: str) -> bool:
  """Excludes Excel export footers, administrative metadata, timestamps, and empty rows."""
  if val is None or pd.isna(val):
    return False

  s_val = str(val).strip().upper()

  # Filter out administrative/report footers & metadata
  junk_keywords = [
      "GENERATED",
      "BY ADMINISTRATOR",
      "TOTAL",
      "PAGE",
      "REPORT",
      "ADMIN",
      "SYSTEM",
      "TIMESTAMP",
      "PRINTED",
      "NAN",
      "NONE",
  ]
  if any(keyword in s_val for keyword in junk_keywords):
    return False

  # Key must contain alphanumeric content
  if not re.search(r"[A-Za-z0-9]", s_val):
    return False

  return True


def clean_card_identifier(val) -> str:
  """Strips decimal artifacts (.0 / .00), spaces, non-numeric junk, and leading zeros from card keys."""
  if pd.isna(val) or val is None:
    return ""

  val_str = str(val).replace("\xa0", " ").strip()

  # Remove trailing float decimal strings
  if val_str.endswith(".00"):
    val_str = val_str[:-3]
  elif val_str.endswith(".0"):
    val_str = val_str[:-2]

  val_str = val_str.split(".")[0].strip()

  # Remove non-digits for pure numeric keys
  digits_only = re.sub(r"\D", "", val_str)

  # Strip leading zeros for canonical alignment
  clean_key = digits_only.lstrip("0")

  return clean_key if clean_key else digits_only


def clean_phone_number(val) -> str:
  """Normalizes country code prefixes (+260 / 260 / 0) to standard 10-digit phone strings."""
  if pd.isna(val) or val == "":
    return ""
  val_str = str(val).replace("\xa0", " ").split(".")[0].strip()
  digits = re.sub(r"\D", "", val_str)

  if digits.startswith("260") and len(digits) == 12:
    digits = "0" + digits[3:]
  elif digits.startswith("26") and len(digits) > 8:
    digits = "0" + digits[2:]

  return digits


def normalize_business_text(text: str) -> str:
  """Strips legal entity noise words (LTD, LIMITED, MERCHANT) & punctuation for high-precision business matching."""
  if not text or pd.isna(text):
    return ""
  t = str(text).upper()

  # Remove noise tokens
  t = re.sub(
      r"\b(LIMITED|LTD|MERCHANT|ENTERPRISE|ENTERPRISES|AND|TRADERS|COMPANY|CO)\b",
      "",
      t,
  )
  # Retain alphanumeric characters only
  t = re.sub(r"[^A-Z0-9]", "", t)
  return t


def detect_header_row(file_bytes: bytes, is_csv: bool, max_scan_rows: int = 10) -> int:
  """Scans first N rows to dynamically identify the true header row index."""
  try:
    if is_csv:
      df_raw = pd.read_csv(
          io.BytesIO(file_bytes), nrows=max_scan_rows, header=None
      )
    else:
      df_raw = pd.read_excel(
          io.BytesIO(file_bytes),
          nrows=max_scan_rows,
          header=None,
          engine="openpyxl",
      )

    string_counts = df_raw.apply(
        lambda row: row.map(
            lambda x: isinstance(x, str)
            and not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ).sum(),
        axis=1,
    )
    return int(string_counts.idxmax())
  except Exception:
    return 0


def normalize_value(val, col_name: str = "", is_phone: bool = False) -> str:
  """Precision Column Normalizer."""
  if pd.isna(val) or val is None:
    return ""

  col_lower = col_name.lower()

  if any(k in col_lower for k in ["card", "id", "code", "account", "number"]):
    return clean_card_identifier(val)

  if is_phone or any(
      k in col_lower
      for k in ["phone", "mobile", "cell", "contact", "tel"]
  ):
    return clean_phone_number(val)

  val_str = str(val).replace("\xa0", " ")
  val_str = re.sub(r"\s+", " ", val_str).strip().upper()

  # Numeric & Currency formatting (strictly 2 decimals for balance)
  try:
    num_val = float(val_str)
    if np.isnan(num_val):
      return ""
    if num_val.is_integer():
      return f"{int(num_val)}"
    return f"{num_val:.2f}"
  except ValueError:
    pass

  if val_str.endswith(".00"):
    val_str = val_str[:-3]
  elif val_str.endswith(".0"):
    val_str = val_str[:-2]

  return val_str


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
  """Applies deep cleaning, column trimming, and decimal suppression across all fields."""
  df_clean = df.copy()

  df_clean.columns = [
      re.sub(r"\s+", " ", str(col).replace("\xa0", " ")).strip()
      for col in df_clean.columns
  ]

  for col in df_clean.columns:
    df_clean[col] = df_clean[col].apply(
        lambda x, c=col: normalize_value(x, col_name=c)
    )

  return df_clean


# --- STAGE 2: AI SEMANTIC & HYBRID MATCHING ENGINE ---

SEMANTIC_MAP = {
    "DEBIT": "DR",
    "DR": "DR",
    "D": "DR",
    "PURCHASE": "DR",
    "PAYMENT": "DR",
    "CREDIT": "CR",
    "CR": "CR",
    "C": "CR",
    "REFUND": "CR",
    "DEPOSIT": "CR",
    "STARDAND": "STANDARD",
    "STANDARD": "STANDARD",
}


def ai_similarity_score(val_a: str, val_b: str) -> float:
  """Calculates similarity score combining exact, semantic, and fuzzy matching algorithms."""
  if val_a == val_b:
    return 1.0

  mapped_a = SEMANTIC_MAP.get(val_a, val_a)
  mapped_b = SEMANTIC_MAP.get(val_b, val_b)
  if mapped_a == mapped_b:
    return 1.0

  # Check normalized business text identity
  norm_a = normalize_business_text(val_a)
  norm_b = normalize_business_text(val_b)
  if norm_a and norm_b and norm_a == norm_b:
    return 1.0

  if HAS_RAPIDFUZZ:
    dist = Levenshtein.distance(val_a, val_b)
    max_len = max(len(val_a), len(val_b))
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0

  return SequenceMatcher(None, val_a, val_b).ratio()


def auto_detect_primary_keys(
    df_a: pd.DataFrame, df_b: pd.DataFrame
) -> list[str]:
  """Evaluates column uniqueness ratios across datasets to identify optimal primary key candidate(s)."""
  common_cols = [col for col in df_a.columns if col in df_b.columns]
  best_key = None
  highest_score = -1.0

  for col in common_cols:
    unique_ratio_a = df_a[col].nunique() / len(df_a) if len(df_a) > 0 else 0
    unique_ratio_b = df_b[col].nunique() / len(df_b) if len(df_b) > 0 else 0
    avg_score = (unique_ratio_a + unique_ratio_b) / 2.0

    col_lower = col.lower()
    if any(
        k in col_lower
        for k in [
            "card",
            "id",
            "number",
            "code",
            "phone",
            "account",
            "email",
            "identifier",
            "reference",
        ]
    ):
      avg_score += 0.25

    if avg_score > highest_score:
      highest_score = avg_score
      best_key = col

  return [best_key] if best_key else []


# --- STAGE 3: VECTORIZED HIGH-PERFORMANCE COMPARISON ENGINE ---


def run_precision_comparison_v2(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    primary_keys: list[str],
    similarity_threshold: float = 0.88,
):
  """Executes composite key record alignment via vectorized merging.

  Excludes footer junk and decimal floating artifacts from keys.
  """
  summary_df = pd.DataFrame()
  mismatched_records = []

  common_cols = [col for col in df_a.columns if col in df_b.columns]
  if not common_cols or not all(k in common_cols for k in primary_keys):
    return summary_df, mismatched_records, [], [], 0

  df_a_aligned = df_a[common_cols].copy()
  df_b_aligned = df_b[common_cols].copy()

  # Build alignment key
  if len(primary_keys) == 1:
    pk_col = primary_keys[0]
    df_a_aligned["_ALIGN_KEY"] = df_a_aligned[pk_col].apply(
        clean_card_identifier
    )
    df_b_aligned["_ALIGN_KEY"] = df_b_aligned[pk_col].apply(
        clean_card_identifier
    )
  else:
    df_a_aligned["_ALIGN_KEY"] = (
        df_a_aligned[primary_keys]
        .astype(str)
        .apply(
            lambda row: "_".join([clean_card_identifier(x) for x in row]),
            axis=1,
        )
    )
    df_b_aligned["_ALIGN_KEY"] = (
        df_b_aligned[primary_keys]
        .astype(str)
        .apply(
            lambda row: "_".join([clean_card_identifier(x) for x in row]),
            axis=1,
        )
    )

  # Filter out non-record metadata and footers
  df_a_aligned = df_a_aligned[
      df_a_aligned["_ALIGN_KEY"].apply(is_valid_record_key)
  ].copy()
  df_b_aligned = df_b_aligned[
      df_b_aligned["_ALIGN_KEY"].apply(is_valid_record_key)
  ].copy()

  df_a_unique = df_a_aligned.drop_duplicates(subset=["_ALIGN_KEY"]).copy()
  df_b_unique = df_b_aligned.drop_duplicates(subset=["_ALIGN_KEY"]).copy()

  keys_a = set(df_a_unique["_ALIGN_KEY"])
  keys_b = set(df_b_unique["_ALIGN_KEY"])

  raw_missing_b = sorted(list(keys_a - keys_b))
  raw_missing_a = sorted(list(keys_b - keys_a))

  missing_b = [k for k in raw_missing_b if is_valid_record_key(k)]
  missing_a = [k for k in raw_missing_a if is_valid_record_key(k)]

  merged = pd.merge(
      df_a_unique,
      df_b_unique,
      on="_ALIGN_KEY",
      suffixes=("_A", "_B"),
      how="inner",
  )

  total_matched_keys = len(merged)
  if total_matched_keys == 0:
    return summary_df, mismatched_records, missing_b, missing_a, 0

  compare_cols = [
      c for c in common_cols if c not in primary_keys and c != "_ALIGN_KEY"
  ]

  # Row Delta Evaluation
  for _, row in merged.iterrows():
    key_val = row["_ALIGN_KEY"]
    row_diffs = {}

    for col in compare_cols:
      val_a = str(row[f"{col}_A"]) if pd.notna(row[f"{col}_A"]) else ""
      val_b = str(row[f"{col}_B"]) if pd.notna(row[f"{col}_B"]) else ""

      if val_a != val_b:
        score = ai_similarity_score(val_a, val_b)
        if score < similarity_threshold:
          row_diffs[col] = {
              "File_A": val_a,
              "File_B": val_b,
              "Match_Score": f"{score * 100:.1f}%",
          }

    if row_diffs:
      rec_a = {col: row[f"{col}_A"] for col in common_cols}
      rec_b = {col: row[f"{col}_B"] for col in common_cols}

      mismatched_records.append({
          "Primary_Key": key_val,
          "Mismatched_Columns": ", ".join(row_diffs.keys()),
          "Deltas": row_diffs,
          "Full_A": rec_a,
          "Full_B": rec_b,
      })

  # Column Parity Metrics
  col_summary = []
  for col in compare_cols:
    col_a = merged[f"{col}_A"].fillna("").astype(str).values
    col_b = merged[f"{col}_B"].fillna("").astype(str).values

    exact_matches = col_a == col_b
    fuzzy_matches = np.zeros(len(col_a), dtype=bool)

    mismatch_indices = np.where(~exact_matches)[0]
    for idx in mismatch_indices:
      if (
          ai_similarity_score(col_a[idx], col_b[idx])
          >= similarity_threshold
      ):
        fuzzy_matches[idx] = True

    total_matches = int(np.sum(exact_matches | fuzzy_matches))
    mismatches_count = total_matched_keys - total_matches
    confidence = (
        (total_matches / total_matched_keys * 100)
        if total_matched_keys > 0
        else 0.0
    )

    col_summary.append({
        "Column Name": col,
        "Matched Records": total_matches,
        "Mismatched Records": mismatches_count,
        "Confidence Rating": f"{confidence:.2f}%",
    })

  summary_df = pd.DataFrame(col_summary)
  return (
      summary_df,
      mismatched_records,
      missing_b,
      missing_a,
      total_matched_keys,
  )


# --- STAGE 4: EXCEL REPORT GENERATOR ---


def generate_excel_mismatch_report_v2(
    summary_df,
    mismatches,
    missing_b,
    missing_a,
    primary_keys,
    context_cols,
):
  """Generates styled context-aware multi-tab Excel workbook with decimal-free card formatting."""
  output = io.BytesIO()
  wb = openpyxl.Workbook()

  header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
  header_fill = PatternFill(
      start_color="1F4E78", end_color="1F4E78", fill_type="solid"
  )
  context_fill = PatternFill(
      start_color="2D6B9E", end_color="2D6B9E", fill_type="solid"
  )
  red_fill = PatternFill(
      start_color="FFC7CE", end_color="FFC7CE", fill_type="solid"
  )
  red_font = Font(name="Calibri", size=11, color="9C0006")
  center_align = Alignment(horizontal="center", vertical="center")
  left_align = Alignment(horizontal="left", vertical="center")

  # --- TAB 1: PARITY SUMMARY ---
  ws_summary = wb.active
  ws_summary.title = "Parity Summary"
  ws_summary.views.sheetView[0].showGridLines = True

  summary_headers = [
      "Column Name",
      "Matched Records",
      "Mismatched Records",
      "Confidence Rating",
  ]
  ws_summary.append(summary_headers)

  for col_num, header in enumerate(summary_headers, 1):
    cell = ws_summary.cell(row=1, column=col_num)
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = center_align

  for row_idx, row_data in enumerate(summary_df.to_dict("records"), 2):
    ws_summary.cell(
        row=row_idx, column=1, value=row_data["Column Name"]
    ).alignment = left_align
    ws_summary.cell(
        row=row_idx, column=2, value=row_data["Matched Records"]
    ).alignment = center_align

    m_cell = ws_summary.cell(
        row=row_idx, column=3, value=row_data["Mismatched Records"]
    )
    m_cell.alignment = center_align
    if row_data["Mismatched Records"] > 0:
      m_cell.fill = red_fill
      m_cell.font = red_font

    ws_summary.cell(
        row=row_idx, column=4, value=row_data["Confidence Rating"]
    ).alignment = center_align

  # --- TAB 2: SIDE-BY-SIDE DELTAS ---
  ws_side = wb.create_sheet(title="Side-by-Side Deltas")
  ws_side.views.sheetView[0].showGridLines = True

  all_mismatched_fields = sorted(
      list(
          set(
              field
              for item in mismatches
              for field in item["Deltas"].keys()
          )
      )
  )

  pk_label = ", ".join(primary_keys)
  side_headers = [f"Primary Key ({pk_label})"]
  for ctx in context_cols:
    side_headers.append(f"Context: {ctx}")

  for field in all_mismatched_fields:
    side_headers.append(f"{field} (File A)")
    side_headers.append(f"{field} (File B)")

  ws_side.append(side_headers)

  for col_num in range(1, len(side_headers) + 1):
    cell = ws_side.cell(row=1, column=col_num)
    cell.font = header_font
    cell.fill = (
        context_fill
        if "Context:" in side_headers[col_num - 1]
        else header_fill
    )
    cell.alignment = center_align

  for r_idx, item in enumerate(mismatches, 2):
    key_val = clean_card_identifier(item["Primary_Key"])
    deltas = item["Deltas"]
    full_a = item["Full_A"]

    c_key = ws_side.cell(row=r_idx, column=1, value=key_val)
    c_key.alignment = center_align
    c_key.number_format = "@"  # Force text format in Excel

    col_cursor = 2
    for ctx in context_cols:
      ctx_val = full_a.get(ctx, "")
      c_ctx = ws_side.cell(row=r_idx, column=col_cursor, value=ctx_val)
      c_ctx.alignment = left_align
      col_cursor += 1

    for field in all_mismatched_fields:
      c_a = ws_side.cell(row=r_idx, column=col_cursor)
      c_b = ws_side.cell(row=r_idx, column=col_cursor + 1)

      if field in deltas:
        c_a.value = deltas[field]["File_A"]
        c_b.value = deltas[field]["File_B"]
        c_a.fill = red_fill
        c_b.fill = red_fill
      else:
        c_a.value = "-"
        c_b.value = "-"
        c_a.alignment = center_align
        c_b.alignment = center_align

      col_cursor += 2

  # --- TAB 3: MISSING KEYS ---
  ws_missing = wb.create_sheet(title="Missing Keys")
  ws_missing.views.sheetView[0].showGridLines = True

  ws_missing.cell(
      row=1, column=1, value="Keys in File A (Missing in B)"
  ).font = header_font
  ws_missing.cell(row=1, column=1).fill = header_fill
  ws_missing.cell(
      row=1, column=2, value="Keys in File B (Missing in A)"
  ).font = header_font
  ws_missing.cell(row=1, column=2).fill = header_fill

  clean_missing_b = [
      clean_card_identifier(k) for k in missing_b if is_valid_record_key(k)
  ]
  clean_missing_a = [
      clean_card_identifier(k) for k in missing_a if is_valid_record_key(k)
  ]

  max_missing_len = max(len(clean_missing_b), len(clean_missing_a), 1)
  for i in range(max_missing_len):
    val_b = clean_missing_b[i] if i < len(clean_missing_b) else ""
    val_a = clean_missing_a[i] if i < len(clean_missing_a) else ""

    cell_b = ws_missing.cell(row=i + 2, column=1, value=val_b)
    cell_a = ws_missing.cell(row=i + 2, column=2, value=val_a)

    cell_b.alignment = center_align
    cell_a.alignment = center_align
    cell_b.number_format = "@"
    cell_a.number_format = "@"

  # Auto-fit column widths
  for ws in [ws_summary, ws_side, ws_missing]:
    for col in ws.columns:
      max_len = max(len(str(cell.value or "")) for cell in col)
      col_letter = get_column_letter(col[0].column)
      ws.column_dimensions[col_letter].width = max(max_len + 3, 15)

  wb.save(output)
  output.seek(0)
  return output


# --- STAGE 5: DASHBOARD UI & COMMAND CENTER ---

st.title("⚡ Migratick | Data Comparison Tool")
st.caption(
    "V-lookup and If Formula on steroids (Excludes"
    " Footers & Suppresses Card Decimals)"
)

col1, col2 = st.columns(2)
with col1:
  file_a = st.file_uploader(
      "📁 Upload Dataset A (Origin / ETC)", type=["csv", "xlsx"]
  )
with col2:
  file_b = st.file_uploader(
      "📁 Upload Dataset B (Target / UTS)", type=["csv", "xlsx"]
  )

if file_a and file_b:
  st.divider()

  is_csv_a = file_a.name.endswith(".csv")
  is_csv_b = file_b.name.endswith(".csv")
  bytes_a, bytes_b = file_a.getvalue(), file_b.getvalue()

  header_idx_a = detect_header_row(bytes_a, is_csv_a)
  header_idx_b = detect_header_row(bytes_b, is_csv_b)

  df_a_raw = (
      pd.read_csv(io.BytesIO(bytes_a), header=header_idx_a)
      if is_csv_a
      else pd.read_excel(
          io.BytesIO(bytes_a), header=header_idx_a, engine="openpyxl"
      )
  )
  df_b_raw = (
      pd.read_csv(io.BytesIO(bytes_b), header=header_idx_b)
      if is_csv_b
      else pd.read_excel(
          io.BytesIO(bytes_b), header=header_idx_b, engine="openpyxl"
      )
  )

  df_a = sanitize_dataframe(df_a_raw)
  df_b = sanitize_dataframe(df_b_raw)

  detected_keys = auto_detect_primary_keys(df_a, df_b)
  common_columns = [col for col in df_a.columns if col in df_b.columns]

  st.info(
      f"🔍 **Auto-Detection Active:** Headers located at Row {header_idx_a + 1}"
      f" (File A) & Row {header_idx_b + 1} (File B)."
      f" {'⚡ RapidFuzz Acceleration Active.' if HAS_RAPIDFUZZ else '⚠️ Using standard difflib engine.'}"
  )

  c_key, c_context, c_thresh = st.columns([2, 2, 1])
  with c_key:
    selected_keys = st.multiselect(
        "📌 Select Primary Alignment Key(s) [Supports Composite Keys]:",
        options=common_columns,
        default=detected_keys if detected_keys else [common_columns[0]],
    )

  available_context_cols = [c for c in common_columns if c not in selected_keys]
  default_contexts = [
      c
      for c in available_context_cols
      if any(
          k in c.lower()
          for k in [
              "merchant",
              "distributor",
              "name",
              "type",
              "date",
              "account",
          ]
      )
  ]

  with c_context:
    selected_context_cols = st.multiselect(
        "🏷️ Select Columns for Report:",
        options=available_context_cols,
        default=default_contexts[:2]
        if default_contexts
        else available_context_cols[:2],
        help="Included in exported sheets to contextualize row-level deltas.",
    )

  with c_thresh:
    ai_threshold = st.slider(
        "🤖 AI Accuracy:",
        min_value=0.70,
        max_value=1.00,
        value=0.88,
        step=0.02,
    )

  if selected_keys:
    (
        summary_df,
        mismatches,
        missing_b,
        missing_a,
        total_keys,
    ) = run_precision_comparison_v2(
        df_a, df_b, selected_keys, similarity_threshold=ai_threshold
    )

    # Metrics Overview
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("File A Rows", f"{len(df_a):,}")
    m2.metric("File B Rows", f"{len(df_b):,}")
    m3.metric("Aligned Key Pairs", f"{total_keys:,}")
    m4.metric("Row Mismatches", f"{len(mismatches):,}", delta_color="inverse")
    m5.metric(
        "Missing Keys",
        f"{len(missing_b) + len(missing_a):,}",
        delta_color="inverse",
    )

    st.divider()

    if not summary_df.empty:
      excel_data = generate_excel_mismatch_report_v2(
          summary_df,
          mismatches,
          missing_b,
          missing_a,
          selected_keys,
          selected_context_cols,
      )

      st.download_button(
          label="📥 Export Mismatch Report (.xlsx)",
          data=excel_data,
          file_name=f"Migratick_Mismatch_Report_{'_'.join(selected_keys)}.xlsx",
          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          use_container_width=True,
      )
      st.divider()

    # Column Parity Table
    st.subheader("📊 Matching Records & Confidence Matrix")

    def highlight_mismatches(val):
      try:
        if int(val) > 0:
          return (
              "background-color: #ff4b4b22; color: #ff4b4b; font-weight: bold;"
          )
      except (ValueError, TypeError):
        pass
      return ""

    if not summary_df.empty:
      st.dataframe(
          summary_df.style.map(
              highlight_mismatches, subset=["Mismatched Records"]
          ),
          use_container_width=True,
      )

    st.divider()

    tab_mismatches, tab_missing, tab_sanitized = st.tabs([
        "🔴 Data comparison",
        "⚠️ Missing Record Keys",
        "🧹 Sanitized Data Preview",
    ])

    with tab_mismatches:
      if mismatches:
        st.subheader(f"Identified {len(mismatches):,} Deviated Rows")

        side_by_side_preview = []
        for item in mismatches:
          row_dict = {
              f"Primary Key ({', '.join(selected_keys)})": (
                  clean_card_identifier(item["Primary_Key"])
              )
          }

          for ctx in selected_context_cols:
            row_dict[f"Context: {ctx}"] = item["Full_A"].get(ctx, "")

          for col_name, delta in item["Deltas"].items():
            row_dict[f"{col_name} (File A)"] = delta["File_A"]
            row_dict[f"{col_name} (File B)"] = delta["File_B"]

          side_by_side_preview.append(row_dict)

        st.dataframe(
            pd.DataFrame(side_by_side_preview).fillna("-"),
            use_container_width=True,
        )

        st.divider()
        selected_mismatch_key = st.selectbox(
            "🔎 Deep Analysis",
            [clean_card_identifier(m["Primary_Key"]) for m in mismatches],
        )

        if selected_mismatch_key:
          record = next(
              m
              for m in mismatches
              if clean_card_identifier(m["Primary_Key"])
              == selected_mismatch_key
          )
          c_left, c_right = st.columns(2)
          with c_left:
            st.markdown("**File A State:**")
            st.json(record["Full_A"])
          with c_right:
            st.markdown("**File B State:**")
            st.json(record["Full_B"])
      else:
        st.success("🎉 100% Parity Achieved across all aligned primary records!")

    with tab_missing:
      col_m1, col_m2 = st.columns(2)
      with col_m1:
        st.subheader(f"Keys in A, Missing in B ({len(missing_b):,})")
        st.dataframe(
            pd.DataFrame(
                {"Missing Key": [clean_card_identifier(k) for k in missing_b]}
            ),
            use_container_width=True,
        )
      with col_m2:
        st.subheader(f"Keys in B, Missing in A ({len(missing_a):,})")
        st.dataframe(
            pd.DataFrame(
                {"Missing Key": [clean_card_identifier(k) for k in missing_a]}
            ),
            use_container_width=True,
        )

    with tab_sanitized:
      c1, c2 = st.columns(2)
      with c1:
        st.write("Sanitized Dataset A (Head 5):", df_a.head())
      with c2:
        st.write("Sanitized Dataset B (Head 5):", df_b.head())
