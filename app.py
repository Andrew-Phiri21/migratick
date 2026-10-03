import os
import re
import io
from typing import Union, Dict, Any, Tuple
import pandas as pd
from flask import Flask, render_template, request, send_file, flash, redirect, url_for

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "stark-arc-reactor-secret-key-9000")
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB max upload limit


class CardDataProcessor:
    """
    STARK Universal E-Toll Card Data Engine
    Handles schema drift, card number formatting, typos, and phone normalizations.
    """
    
    # Canonical Column Mapping Standard
    COLUMN_MAPPING = {
        'mobile': 'Phone',
        'phone': 'Phone',
        'type': 'Type',
        'distributor': 'Distributor',
        'card number': 'Card_Number',
        'card_number': 'Card_Number',
        'cardnumber': 'Card_Number',
        'balance': 'Balance',
        'customer': 'Customer'
    }

    # Value Normalization Rules
    TYPE_NORMALIZATION = {
        'STARDAND': 'Standard',
        'STANDARD': 'Standard',
        'EXEMPT': 'Exempt',
        'EXEMPTS': 'Exempt'
    }

    @classmethod
    def sanitize_card_number(cls, card_val: Any) -> str:
        """Strips non-digits and leading zeros to create a universal match key."""
        if pd.isna(card_val):
            return ""
        card_str = str(card_val).split('.')[0].strip()
        card_digits = re.sub(r'\D', '', card_str)
        return card_digits.lstrip('0')

    @classmethod
    def format_16digit_card(cls, clean_card_key: str) -> str:
        """Pads canonical card key to 16-digit system format."""
        if not clean_card_key:
            return ""
        return clean_card_key.zfill(16)

    @classmethod
    def sanitize_phone(cls, phone_val: Any) -> str:
        """Normalizes Zambian phone numbers to standardized 10-digit format."""
        if pd.isna(phone_val):
            return ""
        phone_str = str(phone_val).split('.')[0].strip()
        phone_digits = re.sub(r'\D', '', phone_str)
        
        # Handle country code prefix (e.g., 260977123456 -> 0977123456)
        if phone_digits.startswith('260') and len(phone_digits) == 12:
            phone_digits = '0' + phone_digits[3:]
            
        return phone_digits

    @classmethod
    def process_stream(cls, file_stream, filename: str) -> pd.DataFrame:
        """
        Ingests and cleans any vendor E-Toll file stream (Excel or CSV).
        """
        if filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_stream)
        elif filename.lower().endswith('.csv'):
            df = pd.read_csv(file_stream)
        else:
            raise ValueError(f"Unsupported file format for {filename}")
        
        # 1. Column Rename & Schema Normalization
        df.columns = [str(col).strip().lower() for col in df.columns]
        df = df.rename(columns=cls.COLUMN_MAPPING)

        # Ensure required columns exist
        required_cols = ['Type', 'Distributor', 'Card_Number', 'Balance', 'Customer', 'Phone']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        # 2. Card Number Standardization
        df['Card_Key'] = df['Card_Number'].apply(cls.sanitize_card_number)
        df['Formatted_Card_Number'] = df['Card_Key'].apply(cls.format_16digit_card)

        # 3. Categorical Value Normalization (Fixing 'STARDAND' typos)
        df['Type'] = df['Type'].astype(str).str.upper().str.strip()
        df['Type'] = df['Type'].map(lambda x: cls.TYPE_NORMALIZATION.get(x, x.title()))

        # 4. Data Types & Phone Cleaning
        df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').fillna(0.0)
        df['Phone'] = df['Phone'].apply(cls.sanitize_phone)
        df['Customer'] = df['Customer'].astype(str).str.strip()
        df['Distributor'] = df['Distributor'].astype(str).str.strip()

        # Reorder canonical columns
        final_df = df[[
            'Type', 
            'Distributor', 
            'Card_Key', 
            'Formatted_Card_Number', 
            'Balance', 
            'Customer', 
            'Phone'
        ]]

        return final_df


def reconcile_datasets(df_etc: pd.DataFrame, df_uts: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs full reconciliation between ETC and UTS datasets using the sanitized Card_Key.
    """
    merged = pd.merge(
        df_etc, 
        df_uts, 
        on='Card_Key', 
        how='outer', 
        suffixes=('_ETC', '_UTS')
    )

    # Classification logic
    matched = merged.dropna(subset=['Formatted_Card_Number_ETC', 'Formatted_Card_Number_UTS']).copy()
    etc_only = merged[merged['Formatted_Card_Number_UTS'].isna()].copy()
    uts_only = merged[merged['Formatted_Card_Number_ETC'].isna()].copy()

    # Balance discrepancy check on matched cards
    matched['Balance_Diff'] = (matched['Balance_ETC'] - matched['Balance_UTS']).abs()
    balance_mismatches = matched[matched['Balance_Diff'] > 0.01].copy()

    stats = {
        'total_etc': len(df_etc),
        'total_uts': len(df_uts),
        'matched_count': len(matched),
        'etc_only_count': len(etc_only),
        'uts_only_count': len(uts_only),
        'balance_mismatch_count': len(balance_mismatches)
    }

    return merged, stats


# --- Flask Application Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the primary dashboard UI."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_files():
    """Handles file uploads, runs ingestion engine, and generates download reports."""
    if 'etc_file' not in request.files or 'uts_file' not in request.files:
        flash("Error: Please provide both ETC and UTS files.", "danger")
        return redirect(url_for('index'))

    etc_file = request.files['etc_file']
    uts_file = request.files['uts_file']

    if etc_file.filename == '' or uts_file.filename == '':
        flash("Error: Select valid files to process.", "danger")
        return redirect(url_for('index'))

    try:
        # Ingest and sanitize datasets via STARK Processor
        df_etc_clean = CardDataProcessor.process_stream(etc_file, etc_file.filename)
        df_uts_clean = CardDataProcessor.process_stream(uts_file, uts_file.filename)

        # Run Reconciliation Pipeline
        reconciled_df, stats = reconcile_datasets(df_etc_clean, df_uts_clean)

        # Generate In-Memory Excel Workbook output
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            reconciled_df.to_excel(writer, sheet_name='Reconciliation_Full', index=False)
            df_etc_clean.to_excel(writer, sheet_name='ETC_Cleaned', index=False)
            df_uts_clean.to_excel(writer, sheet_name='UTS_Cleaned', index=False)
            
            # Summary stats sheet
            stats_df = pd.DataFrame([stats])
            stats_df.to_excel(writer, sheet_name='Summary_Stats', index=False)

        output_buffer.seek(0)

        return send_file(
            output_buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='EToll_Reconciliation_Report.xlsx'
        )

    except Exception as e:
        flash(f"System Failure during pipeline execution: {str(e)}", "danger")
        return redirect(url_for('index'))


if __name__ == '__main__':
    # Arc Reactor Power Grid Online
    app.run(host='0.0.0.0', port=5000, debug=True)
