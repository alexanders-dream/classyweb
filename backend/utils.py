# backend/utils.py
"""General utility functions for the backend."""

import pandas as pd
from io import BytesIO
import traceback
import logging
from pathlib import Path 
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Assuming config is in the same directory or PYTHONPATH is set correctly
try:
    from . import config # Relative import for package structure
except ImportError:
    import config # Fallback for running script directly (less ideal)

# Setup logger for this module
logger = logging.getLogger(__name__)

# --- Session State Management (REMOVED) ---
# init_session_state() - REMOVED
# restart_session() - REMOVED

# --- Data Handling ---
def load_data(file_path: Path | str, original_filename: str) -> Optional[pd.DataFrame]:
    """Loads data from CSV or Excel file path, handles common errors, using original filename for type detection."""
    if not file_path or not original_filename:
        logger.error("load_data received an empty file path or original filename.")
        return None

    file_path = Path(file_path) # Ensure it's a Path object
    if not file_path.exists():
        logger.error(f"File not found at path: {file_path}")
        return None

    try:
        # Use original_filename for logging and extension check
        logger.info(f"Loading '{original_filename}' from path: {file_path}")
        df = None
        # Use original_filename.lower() for case-insensitive extension check
        if original_filename.lower().endswith('.csv'):
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decoding failed for '{original_filename}', trying latin1...")
                df = pd.read_csv(file_path, encoding='latin1')
            except pd.errors.EmptyDataError:
                 logger.warning(f"CSV file '{original_filename}' is empty.")
                 return pd.DataFrame()
            except Exception as e_csv:
                 logger.error(f"Error reading CSV file '{original_filename}': {e_csv}", exc_info=True)
                 return None
        elif original_filename.lower().endswith(('.xls', '.xlsx')):
             try:
                df = pd.read_excel(file_path, engine='openpyxl')
             except Exception as e_excel:
                 logger.error(f"Error reading Excel file '{original_filename}': {e_excel}", exc_info=True)
                 return None
        else:
            # Log error using original_filename
            logger.error(f"Unsupported file format for '{original_filename}'. Please use CSV or Excel.")
            return None # Explicitly return None for unsupported format

        if df is None:
             logger.error(f"DataFrame remained None after attempting to load '{original_filename}'.")
             return None

        # Basic cleaning (remains the same)
        original_shape = df.shape
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        if df.shape != original_shape:
             logger.info(f"Dropped empty rows/columns from '{original_filename}'. Original: {original_shape}, New: {df.shape}")

        logger.info(f"✅ Successfully loaded '{original_filename}' ({df.shape[0]} rows, {df.shape[1]} columns)")
        df.columns = df.columns.astype(str)
        return df

    except Exception as e:
        logger.error(f"Generic error loading file '{original_filename}': {e}")
        logger.error(traceback.format_exc())
        return None


# Removed @st.cache_data
def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Converts DataFrame to Excel bytes using xlsxwriter, falls back to openpyxl."""
    output = BytesIO()
    engine_to_use = 'xlsxwriter'
    try:
        # Try with xlsxwriter first
        import xlsxwriter # Ensure it's available or handle import error
        logger.info("Attempting to write Excel using xlsxwriter.")
        with pd.ExcelWriter(output, engine=engine_to_use) as writer:
            df.to_excel(writer, index=False, sheet_name='ClassificationResults')
        logger.info("Excel conversion with xlsxwriter successful.")
        return output.getvalue()
    except ImportError:
        logger.warning("`xlsxwriter` not found, falling back to `openpyxl`.")
        engine_to_use = 'openpyxl'
    except Exception as e_xlsx:
        logger.error(f"Error generating Excel file with {engine_to_use}: {e_xlsx}", exc_info=True)
        # If xlsxwriter failed, try openpyxl anyway
        if engine_to_use == 'xlsxwriter':
             logger.info("Retrying Excel generation with openpyxl.")
             engine_to_use = 'openpyxl'
        else:
            return b"" # Already failed with openpyxl

    # Try (or retry) with openpyxl
    if engine_to_use == 'openpyxl':
         try:
             # Ensure openpyxl is installed if needed
             import openpyxl
             logger.info("Attempting to write Excel using openpyxl.")
             output = BytesIO() # Reset buffer
             with pd.ExcelWriter(output, engine='openpyxl') as writer:
                 df.to_excel(writer, index=False, sheet_name='ClassificationResults')
             logger.info("Excel conversion with openpyxl successful.")
             return output.getvalue()
         except ImportError:
              logger.error("Neither xlsxwriter nor openpyxl seem to be installed. Cannot generate Excel.")
              return b""
         except Exception as e_openpyxl:
              logger.error(f"Error generating Excel file with openpyxl: {e_openpyxl}", exc_info=True)
              return b""
    return b"" # Should not be reached normally

# --- Hierarchy Manipulation (Keep as is for now) ---
# build_hierarchy_from_df - Keep
# flatten_hierarchy - Keep
# parse_predicted_labels_to_columns - Keep
# (Definitions omitted for brevity, assume they are copied and don't use st.*)
# Make sure imports like defaultdict and config are correct within these functions
def build_hierarchy_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Converts a flat hierarchy DataFrame back into a nested dictionary structure.
    (Copied from original, ensure no st.* calls remain)
    """
    if df is None or df.empty:
        return {'themes': []}

    hierarchy = {'themes': []}
    themes_dict = defaultdict(lambda: {'name': '', 'categories': defaultdict(lambda: {'name': '', 'segments': defaultdict(lambda: {'name': '', 'subsegments': []})})})

    required_base_cols = ['Theme', 'Category', 'Segment']
    subsegment_key = "Subsegment"
    subsegment_display_key = "Sub-Segment"
    keywords_key = "Keywords"

    df_processed = df.copy()

    actual_subsegment_col = None
    if subsegment_key in df_processed.columns: actual_subsegment_col = subsegment_key
    elif subsegment_display_key in df_processed.columns: actual_subsegment_col = subsegment_display_key
    else:
        logger.error(f"Build Hierarchy Error: Neither '{subsegment_key}' nor '{subsegment_display_key}' found.")
        return {'themes': []} # Indicate error

    all_expected_cols = required_base_cols + [actual_subsegment_col, keywords_key]

    for col in all_expected_cols:
        if col not in df_processed.columns:
            logger.warning(f"Build Hierarchy Warning: Column '{col}' missing in input DataFrame. Creating empty column.")
            df_processed[col] = ''

    # Ensure only expected columns remain and fill NaN/astype string
    df_processed = df_processed[all_expected_cols].fillna('').astype(str)

    processed_rows, skipped_rows = 0, 0
    for _, row in df_processed.iterrows():
        theme_name = row['Theme'].strip()
        cat_name = row['Category'].strip()
        seg_name = row['Segment'].strip()
        sub_seg_name = row[actual_subsegment_col].strip()

        # Skip row if any essential part of the path is missing
        if not all([theme_name, cat_name, seg_name, sub_seg_name]):
            skipped_rows += 1
            continue

        keywords_raw = row.get(keywords_key, '')
        keywords = [k.strip() for k in keywords_raw.split(',') if k.strip()]

        # Build nested structure using defaultdict
        themes_dict[theme_name]['name'] = theme_name
        categories_dict = themes_dict[theme_name]['categories']
        categories_dict[cat_name]['name'] = cat_name
        segments_dict = categories_dict[cat_name]['segments']
        segments_dict[seg_name]['name'] = seg_name

        subsegments_list = segments_dict[seg_name]['subsegments']
        # Avoid duplicate subsegments within the same segment
        if not any(ss['name'] == sub_seg_name for ss in subsegments_list):
             subsegments_list.append({'name': sub_seg_name, 'keywords': keywords})
        processed_rows += 1

    if skipped_rows > 0:
        logger.info(f"Build Hierarchy: Skipped {skipped_rows} rows due to missing required path names (Theme, Category, Segment, Subsegment).")

    # Convert defaultdict structure back to regular dicts/lists
    final_themes = []
    for theme_name, theme_data in themes_dict.items():
        final_categories = []
        for cat_name, cat_data in theme_data['categories'].items():
            final_segments = []
            for seg_name, seg_data in cat_data['segments'].items():
                 if seg_data['subsegments']: # Only add segments that have subsegments
                     final_segments.append({'name': seg_data['name'], 'subsegments': seg_data['subsegments']})
            if final_segments: # Only add categories that have valid segments
                 final_categories.append({'name': cat_data['name'], 'segments': final_segments})
        if final_categories: # Only add themes that have valid categories
             final_themes.append({'name': theme_data['name'], 'categories': final_categories})

    final_hierarchy = {'themes': final_themes}
    if not final_themes and processed_rows > 0:
        logger.warning("Build Hierarchy: Processed rows, but the resulting nested hierarchy is empty. Check input data structure.")
    elif not final_themes:
        logger.info("Build Hierarchy: Input DataFrame resulted in an empty hierarchy.")

    return final_hierarchy


def flatten_hierarchy(nested_hierarchy: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts AI-generated nested hierarchy dict to a flat DataFrame.
    (Copied from original, ensure no st.* calls remain)
    """
    rows = []
    required_cols = ['Theme', 'Category', 'Segment', 'Subsegment', 'Keywords']

    if not nested_hierarchy or 'themes' not in nested_hierarchy:
        logger.warning("Flatten Hierarchy: Input is None or missing 'themes' key.")
        return pd.DataFrame(columns=required_cols)

    try:
        for theme in nested_hierarchy.get('themes', []):
            theme_name = str(theme.get('name', '')).strip()
            if not theme_name: continue

            for category in theme.get('categories', []):
                cat_name = str(category.get('name', '')).strip()
                if not cat_name: continue

                for segment in category.get('segments', []):
                    seg_name = str(segment.get('name', '')).strip()
                    if not seg_name: continue

                    if not segment.get('subsegments'): continue

                    for sub_segment in segment.get('subsegments', []):
                        sub_seg_name = str(sub_segment.get('name', '')).strip()
                        if not sub_seg_name: continue

                        keywords_list = [str(k).strip() for k in sub_segment.get('keywords', []) if str(k).strip()]
                        keywords_str = ', '.join(keywords_list)

                        rows.append({
                            'Theme': theme_name,
                            'Category': cat_name,
                            'Segment': seg_name,
                            'Subsegment': sub_seg_name,
                            'Keywords': keywords_str
                        })
    except Exception as e:
        logger.error(f"Error during hierarchy flattening: {e}", exc_info=True)
        return pd.DataFrame(columns=required_cols) # Return empty on error

    return pd.DataFrame(rows, columns=required_cols)


def parse_predicted_labels_to_columns(predicted_labels_list: List[List[str]]) -> List[Dict[str, Optional[str]]]:
    """
    Parses lists of potentially prefixed predicted labels into structured dictionaries.
    (Copied from original, ensure no st.* calls remain)
    """
    structured_results = []
    # Use HIERARCHY_LEVELS from config
    prefixes = {level: f"{level.lower()}:" for level in config.HIERARCHY_LEVELS}
    hierarchy_levels = config.HIERARCHY_LEVELS # Local copy for clarity

    for labels in predicted_labels_list:
        row_dict: Dict[str, Optional[str]] = {level: None for level in hierarchy_levels}
        if not labels:
            structured_results.append(row_dict)
            continue

        # Use a dictionary to store the *first* label found for each level
        first_found_label = {level: None for level in hierarchy_levels}

        for label in labels:
            if not isinstance(label, str) or not label.strip(): continue # Skip non-strings or empty strings

            label_lower = label.lower()
            label_processed = False
            for level in hierarchy_levels:
                prefix_lower = prefixes[level]
                if label_lower.startswith(prefix_lower):
                    # Extract value after "Level: "
                    value = label[len(level) + 2:].strip()
                    if value and first_found_label[level] is None: # Only store the first one found
                        first_found_label[level] = value
                    label_processed = True
                    break # Move to next label once prefix match found

            # Optional: Handle labels *without* a prefix (e.g., assign to lowest level if appropriate?)
            # This depends on the expected output format from the models.
            # If no prefix match, could potentially try assigning to 'Theme' or 'Subsegment' if empty?
            # Example (use with caution):
            # if not label_processed and first_found_label["Subsegment"] is None:
            #     first_found_label["Subsegment"] = label.strip() # Assign non-prefixed to lowest level

        # Assign the first found label for each level to the result dict
        for level in hierarchy_levels:
            row_dict[level] = first_found_label[level]

        structured_results.append(row_dict)

    return structured_results


# --- Statistics Display (REMOVED) ---
# display_hierarchical_stats() - REMOVED (Logic might be reused in an API endpoint later)