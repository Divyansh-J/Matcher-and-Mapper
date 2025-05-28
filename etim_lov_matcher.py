import pandas as pd
import numpy as np
from pathlib import Path
from fuzzywuzzy import fuzz
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ETIMtoLOVMatcher:
    def __init__(self, etim_path, lov_path, output_dir="output"):
        """
        Initialize the ETIM to LOV matcher with file paths
        
        Args:
            etim_path (str): Path to ETIM Excel file
            lov_path (str): Path to LOV Excel file
            output_dir (str): Directory to save output files
        """
        self.etim_path = Path(etim_path)
        self.lov_path = Path(lov_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize dataframes
        self.etim_df = None
        self.lov_df = None
        self.matched_results = []
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess the input data"""
        logger.info("Loading ETIM data...")
        self.etim_df = pd.read_excel(self.etim_path, sheet_name=0)
        
        # Clean column names (remove extra spaces, make lowercase)
        self.etim_df.columns = [col.strip().lower().replace(' ', '_') for col in self.etim_df.columns]
        
        logger.info("Loading LOV data...")
        self.lov_df = pd.read_excel(self.lov_path, sheet_name=0)
        
        # Clean column names for LOV data
        self.lov_df.columns = [col.lower() for col in self.lov_df.columns]
        
        # Preprocess text data for better matching
        self._preprocess_data()
    
    def _preprocess_text(self, text):
        """Preprocess text for better matching"""
        if pd.isna(text):
            return ""
        # Convert to string, lowercase, and remove extra spaces
        return str(text).strip().lower()
    
    def _preprocess_data(self):
        """Preprocess data for matching"""
        logger.info("Preprocessing data...")
        
        # Preprocess ETIM data
        self.etim_df['class'] = self.etim_df['class'].apply(self._preprocess_text)
        self.etim_df['feature'] = self.etim_df['feature'].apply(self._preprocess_text)
        self.etim_df['value'] = self.etim_df['value'].apply(self._preprocess_text)
        
        # Preprocess LOV data
        self.lov_df['main_icc'] = self.lov_df['main_icc'].apply(self._preprocess_text)
        self.lov_df['attr_display_name'] = self.lov_df['attr_display_name'].apply(self._preprocess_text)
        self.lov_df['meaning_english'] = self.lov_df['meaning_english'].apply(self._preprocess_text)
    
    def _calculate_similarity(self, str1, str2):
        """Calculate similarity score between two strings"""
        if not str1 or not str2:
            return 0
        return fuzz.token_sort_ratio(str1, str2)
    
    def _find_best_match(self, etim_feature, etim_value):
        """
        Find the best matching LOV entry for a given ETIM feature and value
        
        Returns:
            tuple: (best_match, match_score, match_type)
        """
        best_match = None
        best_score = 0
        match_type = "no_match"
        
        # First try exact matching
        exact_matches = self.lov_df[
            (self.lov_df['attr_display_name'] == etim_feature) & 
            (self.lov_df['meaning_english'] == etim_value)
        ]
        
        if not exact_matches.empty:
            best_match = exact_matches.iloc[0]
            return best_match, 100, "exact"
        
        # If no exact match, try fuzzy matching
        for _, row in self.lov_df.iterrows():
            # Calculate feature similarity
            feature_sim = self._calculate_similarity(row['attr_display_name'], etim_feature)
            value_sim = self._calculate_similarity(str(row['meaning_english']), str(etim_value))
            
            # Calculate overall score (weighted average)
            score = (feature_sim * 0.6) + (value_sim * 0.4)
            
            if score > best_score:
                best_score = score
                best_match = row
                
                if score > 90:
                    match_type = "high_confidence"
                elif score > 70:
                    match_type = "medium_confidence"
                else:
                    match_type = "low_confidence"
        
        return best_match, best_score, match_type
    
    def generate_etimalism(self):
        """Generate ETIMALISM mapping by matching ETIM data with LOV"""
        logger.info("Starting ETIMALISM generation...")
        
        # Group ETIM data by class and feature
        grouped = self.etim_df.groupby(['class_id', 'class', 'feature_id', 'feature'])
        
        for (class_id, class_name, feature_id, feature), group in grouped:
            logger.info(f"Processing class: {class_name} - Feature: {feature}")
            
            for _, row in group.iterrows():
                etim_value = row['value']
                if pd.isna(etim_value) or not str(etim_value).strip():
                    continue
                
                # Find best matching LOV entry
                lov_match, score, match_type = self._find_best_match(feature, etim_value)
                
                # Create result entry
                result = {
                    'etim_class_id': class_id,
                    'etim_class_name': class_name,
                    'etim_feature_id': feature_id,
                    'etim_feature': feature,
                    'etim_value_id': row.get('value_id', ''),
                    'etim_value': etim_value,
                    'match_type': match_type,
                    'confidence_score': score,
                    'lov_icc_name': lov_match['main_icc'] if lov_match is not None else '',
                    'pdh_attr_name': lov_match['attr_display_name'] if lov_match is not None else '',
                    'pdh_attr_value': lov_match['meaning_english'] if lov_match is not None else '',
                    'lov_lookup_code': lov_match.get('lookup_code', '') if lov_match is not None else ''
                }
                
                self.matched_results.append(result)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.matched_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"etimalism_mapping_{timestamp}.xlsx"
        
        # Create final ETIMALISM format
        etimalism_df = results_df.rename(columns={
            'etim_class_name': 'ICC_NAME',
            'etim_class_id': 'CLASS_CODE',
            'etim_feature_id': 'FEATURE_CODE',
            'etim_feature': 'FEATURE_NAME',
            'etim_value_id': 'VALUE_CODE',
            'etim_value': 'VALUE_NAME',
            'pdh_attr_name': 'PDH_ATTR_NAME',
            'pdh_attr_value': 'PDH_ATTR_VALUE'
        })
        
        # Select and reorder columns to match ETIMALISM format
        etimalism_columns = [
            'ICC_NAME', 'CLASS_CODE', 'FEATURE_CODE', 'FEATURE_NAME',
            'VALUE_CODE', 'VALUE_NAME','PDH_ATTR_NAME', 'PDH_ATTR_VALUE'
        ]
        
        # Add any missing columns with empty values
        for col in etimalism_columns:
            if col not in etimalism_df.columns:
                etimalism_df[col] = ''
        
        # Reorder and select only the required columns
        etimalism_df = etimalism_df[etimalism_columns]
        
        # Save to Excel
        etimalism_df.to_excel(output_path, index=False)
        logger.info(f"ETIMALISM mapping saved to: {output_path}")
        
        # Save detailed results for review
        detailed_path = self.output_dir / f"detailed_mapping_{timestamp}.xlsx"
        results_df.to_excel(detailed_path, index=False)
        logger.info(f"Detailed mapping saved to: {detailed_path}")
        
        return etimalism_df, results_df

def main():
    # Define file paths
    base_dir = Path(__file__).parent
    etim_path = base_dir / "src/ETIM-9.0-ALL-SECTORS-EXCEL-METRIC-EI-2022-12-05.xlsx"
    lov_path = base_dir / "src/Lov.xlsx"
    output_dir = base_dir / "src/output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize the matcher
        logger.info("Initializing ETIM to LOV matcher...")
        matcher = ETIMtoLOVMatcher(
            etim_path=etim_path,
            lov_path=lov_path,
            output_dir=output_dir
        )
        
        # Generate ETIMALISM mapping
        logger.info("Generating ETIMALISM mapping...")
        etimalism_df, detailed_df = matcher.generate_etimalism()
        
        # Generate summary report
        logger.info("Generating summary report...")
        summary = generate_summary(detailed_df)
        
        # Print summary with proper encoding
        try:
            print("\n" + "="*50)
            print("ETIMALISM GENERATION SUMMARY")
            print("="*50)
            print(summary.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
            print("="*50)
        except Exception as e:
            logger.warning(f"Could not print full summary to console: {str(e)}")
            print("\nSummary generated successfully. Check the output files for complete details.")
            print(f"Output files saved to: {output_dir}")
        
        return etimalism_df, detailed_df
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

def generate_summary(detailed_df):
    """Generate a summary report of the matching results."""
    if detailed_df.empty:
        return "No results to summarize."
    
    total = len(detailed_df)
    matched = len(detailed_df[detailed_df['match_type'] != 'no_match'])
    exact = len(detailed_df[detailed_df['match_type'] == 'exact'])
    high_conf = len(detailed_df[detailed_df['match_type'] == 'high_confidence'])
    med_conf = len(detailed_df[detailed_df['match_type'] == 'medium_confidence'])
    low_conf = len(detailed_df[detailed_df['match_type'] == 'low_confidence'])
    no_match = len(detailed_df[detailed_df['match_type'] == 'no_match'])
    
    summary = f"""
    Total ETIM records processed: {total:,}
    Successfully matched: {matched:,} ({matched/total*100:.1f}%)
      - Exact matches: {exact:,}
      - High confidence matches: {high_conf:,}
      - Medium confidence matches: {med_conf:,}
      - Low confidence matches: {low_conf:,}
    No match found: {no_match:,} ({no_match/total*100:.1f}%)
    
    Match confidence distribution:
    {generate_confidence_distribution(detailed_df)}
    
    Top unmatched features:
    {get_top_unmatched(detailed_df, 5)}
    """
    
    return summary

def generate_confidence_distribution(df):
    """Generate a text-based histogram of confidence scores."""
    if df.empty:
        return "No data available"
    
    # Filter out no match entries
    matched_df = df[df['match_type'] != 'no_match']
    if matched_df.empty:
        return "No matches found"
    
    # Create bins for confidence scores
    bins = [0, 60, 70, 80, 90, 100]
    labels = ['0-60%', '61-70%', '71-80%', '81-90%', '91-100%']
    
    # Categorize confidence scores
    matched_df['confidence_bin'] = pd.cut(
        matched_df['confidence_score'],
        bins=bins,
        labels=labels,
        right=True
    )
    
    # Count occurrences in each bin
    distribution = matched_df['confidence_bin'].value_counts().sort_index()
    
    # Generate histogram
    max_count = distribution.max()
    scale = 50.0 / max_count if max_count > 0 else 1
    
    hist = []
    for label in labels:
        count = distribution.get(label, 0)
        bar = '█' * int(count * scale)
        hist.append(f"{label}: {bar} {count:,}")
    
    return "\n".join(hist)

def get_top_unmatched(df, n=5):
    """Get top N unmatched features with their counts."""
    unmatched = df[df['match_type'] == 'no_match']
    if unmatched.empty:
        return "All features were matched successfully."
    
    top_unmatched = unmatched['etim_feature'].value_counts().head(n)
    return "\n".join([f"  - {feat}: {count:,} occurrences" for feat, count in top_unmatched.items()])

if __name__ == "__main__":
    main()