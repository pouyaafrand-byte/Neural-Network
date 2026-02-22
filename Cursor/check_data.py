"""
Quick script to check the data structure before running the full comparison.
"""

import pandas as pd
import numpy as np

def check_data():
    """Check the structure of the Excel file."""
    try:
        df = pd.read_excel('DATA.xlsx')
        
        print("="*60)
        print("DATA FILE STRUCTURE")
        print("="*60)
        print(f"\nShape: {df.shape}")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        print("\nColumn Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nData types:")
        print(df.dtypes)
        
        print("\nMissing values:")
        print(df.isnull().sum())
        
        print("\nBasic statistics:")
        print(df.describe())
        
        # Check for expected columns
        df_lower = df.copy()
        df_lower.columns = df_lower.columns.str.strip().str.lower()
        
        input_cols = ['fx', 'fy', 'fz', 'mz', 'q']
        output_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
        
        print("\n" + "="*60)
        print("COLUMN MATCHING")
        print("="*60)
        
        found_inputs = [col for col in input_cols if col in df_lower.columns]
        found_outputs = [col for col in output_cols if col in df_lower.columns]
        
        print(f"\nInput columns found: {found_inputs}")
        print(f"Input columns missing: {[col for col in input_cols if col not in df_lower.columns]}")
        
        print(f"\nOutput columns found: {found_outputs}")
        print(f"Output columns missing: {[col for col in output_cols if col not in df_lower.columns]}")
        
        if len(found_inputs) == len(input_cols) and len(found_outputs) == len(output_cols):
            print("\n[OK] All expected columns found!")
        else:
            print("\n[WARNING] Some columns are missing. Please check column names.")
        
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_data()
