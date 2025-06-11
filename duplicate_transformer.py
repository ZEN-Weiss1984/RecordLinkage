import linktransformer as lt
import numpy as np
import pandas as pd
import os
import json
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Transformer deduplication')
    parser.add_argument('--in', dest='input_file', required=True, help='Input CSV file path')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2', help='Model name for deduplication')
    args = parser.parse_args()

    # Read input file
    df = pd.read_csv(args.input_file)
    print("Available columns in the CSV file:", df.columns.tolist())
    print("\nOriginal number of rows:", len(df))

    # Perform deduplication
    df_dedup = lt.dedup_rows(df, on="NAME", model=args.model, cluster_type="agglomerative",
        cluster_params={'threshold': 0.55})
    print("\nNumber of rows after deduplication:", len(df_dedup))
    print("\nNumber of rows removed:", len(df) - len(df_dedup))

    # Get duplicates
    duplicates = df[~df.index.isin(df_dedup.index)]
    duplicates_dict = duplicates[['ID', 'NAME']].to_dict('records')

    # Save duplicates to JSON
    with open("./duplicate_delete.json", "w", encoding='utf-8') as f:
        json.dump(duplicates_dict, f, ensure_ascii=False, indent=4)

    print(f"\nSaved {len(duplicates_dict)} duplicate records to duplicate_delete.json")

    # Save deduplicated data
    output_file = os.path.splitext(args.input_file)[0] + "_deduplicated.csv"
    df_dedup.to_csv(output_file, index=False)
    print(f"\nDeduplicated data saved to '{output_file}'")

if __name__ == "__main__":
    main()

