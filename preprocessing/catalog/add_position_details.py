import requests
import argparse
from urllib.parse import urljoin
import os
import gzip
import csv
import pandas as pd
from tess_stars2px import tess_stars2px_function_entry
import hashlib
from tqdm import tqdm
from io import BytesIO

def parse_args():
    """
    Parses commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir1",
                    type=str,
                    default='/cluster/research-groups/hutchinson/data/ml_astro/tess/catalog/arch2/preds'
                )
    parser.add_argument("-pred_dir2",
                    type=str,
                    default='/cluster/research-groups/hutchinson/data/ml_astro/tess/catalog/preds'
                )
    parser.add_argument('--save_path', type=str,
                        default="/cluster/research-groups/hutchinson/data/ml_astro/tess/catalog/cat_preds.csv")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Initialize a list to store DataFrame objects
    frames = []

    for directory_path in (args.pred_dir1, args.pred_dir2):
        for file in os.listdir(directory_path):
            if file.endswith('.csv'):
                file_path = os.path.join(directory_path, file)
                df = pd.read_csv(file_path)
                frames.append(df)

    concatenated_df = pd.concat(frames, ignore_index=True)

    # Write the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(args.save_path, index=False)

    print(f"All CSV files have been concatenated into {args.save_path}")


if __name__ == '__main__':
    main()