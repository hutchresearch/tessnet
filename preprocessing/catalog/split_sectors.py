import argparse
import os
import pandas as pd
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str,
                        default="/cluster/research-groups/hutchinson/data/ml_astro/tess/catalog/cat_preds.csv",
                        help="Path to the input CSV file")
    parser.add_argument('--save_dir', type=str,
                        default="/cluster/research-groups/hutchinson/data/ml_astro/tess/catalog/sectors",
                        help="Directory to save the split sector CSV files")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    csv_path = args.csv_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    grouped = df.groupby('sector')

    for sector, group in tqdm(grouped, desc="Processing sectors"):
        filename = f'sector_{sector}.csv'
        file_path = os.path.join(save_dir, filename)
        group.to_csv(file_path, index=False)
        print(f'Saved {file_path}')


if __name__ == '__main__':
    main()
