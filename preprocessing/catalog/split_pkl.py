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
    parser.add_argument("--pkl_dir",
                    type=str,
                    default='/Users/harryqiang/Documents/Personal/Code/astro_data/pkl'
                )
    parser.add_argument("--csv_dir",
                    type=str,
                    default='/Users/harryqiang/Documents/Personal/Code/astro_data/csv'
                )
    parser.add_argument("--header_path",
                        type=str,
                        default="metadata/header.csv"
                        )
    parser.add_argument('--save_dir', type=str,
                        default="/Users/harryqiang/Documents/Personal/Code/astro_data/processed_pkl")
    parser.add_argument('--chunk_size', type=int, default=150000)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    names = []
    types = {}
    type_mapping = {
        "integer": int,
        "float": float,
        "string": str
    }

    with open(args.header_path, "r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            for key in row:
                name, type_str = key.split(':')
                name = name[1:][:-1]
                types[name] = type_mapping[type_str.lower()]
                names.append(name)

    for filename in tqdm(os.listdir(args.pkl_dir)):
        # Check if the file ends with .pkl
        if not filename.endswith('.pkl'):
            continue

        filename_no_ext = filename[:-4]
        match = False
        for dest_filename in os.listdir(args.save_dir):
            if not dest_filename.endswith('.pkl'):
                continue

            if dest_filename.startswith(filename_no_ext):
                match = True
                break

        if match:
            print(f"Match found for {filename}")
            continue

        gz_filename = filename.replace('.pkl', '.csv.gz')

        gz_f_path = os.path.join(args.csv_dir, gz_filename)
        pkl_f_path = os.path.join(args.pkl_dir, filename)

        tess_data = pd.read_pickle(pkl_f_path)

        with gzip.open(gz_f_path, "rt", encoding="utf-8") as gz_file:
            contents = gz_file.read().encode()
            # Read the decompressed data as a CSV
            df = pd.read_csv(BytesIO(contents), names=names)
            df = df[df['Tmag'] < 16]

            df = df[['ID', 'ra', 'dec']]
            df.rename(columns={'ID': 'outID'}, inplace=True)

            tess_data = tess_data.merge(df, on='outID', how='left')
            tess_data = tess_data.sort_values(by=['outID', 'outSec', 'outCam', 'outCcd'])
            tess_data = tess_data.reset_index(drop=True)

            for i in range(0, len(tess_data), args.chunk_size):
                chunk = tess_data.iloc[i:i + args.chunk_size]

                out_filename = '{}-{}-{}.pkl'.format(filename.split('.')[0], i, i + len(chunk))
                out_path = os.path.join(args.save_dir, out_filename)

                if not os.path.exists(out_path):
                    chunk.to_pickle(out_path, protocol=4)
                else:
                    print("exists:", out_path)


if __name__ == '__main__':
    main()
