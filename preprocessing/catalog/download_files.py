import requests
import argparse
from urllib.parse import urljoin
import os
import gzip
import csv
import pandas as pd
from tess_stars2px import tess_stars2px_function_entry
import hashlib
from io import BytesIO

URL_BASE = 'https://archive.stsci.edu/missions/tess/catalogs/tic_v82/'
MD5_SUM_PATH = 'metadata/md5sum.txt'


def parse_args():
    """
    Parses commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir",
                    type=str,
                    default="C:/Users/harry/TESSNET-2/data/csv"
                )
    parser.add_argument("--header_path",
                        type=str,
                        default="metadata/header.csv"
                        )
    parser.add_argument('--save_dir', type=str,
                        default="C:/Users/harry/TESSNET-2/data/pkl")
    parser.add_argument('--reverse', action="store_true")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    with open(MD5_SUM_PATH, "r") as file:
        lines = file.readlines()

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

    if args.reverse:
        lines = lines[::-1]

    for line in lines:
        print(line)
        md5_sum, f_name = line.split()
        out_f_name = f_name[1:]
        f_name = out_f_name + '.gz'
        url_path = urljoin(URL_BASE, f_name)

        gz_f_path = os.path.join(args.download_dir, f_name)

        output_path = os.path.join(args.save_dir, out_f_name[:-4] + '.pkl')

        if os.path.isfile(output_path):
            continue

        print(gz_f_path, output_path)

        response = requests.get(url_path)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open the local file for binary writing and write the content from the response
            with open(gz_f_path, "wb") as file:
                file.write(response.content)
            print(f"Downloaded '{url_path}' to '{args.download_dir}'")
        else:
            print(f"Failed to download '{url_path}' (status code: {response.status_code})")
            continue

        with gzip.open(gz_f_path, "rt", encoding="utf-8") as gz_file:
            contents = gz_file.read().encode()
            hashsum = hashlib.md5(contents).hexdigest()
            if hashsum != md5_sum:
                print("ERROR: HASH SUMS DO NOT MATCH. {} != {} SKIPPING".format(hashsum, md5_sum))
                continue
            else:
                print("HASH SUMS MATCH: {} == {}".format(hashsum, md5_sum))

            # Read the decompressed data as a CSV
            df = pd.read_csv(BytesIO(contents), names=names)
            df = df[df['Tmag'] < 16]
            outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, outColPix, outRowPix, scinfo = tess_stars2px_function_entry(df['ID'].values, df['ra'].values, df['dec'].values)

            df = df[['ID', 'Tmag', 'Hmag', 'gmag', 'plx', 'gaiabp', 'gaiarp']]

            tess_data = pd.DataFrame({
                'ID': outID,
                'outSec': outSec,
                'outCam': outCam,
                'outCcd': outCcd,
                'rowPix': outRowPix,
                'colPix': outColPix
            })

            tess_data = tess_data.merge(df, on='ID', how='left')
            tess_data = tess_data.rename(columns={'ID': 'outID', 'Tmag': 'tmag',
                                                  'Hmag': 'hmag', 'gaiabp': 'BPmag', 'gaiarp': 'RPmag'})

            tess_data.to_pickle(output_path, protocol=4)


if __name__ == '__main__':
    main()
