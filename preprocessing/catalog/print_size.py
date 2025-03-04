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

    bytes = 0
    for line in lines:
        print(line)
        md5_sum, f_name = line.split()
        out_f_name = f_name[1:]
        f_name = out_f_name + '.gz'
        url_path = urljoin(URL_BASE, f_name)

        gz_f_path = os.path.join(args.download_dir, f_name)

        output_path = os.path.join(args.save_dir, out_f_name[:-4] + '.pkl')

        response = requests.head(url_path)

        # Check if the Content-Length header is present
        if 'Content-Length' in response.headers:
            fsize = int(response.headers['Content-Length'])
            bytes += fsize
            print(f"The size of the file is: {fsize} bytes")
            print(bytes)
        else:
            print("The size of the file could not be determined")



if __name__ == '__main__':
    main()