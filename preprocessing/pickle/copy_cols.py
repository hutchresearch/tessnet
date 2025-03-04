import argparse
import pickle as pkl
import os
import numpy as np
from tqdm import tqdm
import pandas as pd


def main():
    args = parse_args()
    tess_data = copy_cols(args.source_pkl_path, args.target_pkl_path)
    tess_data.to_pickle(args.output_pkl_path, protocol=4)


def copy_cols(source_pkl_path, target_pkl_path):
    """Copies cols"""
    with open(source_pkl_path, "rb") as pkl_handle:
        source_data = pkl.load(pkl_handle)
        source_data = source_data[['outSec', 'outCam', 'outCcd', 'outID', 'BPmag', 'RPmag']]

    with open(target_pkl_path, "rb") as pkl_handle:
        target_data = pkl.load(pkl_handle)

    copied_data = target_data.merge(source_data, how='left', on=['outSec', 'outCam', 'outCcd', 'outID'])

    #print(copied_data[copied_data['BPmag'].isna()].sort_values(by=['outID']))
    #assert len(copied_data.columns[copied_data.isna().any()].tolist()) == 0

    return copied_data


def parse_args():
    """
    Parses commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pkl_path",
                    type=str,
                    help="Path to the source file, including filename",
                )
    parser.add_argument("target_pkl_path",
                    type=str,
                    help='Path of the target pickle, including filename'
                )
    parser.add_argument("output_pkl_path",
                    type=str,
                    help="Path of the output pickle, including filename"
                )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()