import argparse
import pickle as pkl
import os
import numpy as np
from tqdm import tqdm
import pandas as pd


def main():
    args = parse_args()
    tess_data = copy_splits(args.source_pkl_path, args.target_pkl_path)
    tess_data.to_pickle(args.output_pkl_path, protocol=4)


def copy_splits(source_pkl_path, target_pkl_path):
    """Copies splits"""
    with open(source_pkl_path, "rb") as pkl_handle:
        source_data = pkl.load(pkl_handle)
        source_data = source_data[['outSec', 'outCam', 'outCcd', 'outID',
                                  "train_set", 'dev_set', 'test1_set', 'test2_set']]

    with open(target_pkl_path, "rb") as pkl_handle:
        target_data = pkl.load(pkl_handle)
        target_data = target_data.drop(["train_set", 'dev_set', 'test1_set',
                                        'test2_set', 'test_set'], axis=1, errors='ignore')

    copied_data = target_data.merge(source_data, how='left', on=['outSec', 'outCam', 'outCcd', 'outID'])

    assert len(copied_data.columns[copied_data.isna().any()].tolist()) == 0
    assert len(copied_data[~copied_data['train_set'] & ~copied_data['dev_set']
                           & ~copied_data['test1_set'] & ~copied_data['test2_set']]) == 0

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