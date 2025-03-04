import argparse
import pickle as pkl
import os
import numpy as np
from tqdm import tqdm
import pandas as pd


def main():
    args = parse_args()
    tess_data = add_splits(args.pkl_path, args.t2sectors, args.d2sectors, args.splits)
    output_path = os.path.join(args.output_dir, 'tess_stars_redbright.pd.pkl')
    tess_data.to_pickle(output_path, protocol=4)


def add_splits(data_pd_pkl_path, t2sectors, d2sectors, splits):
    """Filters + adds splits"""
    with open(data_pd_pkl_path, "rb") as pkl_handle:
        tess_data = pkl.load(pkl_handle)

    tess_data["mmtype"] = tess_data["mmtype"].fillna("")
    tess_data = tess_data[tess_data["mmtype"] != ""]
    tess_data["period"] = tess_data["period"].fillna(0.0)
    tess_data['mmtype'] = tess_data['mmtype'].replace('eb', 'eclipse')

    tess_data["train_set"] = np.full(len(tess_data), False)
    tess_data["dev1_set"] = np.full(len(tess_data), False)
    tess_data["test1_set"] = np.full(len(tess_data), False)
    tess_data["test2_set"] = np.full(len(tess_data), False)

    main_tess = tess_data[~tess_data['outSec'].isin(t2sectors + d2sectors)].reset_index(drop=True)
    t2_tess = tess_data[tess_data['outSec'].isin(t2sectors)].reset_index(drop=True)
    d2_tess = tess_data[tess_data['outSec'].isin(d2sectors)].reset_index(drop=True)

    t2_tess["test2_set"] = np.full(len(t2_tess), True)
    d2_tess["dev2_set"] = np.full(len(d2_tess), True)

    test_size = int(len(main_tess) * splits[2])
    dev_size = int(len(main_tess) * splits[1])
    train_size = len(main_tess) - test_size - dev_size

    labels = np.array([0] * train_size + [1] * dev_size + [2] * test_size)

    assert len(labels) == len(main_tess)

    np.random.shuffle(labels)

    train_idx = np.where(labels == 0)[0]
    dev_idx = np.where(labels == 1)[0]
    test_idx = np.where(labels == 2)[0]

    main_tess.loc[train_idx, 'train_set'] = True
    main_tess.loc[dev_idx, 'dev1_set'] = True
    main_tess.loc[test_idx, 'test1_set'] = True

    tess_data = pd.concat([main_tess, t2_tess, d2_tess])
    return tess_data.reset_index(drop=True)


def parse_args():
    """
    Parses commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path",
                    type=str,
                    help="Path to the pickle file, including filename",
                )
    parser.add_argument("output_dir",
                    type=str,
                    help="Directory of the output pickle"
                )
    parser.add_argument("--t2sectors",
                        type=int,
                        nargs='+',
                        help="Sectors to include in test2, given as a space-separated list. "
                             "Example: --t2sectors 1 9 21",
                        required=True
                        )
    parser.add_argument("--d2sectors",
                        type=int,
                        nargs='+',
                        help="Sectors to include in dev2, given as a space-separated list. "
                             "Example: --d2sectors 3",
                        required=True
                        )
    parser.add_argument("--splits",
                        type=float,
                        nargs='+',
                        help="Data splits, given as a space-separated list, must add up to 1. "
                             "Example: --split 0.8 0.1 0.1",
                        required=True
                        )
    args = parser.parse_args()

    if len(args.splits) != 3 or sum(args.splits) != 1.:
        raise ValueError("Data splits incorrectly configured.")

    return args
if __name__ == '__main__':
    main()