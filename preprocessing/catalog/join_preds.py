import argparse
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

def parse_args():
    """
    Parses commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir",
                    type=str,
                    default='/cluster/research-groups/hutchinson/data/ml_astro/tess/catalog/arch2/preds'
                )
    parser.add_argument("-processed_pkl_dir",
                    type=str,
                    default='/cluster/research-groups/hutchinson/data/ml_astro/tess/catalog/processed_pkl'
                )
    parser.add_argument('--save_path', type=str,
                        default="/cluster/research-groups/hutchinson/data/ml_astro/tess/catalog/cat_preds.csv")
    args = parser.parse_args()

    return args



def overflow(values):
    return np.array(values) % (2**32)

def round_to_sig_figs(x, sig_figs=3):
    if x != 0:
        return round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)
    return 0

def main():
    args = parse_args()

    # Initialize a list to store DataFrame objects
    frames = []
    sectors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    tolerance = 1e-5
    for csv_file in tqdm(os.listdir(args.pred_dir)):
        pkl_file = csv_file[:-4] + '.pkl'

        df1 = pd.read_csv(os.path.join(args.pred_dir, csv_file))

        df2 = pd.read_pickle(os.path.join(args.processed_pkl_dir, pkl_file))

        df2 = df2[df2['outSec'].isin(sectors)]
        df2 = df2.rename(columns={"outSec": "sector", "outID": "ID", 'rowPix': 'row_pix', 'colPix': 'col_pix'})

        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)

        match_mask = (df1['sector'] == df2['sector']) & \
                      np.isclose(df1['ra'], df2['ra'], atol=tolerance) & \
                      np.isclose(df1['dec'], df2['dec'], atol=tolerance) & \
                      np.isclose(df1['tmag'], df2['tmag'], atol=tolerance)

        df2['modID'] = overflow(df2['ID'])
        mod_match_mask = (df1['ID'] == df2['modID'])

        if not match_mask.all():
            print("ERRORmatch", csv_file)
        if not mod_match_mask.all():
            print("ERRORmod", csv_file)

        df1['ID'] = df2['ID']
        df1['row_pix'] = df2['row_pix']
        df1['col_pix'] = df2['col_pix']

        sf_cols = \
            ['eb_prob', 'pulse_prob', 'rot_prob', 'nonvar_prob', 'cont_pred', 'cont_full_pred']

        round_cols = {'row_pix': 1, 'col_pix': 1}

        for col in sf_cols:
            df1[col] = df1[col].apply(lambda x: round_to_sig_figs(x, 4))
        df1 = df1.round(round_cols)
        frames.append(df1)


    concatenated_df = pd.concat(frames, ignore_index=True)

    # Write the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(args.save_path, index=False)

    print(f"All CSV files have been concatenated into {args.save_path}")


if __name__ == '__main__':
    main()
