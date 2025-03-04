import os
import pandas as pd
import numpy as np
from collections import Counter

PRED_DIR = '.'
PRED_FNAMES = ['08Mar-18_35_15_predictions.csv', '09Mar-21_33_02_predictions.csv', '12Mar-15_56_41_predictions.csv']
OUT_FNAME = 'blend.csv'
MERGE_REG = False
MERGE_CLASS = True

if __name__ == '__main__':
    dfs = []
    for fname in PRED_FNAMES:
        fpath = os.path.join(PRED_DIR, fname)
        df = pd.read_csv(fpath)
        dfs.append(df)

    sum_df = dfs[0].copy()

    if MERGE_REG:
        for i, df in enumerate(dfs[1:]):
            pre = dfs[i - 1]
            assert len(np.where(pre['ID'] != df['ID'])[0]) == 0
            sum_df['period_preds'] += df['period_preds']

        sum_df['period_preds'] /= len(dfs)

    if MERGE_CLASS:
        for i in range(len(sum_df)):
            c = Counter()
            for df in dfs:
                pred = df.at[i, 'class_preds']
                c.update([pred])
            maj_class = c.most_common(1)[0][0]
            sum_df.at[i, 'class_preds'] = maj_class

    sum_df.to_csv(OUT_FNAME, index=False)


