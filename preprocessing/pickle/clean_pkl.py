import argparse
import pickle as pkl
import os


def main():
    args = parse_args()
    tess_data = clean_data(args.marked_pd_pkl_path, threshold=args.threshold)
    output_path = os.path.join(args.output_dir, 'tess_stars_rb_cleaned_new.pd.pkl')
    tess_data.to_pickle(output_path, protocol=4)


def clean_data(data_pd_pkl_path, threshold=1):
    """Cleans data at corruption threshold"""
    with open(data_pd_pkl_path, "rb") as pkl_handle:
        tess_data = pkl.load(pkl_handle)

    tess_data = tess_data[tess_data['corruption'] <= threshold]
    tess_data = tess_data.reset_index(drop=True)

    return tess_data


def parse_args():
    """
    Parses commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("marked_pd_pkl_path",
                    type=str,
                    help="Path to the marked pickle file, including filename",
                    default='C:/Users/harry/TESSNET-2/data/tess_stars.pd.pkl'
                )
    parser.add_argument("output_dir",
                    type=str,
                    help="Directory of the output marked pickle",
                    default='C:/Users/harry/TESSNET-2/data'
                )
    parser.add_argument("--threshold",
                        type=float,
                        default=0.2,
                        help="Threshold of proportion of zero values at which a crop is excluded."
                        )

    return parser.parse_args()


if __name__ == '__main__':
    main()
