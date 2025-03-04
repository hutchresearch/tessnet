import argparse
import os
import pickle as pkl
from tqdm import tqdm

# path enables running from command line
import sys
sys.path.append('../..')
from preprocessing.util import zarr_filepath, get_crop


def main():
    args = parse_args()
    tess_data = prepare_data(args.input_pd_pkl_path, args.o, args.sectors)
    corruption_values = find_corruption(tess_data, args.cubes_dir, args.crop_size)
    tess_data['corruption'] = corruption_values
    crop_sizes = [args.crop_size] * len(tess_data)
    tess_data['cropSize'] = crop_sizes

    output_path = os.path.join(args.output_dir, 'tess_stars_rb_marked_test_new.pd.pkl')
    tess_data.to_pickle(output_path, protocol=4)


def prepare_data(data_pd_pkl_path, include_other_class, sectors):
    """Opens, filters, and sorts pandas dataframe to be marked."""
    with open(data_pd_pkl_path, "rb") as pkl_handle:
        tess_data = pkl.load(pkl_handle)

    '''
    tess_data["mmtype"] = tess_data["mmtype"].fillna("other")
    tess_data = tess_data if include_other_class else tess_data[tess_data["mmtype"] != "other"]
    tess_data["period"] = tess_data["period"].fillna(0.0)
    tess_data = tess_data[tess_data["period"] < 30]
    tess_data = tess_data[tess_data["tmag"] < 16]
    columns_to_load = ["mmtype", "period", "outSec", "outCam", "outCcd",
                       "colPix", "rowPix", "train_set", "dev_set", "tmag", "outID"]
    tess_data = tess_data.filter(columns_to_load)
    '''

    tess_data = tess_data[tess_data["outSec"].isin(sectors)]
    tess_data = tess_data.sort_values(by=['outSec', 'outCam', 'outCcd'])
    tess_data = tess_data.reset_index(drop=True)

    return tess_data


def find_corruption(tess_data, cubes_dir, crop_size):
    corruption_values = []

    for star in tqdm(tess_data.itertuples(), total=len(tess_data)):
        sec, cam, ccd, star_id = star.outSec, star.outCam, star.outCcd, star.outID
        zarr_fp = zarr_filepath(cubes_dir, sec, cam, ccd)
        _, corruption = get_crop(zarr_fp, crop_size, star.rowPix, star.colPix)
        corruption_values.append(corruption)

    return corruption_values


def parse_args():
    """
    Parses commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pd_pkl_path",
                    type=str,
                    help="Path to the input pickle file, including filename",
                    default='C:/Users/harry/TESSNET-2/data/tess_stars.pd.pkl'
                )
    parser.add_argument("cubes_dir",
                        type=str,
                        help="Path to the datacubes root directory"
                        )
    parser.add_argument("output_dir",
                    type=str,
                    help="Directory of the output marked pickle",
                    default='C:/Users/harry/TESSNET-2/data'
                )
    parser.add_argument("-o", action="store_true",
                        help="Include other class")
    parser.add_argument("--sectors",
                        type=int,
                        nargs='+',
                        help="Sectors to include, given as a space-separated list. Example: --sectors 1 9 21",
                        required=True
                        )
    parser.add_argument("--crop_size",
                        type=int,
                        default=6,
                        help="Size of crop when calculating corruption. Stored in dataframe and used in zarr_to_hdf5."
                        )

    return parser.parse_args()


if __name__ == '__main__':
    main()
