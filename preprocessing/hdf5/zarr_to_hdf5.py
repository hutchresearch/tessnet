import argparse
import pickle as pkl
import os
import h5py
from tqdm import tqdm

# path enables running from command line
import sys
sys.path.append('../..')
from preprocessing.util import zarr_filepath, get_crop


def main():
    args = parse_args()
    with open(args.cleaned_pd_pkl_path, "rb") as pkl_handle:
        tess_data = pkl.load(pkl_handle)

    for sector in args.sectors:
        sec_label = str(sector).zfill(2)
        h5_path = os.path.join(args.hdf5_dir, 'sec_{}.hdf5'.format(sec_label))
        h5_file = h5py.File(h5_path, "w")

        sector_data = tess_data[tess_data["outSec"] == sector]
        sector_data = sector_data.sort_values(by=['outCam', 'outCcd'])
        sector_data = sector_data.reset_index(drop=True)
        print("Starting sector {}".format(sector))
        for star in tqdm(sector_data.itertuples(), total=len(sector_data)):
            sec, cam, ccd, star_id = star.outSec, star.outCam, star.outCcd, star.outID
            zarr_fp = zarr_filepath(args.cubes_dir, sec, cam, ccd)
            crop, corruption = get_crop(zarr_fp, 6, star.rowPix, star.colPix)

            if corruption > 0.2:
                continue

            subgroup = '{}.{}.{}'.format(sec, cam, ccd)
            dataset_path = '{}/{}'.format(subgroup, star_id)

            h5_file[dataset_path] = crop
        print("Sector finished.")


def parse_args():
    """
    Parses commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("cleaned_pd_pkl_path",
                        type=str,
                        help="Path to the cleaned pkl file"
                        )
    parser.add_argument("cubes_dir",
                        type=str,
                        help="Path to the datacubes root directory"
                        )
    parser.add_argument("hdf5_dir",
                        type=str,
                        help="Path to the hdf5 directory"
                        )
    parser.add_argument("--sectors",
                        type=int,
                        nargs='+',
                        help="Sectors to convert, given as a space-separated list. Example: --sectors 1 9 21",
                        required=True
                        )
    return parser.parse_args()


if __name__ == "__main__":
    main()

