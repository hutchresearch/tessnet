import argparse
import pickle as pkl
import os
from astropy.io import fits
import pandas as pd


def main():
    args = parse_args()
    tess_data = get_data(args.fits_path)
    output_path = os.path.join(args.output_dir, 'tess_stars_v1_pre.pd.pkl')
    tess_data.to_pickle(output_path, protocol=4)


def get_data(fits_path):
    d = {}
    with fits.open(fits_path, mode="readonly") as hdul:
        d['mmtype'] = hdul[1].data['type']
        d['period'] = hdul[1].data['period']
        d['outSec'] = hdul[1].data['Sector']
        d['outCam'] = hdul[1].data['Camera']
        d['outCcd'] = hdul[1].data['Ccd']
        d['colPix'] = hdul[1].data['ColPix']
        d['rowPix'] = hdul[1].data['RowPix']
        d['tmag'] = hdul[1].data['Tmag']
        d['outID'] = hdul[1].data['TIC']
        d['cont'] =hdul[1].data['cont']
        d['contFull'] = hdul[1].data['cont_full']

    tess_data = pd.DataFrame(d)
    # do not modify dataframe here, issues with endianness

    return tess_data


def parse_args():
    """
    Parses commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("fits_path",
                    type=str,
                    help="Path to the fits file with new stars",
                    default='/Users/harryqiang/downloads/tess_labels_v1_with_point.fits'
                )
    parser.add_argument("output_dir",
                    type=str,
                    help="Directory of the output pickle",
                    default='/Users/harryqiang/documents/data'
                )

    return parser.parse_args()


if __name__ == '__main__':
    main()
