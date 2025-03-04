#!/usr/bin/env python3

"""
save_fits_header_timestamps.py
    Takes a fits image dir full and creates a text file containing
    timestamps in chronological order. Used once per sector, the timestamps
    are later used in model training.
"""

import argparse
import os
import sys
from datetime import datetime


def main():
    """
    main
        Drives the script
    """
    args = parse_args()
    sec_dir = args.fits_images_dir.strip("/").split("/")[-1]
    if "sec_" not in sec_dir:
        print("fits_images_dir must be dir for a SINGLE sector! Aborting.")
        sys.exit()
    timestamps = make_timestamps(args.fits_images_dir)
    save_timestamps(timestamps, args.timestamp_dir, sec_dir)


def save_timestamps(timestamps, timestamp_dir, sec_dir):
    """
    save_timestamps - saves true timestamps as strings in ISO format to sec_**.txt in timestamp_dir
    """
    timestamp_filehandle = os.path.join(timestamp_dir, os.path.basename(sec_dir)+".txt")

    with open(timestamp_filehandle, "w+") as timestamp_file:
        first = True
        for stamp in timestamps:
            timestamp_str = str(stamp.isoformat())
            if first:
                first = False
                timestamp_file.write(timestamp_str)
            else:
                timestamp_file.write("\n"+timestamp_str)


def make_timestamps(fits_dir):
    """
    make_timestamps - Builds list of datetime timestamps from info in header files of fits files.
                    All cam and ccd frametimes across a sector are consistent so only 1 list is
                    needed per sector.
    """
    timestamps = []
    for fits_file in sorted(os.listdir(fits_dir)):
        if ".fits" in fits_file:
            file_name_split = fits_file.split("-")
            timestr = file_name_split[0][4:]
            cam = int(file_name_split[2])
            ccd = int(file_name_split[3])
            # Only grab one sec_cam_ccd per sector (frametimes are all the same for a sector)
            if cam == 1 and ccd == 1:
                year = timestr[:4]
                day = timestr[4:7]
                hour = timestr[7:9]
                minute = timestr[9:11]
                second = timestr[11:]
                timestr = " ".join([year, day, hour, minute, second])
                time = datetime.strptime(timestr, "%Y %j %H %M %S")
                timestamps.append(time)
    return timestamps


def parse_args():
    """
    Parses commandline arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("fits_images_dir",
                    type=str,
                    help="Path to the fits images dir for a SINGLE sector (str)\n"+\
                    "[example: /cluster/research-groups/hutchinson/data/ml_astro/tess/fits/sec_01]"
                )
    parser.add_argument("timestamp_dir",
                    type=str,
                    help="Path to the folder that holds all cubes timestamp files\n"+\
                            "[default: "+\
                            "/cluster/research-groups/hutchinson/data/ml_astro/tess/timestamps]"
                )
    return parser.parse_args()


if __name__ == "__main__":
    main()
