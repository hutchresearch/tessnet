#!/usr/bin/env python3

"""
build_datacubes.py
    Given a fits images sector dir, builds datacubes for a sector
"""

import argparse
import os

import numpy as np
import zarr
from astropy.io import fits

def main():
    """
    main
        Handles control flow for building the datacubes for a sector
    """
    args = parse_args()
    build_numpy_cubes_from_fits(args.fits_images_dir, args.cube_output_dir)
    return

def build_numpy_cubes_from_fits(fits_dir, cubes_dir):
    """
    build_numpy_cubes_from_fits
        Goes through a fits image file dir one sec_cam_ccd at a time,
        pulls out the images in order from each of the files,
        stacks them together along the 3rd dimension, then saves the 3-D cube 
        as a zarr array into the cubes_dir
    """
    print(2)
    logfile = os.path.join(cubes_dir, "log.txt")
    print("Starting")
    # Build list of sec_cam_ccd_id_list
    sec_cam_ccd_id_list = set()
    seen_id_list = []
    for cube_file in sorted(os.listdir(cubes_dir)):
        if ".zarr" in cube_file:
            seen_id_list.append(cube_file.split(".zarr")[0])

    for fits_file in sorted(os.listdir(fits_dir)):
        if ".fits" in fits_file:
            sec_cam_ccd_id = "_".join(fits_file.split("-")[1:-2])
            sec_cam_ccd_id = sec_cam_ccd_id[1:].strip("0")
            sec_cam_ccd_id_list.add(sec_cam_ccd_id)

    sec_cam_ccd_id_list = list(sec_cam_ccd_id_list)
    # Build and then save each sec_cam_ccd datacube
    sec_cam_ccd_id_list = ['16_3_4']
    print(sec_cam_ccd_id_list, seen_id_list)

    for idx, sec_cam_ccd_id in enumerate(sec_cam_ccd_id_list):
        #if sec_cam_ccd_id in seen_id_list:
         #   continue

        print("Processing sccid {}".format(sec_cam_ccd_id))
        cube_list = []
        image_count = 0
        for fits_file in sorted(os.listdir(fits_dir)):
            if ".fits" in fits_file:
                file_sec_cam_ccd_id = "_".join(fits_file.split("-")[1:-2])
                file_sec_cam_ccd_id = file_sec_cam_ccd_id[1:].strip("0")
                if file_sec_cam_ccd_id == sec_cam_ccd_id:
                    image_count += 1
                    with open(logfile, "w") as log:
                        log.write("Building cube {}. Cube:({}/{}). Image #{}. Fits:{}\n".format(sec_cam_ccd_id, idx+1, len(sec_cam_ccd_id_list), image_count, fits_file))
                    with fits.open(os.path.join(fits_dir,fits_file), ignore_missing_simple=True) as hdul:
                        cube_list.append(np.array(hdul[1].data, dtype=np.float32))
        datacube = np.stack(cube_list, axis=2)
        # Put the cube into LOG-SPACE so it can fit into the float16 array
        print("Cleaning up the cube and putting into log space")
        fi16 = np.finfo(np.float16)
        datacube[datacube<1.0] = 1.0
        datacube = np.log(datacube)
        np.around(datacube, decimals=fi16.precision, out=datacube)
        # Save the datacube
        cube_save_filepath = os.path.join(cubes_dir, sec_cam_ccd_id+".zarr")
        zarr.save_array(cube_save_filepath, datacube.astype(np.float16), dtype="float16")
    print("Done building cubes for sector.")
    return

def parse_args():
    """
    Parses commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("fits_images_dir",
                    type=str,
                    help="Path to the fits images dir for a single sector"
                )
    parser.add_argument("cube_output_dir",
                    type=str,
                    help="Path to the output datacube save dir for a single sector"
                )
    return parser.parse_args()



"""
Main Function Catch
"""
if __name__ == "__main__":
    print(1)
    main()
