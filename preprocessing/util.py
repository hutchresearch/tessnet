import os
import zarr
import numpy as np


def zarr_filepath(cubes_dir, sec, cam, ccd):
    """
    zarr_filepath
        Generates the path and name for the zarr file associated with the sec_cam_ccd
    """
    sec, cam, ccd = str(sec), str(cam), str(ccd)
    sec_cam_ccd = "_".join([sec, cam, ccd])
    sec_zpadded = sec.zfill(2)
    filepath = os.path.join(cubes_dir, "sec_" + sec_zpadded, sec_cam_ccd + ".zarr")
    return filepath


def get_crop(filepath, crop_size, row_pix, col_pix):
    """
    get_crop
        Grabs the cropped area from datacube specified and returns a 3d HxWxT np float32 array
    """
    datacube = zarr.open(filepath, mode="r")
    width, height = datacube.shape[0], datacube.shape[1]
    half_crop_size = crop_size // 2
    x_pixel = round(row_pix)
    y_pixel = round(col_pix)
    left = x_pixel-half_crop_size if x_pixel-half_crop_size >= 0 else 0
    right = x_pixel+half_crop_size if x_pixel+half_crop_size <= width else width
    top = y_pixel-half_crop_size if y_pixel-half_crop_size >= 0 else 0
    bottom = y_pixel+half_crop_size if y_pixel+half_crop_size <= height else height
    crop = datacube[left:right, top:bottom]

    if x_pixel-half_crop_size < 0:
        crop = np.pad(crop, ((abs(x_pixel-half_crop_size), 0), (0, 0), (0, 0)))
    if x_pixel+half_crop_size+1 > width:
        crop = np.pad(crop, ((0, abs(x_pixel+half_crop_size+1-width)), (0, 0), (0, 0)))
    if y_pixel-half_crop_size < 0:
        crop = np.pad(crop, ((0, 0), (abs(y_pixel-half_crop_size), 0), (0, 0)))
    if y_pixel+half_crop_size+1 > height:
        crop = np.pad(crop, ((0, 0), (0, abs(y_pixel+half_crop_size+1-height)), (0,)))

    nonzero_prop = np.count_nonzero(crop) / crop.size
    corruption = 1 - nonzero_prop

    return crop, corruption