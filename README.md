# TESS Data Processing Pipeline

## Labels
1. Run `preprocessing/pickle/fits_to_pkl.py` on the FITS file.
2. Run `preprocessing/pickle/add_splits.py` to add training splits.
3. Run `preprocessing/pickle/mark_pkl.py` to compute corruption and filter data based on star properties.
4. Run `preprocessing/pickle/clean_pkl.py` to filter data based on corruption.

## Training

1. Download the TESS FFI scripts for the desired sectors from [TESS Bulk Downloads](https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html).
2. Create a directory for the FFIs. Inside, create folders for each sector and place the respective download script in each folder.
3. Run `preprocessing/cubes/download_sectors.py` with the directory and sector names to start downloading.
4. Run `preprocessing/cubes/check_download_completed.py` to verify downloads.
5. Run `preprocessing/cubes/build_datacubes.py` to convert FITS frames into datacubes.
6. Run `preprocessing/hdf5/zarr_to_hdf5.py` to store datapoints efficiently. *(Requires cleaned labels pickle file.)*
7. Run `preprocessing/cubes/save_fits_header_timestamps.py` to save timestamps for each sector.
8. Update `training/configs/dataset` with the correct data paths and adjust hyperparameters if needed.
9. Run `training/pipeline.py` to train the model.

## Evaluation

1. Run `preprocessing/catalog/download_files.py` to download files from the [TESS Catalog](https://archive.stsci.edu/missions/tess/catalogs/tic_v82/). The provided header.csv and md5sum.txt within the repository are provided from this url.
2. Run `preprocessing/catalog/split_pkl.py` to clean and split input pickle files into manageable chunks.
3. Run `training/eval_catalog.py` to process input chunks and generate outputs.
4. (Optional) Run `preprocessing/catalog/join_preds.py` to merge prediction chunks into a single file.
