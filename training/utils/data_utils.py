import h5py
import os
import pickle as pkl
import numpy as np
from dateutil.parser import parse
from tqdm import tqdm


def load_star_crop(h5_dir, sec, cam, ccd, star_id):
    path = os.path.join(h5_dir, 'sec_{}.hdf5'.format(str(sec).zfill(2)))

    f = h5py.File(path, "r")
    subgroup = '{}.{}.{}'.format(sec, cam, ccd)
    dataset_path = '{}/{}'.format(subgroup, star_id)
    star_crop = f[dataset_path][:]
    f.close()

    return star_crop


def partial_eager_load(time_series, tess_data, dataset_cfg):
    num_eager_load = dataset_cfg.num_eager_load
    if num_eager_load == -1:
        num_eager_load = len(tess_data)

    preloaded_stars = {}
    load_data = tess_data.head(num_eager_load)
    for star in tqdm(load_data.itertuples(), total=len(load_data)):
        sec, cam, ccd, star_id = star.outSec, star.outCam, star.outCcd, star.outID
        star_crop = load_star_crop(dataset_cfg.crop_hdf5_dir, sec, cam, ccd, star_id)
        star_crop = fix_star(time_series, star_crop, sec, dataset_cfg)
        star_key = '{}.{}.{}.{}'.format(sec, cam, ccd, star_id)

        assert star_key not in preloaded_stars
        preloaded_stars[star_key] = star_crop
    return preloaded_stars


def handle_preloading(time_series, dataset_cfg, preload=True, filter_none=False):
    with open(dataset_cfg.data_pd_pkl_path, "rb") as pkl_handle:
        tess_data = pkl.load(pkl_handle)
        if filter_none:
            return tess_data, {}

        tess_data = tess_data[tess_data['outSec'].isin(dataset_cfg.sectors)]
        if not dataset_cfg.include_nonvar:
            tess_data = tess_data[tess_data['mmtype'] != 'nonvar']

        if 'cont' in tess_data:
            tess_data = tess_data[tess_data['cont'] >= dataset_cfg.min_cont]
            tess_data = tess_data[tess_data['contFull'] >= dataset_cfg.min_cont_full]
        else:
            print("Cont missing, skipping")

        if dataset_cfg.min_bp_diff > 0:
            tess_data = tess_data[~tess_data['BPmag'].isna() & ~tess_data['RPmag'].isna()]
            if 'mmtype' in tess_data:
                tess_data = tess_data[((tess_data['BPmag'] - tess_data['RPmag']) > dataset_cfg.min_bp_diff) | (tess_data['mmtype'] != 'nonvar')]
            else:
                print("mmtype missing, skipping mag filter")

            if dataset_cfg.filter_giants:
                print("Filtering giants..")
                tess_data = tess_data[tess_data['plx'] > 0]
                tess_data = tess_data[~tess_data['gmag'].isna()]
                if 'mmtype' in tess_data:
                    tess_data = tess_data[(tess_data['mmtype'] != 'nonvar') | (tess_data['gmag'] - 5 * np.log10(1000 / tess_data['plx']) + 5 > 2)]
                else:
                    print("mmtype missing, skipping giant filter")
     
    if dataset_cfg.num_eager_load != 0:
        print('Partial eager loading...', preload)

    if preload:
        preloaded_stars = partial_eager_load(time_series, tess_data, dataset_cfg)
    else:
        preloaded_stars = {}

    return tess_data, preloaded_stars


def interpolate_section(star_crop, crop_size, gap_idx, gap_len):
    interpolated = np.empty([crop_size, crop_size, gap_len], dtype=np.float16)

    for i in range(crop_size):
        for j in range(crop_size):
            # note to self: slicing is non inclusive at the end
            interpolated[i, j, :] = np.interp(np.arange(gap_len), [-1, gap_len], star_crop[i, j, gap_idx-1:gap_idx+1])

    return interpolated


INTERVAL = 1800
FORGIVENESS = 5


def find_gaps(series):
    prev_time = None
    gaps = []
    for i, time in enumerate(series):
        if prev_time is not None:
            elapsed = time - prev_time
            if abs(elapsed - INTERVAL) > FORGIVENESS:
                gap_size = round(elapsed / INTERVAL) - 1
                assert gap_size > 0

                # (index of insertion, n frames to insert)
                gaps.append((i, gap_size))
        prev_time = time

    return gaps


def remove_zero_slices(star_crop, series):
    # this changes series into a np array, but that doesn't matter
    assert star_crop.shape[2] == len(series)

    zero_indices = np.where(~star_crop.any(axis=(0, 1)))[0]

    cleaned_crop = np.delete(star_crop, zero_indices, axis=2)
    cleaned_series = np.delete(series, zero_indices)

    return cleaned_crop, cleaned_series


def load_time_series(sectors, timestamps_dir):
    time_series = {}
    timestamp_files = ['sec_{}.txt'.format(str(sec).zfill(2)) for sec in sectors]

    for i, timestamp_fn in enumerate(timestamp_files):
        timestamp_fp = os.path.join(timestamps_dir, timestamp_fn)
        sector_number = sectors[i]
        time_series[sector_number] = []
        with open(timestamp_fp, "r") as handle:
            first_time = None
            for j, timestamp_str in enumerate(handle.readlines()):
                time = parse(timestamp_str)
                if first_time is None:
                    first_time = time

                elapsed = (time - first_time).total_seconds()
                time_series[sector_number].append(elapsed)
    return time_series


def fix_star(time_series, star_crop, sec, dataset_cfg):
    # fixes issues related to inconsistency: zero slices near gaps, and fills gaps according to strategy
    # also increases precision
    series = time_series[sec]
    if dataset_cfg.remove_zero_slices:
        star_crop, series = remove_zero_slices(star_crop, series)

    if dataset_cfg.time_strategy is None:
        return star_crop

    elif dataset_cfg.time_strategy in ['zeros', 'interpolate']:
        chunks = []
        gap_data = find_gaps(series)
        crop_size = star_crop.shape[0]

        start = 0
        for gap_idx, gap_len in gap_data:
            if dataset_cfg.time_strategy == 'zeros':
                # fill with zeros
                insert_block = np.zeros((crop_size, crop_size, gap_len), dtype=np.float16)
            else:
                # interpolate between nearest frames
                insert_block = interpolate_section(star_crop, crop_size, gap_idx, gap_len)

            chunks.append(star_crop[:, :, start:gap_idx])
            chunks.append(insert_block)
            start = gap_idx
        chunks.append(star_crop[:, :, start:])

        return np.concatenate(chunks, axis=2)
    else:
        raise ValueError
