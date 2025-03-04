import torch
import pickle as pkl
from torch.utils.data import Dataset
from utils.data_utils import load_star_crop, fix_star
import numpy as np
from utils.cubes import get_crop, zarr_filepath
import os
import datetime


class TESSDataset(Dataset):
    def __init__(self, subset, tess_data, preloaded_stars, time_series, dataset_cfg,
                 normalizers: dict = None, augmentors: list = None, use_zarr_cubes=False):
        super().__init__()
        if augmentors is None:
            augmentors = []
        if normalizers is None:
            normalizers = {}

        self.dataset_cfg = dataset_cfg
        self.sectors = dataset_cfg.sectors
        self.use_zarr_cubes = use_zarr_cubes

        if subset is not None:
            set_name = '{}_set'.format(subset)
            self.set_data = tess_data[tess_data[set_name]].reset_index(drop=True)
        else:
            self.set_data = tess_data.reset_index(drop=True)

        self.n_stars = len(self.set_data)
        self.timestamps_dir = dataset_cfg.timestamps_dir
        self.remove_zero_slices = dataset_cfg.remove_zero_slices
        self.cont_full = dataset_cfg.cont_full

        star_map = []
        aug_map = []

        for i, augmentor in enumerate(augmentors):
            aug_stars = int(self.n_stars * augmentor.proportion)
            star_map.extend(np.random.randint(self.n_stars, size=aug_stars))
            aug_map.extend([i] * aug_stars)

        self.aug_mapping = list(zip(star_map, aug_map))
        np.random.shuffle(self.aug_mapping)

        self.n_total = self.n_stars + len(self.aug_mapping)
        self.augmentors = augmentors

        self.h5_dir = dataset_cfg.crop_hdf5_dir
        self.cubes_dir = dataset_cfg.cubes_dir
        self.normalizers = normalizers

        self.preloaded_stars = preloaded_stars

        self.star_type_onehot_map = {
            "eclipse": 0.0,
            "pulse": 1.0,
            "rot": 2.0,
            "nonvar": 3.0
        }

        self.time_series = time_series
        self.time_strategy = dataset_cfg.time_strategy

    def __len__(self):
        return self.n_total

    def __getitem__(self, index):
        if index < self.n_stars:
            star = self.set_data.iloc[index]
            augmentor = None
        else:
            star_idx, aug_idx = self.aug_mapping[index - self.n_stars]
            star = self.set_data.iloc[star_idx]
            augmentor = self.augmentors[aug_idx]

        sec, cam, ccd, star_id, ra, dec = star.outSec, star.outCam, star.outCcd, star.outID, star.ra, star.dec
        star_key = '{}.{}.{}.{}'.format(sec, cam, ccd, star_id)

        if 'mmtype' in self.set_data:
            star_type: float = self.star_type_onehot_map[star.mmtype]
        else:
            star_type: float = 0.

        if 'period' in self.set_data:
            period = float(star.period)
            if 'period' in self.normalizers:
                period = self.normalizers['period'].normalize(period)
        else:
            period = 0.

        tmag = float(star.tmag)

        if star_key in self.preloaded_stars:
            # preloaded stars are already fixed
            star_crop = self.preloaded_stars[star_key]
        else:
            if self.use_zarr_cubes:
                filepath = zarr_filepath(self.cubes_dir, int(sec), int(cam), int(ccd))
                star_crop, _ = get_crop(filepath, 6, int(star.rowPix), int(star.colPix))
            else:
                star_crop = load_star_crop(self.h5_dir, sec, cam, ccd, star_id)
            star_crop = fix_star(self.time_series, star_crop, sec, self.dataset_cfg)

        # increase precision before augment/normalize
        star_crop = star_crop.astype(np.float32)

        if augmentor is not None:
            # pycharm thinks this is a list, its not
            star_crop = augmentor(star_crop, index)

            # clip in case of negative from mag shift
            star_crop = star_crop.clip(0, None)

        if 'input' in self.normalizers:
            star_crop = self.normalizers['input'].normalize(star_crop)

        star_crop = np.transpose(star_crop, (2, 0, 1))  # HxWxT -> TxHxW
        model_input = np.expand_dims(star_crop, axis=0)  # TxHxW -> 1xTxHxW

        if 'cont' in self.set_data:
            star_cont = star.contFull if self.cont_full else star.cont
        else:
            star_cont = 0.

        return model_input, star_type, period, sec, tmag, star_cont, star_id, ra, dec

    def get_sampler_weights(self) -> np.array:
        """
        get_sampler_weights
            Generates the per-element weighting for stratified class sampling
        """
        star_types = self.set_data["mmtype"]
        class_counts = star_types.value_counts()
        sample_weights = 1.0 / (class_counts / len(star_types))
        sample_weights = sample_weights / sample_weights.sum()
        sample_weights = sample_weights.to_dict()

        type_list = star_types.values.tolist()
        augment_type_list = [star_types.iloc[aug_map[0]] for aug_map in self.aug_mapping]

        weight_vector = np.vectorize(sample_weights.get)(type_list + augment_type_list)

        assert len(weight_vector) == len(self)
        return weight_vector
