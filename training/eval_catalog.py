import os
import hydra
import sys
from dataset import TESSDataset
from utils.data_utils import handle_preloading, load_time_series
from utils.normalizer import LogNormalizer, ScaleNormalizer
from utils.misc import save_predictions
from utils.predictor import AverageEnsemblePredictor, AverageEnsemblePredictorNoCont
from utils.training import collate_fn_padd, only_valid_regression_loss
from utils.model_utils import initialize_model
from utils.catalog_util import get_available_file
import torch
import numpy as np
import yaml
import random
from catalog_runner import CatalogRunner
from argparse import Namespace
import pandas as pd
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')


@hydra.main(config_path=DEFAULT_PATH, config_name="catalog_cfg", version_base="1.2")
def pipeline(cfg):
    while True:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Eval start ({cfg.logger.name}) | device: ({device})...", file=sys.stdout)

        normalizers = {
            'period': LogNormalizer(cfg.dataset.period_max_value),
            'input': ScaleNormalizer(cfg.dataset.crop_90p_value)
        }

        pkl_filename, pkl_filepath = get_available_file(cfg.processed_pkl_dir, cfg.status_dir, cfg.pred_dir)
        if pkl_filepath is None:
            print("No more files left!")
            quit()
        cfg.dataset.data_pd_pkl_path = pkl_filepath
        print("Found path: ", cfg.dataset.data_pd_pkl_path)
        pred_filename = pkl_filename.replace('.pkl', '.csv')

        time_series = load_time_series(cfg.dataset.sectors, cfg.dataset.timestamps_dir)

        tess_data, preloaded_stars = handle_preloading(time_series, cfg.dataset, preload=False, filter_none=True)
        missing_period_value = normalizers['period'].normalize(0.)
        if not cfg.model.classifier_head:
            tess_data = tess_data[tess_data['period'] != missing_period_value]
            mse_loss_crit = torch.nn.MSELoss()
        else:
            mse_loss_crit = lambda y_pred, y: only_valid_regression_loss(y_pred, y, missing_period_value)

        tess_dev = TESSDataset(None, tess_data, preloaded_stars, time_series, cfg.dataset,
                               normalizers=normalizers, use_zarr_cubes=True)
        print(tess_dev.set_data)
        print(len(tess_dev))
        
        if len(tess_dev) == 0:
            raise ValueError
            pred_filepath = os.path.join(cfg.pred_dir, pred_filename)
            column_names = ['class_pred', 'eb_prob', 'pulse_prob', 'rot_prob', 'nonvar_prob', 'cont_pred', 'cont_full_pred', 'sector', 'tmag', 'ra', 'dec', 'ID']
            empty_df = pd.DataFrame(columns=column_names)
            empty_df.to_csv(pred_filepath, index=False)
            quit()

        dev_loader = torch.utils.data.DataLoader(
            dataset=tess_dev,
            batch_size=cfg.dataloader.dev.batch_size,
            shuffle=False,
            collate_fn=collate_fn_padd,
            pin_memory=cfg.dataloader.pin_memory
        )

        ROOT_DIR = '/cluster/research-groups/hutchinson/projects/ml_asto_tess/harry/0/ml_astro_tess20/training/catalog_models'
        MODEL_PATHS = [
            'classification/03Aug-11_49_33_CNNAllTrain_model.pt',
            'regression_cont/14Sep-11_22_57_CNNCont_model.pt',
            'regression_cont_full/14Sep-13_38_19_CNNContFull_model.pt'
        ]
        CONFIG_PATHS = [
            'classification/03Aug-09_47_13_CNNAllTrain_args.yaml',
            'regression_cont/14Sep-09_20_01_CNNCont_args.yaml',
            'regression_cont_full/14Sep-12_43_03_CNNContFull_args.yaml'
        ]

        predictors = []
        for i in range(len(MODEL_PATHS)):
            models = []
            with open(os.path.join(ROOT_DIR, CONFIG_PATHS[i])) as stream:
                model_cfg = yaml.safe_load(stream)
                formatted_cfg = Namespace(**model_cfg['model'])

                model = initialize_model(
                    model_cfg=formatted_cfg,
                    state_dict=torch.load(os.path.join(ROOT_DIR, MODEL_PATHS[i]))
                ).to(device)
                models.append(model)

            predictor = AverageEnsemblePredictor(models)
            predictors.append(predictor)

        dev_runner = CatalogRunner(
            loader=dev_loader,
            class_predictor=predictors[0],
            cont_predictor=predictors[1],
            cont_full_predictor=predictors[2],
            class_loss_criterion=torch.nn.functional.cross_entropy,
            mse_loss_criterion=mse_loss_crit,
            class_head_weight=cfg.training.class_head_weight,
            normalizers=normalizers,
            device=device,
        )

        predictions = dev_runner.run()
        f_name = os.path.basename(cfg.dataset.data_pd_pkl_path)
        pre, _ = os.path.splitext(f_name)

        pred_filepath = os.path.join(cfg.pred_dir, pred_filename)
        predictions.to_csv(pred_filepath, index=False)


if __name__ == "__main__":
    pipeline()
