import os
import hydra
import sys
from dataset import TESSDataset
from utils.data_utils import handle_preloading, load_time_series
from utils.normalizer import LogNormalizer, ScaleNormalizer
from utils.augment import get_augmentors
from utils.early_stopping import EarlyStopping
from utils.predictor import ModelPredictor
from utils.training import collate_fn_padd, init_optimizer, only_valid_regression_loss, train_model, init_scheduler
from utils.model_utils import initialize_model
from logger import WandBAstroLogger
import torch
import numpy as np
import random
from runner import AstroRunner

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')


@hydra.main(config_path=DEFAULT_PATH, config_name="config", version_base="1.2")
def pipeline(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Pipeline start ({cfg.logger.name}) | device: ({device})...", file=sys.stdout)

    logger = WandBAstroLogger(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        config=cfg,
        name=cfg.logger.name,
        notes=cfg.logger.notes,
        save_path=cfg.save_dir
    )

    normalizers = {
        'period': LogNormalizer(cfg.dataset.period_max_value),
        'input': ScaleNormalizer(cfg.dataset.crop_90p_value)
    }

    augmentors = get_augmentors(cfg.augment)

    time_series = load_time_series(cfg.dataset.sectors, cfg.dataset.timestamps_dir)

    # preloading must be done outside of dataset
    tess_data, preloaded_stars = handle_preloading(time_series, cfg.dataset)

    missing_period_value = normalizers['period'].normalize(0.)
    if not cfg.model.classifier_head:
        tess_data = tess_data[tess_data['period'] != 0.]
        mse_loss_crit = torch.nn.MSELoss()
    else:
        mse_loss_crit = lambda y_pred, y: only_valid_regression_loss(y_pred, y, missing_period_value)

    tess_train = TESSDataset('train', tess_data, preloaded_stars, time_series, cfg.dataset,
                             augmentors=augmentors, normalizers=normalizers)

    tess_dev = TESSDataset('dev1', tess_data, preloaded_stars, time_series, cfg.dataset,
                           normalizers=normalizers)

    print('Size of augmented train set: {}'.format(len(tess_train)))

    train_sampler = None
    if cfg.dataloader.train.stratified:
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=tess_train.get_sampler_weights(),
            num_samples=len(tess_train),
            replacement=False
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=tess_train,
        batch_size=cfg.dataloader.train.batch_size,
        shuffle=(train_sampler is None), # Only shuffle when there is no sampler.
        sampler=train_sampler,
        collate_fn=collate_fn_padd,
        pin_memory=cfg.dataloader.pin_memory
    )

    dev_loader = torch.utils.data.DataLoader(
        dataset=tess_dev,
        batch_size=cfg.dataloader.dev.batch_size,
        shuffle=False,
        collate_fn=collate_fn_padd,
        pin_memory=cfg.dataloader.pin_memory
    )

    model = initialize_model(cfg.model).to(device)

    predictor = ModelPredictor(model)

    optimizer = init_optimizer(model.parameters(), cfg.training)

    scheduler = init_scheduler(optimizer, cfg.training)

    train_runner = AstroRunner(
        loader=train_loader,
        predictor=predictor,
        class_loss_criterion=torch.nn.functional.cross_entropy,
        mse_loss_criterion=mse_loss_crit,
        class_head_weight=cfg.training.class_head_weight,
        optimizer=optimizer,
        scheduler=scheduler,
        normalizers=normalizers,
        device=device,
    )

    dev_runner = AstroRunner(
        loader=dev_loader,
        predictor=predictor,
        class_loss_criterion=torch.nn.functional.cross_entropy,
        mse_loss_criterion=mse_loss_crit,
        class_head_weight=cfg.training.class_head_weight,
        normalizers=normalizers,
        device=device,
    )

    early_stopper = EarlyStopping(
        patience=cfg.early_stopping.patience,
        verbose=cfg.early_stopping.verbose,
        delta=cfg.early_stopping.delta,
        path=cfg.save_dir,
        save_name=logger.name
    )

    logger.save_config()

    train_model(
        train_runner=train_runner,
        dev_runner=dev_runner,
        logger=logger,
        epochs=cfg.training.epochs,
        device=device,
        early_stopper=early_stopper,
    )


if __name__ == "__main__":
    pipeline()
