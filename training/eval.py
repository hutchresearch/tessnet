import os
import hydra
import sys
from dataset import TESSDataset
from utils.data_utils import handle_preloading, load_time_series
from utils.normalizer import LogNormalizer, ScaleNormalizer
from utils.misc import save_predictions
from utils.predictor import AverageEnsemblePredictor
from utils.training import collate_fn_padd, only_valid_regression_loss
from utils.model_utils import initialize_model
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
    print(f"Eval start ({cfg.logger.name}) | device: ({device})...", file=sys.stdout)

    normalizers = {
        'period': LogNormalizer(cfg.dataset.period_max_value),
        'input': ScaleNormalizer(cfg.dataset.crop_90p_value)
    }

    time_series = load_time_series(cfg.dataset.sectors, cfg.dataset.timestamps_dir)

    tess_data, preloaded_stars = handle_preloading(time_series, cfg.dataset, preload=False)

    missing_period_value = normalizers['period'].normalize(0.)
    if not cfg.model.classifier_head:
        tess_data = tess_data[tess_data['period'] != missing_period_value]
        mse_loss_crit = torch.nn.MSELoss()
    else:
        mse_loss_crit = lambda y_pred, y: only_valid_regression_loss(y_pred, y, missing_period_value)

    tess_dev = TESSDataset('test2', tess_data, preloaded_stars, time_series, cfg.dataset,
                           normalizers=normalizers)
    
    print(len(tess_dev))
    dev_loader = torch.utils.data.DataLoader(
        dataset=tess_dev,
        batch_size=cfg.dataloader.dev.batch_size,
        shuffle=False,
        collate_fn=collate_fn_padd,
        pin_memory=cfg.dataloader.pin_memory
    )

    models = []
    ROOT_DIR = '/cluster/research-groups/hutchinson/projects/ml_asto_tess/harry/0/ml_astro_tess20/training/eval_models'
    for file in os.listdir(ROOT_DIR):
        filename = os.fsdecode(file)
        if filename.endswith(".pt"):
            model = initialize_model(
                model_cfg=cfg.model,
                state_dict=torch.load(os.path.join(ROOT_DIR, filename))
            ).to(device)
            models.append(model)

    predictor = AverageEnsemblePredictor(models)

    dev_runner = AstroRunner(
        loader=dev_loader,
        predictor=predictor,
        class_loss_criterion=torch.nn.functional.cross_entropy,
        mse_loss_criterion=mse_loss_crit,
        class_head_weight=cfg.training.class_head_weight,
        normalizers=normalizers,
        device=device,
    )

    predictions = dev_runner.run(gather_preds=True)

    if dev_runner.classifier_head:
        print("\nDEV CONFUSION MATRIX: ")
        print(dev_runner.class_metrics)
        print("dev_class_accuracy: {}".format(dev_runner.class_metrics.accuracy))
        print("dev_class_precision: {}".format(dev_runner.class_metrics.precision))
        print("dev_class_recall: {}".format(dev_runner.class_metrics.recall))
        print("dev_class_f1score: {}".format(dev_runner.class_metrics.f1score))

    save_predictions(cfg.save_dir, 'eval', predictions)


if __name__ == "__main__":
    pipeline()
