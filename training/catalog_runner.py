import torch
from typing import Optional
from metrics import AverageMeter, ClassifierMetricMeter
from utils.predictor import Predictor, ModelPredictor
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torchmetrics import functional
from tqdm import tqdm
import pandas as pd
from enum import Enum, auto
import numpy as np

class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    DEV = auto()


class CatalogRunner:
    def __init__(
            self,
            loader: torch.utils.data.DataLoader,
            class_predictor: Predictor,
            cont_predictor: Predictor,
            cont_full_predictor: Predictor,
            class_loss_criterion,
            mse_loss_criterion,
            class_head_weight: float,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional = None,
            normalizers: Optional[dict] = None,
            device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:

        if normalizers is None:
            normalizers = {}

        self.loader = loader
        self.class_predictor = class_predictor
        self.cont_predictor = cont_predictor
        self.cont_full_predictor = cont_full_predictor
        self.cont_loss_crit = torch.nn.MSELoss()

        self.normalizers = normalizers

        self.class_loss_criterion = class_loss_criterion
        self.mse_loss_criterion = mse_loss_criterion
        self.class_head_weight = class_head_weight

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stage = Stage.DEV if optimizer is None else Stage.TRAIN
        self.device = device
        self.global_step = 0
        self._reset_metrics()

    def run(self) -> pd.DataFrame:
        self._reset_metrics()

        self.class_predictor.train(False)
        self.cont_predictor.train(False)
        self.cont_full_predictor.train(False)

        predictions_dict = {
            'class_pred': [],
            'eb_prob': [],
            'pulse_prob': [],
            'rot_prob': [],
            'nonvar_prob': [],
            'cont_pred': [],
            'cont_full_pred': [],
            "sector": [],
            "tmag": [],
            'ra': [],
            'dec': [],
            "ID": [],
        }
        loader = tqdm(self.loader, unit='batch')

        for cube_inputs, class_target, period_target, sectors, tmags, conts, IDs, ras, decs in loader:
            loss, class_loss, mse_loss, denorm_mse_loss, cont_loss, class_pred, period_pred, cont_pred, class_pred_probs = \
                self._run_batch(self.class_predictor, cube_inputs, class_target, period_target, conts, classifier_head=True)
            
            np_probs = class_pred_probs.cpu().detach().numpy()
            predictions_dict['class_pred'].append(class_pred.cpu().detach().numpy())
            predictions_dict['sector'].append(sectors.numpy())
            predictions_dict['tmag'].append(tmags.numpy())
            predictions_dict['ID'].append(IDs.numpy())
            predictions_dict['eb_prob'].append(np_probs[:, 0])
            predictions_dict['pulse_prob'].append(np_probs[:, 1])
            predictions_dict['rot_prob'].append(np_probs[:, 2])
            predictions_dict['nonvar_prob'].append(np_probs[:, 3])
            predictions_dict['ra'].append(ras.numpy())
            predictions_dict['dec'].append(decs.numpy())

            loss, class_loss, mse_loss, denorm_mse_loss, cont_loss, class_pred, period_pred, cont_pred, class_pred_probs = \
                self._run_batch(self.cont_predictor, cube_inputs, class_target, period_target, conts, cont_head=True)

            cont_pred = cont_pred.squeeze().cpu().detach()
            predictions_dict['cont_pred'].append(cont_pred.numpy())

            loss, class_loss, mse_loss, denorm_mse_loss, cont_loss, class_pred, period_pred, cont_pred, class_pred_probs = \
                self._run_batch(self.cont_full_predictor, cube_inputs, class_target, period_target, conts, cont_head=True)
            cont_pred = cont_pred.squeeze().cpu().detach()
            predictions_dict['cont_full_pred'].append(cont_pred.numpy())

            self.global_step += 1

        for k in predictions_dict:
            try:
                k_list = predictions_dict[k]
                k_fix = [entry.reshape(1) if entry.ndim == 0 else entry for entry in k_list]
                predictions_dict[k] = np.concatenate(k_fix)
            except Exception as e:
                print(e)
                print(k)
                print(predictions_dict[k][0], type(predictions_dict[k][0]))
                print(predictions_dict[k][-1], type(predictions_dict[k][-1]))
                raise ValueError
        return pd.DataFrame(predictions_dict)

    def _run_batch(
            self,
            predictor: Predictor,
            cube_inputs: torch.Tensor,
            class_target_t: torch.Tensor,
            period_target_t: torch.Tensor,
            cont_target_t: torch.Tensor,
            classifier_head=False,
            period_head=False,
            cont_head=False
    ):
        self.run_count += 1

        cube_inputs = cube_inputs.to(self.device)
        class_target_t = class_target_t.to(self.device).long()
        period_target_t = period_target_t.reshape(-1, 1).to(self.device)
        cont_target_t = cont_target_t.reshape(-1, 1).to(self.device)

        class_pred, period_pred, cont_pred = predictor(cube_inputs)

        if classifier_head:
            class_loss = self.class_loss_criterion(class_pred, class_target_t).to(self.device)
        else:
            class_loss = torch.Tensor([float("nan")])

        if period_head:
            mse_loss = self.mse_loss_criterion(period_pred, period_target_t).to(self.device)

            denorm_mse_loss = torch.Tensor([float("nan")])
            if 'period' in self.normalizers:
                denorm_period_pred = self.normalizers['period'].denormalize(period_pred)
                denorm_period_targ = self.normalizers['period'].denormalize(period_target_t)
                denorm_mse_loss = self.mse_loss_criterion(denorm_period_pred, denorm_period_targ)
        else:
            mse_loss = torch.Tensor([float("nan")])
            denorm_mse_loss = torch.Tensor([float("nan")])

        if cont_head:
            cont_loss = self.cont_loss_crit(cont_pred, cont_target_t).to(self.device)
        else:
            cont_loss = torch.Tensor([float("nan")])

        loss = torch.tensor(0.).to(self.device)

        if period_head:
            loss += mse_loss
        if classifier_head:
            loss += class_loss * self.class_head_weight
        if cont_head:
            loss += cont_loss
        
        if not period_head and not classifier_head and not cont_head:
            raise Exception('Headless!')
        
        class_pred_label = torch.argmax(class_pred, dim=1)
        class_pred_probs = torch.nn.functional.softmax(class_pred, dim=1)
        return loss, class_loss, mse_loss, denorm_mse_loss, cont_loss, class_pred_label, period_pred, cont_pred, class_pred_probs

    def _reset_metrics(self) -> None:
        self.run_count = 0
        self.loss = AverageMeter()
        self.class_loss = AverageMeter()
        self.mse_loss = AverageMeter()
        self.denorm_mse = AverageMeter()
        self.rmse_loss = AverageMeter()
        self.denorm_rmse = AverageMeter()
        self.cont_loss = AverageMeter()
        self.class_metrics = ClassifierMetricMeter()

    def step_scheduler(self, loss):
        assert self.scheduler is not None

        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(loss)
        else:
            self.scheduler.step()

    def get_lr(self):
        assert self.scheduler is not None

        if isinstance(self.scheduler, ReduceLROnPlateau):
            return self.scheduler.optimizer.param_groups[0]['lr']
        else:
            return self.scheduler.get_last_lr()[0]
