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


class AstroRunner:
    def __init__(
            self,
            loader: torch.utils.data.DataLoader,
            predictor: Predictor,
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
        self.predictor = predictor
        self.model = predictor.model if isinstance(predictor, ModelPredictor) else None
        self.classifier_head = predictor.classifier_head
        self.period_head = predictor.period_head
        self.cont_head = predictor.cont_head
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

    def run(self, gather_preds=False) -> pd.DataFrame:
        self._reset_metrics()

        self.predictor.train(self.stage is Stage.TRAIN)

        predictions_dict = {
            'class_pred': [],
            'true_class': [],
            'period_pred': [],
            'true_period': [],
            'cont_pred': [],
            'true_cont': [],
            "sector": [],
            "tmag": [],
            "ID": []
        }
        loader = tqdm(self.loader, unit='batch')

        for cube_inputs, class_target, period_target, sectors, tmags, conts, IDs, ras, decs in loader:
            batch_size = cube_inputs.shape[0]
            loss, class_loss, mse_loss, denorm_mse_loss, cont_loss, class_pred, period_pred, cont_pred = self._run_batch(cube_inputs,
                                                                                                   class_target,
                                                                                                   period_target, conts)
            period_pred = period_pred.squeeze().cpu().detach()
            cont_pred = cont_pred.squeeze().cpu().detach()
            if 'period' in self.normalizers:
                period_pred = self.normalizers['period'].denormalize(period_pred)
                period_target = self.normalizers['period'].denormalize(period_target)

            if gather_preds:
                predictions_dict['class_pred'].append(class_pred.cpu().detach().numpy())
                predictions_dict['true_class'].append(class_target.numpy())
                predictions_dict['period_pred'].append(period_pred.numpy())
                predictions_dict['true_period'].append(period_target.numpy())
                predictions_dict['cont_pred'].append(cont_pred.numpy())
                predictions_dict['true_cont'].append(conts.numpy())
                predictions_dict['sector'].append(sectors.numpy())
                predictions_dict['tmag'].append(tmags.numpy())
                predictions_dict['ID'].append(IDs.numpy())

            if self.optimizer:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.loss.accum(float(loss.detach()), batch_size)

            if self.classifier_head:
                self.class_loss.accum(float(class_loss.detach()), batch_size)
                self.class_metrics.accum(class_target.detach().cpu(), class_pred.detach().cpu())

            if self.period_head:
                rmse_loss = torch.sqrt(mse_loss.detach())
                denorm_rmse_loss = torch.sqrt(denorm_mse_loss.detach())

                self.mse_loss.accum(float(mse_loss.detach()), batch_size)
                self.denorm_mse.accum(float(denorm_mse_loss.detach()), batch_size)
                self.rmse_loss.accum(float(rmse_loss), batch_size)
                self.denorm_rmse.accum(float(denorm_rmse_loss), batch_size)

            if self.cont_head:
                self.cont_loss.accum(float(cont_loss.detach()), batch_size)

            self.global_step += 1

        if gather_preds:
            for k in predictions_dict:
                predictions_dict[k] = np.concatenate(predictions_dict[k])
            return pd.DataFrame(predictions_dict)

    def _run_batch(
            self,
            cube_inputs: torch.Tensor,
            class_target_t: torch.Tensor,
            period_target_t: torch.Tensor,
            cont_target_t: torch.Tensor
    ):
        self.run_count += 1

        cube_inputs = cube_inputs.to(self.device)
        class_target_t = class_target_t.to(self.device).long()
        period_target_t = period_target_t.reshape(-1, 1).to(self.device)
        cont_target_t = cont_target_t.reshape(-1, 1).to(self.device)

        class_pred, period_pred, cont_pred = self.predictor(cube_inputs)

        if self.classifier_head:
            class_loss = self.class_loss_criterion(class_pred, class_target_t).to(self.device)
        else:
            class_loss = torch.Tensor([float("nan")])

        if self.period_head:
            mse_loss = self.mse_loss_criterion(period_pred, period_target_t).to(self.device)

            denorm_mse_loss = torch.Tensor([float("nan")])
            if 'period' in self.normalizers:
                denorm_period_pred = self.normalizers['period'].denormalize(period_pred)
                denorm_period_targ = self.normalizers['period'].denormalize(period_target_t)
                denorm_mse_loss = self.mse_loss_criterion(denorm_period_pred, denorm_period_targ)
        else:
            mse_loss = torch.Tensor([float("nan")])
            denorm_mse_loss = torch.Tensor([float("nan")])

        if self.cont_head:
            cont_loss = self.cont_loss_crit(cont_pred, cont_target_t).to(self.device)
        else:
            cont_loss = torch.Tensor([float("nan")])

        loss = torch.tensor(0.).to(self.device)

        if self.period_head:
            loss += mse_loss
        if self.classifier_head:
            loss += class_loss * self.class_head_weight
        if self.cont_head:
            loss += cont_loss
        
        if not self.period_head and not self.classifier_head and not self.cont_head:
            raise Exception('Headless!')

        class_pred_label = torch.argmax(class_pred, dim=1)
        return loss, class_loss, mse_loss, denorm_mse_loss, cont_loss, class_pred_label, period_pred, cont_pred

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
