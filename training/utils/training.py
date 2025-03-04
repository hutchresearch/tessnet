import torch
from torch import nn
import math
from torch.optim import lr_scheduler


def collate_fn_padd(batch):
    """
    collate_fn_padd
        Collects samples for a batch and pads each sample to the
        max length sequence.
        Written by Sally Bass & Logan Sizemore, July 2022
    """

    # Get the inputs, class labels, and regression labels
    inputs = [x[0] for x in batch]
    class_labels = torch.Tensor([x[1] for x in batch])
    reg_labels = torch.Tensor([x[2] for x in batch])
    sectors = torch.tensor([x[3] for x in batch], dtype=torch.int32)
    tmags = torch.Tensor([x[4] for x in batch])
    conts = torch.Tensor([x[5] for x in batch])
    ids = torch.tensor([x[6] for x in batch], dtype=torch.int64)
    ras = torch.tensor([x[7] for x in batch])
    decs = torch.tensor([x[8] for x in batch])

    # compute ragged inputs
    inputs_ragged = [x.transpose((1, 0, 2, 3)) for x in inputs]  # x transpose is to get dim that we want padded first
    inputs_ragged = [torch.Tensor(t) for t in inputs_ragged]

    # pad the input
    inputs = torch.nn.utils.rnn.pad_sequence(inputs_ragged)

    # permute the inputs
    inputs = inputs.permute((1, 2, 0, 3, 4))  # once padded, it gains a dim, which is the batch dim

    return inputs, class_labels, reg_labels, sectors, tmags, conts, ids, ras, decs


def init_optimizer(parameters, train_cfg) -> torch.optim.Optimizer:
    optim_name = train_cfg.optimizer
    if optim_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=train_cfg.lr)
    elif optim_name == "adamax":
        optimizer = torch.optim.Adamax(parameters, lr=train_cfg.lr)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=train_cfg.lr)
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=train_cfg.lr)
    elif optim_name == "asgd":
        optimizer = torch.optim.ASGD(parameters, lr=train_cfg.lr)
    else:
        raise Exception("No valid optimizer selected!")

    return optimizer


def init_scheduler(optimizer, train_cfg):
    if train_cfg.scheduler is None:
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif train_cfg.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=train_cfg.patience,
                                                   factor=train_cfg.factor)
    elif train_cfg.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, gamma=train_cfg.factor, step_size=train_cfg.step_size)
    elif train_cfg.scheduler == 'multi_step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=train_cfg.factor, milestones=train_cfg.milestones)
    else:
        raise ValueError

    return scheduler


def only_valid_regression_loss(predictions, targets, missing_period_value, rmse=False):
    """
    only_valid_regression_loss
        Performs regression loss for only prediction-target pairs that
        have targets not equal to the missing_period_value.
    """
    mask = targets.ne(missing_period_value)
    targets = torch.masked_select(targets, mask)
    predictions = torch.masked_select(predictions.squeeze(), mask)

    if len(targets) == 0:
        return torch.tensor([0.0], requires_grad=True)
    elif rmse:
        return torch.sqrt(nn.functional.mse_loss(predictions, targets))
    else:
        return nn.functional.mse_loss(predictions, targets)


def train_model(train_runner, dev_runner, logger, epochs, device, early_stopper=None):
    for epoch in range(epochs):
        train_runner.run()
        print(f"Train Loss (Process: {device}): {train_runner.loss}")

        predictions = dev_runner.run(gather_preds=True)
        print(f"Dev Loss (Process: {device}): {dev_runner.loss}")

        if dev_runner.classifier_head:
            print("\nTRAIN CONFUSION MATRIX: ")
            print(train_runner.class_metrics)

            print("\nDEV CONFUSION MATRIX: ")
            print(dev_runner.class_metrics)

        # Log Epoch Metrics
        train_loss = float(train_runner.loss.mean)

        logger.log_metrics({
            "train_loss": train_loss,
            "train_class_loss": float(train_runner.class_loss.mean),

            "train_mse_loss": float(train_runner.mse_loss.mean),
            "train_rmse_loss": float(train_runner.rmse_loss.mean),

            "train_denorm_mse_loss": float(train_runner.denorm_mse.mean),
            "train_denorm_rmse_loss": float(train_runner.denorm_rmse.mean),

            "train_class_accuracy": float(train_runner.class_metrics.accuracy),
            "train_class_precision": float(train_runner.class_metrics.precision),
            "train_class_recall": float(train_runner.class_metrics.recall),
            "train_class_f1score": float(train_runner.class_metrics.f1score),

            "train_cont_loss": float(train_runner.cont_loss.mean),

            "dev_loss": float(dev_runner.loss.mean),
            "dev_class_loss": float(dev_runner.class_loss.mean),

            "dev_mse_loss": float(dev_runner.mse_loss.mean),
            "dev_rmse_loss": float(dev_runner.rmse_loss.mean),

            "dev_denorm_mse_loss": float(dev_runner.denorm_mse.mean),
            "dev_denorm_rmse_loss": float(dev_runner.denorm_rmse.mean),

            "dev_class_accuracy": float(dev_runner.class_metrics.accuracy),
            "dev_class_precision": float(dev_runner.class_metrics.precision),
            "dev_class_recall": float(dev_runner.class_metrics.recall),
            "dev_class_f1score": float(dev_runner.class_metrics.f1score),

            "dev_cont_loss": float(dev_runner.cont_loss.mean),

            "learning_rate": train_runner.get_lr(),
            "epoch": epoch
        }, train_runner.global_step)

        train_runner.step_scheduler(train_loss)

        if dev_runner.classifier_head and not dev_runner.period_head and not dev_runner.cont_head:
            print("Early Stopping with class only")
            early_stopper(-dev_runner.class_metrics.accuracy, dev_runner.model, predictions)
        else:
            print("Early stopping with regression loss")
            early_stopper(dev_runner.loss.mean, dev_runner.model, predictions)

        if early_stopper.early_stop:
            break
