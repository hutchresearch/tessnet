from utils.misc import skip_save, save_predictions
from typing import List
import pandas as pd
import os
import torch
import numpy as np
import datetime


class EarlyStopping:
    """
        Early stops the training if validation loss doesn't improve after a given patience.
        https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='./', save_name=""):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = float('-inf')
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path if path else "./"
        self.save_name = save_name

        self._now: str = datetime.datetime.now().strftime("%d%b-%H_%M_%S")

    def __call__(self, val_loss, model, predictions):
        score = -val_loss

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). '
                      f'Saving model and predictions...')

            save_predictions(self.path, self._now, predictions)
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    @skip_save
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        model_save_name = os.path.join(self.path, f"{self._now}_{self.save_name}_model.pt")
        torch.save(model.state_dict(), model_save_name)

        self.val_loss_min = val_loss
