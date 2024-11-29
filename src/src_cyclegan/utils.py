# src/src_cyclegan/utils.py

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
            verbose (bool): If True, prints a message for each validation metric improvement.
            delta (float): Minimum change to qualify as an improvement.
            path (str): Path to save the checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_metric, model):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        '''Saves model when validation metric improves.'''
        if self.verbose:
            print(f'Validation metric improved ({self.val_metric_min:.6f} --> {val_metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_metric_min = val_metric
