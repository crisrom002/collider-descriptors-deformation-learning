
import numpy as np
import torch
from torch import nn

class BaseLoss(nn.Module):

    def __init__(self):
        super(BaseLoss, self).__init__()

    def setNormalization(self, value):
        raise NotImplementedError

    def clearNormalization(self):
        raise NotImplementedError

    def accumNormalization(self, output, target):
        raise NotImplementedError

    def divideNormalization(self, size):
        raise NotImplementedError

class MSENLoss(BaseLoss):

    def __init__(self):
        super(MSENLoss, self).__init__()

        self.mse_loss = nn.MSELoss()

        self.mse_loss_norm = nn.Parameter(torch.ones(1), requires_grad = False)

    def setNormalization(self, value):
        self.mse_loss_norm[:] = value

    def clearNormalization(self):   
        self.mse_loss_norm[:] = 0.0

    def accumNormalization(self, output, target):
        self.mse_loss_norm += self.mse_loss(output, target).detach().item()

    def divideNormalization(self, size):
        self.mse_loss_norm /= size
        
        if self.mse_loss_norm.item() == 0.0:
            self.mse_loss_norm[:] = 1.0

    def forward(self, output, target):
        loss = self.mse_loss(output, target) / self.mse_loss_norm

        return loss

class EarlyStopper:

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def step(self, validation_loss, epochs_from_last):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += epochs_from_last
            if self.counter >= self.patience:
                return True
        return False








