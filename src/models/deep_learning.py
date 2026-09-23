from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn 

from torch.utils.data import DataLoader, Dataset,TensorDataset

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

class churnANN(nn.Module):
    def __init__(self,input_dim:int,hidden_dims:List[int] = (64,32),dropout:float  = .3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers+= [
                nn.Linear(prev_dim,h),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(h),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim,1))
        self.net = nn.Sequential(*layers)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.net(x).squeeze(-1)

@dataclass
class TrainingHistory:
    train_loss:List[float]
    val_loss:List[float]

def train_ann(
        X_train:np.ndarray,
        y_train:np.ndarray,
        X_val:np.ndarray,
        y_val:np.ndarray,
        hidden_dims:List[int] = (64,32),
        dropout:float = .3,
        lr:float = 1e-3,
        max_epochs:int = 200,
        patience:int = 10,
        batch_size:int = 64,
)->Tuple[churnANN,TrainingHistory]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = churnANN(input_dim  = X_train.shape[1],hidden_dims = hidden_dims,dropout = dropout).to(device)
    pos_weight = torch.tensor
        





