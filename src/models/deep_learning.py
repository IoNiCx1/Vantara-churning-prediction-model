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
    pos_weight = torch.tensor(
        [(y_train == 0).sum()/max((y_train == 1).sum(),1)],dtype=torch.float32

    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    optimizier = torch.optim.Adam(model.parameter)
    train_ds = TensorDataset(
        torch.tensor(X_train,dtype = torch.float32),torch.tensor(y_train,dtype = torch.float32)

    )
    train_loader = DataLoader(train_ds,batch_size = batch_size,shuffle = True)
    x_val_t = torch.tensor(X_val,dtype = torch.float32).to(device)
    y_val_t  = torch.tensor(y_val,dtype =torch.float32 ).to(device)
    history  = TrainingHistory(train_loss = [],val_loss   =[])
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement  = 0


    for epoch in range(max_epochs):
        model.train()
        epoch_loss  =.0
        for xb,yb in train_loader:
            xb,yb = xb.to(device),yb.to(device)
            optimizier.zero_grad()
            logits = model(xb)
            loss = criterion(logits,yb)
            loss.backward()
            optimizier.step()
            epoch_loss += loss.item()*len(xb)
        train_loss = epoch_loss/len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = criterion(val_logits,y_val_t).item()
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if val_loss<best_val_loss-1e-4:
            best_val_loss = val_loss
            best_state = {k:v.clone() for k,v in model.state_dict().item()}
            epochs_without_improvement =0
        else:
            epochs_without_imporvement +=1
            if epochs_without_improvement>=patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model,history


def predict_ann_proba(model:churnANN,X:np.ndarray)->np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(X,dtype = torch.float32).to(device))
        return torch.sigmoid(logits).cpu().numpy()



class PurchaseSequenceDataset(Dataset):
    def __init__(self,sequence:List[np.ndarray],targets:np.ndarray,max_len:Optional[int]=None):
        self.max_len = max_len or max(len(s) for s in sequence)
        self.sequence = sequence
        self.targets = targets

    def __len__(self)->int:
        return len(self.sequence)
    def __getitem__(self,idx:int):
        seq = self.sequence[idx]
        length = min(len(seq),self.max_len)
        padded = np.zeros((self.max_len,seq.shape[1]),dtype = np.float32)
        padded[:length] = seq[:length]
        return(
            torch.tensor(padded,dtype = torch.float32),
            torch.tensor(length,dtype = torch.long),
            torch.tensor(self.target[idx],dtype = torch.float32),

        )
class PurchaseLSTM(nn.Module):
    
 
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, 1))
 
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]  
        return self.head(last_hidden).squeeze(-1)


def train_lstm(
    sequences: List[np.ndarray],
    targets: np.ndarray,
    val_sequences: List[np.ndarray],
    val_targets: np.ndarray,
    hidden_dim: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 8,
    batch_size: int = 32,
) -> Tuple[PurchaseLSTM, TrainingHistory]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = sequences[0].shape[1] if sequences else 3
    model = PurchaseLSTM(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    train_ds = PurchaseSequenceDataset(sequences, targets)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds = PurchaseSequenceDataset(val_sequences, val_targets, max_len=train_ds.max_len)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
 
    history = TrainingHistory(train_loss=[], val_loss=[])
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
 
    for epoch in range(max_epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        for xb, lengths, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb, lengths)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
            n += len(xb)
        train_loss = epoch_loss / max(n, 1)
 
        model.eval()
        val_loss_total, n_val = 0.0, 0
        with torch.no_grad():
            for xb, lengths, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb, lengths)
                val_loss_total += criterion(preds, yb).item() * len(xb)
                n_val += len(xb)
        val_loss = val_loss_total / max(n_val, 1)
 
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
 
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
 
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

class SpendingAutoencoder(nn.Module):
    def __init__(self,input_dim:int,bottleneck_dim:int =4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,16),nn.ReLU(),
            nn.Linear(16,bottleneck_dim),nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim,16),nn.ReLU(),
            nn.Linear(16,input_dim),
        )
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.decoder(self.encoder(x))

def train_autoencoder(
    X_train:np.ndarray,
    X_val:np.ndarray,
    bottleneck_dim:int = 4,
    lr:float = 1e-3,
    max_epochs:int = 150,
    patience:int = 10,
    batch_size:int = 64,


)-> Tuple[SpendingAutoencoder,TrainingHistory,float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpendingAutoencoder(input_dim = X_train.shape[1],bottleneck_dim = bottleneck_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    train_ds = TensorDataset(torch.tensor(X_train,dtype = torch.float32))
    train_loader  = DataLoader(train_ds,batch_size = batch_size,shuffle = True)
    X_val_t = torch.tensor(X_val,dtype = torch.float32).to(device)
    history = TrainingHistory(train_loss = [],val_loss= [])
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epochs in range(max_epochs):
        model.train()
        epoch_loss = .0
        for (xb,)  in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon  = model(xb)
            loss = criterion(recon,xb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*len(xb)
        train_loss = epoch_loss/len(train_ds)
        model.eval()
        with torch.no_grad():
            val_recon = model(X_val_t)
            val_loss = criterion(val_recon,X_val_t).item()
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if val_loss<best_val_loss-1e-5:
            best_val_loss = val_loss
            best_state = {k:v.clone() for k,v in model.state_dict().items()}





