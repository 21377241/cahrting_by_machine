"""PyTorch implementations of neural network architectures."""

import os as _os

# Windows 跨盘符 DLL 搜索路径修复（WinError 1114）
_dll_dirs = []
for _p in [
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\lib",
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\bin",
]:
    if _os.path.isdir(_p) and hasattr(_os, "add_dll_directory"):
        _dll_dirs.append(_os.add_dll_directory(_p))

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from tqdm import tqdm

from cbm.ml.models.base import BaseNeuralNetworkModel


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for PyTorch."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


# ==================== Neural Network Modules ====================

class FNNModule(nn.Module):
    """Feed-forward Neural Network module."""
    
    def __init__(
        self,
        input_size: int = 12,
        hidden_sizes: List[int] = [64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class CNNModule(nn.Module):
    """Convolutional Neural Network module for sequential data."""
    
    def __init__(
        self,
        input_size: int = 12,
        n_filters: int = 32,
        kernel_size: int = 3,
        hidden_size: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size, padding=kernel_size // 2)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(n_filters * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.squeeze(-1)


class LSTMModule(nn.Module):
    """Long Short-Term Memory module."""
    
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features) -> (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # LSTM output
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        out = self.dropout(h_n[-1])
        out = self.fc(out)
        
        return out.squeeze(-1)


class CNNLSTMModule(nn.Module):
    """
    Combined CNN-LSTM module.
    
    This is the architecture that performed best in the paper:
    CNN for local pattern extraction followed by LSTM for sequential modeling.
    """
    
    def __init__(
        self,
        input_size: int = 12,
        cnn_filters: int = 32,
        cnn_kernel_size: int = 3,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, cnn_filters, cnn_kernel_size, padding=cnn_kernel_size // 2)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, cnn_kernel_size, padding=cnn_kernel_size // 2)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_size, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        batch_size = x.size(0)
        
        # CNN: (batch, features) -> (batch, 1, features) -> (batch, filters, features)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Reshape for LSTM: (batch, filters, features) -> (batch, features, filters)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        out = self.dropout(h_n[-1])
        out = self.fc(out)
        
        return out.squeeze(-1)


# ==================== Model Wrappers ====================

class PyTorchModelBase(BaseNeuralNetworkModel):
    """Base class for PyTorch model wrappers."""
    
    def __init__(self, device: str = "auto"):
        self.device = get_device(device)
        self.model: Optional[nn.Module] = None
        self._is_fitted = False
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 10,
        loss_function: str = "mse",
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        self.model.to(self.device)
        self.model.train()
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        if weights is not None:
            weights_tensor = torch.FloatTensor(weights).to(self.device)
        else:
            weights_tensor = torch.ones(len(X)).to(self.device)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Loss function
        if loss_function == "mse":
            criterion = nn.MSELoss(reduction='none')
        elif loss_function == "mae":
            criterion = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Training loop
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        
        for epoch in iterator:
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_X, batch_y, batch_w in dataloader:
                optimizer.zero_grad()
                
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Apply sample weights
                weighted_loss = (loss * batch_w).mean()
                
                weighted_loss.backward()
                # Gradient clipping (prevents exploding gradients with large batches)
                grad_clip = kwargs.get("grad_clip_norm", 1.0)
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()
                
                epoch_loss += weighted_loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_train_loss)
            
            # Validation — chunked to avoid OOM on large validation sets
            if validation_data is not None:
                self.model.eval()
                val_chunk = min(batch_size * 8, 4096)
                with torch.no_grad():
                    chunks = []
                    for s in range(0, len(X_val_tensor), val_chunk):
                        chunks.append(self.model(X_val_tensor[s: s + val_chunk]))
                    val_pred = torch.cat(chunks)
                    val_loss = criterion(val_pred, y_val_tensor).mean().item()
                history["val_loss"].append(val_loss)
                self.model.train()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if verbose:
                    iterator.set_postfix({
                        "train_loss": f"{avg_train_loss:.4f}",
                        "val_loss": f"{val_loss:.4f}",
                    })
            else:
                if verbose:
                    iterator.set_postfix({"train_loss": f"{avg_train_loss:.4f}"})
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self._is_fitted = True
        
        return history
    
    def predict(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Generate predictions (chunked to avoid OOM on large inputs)."""
        if self.model is None or not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        self.model.to(self.device)
        self.model.eval()

        chunks = []
        with torch.no_grad():
            for s in range(0, len(X), batch_size):
                x_chunk = torch.FloatTensor(X[s: s + batch_size]).to(self.device)
                chunks.append(self.model(x_chunk).cpu())

        return torch.cat(chunks).numpy()
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "is_fitted": self._is_fitted,
        }, path)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._is_fitted = checkpoint["is_fitted"]


class PyTorchFNN(PyTorchModelBase):
    """Feed-forward Neural Network wrapper."""
    
    def build(
        self,
        input_size: int = 12,
        hidden_sizes: List[int] = [64, 32],
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        self.model = FNNModule(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )


class PyTorchCNN(PyTorchModelBase):
    """Convolutional Neural Network wrapper."""
    
    def build(
        self,
        input_size: int = 12,
        n_filters: int = 32,
        kernel_size: int = 3,
        hidden_size: int = 64,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        self.model = CNNModule(
            input_size=input_size,
            n_filters=n_filters,
            kernel_size=kernel_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )


class PyTorchLSTM(PyTorchModelBase):
    """Long Short-Term Memory Network wrapper."""
    
    def build(
        self,
        input_size: int = 12,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        self.model = LSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )


class PyTorchCNNLSTM(PyTorchModelBase):
    """
    CNN-LSTM combined architecture wrapper.
    
    This is the best-performing architecture from Murray, Xia, Xiao (2024).
    """
    
    def build(
        self,
        input_size: int = 12,
        cnn_filters: int = 32,
        cnn_kernel_size: int = 3,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        self.model = CNNLSTMModule(
            input_size=input_size,
            cnn_filters=cnn_filters,
            cnn_kernel_size=cnn_kernel_size,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            dropout=dropout,
        )
