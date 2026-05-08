"""Abstract base class for neural network models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseNeuralNetworkModel(ABC):
    """
    Abstract base class for neural network models.
    
    This class defines the interface that all neural network implementations
    must follow, enabling framework-agnostic model handling.
    
    Subclasses implement specific architectures:
    - FNN: Feed-forward Neural Network
    - CNN: Convolutional Neural Network
    - LSTM: Long Short-Term Memory Network
    - CNN-LSTM: Combined CNN and LSTM
    """
    
    @abstractmethod
    def build(self, input_size: int, **kwargs) -> None:
        """
        Build the model architecture.
        
        Parameters
        ----------
        input_size : int
            Number of input features (typically 12 for cumulative returns).
        **kwargs
            Architecture-specific parameters.
        """
        pass
    
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the model on provided data.
        
        Parameters
        ----------
        X : np.ndarray
            Training features with shape (n_samples, n_features).
        y : np.ndarray
            Training targets with shape (n_samples,).
        weights : np.ndarray, optional
            Sample weights with shape (n_samples,).
        validation_data : tuple, optional
            Validation data as (X_val, y_val).
        **kwargs
            Additional training parameters.
            
        Returns
        -------
        dict
            Training history and metrics.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input features.
        
        Parameters
        ----------
        X : np.ndarray
            Input features with shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Predictions with shape (n_samples,).
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Parameters
        ----------
        path : str
            Path to save the model.
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Parameters
        ----------
        path : str
            Path to load the model from.
        """
        pass
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        pass
