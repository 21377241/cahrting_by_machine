"""Data preprocessing for ML models."""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from loguru import logger


class DataPreprocessor:
    """
    Preprocessor for ML input features.
    
    Handles feature normalization and transformation following the paper methodology.
    Uses cross-sectional standardization (z-score) by default.
    
    Parameters
    ----------
    method : str, optional
        Normalization method ("standard", "minmax", "none"). Default: "standard".
        
    Example
    -------
    >>> preprocessor = DataPreprocessor()
    >>> X_train_scaled = preprocessor.fit_transform(X_train)
    >>> X_test_scaled = preprocessor.transform(X_test)
    """
    
    def __init__(self, method: str = "standard"):
        self.method = method
        self.scaler: Optional[StandardScaler] = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> "DataPreprocessor":
        """
        Fit the preprocessor on training data.
        
        Parameters
        ----------
        X : np.ndarray
            Training features with shape (n_samples, n_features).
            
        Returns
        -------
        DataPreprocessor
            The fitted preprocessor.
        """
        if self.method == "standard":
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        elif self.method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            self.scaler.fit(X)
        elif self.method == "none":
            pass
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self._is_fitted = True
        logger.debug(f"Preprocessor fitted with method: {self.method}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted parameters.
        
        Parameters
        ----------
        X : np.ndarray
            Features to transform with shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Transformed features.
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        if self.method == "none":
            return X
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : np.ndarray
            Features with shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Transformed features.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform to original scale.
        
        Parameters
        ----------
        X : np.ndarray
            Transformed features.
            
        Returns
        -------
        np.ndarray
            Features in original scale.
        """
        if self.method == "none":
            return X
        
        return self.scaler.inverse_transform(X)
    
    def save(self, path: str) -> None:
        """Save preprocessor state."""
        import joblib
        joblib.dump({
            "method": self.method,
            "scaler": self.scaler,
            "is_fitted": self._is_fitted,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """Load preprocessor from disk."""
        import joblib
        state = joblib.load(path)
        
        preprocessor = cls(method=state["method"])
        preprocessor.scaler = state["scaler"]
        preprocessor._is_fitted = state["is_fitted"]
        
        return preprocessor


def compute_sample_weights(
    dates: np.ndarray,
    market_caps: Optional[np.ndarray] = None,
    scheme: str = "ewpm",
) -> np.ndarray:
    """
    Compute sample weights following paper methodology.
    
    Parameters
    ----------
    dates : np.ndarray
        Date indices for each sample.
    market_caps : np.ndarray, optional
        Market capitalization for each sample.
    scheme : str
        Weighting scheme:
        - "ew": Equal-weighted
        - "ewpm": Equal-weighted per month
        - "ewpmvw": Equal-weighted per month, value-weighted within month
        
    Returns
    -------
    np.ndarray
        Sample weights normalized to sum to 1.
    """
    n_samples = len(dates)
    
    if scheme == "ew":
        # Equal weights
        weights = np.ones(n_samples)
    
    elif scheme == "ewpm":
        # Equal-weighted per month
        # Each month contributes equally regardless of number of stocks
        unique_dates = np.unique(dates)
        weights = np.zeros(n_samples)
        
        for date in unique_dates:
            mask = dates == date
            n_stocks = mask.sum()
            weights[mask] = 1.0 / n_stocks
        
        # Normalize so each month contributes equally
        weights = weights / len(unique_dates)
    
    elif scheme == "ewpmvw":
        # Equal-weighted per month, value-weighted within month
        if market_caps is None:
            logger.warning("No market caps provided, falling back to EWPM")
            return compute_sample_weights(dates, market_caps, "ewpm")
        
        unique_dates = np.unique(dates)
        weights = np.zeros(n_samples)
        
        for date in unique_dates:
            mask = dates == date
            month_caps = market_caps[mask]
            
            # Handle missing market caps
            valid_caps = ~np.isnan(month_caps)
            if valid_caps.any():
                # Value-weighted within month
                month_weights = np.zeros(mask.sum())
                month_weights[valid_caps] = month_caps[valid_caps] / month_caps[valid_caps].sum()
                weights[mask] = month_weights
            else:
                # Fall back to equal weights if no market caps
                weights[mask] = 1.0 / mask.sum()
        
        # Normalize so each month contributes equally
        weights = weights / len(unique_dates)
    
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    
    # Normalize to sum to 1
    weights = weights / weights.sum()
    
    return weights
