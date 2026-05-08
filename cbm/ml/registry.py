"""Model registry for versioning and persistence."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json

from loguru import logger


class ModelRegistry:
    """
    Registry for model versioning and persistence.
    
    Handles saving and loading models with metadata for reproducibility.
    Can optionally integrate with MLflow for enterprise-grade tracking.
    
    Parameters
    ----------
    path : str
        Base directory for model storage.
    use_mlflow : bool, optional
        Whether to use MLflow for tracking. Default: False.
    mlflow_uri : str, optional
        MLflow tracking URI.
        
    Example
    -------
    >>> registry = ModelRegistry(path="./models")
    >>> registry.save("model_v1", model_data)
    >>> model_id, model_data = registry.load("./models/model_v1")
    """
    
    def __init__(
        self,
        path: str = "./models",
        use_mlflow: bool = False,
        mlflow_uri: Optional[str] = None,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        
        self.use_mlflow = use_mlflow
        self.mlflow_uri = mlflow_uri
        
        if use_mlflow:
            self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        try:
            import mlflow
            
            if self.mlflow_uri:
                mlflow.set_tracking_uri(self.mlflow_uri)
            
            logger.info(f"MLflow tracking initialized: {mlflow.get_tracking_uri()}")
        except ImportError:
            logger.warning("MLflow not installed. Falling back to local storage.")
            self.use_mlflow = False
    
    def save(
        self,
        model_id: str,
        model_data: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save model to registry.
        
        Parameters
        ----------
        model_id : str
            Unique model identifier.
        model_data : dict
            Model data including model object and config.
        metrics : dict, optional
            Training metrics to log.
        params : dict, optional
            Hyperparameters to log.
            
        Returns
        -------
        str
            Path to saved model.
        """
        model_path = self.path / model_id
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model = model_data.get("model")
        if hasattr(model, "models"):  # EnsembleModel
            for i, m in enumerate(model.models):
                m.save(str(model_path / f"model_{i}.pt"))
            
            # Save preprocessor
            if hasattr(model, "preprocessor"):
                model.preprocessor.save(str(model_path / "preprocessor.joblib"))
        
        # Save metadata
        metadata = {
            "model_id": model_id,
            "config": model_data.get("config", {}),
            "metrics": model_data.get("metrics", metrics or {}),
        }
        
        with open(model_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {model_path}")
        
        # Log to MLflow if enabled
        if self.use_mlflow:
            self._log_to_mlflow(model_id, metrics, params)
        
        return str(model_path)
    
    def load(self, path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Load model from registry.
        
        Parameters
        ----------
        path : str
            Path to model directory.
            
        Returns
        -------
        tuple
            (model_id, model_data)
        """
        model_path = Path(path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        # Load metadata
        with open(model_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load preprocessor
        from cbm.ml.preprocessing import DataPreprocessor
        preprocessor_path = model_path / "preprocessor.joblib"
        preprocessor = None
        if preprocessor_path.exists():
            preprocessor = DataPreprocessor.load(str(preprocessor_path))
        
        # Load model weights
        from cbm.ml.models.pytorch_impl import PyTorchCNNLSTM
        from cbm.ml.trainer import EnsembleModel
        
        models = []
        i = 0
        while (model_path / f"model_{i}.pt").exists():
            model = PyTorchCNNLSTM()
            model.build()
            model.load(str(model_path / f"model_{i}.pt"))
            models.append(model)
            i += 1
        
        ensemble = None
        if models and preprocessor:
            ensemble = EnsembleModel(models=models, preprocessor=preprocessor)
        
        model_data = {
            "model": ensemble,
            "config": metadata.get("config", {}),
            "metrics": metadata.get("metrics", {}),
        }
        
        logger.info(f"Model loaded from {model_path}")
        
        return metadata["model_id"], model_data
    
    def _log_to_mlflow(
        self,
        model_id: str,
        metrics: Optional[Dict[str, float]],
        params: Optional[Dict[str, Any]],
    ) -> None:
        """Log model to MLflow."""
        import mlflow
        
        with mlflow.start_run(run_name=model_id):
            if params:
                mlflow.log_params(params)
            
            if metrics:
                mlflow.log_metrics(metrics)
            
            mlflow.log_artifact(str(self.path / model_id))
    
    def list_models(self) -> list:
        """List all saved models."""
        models = []
        
        for model_dir in self.path.iterdir():
            if model_dir.is_dir() and (model_dir / "metadata.json").exists():
                with open(model_dir / "metadata.json", "r") as f:
                    metadata = json.load(f)
                models.append({
                    "model_id": metadata["model_id"],
                    "path": str(model_dir),
                    "metrics": metadata.get("metrics", {}),
                })
        
        return models
    
    def delete_model(self, model_id: str) -> None:
        """Delete a model from the registry."""
        import shutil
        
        model_path = self.path / model_id
        if model_path.exists():
            shutil.rmtree(model_path)
            logger.info(f"Model {model_id} deleted")
        else:
            logger.warning(f"Model {model_id} not found")
