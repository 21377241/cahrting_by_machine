"""Main orchestration engine for the charting-by-machines package."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import polars as pl
from loguru import logger

from cbm.core.config import CBMConfig
from cbm.core.types import (
    Architecture,
    BacktestResult,
    FeatureSet,
    Forecast,
    PerformanceMetrics,
    Portfolio,
    PortfolioSet,
    StockData,
)


class PortfolioEngine:
    """
    Main engine for ML-based portfolio selection.
    
    This class orchestrates the entire pipeline from data loading through
    portfolio construction and analysis, following the methodology from
    Murray, Xia, and Xiao (2024).
    
    Example
    -------
    >>> engine = PortfolioEngine()
    >>> engine.load_data(universe="sp500", start_date="2010-01-01")
    >>> model_id = engine.train_model(architecture="cnn_lstm")
    >>> forecasts = engine.forecast(model_id=model_id)
    >>> portfolios = engine.construct_portfolios(forecasts)
    >>> performance = engine.analyze_performance(portfolios)
    """
    
    def __init__(self, config: Optional[Union[CBMConfig, str]] = None):
        """
        Initialize the Portfolio Engine.
        
        Parameters
        ----------
        config : CBMConfig or str, optional
            Configuration object or path to YAML config file.
            If None, uses default configuration.
        """
        if config is None:
            self.config = CBMConfig()
        elif isinstance(config, str):
            self.config = CBMConfig.from_yaml(config)
        else:
            self.config = config
        
        self._data: Optional[StockData] = None
        self._features: Optional[FeatureSet] = None
        self._models: Dict[str, Any] = {}
        self._forecasts: Dict[str, Forecast] = {}
        self._portfolios: Dict[str, PortfolioSet] = {}
        
        self._setup_logging()
        logger.info("PortfolioEngine initialized")
    
    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        from cbm.utils.logging import setup_logging
        setup_logging(level=self.config.tracking.log_level)
    
    # ==================== Data Loading ====================
    
    def load_data(
        self,
        tickers: Optional[List[str]] = None,
        universe: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source: Optional[str] = None,
    ) -> StockData:
        """
        Load stock market data from specified source.
        
        Parameters
        ----------
        tickers : list of str, optional
            List of ticker symbols to load.
        universe : str, optional
            Predefined universe (e.g., "sp500", "russell1000").
        start_date : str, optional
            Start date in YYYY-MM-DD format.
        end_date : str, optional
            End date in YYYY-MM-DD format.
        source : str, optional
            Data source ("yahoo", "wrds", "local").
            
        Returns
        -------
        StockData
            Container with prices, returns, and metadata.
        """
        from cbm.data import DataManager
        
        # Override config with provided parameters
        tickers = tickers or self.config.data.tickers
        universe = universe or self.config.data.universe
        start_date = start_date or self.config.data.start_date
        end_date = end_date or self.config.data.end_date
        source = source or self.config.data.source.value
        
        logger.info(f"Loading data from {source}")
        
        manager = DataManager(
            source=source,
            cache_dir=self.config.data.cache_dir,
        )
        
        self._data = manager.load_data(
            tickers=tickers,
            universe=universe,
            start_date=start_date,
            end_date=end_date,
        )
        
        logger.info(
            f"Loaded {len(self._data.tickers)} tickers from "
            f"{self._data.date_range[0]} to {self._data.date_range[1]}"
        )
        
        return self._data
    
    def prepare_features(
        self,
        data: Optional[StockData] = None,
    ) -> FeatureSet:
        """
        Prepare ML features from stock data.
        
        Creates the 12 cumulative return features used as inputs
        to the neural network, following the paper methodology.
        
        Parameters
        ----------
        data : StockData, optional
            Stock data to use. If None, uses previously loaded data.
            
        Returns
        -------
        FeatureSet
            Container with features, targets, and metadata.
        """
        from cbm.data import FeatureEngineer
        
        data = data or self._data
        if data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Preparing ML features")
        
        engineer = FeatureEngineer(
            return_variable=self.config.training.return_variable,
        )
        
        self._features = engineer.create_features(data)
        
        logger.info(f"Created {len(self._features)} feature samples")
        
        return self._features
    
    # ==================== Model Training ====================
    
    def train_model(
        self,
        architecture: Optional[str] = None,
        loss_function: Optional[str] = None,
        weighting: Optional[str] = None,
        optimization_period: Optional[Tuple[str, str]] = None,
        features: Optional[FeatureSet] = None,
    ) -> str:
        """
        Train an ML model for return forecasting.
        
        Parameters
        ----------
        architecture : str, optional
            Neural network architecture ("fnn", "cnn", "lstm", "cnn_lstm").
        loss_function : str, optional
            Loss function ("mse", "mae").
        weighting : str, optional
            Observation weighting scheme ("ew", "ewpm", "ewpmvw").
        optimization_period : tuple of str, optional
            Training period as (start_month, end_month).
        features : FeatureSet, optional
            Pre-computed features. If None, uses stored features.
            
        Returns
        -------
        str
            Unique model identifier for later reference.
        """
        from cbm.ml import ModelTrainer
        
        # Prepare features if not already done
        if features is None:
            if self._features is None:
                self.prepare_features()
            features = self._features
        
        # Get configuration
        arch = Architecture(architecture) if architecture else self.config.model.architecture
        
        logger.info(f"Training {arch.value} model")
        
        trainer = ModelTrainer(
            architecture=arch,
            model_config=self.config.model,
            training_config=self.config.training,
        )
        
        # Filter features to optimization period
        opt_period = optimization_period or self.config.training.optimization_period
        
        # Train ensemble of models
        model, metrics = trainer.train(
            features=features,
            optimization_period=opt_period,
            n_ensemble=self.config.training.n_ensemble,
        )
        
        # Generate model ID and store
        model_id = f"{arch.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._models[model_id] = {
            "model": model,
            "metrics": metrics,
            "config": {
                "architecture": arch.value,
                "optimization_period": opt_period,
            },
        }
        
        logger.info(f"Model trained with ID: {model_id}")
        logger.info(f"Training metrics: {metrics}")
        
        return model_id
    
    # ==================== Forecasting ====================
    
    def forecast(
        self,
        model_id: str,
        test_period: Optional[Tuple[str, str]] = None,
        features: Optional[FeatureSet] = None,
    ) -> Forecast:
        """
        Generate ML-based return forecasts.
        
        Parameters
        ----------
        model_id : str
            Identifier of trained model to use.
        test_period : tuple of str, optional
            Test period as (start_month, end_month).
        features : FeatureSet, optional
            Features to forecast. If None, uses stored features.
            
        Returns
        -------
        Forecast
            Container with forecast values and metadata.
        """
        from cbm.ml import Forecaster
        
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found. Train a model first.")
        
        features = features or self._features
        if features is None:
            raise ValueError("No features available. Call prepare_features() first.")
        
        test_period = test_period or self.config.backtest.test_period
        
        logger.info(f"Generating forecasts for period {test_period}")
        
        model_data = self._models[model_id]
        
        forecaster = Forecaster(model=model_data["model"])
        
        forecast = forecaster.predict(
            features=features,
            test_period=test_period,
        )
        forecast.model_id = model_id
        
        self._forecasts[model_id] = forecast
        
        logger.info(f"Generated forecasts for {forecast.values.height} periods")
        
        return forecast
    
    # ==================== Portfolio Construction ====================
    
    def construct_portfolios(
        self,
        forecasts: Optional[Forecast] = None,
        n_portfolios: Optional[int] = None,
        weighting: Optional[str] = None,
        method: Optional[str] = None,
    ) -> PortfolioSet:
        """
        Construct portfolios sorted by ML forecasts.
        
        Parameters
        ----------
        forecasts : Forecast, optional
            Return forecasts to use for sorting.
        n_portfolios : int, optional
            Number of quantile portfolios (default: 10 for deciles).
        weighting : str, optional
            Portfolio weighting ("value" or "equal").
        method : str, optional
            Sorting method ("univariate", "bivariate", "trivariate").
            
        Returns
        -------
        PortfolioSet
            Set of sorted portfolios including long-short portfolio.
        """
        from cbm.portfolio import PortfolioConstructor
        
        # Use most recent forecast if not specified
        if forecasts is None:
            if not self._forecasts:
                raise ValueError("No forecasts available. Call forecast() first.")
            forecasts = list(self._forecasts.values())[-1]
        
        n_portfolios = n_portfolios or self.config.portfolio.n_portfolios
        weighting = weighting or self.config.portfolio.weighting
        
        logger.info(f"Constructing {n_portfolios} portfolios")
        
        constructor = PortfolioConstructor(
            n_portfolios=n_portfolios,
            weighting=weighting,
            use_nyse_breakpoints=self.config.portfolio.use_nyse_breakpoints,
        )
        
        portfolio_set = constructor.construct(
            forecasts=forecasts,
            returns=self._data.returns,
            market_caps=self._data.market_cap,
        )
        
        self._portfolios[forecasts.model_id] = portfolio_set
        
        logger.info(f"Constructed portfolios: {portfolio_set.keys()}")
        
        return portfolio_set
    
    # ==================== Performance Analysis ====================
    
    def analyze_performance(
        self,
        portfolios: Optional[PortfolioSet] = None,
        factor_models: Optional[List[str]] = None,
    ) -> Dict[str, PerformanceMetrics]:
        """
        Analyze portfolio performance with risk adjustment.
        
        Parameters
        ----------
        portfolios : PortfolioSet, optional
            Portfolios to analyze. If None, uses most recent.
        factor_models : list of str, optional
            Factor models for alpha calculation.
            
        Returns
        -------
        dict
            Performance metrics for each portfolio.
        """
        from cbm.portfolio import PerformanceAnalyzer
        
        if portfolios is None:
            if not self._portfolios:
                raise ValueError("No portfolios available. Call construct_portfolios() first.")
            portfolios = list(self._portfolios.values())[-1]
        
        factor_models = factor_models or self.config.backtest.factor_models
        
        logger.info("Analyzing portfolio performance")
        
        analyzer = PerformanceAnalyzer(
            factor_models=factor_models,
            newey_west_lags=self.config.backtest.newey_west_lags,
        )
        
        performance = analyzer.analyze(portfolios)
        
        return performance
    
    # ==================== Full Pipeline ====================
    
    def run_backtest(
        self,
        tickers: Optional[List[str]] = None,
        universe: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run full backtest pipeline.
        
        This method runs the complete pipeline:
        1. Load data
        2. Prepare features
        3. Train model
        4. Generate forecasts
        5. Construct portfolios
        6. Analyze performance
        
        Parameters
        ----------
        tickers : list of str, optional
            Tickers to include.
        universe : str, optional
            Predefined universe to use.
            
        Returns
        -------
        BacktestResult
            Complete backtest results.
        """
        logger.info("Starting full backtest pipeline")
        
        # Step 1: Load data
        self.load_data(tickers=tickers, universe=universe)
        
        # Step 2: Prepare features
        self.prepare_features()
        
        # Step 3: Train model
        model_id = self.train_model()
        
        # Step 4: Generate forecasts
        forecasts = self.forecast(model_id=model_id)
        
        # Step 5: Construct portfolios
        portfolios = self.construct_portfolios(forecasts=forecasts)
        
        # Step 6: Analyze performance
        performance = self.analyze_performance(portfolios=portfolios)
        
        # Calculate cumulative returns as Polars DataFrame
        cumulative_data = {"date": []}
        dates = None
        for name, port in portfolios.portfolios.items():
            cum_returns = (1 + port.returns).cumprod()
            cumulative_data[name] = cum_returns
            if dates is None:
                dates = list(range(len(cum_returns)))
        
        cumulative_data["date"] = dates
        cumulative = pl.DataFrame(cumulative_data)
        
        result = BacktestResult(
            portfolio_set=portfolios,
            performance=performance,
            cumulative_returns=cumulative,
        )
        
        from cbm.utils.results_io import save_pipeline_results
        
        save_pipeline_results(
            engine=self,
            model_id=model_id,
            performance=performance,
            forecasts=forecasts,
            portfolios=portfolios,
            cumulative_returns=cumulative,
            result_dir=self.config.tracking.results_dir,
        )
        
        logger.info("Backtest completed")
        
        return result
    
    # ==================== Model Management ====================
    
    def save_model(self, model_id: str, path: str) -> None:
        """Save a trained model to disk."""
        from cbm.ml import ModelRegistry
        
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found.")
        
        registry = ModelRegistry(path=self.config.tracking.model_dir)
        registry.save(model_id, self._models[model_id])
        logger.info(f"Model {model_id} saved to {path}")
    
    def load_model(self, path: str) -> str:
        """Load a trained model from disk."""
        from cbm.ml import ModelRegistry
        
        registry = ModelRegistry(path=self.config.tracking.model_dir)
        model_id, model_data = registry.load(path)
        self._models[model_id] = model_data
        logger.info(f"Model {model_id} loaded from {path}")
        return model_id
    
    def list_models(self) -> List[str]:
        """List all loaded model IDs."""
        return list(self._models.keys())
    
    # ==================== Properties ====================
    
    @property
    def data(self) -> Optional[StockData]:
        """Currently loaded stock data."""
        return self._data
    
    @property
    def features(self) -> Optional[FeatureSet]:
        """Current feature set."""
        return self._features
