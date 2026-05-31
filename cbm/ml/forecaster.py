"""Forecasting module for generating ML-based return predictions."""

import calendar
from datetime import date, datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from scipy.stats import spearmanr

from cbm.core.types import FeatureSet, Forecast
from cbm.ml.trainer import EnsembleModel


def merge_forecast_wides(wides: list[pl.DataFrame]) -> pl.DataFrame:
    """
    合并多段扩展窗口的预测宽表。

    各窗口 PERMNO 列集合不同，不能直接 ``pl.concat``；先转 long 再 pivot。
    预测月份不重叠时，``(date, permno)`` 唯一。
    """
    if not wides:
        raise ValueError("No forecast tables to merge")
    if len(wides) == 1:
        return wides[0].sort("date")

    long_parts: list[pl.DataFrame] = []
    for wide in wides:
        tickers = [c for c in wide.columns if c != "date"]
        if not tickers:
            continue
        long_parts.append(
            wide.unpivot(on=tickers, index="date", variable_name="permno", value_name="score")
            .drop_nulls("score")
            .filter(pl.col("score").is_not_nan())
        )

    combined_long = pl.concat(long_parts).unique(subset=["date", "permno"], keep="first")
    logger.info(
        f"Merging {len(wides)} forecast windows: "
        f"{combined_long['date'].n_unique()} months, {combined_long['permno'].n_unique()} PERMNO"
    )
    return (
        combined_long.pivot(on="permno", index="date", values="score", aggregate_function="first")
        .sort("date")
    )


def _coerce_date_column(df: pl.DataFrame) -> pl.DataFrame:
    """将 date 列统一转为 pl.Date 类型，兼容 Datetime / String 输入。"""
    if "date" not in df.columns:
        return df
    dtype = df.get_column("date").dtype
    if dtype == pl.Date:
        return df
    if dtype in (pl.Datetime, pl.Datetime("us"), pl.Datetime("ns"), pl.Datetime("ms")):
        return df.with_columns(pl.col("date").cast(pl.Date))
    return df.with_columns(pl.col("date").str.to_date(strict=False))


def _period_str_to_end_date(period: str) -> np.datetime64:
    """将 'YYYY-MM' 转为该月最后一天的 numpy datetime64[D]（含整个末月）。"""
    y, m = map(int, period.split("-"))
    last_day = calendar.monthrange(y, m)[1]
    return np.datetime64(f"{y:04d}-{m:02d}-{last_day:02d}", "D")


class Forecaster:
    """
    Generate ML-based return forecasts (MLER) from trained models.
    
    Produces forecasts following the paper methodology where predictions
    are made using features available at the end of month t-1 to predict
    month t returns.
    
    Parameters
    ----------
    model : EnsembleModel
        Trained ensemble model.
        
    Example
    -------
    >>> forecaster = Forecaster(model=trained_model)
    >>> forecasts = forecaster.predict(features, test_period=("2019-01", "2023-12"))
    """
    
    def __init__(self, model: EnsembleModel):
        self.model = model
    
    def predict(
        self,
        features: FeatureSet,
        test_period: Optional[Tuple[str, str]] = None,
    ) -> Forecast:
        """
        Generate forecasts for the test period.
        
        Parameters
        ----------
        features : FeatureSet
            Feature set with features, dates, and tickers.
        test_period : tuple, optional
            Test period as (start_month, end_month).
            
        Returns
        -------
        Forecast
            Container with forecast values indexed by date and ticker.
        """
        logger.info(f"Generating forecasts for period {test_period}")
        
        # 将 "YYYY-MM" 转为该月末日（含整个末月），避免月精度 datetime64 比较截断
        if test_period is not None:
            start_date = np.datetime64(test_period[0] + "-01", "D")
            end_date = _period_str_to_end_date(test_period[1])
            mask = (features.dates >= start_date) & (features.dates <= end_date)
        else:
            mask = np.ones(len(features), dtype=bool)
        
        X = features.features[mask]
        dates = features.dates[mask]
        tickers = features.tickers[mask]
        
        if len(X) == 0:
            raise ValueError(f"No data found for test period {test_period}")
        
        # Generate predictions
        predictions = self.model.predict(X)
        
        # Organize into Polars DataFrame (date x ticker)
        unique_dates = np.unique(dates)
        unique_tickers = np.unique(tickers)

        def _to_cal_date(d) -> date:
            return pd.Timestamp(d).date()

        unique_dates_list = [_to_cal_date(d) for d in unique_dates]
        date_to_idx = {d: i for i, d in enumerate(unique_dates_list)}
        
        # Build data dict with date column and ticker columns
        data_dict = {"date": unique_dates_list}
        for ticker in unique_tickers:
            data_dict[ticker] = [np.nan] * len(unique_dates_list)
        
        # Fill in predictions
        for dt, ticker, pred in zip(dates, tickers, predictions):
            idx = date_to_idx[_to_cal_date(dt)]
            data_dict[ticker][idx] = pred
        
        forecast_df = pl.DataFrame(data_dict)
        
        logger.info(
            f"Generated forecasts for {len(unique_dates)} months "
            f"and {len(unique_tickers)} tickers"
        )
        
        return Forecast(
            values=forecast_df,
            model_id="",  # Will be set by engine
            created_at=datetime.now().isoformat(),
            metadata={
                "test_period": test_period,
                "n_predictions": len(predictions),
            },
        )
    
    def predict_with_uncertainty(
        self,
        features: FeatureSet,
        test_period: Optional[Tuple[str, str]] = None,
    ) -> Tuple[Forecast, pl.DataFrame]:
        """
        Generate forecasts with uncertainty estimates.
        
        Returns standard deviation across ensemble models as
        a measure of prediction uncertainty.
        
        Returns
        -------
        tuple
            (Forecast, uncertainty_df) where uncertainty_df has same
            structure as forecast values.
        """
        logger.info("Generating forecasts with uncertainty estimates")

        if test_period is not None:
            start_date = np.datetime64(test_period[0] + "-01", "D")
            end_date = _period_str_to_end_date(test_period[1])
            mask = (features.dates >= start_date) & (features.dates <= end_date)
        else:
            mask = np.ones(len(features), dtype=bool)
        
        X = features.features[mask]
        dates = features.dates[mask]
        tickers = features.tickers[mask]
        
        # Generate predictions with uncertainty
        predictions, uncertainties = self.model.predict_with_uncertainty(X)
        
        # Build DataFrames
        unique_dates = np.unique(dates)
        unique_tickers = np.unique(tickers)

        def _to_cal_date_u(d) -> date:
            return pd.Timestamp(d).date()

        unique_dates_list = [_to_cal_date_u(d) for d in unique_dates]
        
        # Create data dicts
        forecast_data = {"date": unique_dates_list}
        uncertainty_data = {"date": unique_dates_list}
        for ticker in unique_tickers:
            forecast_data[ticker] = [np.nan] * len(unique_dates_list)
            uncertainty_data[ticker] = [np.nan] * len(unique_dates_list)
        
        date_to_idx = {d: i for i, d in enumerate(unique_dates_list)}
        
        for dt, ticker, pred, unc in zip(dates, tickers, predictions, uncertainties):
            idx = date_to_idx[_to_cal_date_u(dt)]
            forecast_data[ticker][idx] = pred
            uncertainty_data[ticker][idx] = unc
        
        forecast_df = pl.DataFrame(forecast_data)
        uncertainty_df = pl.DataFrame(uncertainty_data)
        
        forecast = Forecast(
            values=forecast_df,
            model_id="",
            created_at=datetime.now().isoformat(),
            metadata={
                "test_period": test_period,
                "has_uncertainty": True,
            },
        )
        
        return forecast, uncertainty_df
    
    def compute_forecast_correlation(
        self,
        forecast: Forecast,
        actual_returns: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Compute Spearman rank correlation between forecasts and actual returns.
        
        This is the key metric used in the paper to evaluate forecast quality.
        
        Parameters
        ----------
        forecast : Forecast
            ML forecasts.
        actual_returns : pl.DataFrame
            Actual realized returns with date column.
            
        Returns
        -------
        pl.DataFrame
            Monthly Spearman correlations with date and correlation columns.
        """
        correlations = []
        
        fv = _coerce_date_column(forecast.values)
        ar = _coerce_date_column(actual_returns)
        forecast_dates = fv.get_column("date").to_list()
        actual_date_set = set(ar.get_column("date").to_list())
        
        tickers = [c for c in fv.columns if c != "date"]
        
        for date in forecast_dates:
            if date not in actual_date_set:
                continue
            
            # Get forecasts and actuals for this date
            fcst_row = fv.filter(pl.col("date") == date)
            actual_row = ar.filter(pl.col("date") == date)
            
            if fcst_row.height == 0 or actual_row.height == 0:
                continue
            
            # Get common tickers with valid data
            fcst_vals = []
            actual_vals = []
            
            for ticker in tickers:
                if ticker not in actual_row.columns:
                    continue
                fcst_val = fcst_row.get_column(ticker)[0]
                actual_val = actual_row.get_column(ticker)[0]
                
                if fcst_val is not None and actual_val is not None:
                    if not np.isnan(fcst_val) and not np.isnan(actual_val):
                        fcst_vals.append(fcst_val)
                        actual_vals.append(actual_val)
            
            if len(fcst_vals) > 10:  # Require minimum stocks
                corr, _ = spearmanr(fcst_vals, actual_vals)
                correlations.append({"date": date, "correlation": corr})
        
        return pl.DataFrame(correlations)
