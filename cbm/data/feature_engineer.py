"""Feature engineering for ML model inputs."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from loguru import logger
from scipy import stats

from cbm.core.types import FeatureSet, ReturnVariable, StockData


class FeatureEngineer:
    """
    Creates ML features from stock data following Murray, Xia, Xiao (2024).
    
    The main features are 12 cumulative monthly returns:
    - r_{t-1,1}: cumulative return over month t-12
    - r_{t-1,2}: cumulative return over months t-12 to t-11
    - ...
    - r_{t-1,12}: cumulative return over months t-12 to t-1
    
    Parameters
    ----------
    return_variable : ReturnVariable or str
        Target variable transformation. Options:

        ``"ret"``
            Raw excess return, no transformation.
        ``"ret_std"``
            Cross-sectional standardisation by std only.
        ``"ret_norm"`` *(default)*
            Cross-sectional z-score: ``y = (r - mean) / std``.
            Unbounded; sensitive to extreme observations.
        ``"ret_rank_norm"``
            Cross-sectional inverse-normal rank transform (paper method):
            ``y = Phi^{-1}[ rank_i / (N + 1) ]``
            Rank-based → robust to outliers; output in ~(-4, 4).
        ``"ret_pctl"``
            Cross-sectional percentile rank in (0, 1].

    lookback_months : int, optional
        Number of months for cumulative returns. Default: 12.
        
    Example
    -------
    >>> # Default: z-score target
    >>> engineer = FeatureEngineer(return_variable="ret_norm")
    >>> # Paper method: inverse-normal rank
    >>> engineer = FeatureEngineer(return_variable="ret_rank_norm")
    >>> features = engineer.create_features(stock_data)
    """
    
    def __init__(
        self,
        return_variable: ReturnVariable = ReturnVariable.RET_NORM,
        lookback_months: int = 12,
    ):
        if isinstance(return_variable, str):
            return_variable = ReturnVariable(return_variable)
        self.return_variable = return_variable
        self.lookback_months = lookback_months
    
    def create_features(
        self,
        data: StockData,
        risk_free_rate: Optional[pl.DataFrame] = None,
    ) -> FeatureSet:
        """
        Create ML features from stock data.

        Fully vectorised implementation: no Python loops over dates or tickers.
        Handles 20 000+ securities × 400 months in seconds instead of hours.

        Parameters
        ----------
        data : StockData
            Stock data with prices and returns.
        risk_free_rate : pl.DataFrame, optional
            Risk-free rate for calculating excess returns.

        Returns
        -------
        FeatureSet
            Container with features, targets, and metadata.
        """
        logger.info("Creating cumulative return features (vectorised)")

        returns_df = data.returns
        tickers    = data.tickers
        dates      = returns_df.get_column("date").to_numpy()

        # (T, N) float64 matrix – the only large allocation
        returns_arr = returns_df.select(tickers).to_numpy()

        # ── Risk-free rate subtraction (row-wise broadcast) ────────────────
        if risk_free_rate is not None:
            rf_dates = risk_free_rate.get_column("date").to_numpy()
            rf_rates = risk_free_rate.get_column("rate").to_numpy()
            rf_map   = dict(zip(rf_dates, rf_rates))
            rf_vec   = np.array([rf_map.get(d, 0.0) for d in dates])  # (T,)
            returns_arr = returns_arr - rf_vec[:, np.newaxis]

        T, N = returns_arr.shape
        L    = self.lookback_months

        if T <= L:
            raise ValueError(
                f"Not enough months ({T}) for lookback={L}. "
                "Need at least lookback_months + 1 rows in returns."
            )

        logger.info(f"  Returns matrix: {T} months × {N:,} securities")

        # ── Cumulative returns (fully vectorised) ─────────────────────────
        cum_returns = self._calculate_cumulative_returns(returns_arr)
        # cum_returns[tau] shape: (T, N),  tau ∈ [0, L-1]

        # ── Target variable ───────────────────────────────────────────────
        targets = self._create_target_variable(returns_arr, dates)
        # shape: (T, N)

        # ── Build 3-D feature cube (T-L, N, L) in one shot ───────────────
        # For prediction at calendar time-index t (t ∈ [L, T-1]):
        #   feature[tau] = cum_returns[tau][t-1, j]   (use info available at t-1)
        #   target        = targets[t, j]
        #
        # Stacking cum_returns[tau][L-1 : T-1, :] for tau=0..L-1:
        #   axis-0 → prediction period (T-L rows, i.e. t = L..T-1)
        #   axis-1 → securities
        #   axis-2 → tau
        feature_cube = np.stack(
            [cum_returns[tau][L - 1: T - 1, :] for tau in range(L)],
            axis=2,
        )  # (T-L, N, L)

        target_slice = targets[L:, :]        # (T-L, N)
        dates_slice  = dates[L:]             # (T-L,)

        # ── Valid-sample mask: no NaN in any of L features, target non-NaN ─
        feat_valid   = ~np.any(np.isnan(feature_cube), axis=2)  # (T-L, N)
        target_valid = ~np.isnan(target_slice)                  # (T-L, N)
        valid_mask   = feat_valid & target_valid                  # (T-L, N)

        ti, ji = np.where(valid_mask)   # row (time) and col (security) indices

        if len(ti) == 0:
            raise ValueError(
                "Feature extraction produced zero valid samples. "
                "Check that returns data covers at least lookback_months+1 months "
                "and has non-NaN values."
            )

        features_array = feature_cube[ti, ji, :]           # (n_valid, L)
        targets_array  = target_slice[ti, ji]               # (n_valid,)
        excess_array   = returns_arr[L:, :][ti, ji]         # (n_valid,) month-t excess
        dates_array    = dates_slice[ti]                    # (n_valid,)
        tickers_array  = np.array(tickers, dtype=object)[ji]  # (n_valid,)

        # ── Market cap: single numpy extraction, no per-sample Polars calls ─
        market_caps_array: Optional[np.ndarray] = None
        if data.market_cap is not None:
            mc_cols = [c for c in tickers if c in data.market_cap.columns]
            if mc_cols:
                mc_matrix  = data.market_cap.select(mc_cols).to_numpy()  # (T, N_mc)
                mc_slice   = mc_matrix[L:, :]                             # (T-L, N_mc)
                # ji may reference columns not present in mc_cols; use a safe lookup
                mc_col_set = {c: idx for idx, c in enumerate(mc_cols)}
                mc_ji      = np.array(
                    [mc_col_set.get(tickers[j], -1) for j in ji], dtype=np.intp
                )
                valid_mc   = mc_ji >= 0
                market_caps_array = np.full(len(ti), np.nan, dtype=np.float64)
                if valid_mc.any():
                    market_caps_array[valid_mc] = mc_slice[
                        ti[valid_mc], mc_ji[valid_mc]
                    ]

        logger.info(f"  Created {len(features_array):,} feature samples "
                    f"({valid_mask.mean()*100:.1f}% of possible date×ticker pairs)")

        return FeatureSet(
            features=features_array,
            targets=targets_array,
            dates=dates_array,
            tickers=tickers_array,
            market_caps=market_caps_array,
            excess_returns=excess_array,
        )
    
    def _calculate_cumulative_returns(
        self,
        returns: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """
        Calculate cumulative returns for each lookback period.

        Returns r_{t-1,τ} for τ = 1, ..., lookback_months as described in the
        paper.  Fully vectorised: no inner Python loop over time steps.

        Algorithm
        ---------
        Let  log1p_r = log(1 + r)  (NaN where r is NaN).
        A rolling window sum of log-returns equals the log of the cumulative
        product, so  cum_ret = exp(rolling_sum(log1p_r, tau)) - 1.

        Rolling sums are computed via prefix-sum differences in O(T·N) time
        rather than the previous O(T·N·L) inner-loop approach.

        NaN handling: if *any* return inside a window is NaN, the output for
        that cell is NaN (same behaviour as the original code).

        Parameters
        ----------
        returns : np.ndarray
            Shape (T, N).

        Returns
        -------
        Dict[int, np.ndarray]
            Keys 0…L-1, each value has shape (T, N).
        """
        T, N = returns.shape
        L    = self.lookback_months

        nan_mask = np.isnan(returns)

        # log(1+r), substituting 0 for NaN positions so cumsum stays finite
        safe_r     = np.where(nan_mask, 0.0, returns)
        log1p_safe = np.log1p(safe_r)                 # (T, N)

        # Prefix sums: padded with a leading row of zeros → shape (T+1, N)
        log_prefix = np.empty((T + 1, N), dtype=np.float64)
        log_prefix[0, :] = 0.0
        np.cumsum(log1p_safe, axis=0, out=log_prefix[1:])

        # NaN-count prefix sum (to detect any NaN inside a window)
        nan_prefix = np.empty((T + 1, N), dtype=np.int32)
        nan_prefix[0, :] = 0
        np.cumsum(nan_mask.astype(np.int32), axis=0, out=nan_prefix[1:])

        cum_returns: Dict[int, np.ndarray] = {}

        for tau in range(1, L + 1):
            cum_ret = np.full((T, N), np.nan, dtype=np.float64)

            # Valid rows: t in [tau-1, T-1]
            # Window [t-tau+1 .. t] corresponds to prefix indices [t-tau+1 .. t+1]
            # log_sum  = log_prefix[t+1]     - log_prefix[t+1-tau]
            # nan_count= nan_prefix[t+1]     - nan_prefix[t+1-tau]
            end_idx   = np.arange(tau, T + 1)        # t+1, shape (T-tau+1,)
            start_idx = end_idx - tau                  # t+1-tau

            log_sum   = log_prefix[end_idx, :] - log_prefix[start_idx, :]   # (T-tau+1, N)
            nan_count = nan_prefix[end_idx, :] - nan_prefix[start_idx, :]   # (T-tau+1, N)

            values = np.expm1(log_sum)                 # exp(log_sum) - 1
            values[nan_count > 0] = np.nan             # propagate NaN

            cum_ret[tau - 1:, :] = values
            cum_returns[tau - 1] = cum_ret

        return cum_returns
    
    def _create_target_variable(
        self,
        returns: np.ndarray,
        dates: np.ndarray,
    ) -> np.ndarray:
        """
        Create target variable based on configured transformation.
        
        Parameters
        ----------
        returns : np.ndarray
            Returns array with shape (n_dates, n_tickers)
        dates : np.ndarray
            Array of dates
            
        Returns
        -------
        np.ndarray
            Transformed target variable
        """
        if self.return_variable == ReturnVariable.RET:
            # Raw excess return
            return returns
        
        elif self.return_variable == ReturnVariable.RET_STD:
            # Standardized by cross-sectional std
            std = np.nanstd(returns, axis=1, keepdims=True)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            return returns / std
        
        elif self.return_variable == ReturnVariable.RET_NORM:
            # Cross-sectional z-score: y = (r - mean) / std
            mean = np.nanmean(returns, axis=1, keepdims=True)
            std = np.nanstd(returns, axis=1, keepdims=True)
            std = np.where(std == 0, 1, std)
            return (returns - mean) / std
        
        elif self.return_variable == ReturnVariable.RET_RANK_NORM:
            # Cross-sectional inverse-normal rank transform (paper method)
            # y_i,t = Phi^{-1}[ rank_i,t / (N_t + 1) ]
            # Rank-based: robust to outliers; output ≈ N(0,1) in large samples.
            # Vectorised: iterate over T dates (cheap at monthly frequency),
            # scipy.rankdata handles NaN-aware ranking per row.
            n_dates, n_tickers = returns.shape
            result = np.full((n_dates, n_tickers), np.nan, dtype=np.float64)
            for t in range(n_dates):
                row = returns[t, :]
                valid_mask = ~np.isnan(row)
                n_valid = int(valid_mask.sum())
                if n_valid > 0:
                    ranks = stats.rankdata(row[valid_mask])      # 1 … N_t
                    result[t, valid_mask] = stats.norm.ppf(ranks / (n_valid + 1))
            return result
        
        elif self.return_variable == ReturnVariable.RET_PCTL:
            # Percentile rank (cross-sectional)
            n_dates, n_tickers = returns.shape
            result = np.full_like(returns, np.nan)
            for t in range(n_dates):
                row = returns[t, :]
                valid_mask = ~np.isnan(row)
                if valid_mask.sum() > 0:
                    ranks = stats.rankdata(row[valid_mask])
                    pctls = ranks / len(ranks)
                    result[t, valid_mask] = pctls
            return result
        
        else:
            raise ValueError(f"Unknown return variable: {self.return_variable}")
    
    def create_expanding_window_features(
        self,
        data: StockData,
        start_date: str,
        end_date: str,
    ) -> List[Tuple[str, FeatureSet]]:
        """
        Create features using expanding window for time-series cross-validation.
        
        This is used for the expanding window training described in the paper.
        
        Parameters
        ----------
        data : StockData
            Stock data.
        start_date : str
            Start date for expanding window.
        end_date : str
            End date for test period.
            
        Returns
        -------
        list of (str, FeatureSet)
            List of (end_date, features) tuples for each window.
        """
        # Implementation for expanding window
        # Used for rolling forecasts
        windows = []
        
        # Filter to date range
        returns_df = data.returns.filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )
        
        dates = returns_df.get_column("date").to_list()
        
        # For each month after lookback period, create features using only past data
        for i in range(self.lookback_months, len(dates)):
            window_end = dates[i]
            
            # Filter data up to window_end
            window_prices = data.prices.filter(pl.col("date") <= window_end)
            window_returns = data.returns.filter(pl.col("date") <= window_end)
            window_mc = None
            if data.market_cap is not None:
                window_mc = data.market_cap.filter(pl.col("date") <= window_end)
            
            window_data = StockData(
                prices=window_prices,
                returns=window_returns,
                market_cap=window_mc,
                metadata=data.metadata,
            )
            features = self.create_features(window_data)
            windows.append((str(window_end), features))
        
        return windows


class DataValidator:
    """
    Validates data quality and consistency.
    
    Performs checks for:
    - Missing data
    - Outliers
    - Data consistency
    - Survivorship bias indicators
    """
    
    def __init__(
        self,
        max_missing_pct: float = 0.1,
        outlier_threshold: float = 5.0,  # Standard deviations
    ):
        self.max_missing_pct = max_missing_pct
        self.outlier_threshold = outlier_threshold
    
    def validate(self, data: StockData) -> Dict[str, any]:
        """
        Run all validation checks.
        
        Returns
        -------
        dict
            Validation results with issues found.
        """
        results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "stats": {},
        }
        
        # Convert to numpy for statistical calculations
        tickers = data.tickers
        returns_arr = data.returns.select(tickers).to_numpy()
        
        # Check missing data
        missing_pct = np.isnan(returns_arr).mean(axis=0)
        high_missing = np.sum(missing_pct > self.max_missing_pct)
        if high_missing > 0:
            results["warnings"].append(
                f"{high_missing} tickers have >{self.max_missing_pct:.0%} missing data"
            )
        results["stats"]["missing_pct"] = float(np.nanmean(missing_pct))
        
        # Check for outliers
        valid_returns = returns_arr[~np.isnan(returns_arr)]
        if len(valid_returns) > 0:
            z_scores = np.abs((valid_returns - np.mean(valid_returns)) / np.std(valid_returns))
            outlier_pct = np.mean(z_scores > self.outlier_threshold)
            if outlier_pct > 0.01:
                results["warnings"].append(
                    f"{outlier_pct:.1%} of observations are outliers"
                )
            results["stats"]["outlier_pct"] = float(outlier_pct)
        
        # Check date coverage
        dates = data.returns.get_column("date")
        date_range = (dates.max() - dates.min()).days
        results["stats"]["date_range_days"] = int(date_range)
        
        # Check ticker count
        results["stats"]["n_tickers"] = len(tickers)
        results["stats"]["n_observations"] = data.returns.height
        
        logger.info(f"Validation complete: {results['stats']}")
        
        return results
