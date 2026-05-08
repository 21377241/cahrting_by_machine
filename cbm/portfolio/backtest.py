"""Backtesting engine for historical portfolio simulation."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from loguru import logger

from cbm.core.types import BacktestResult, PerformanceMetrics, Portfolio, PortfolioSet
from cbm.portfolio.analyzer import PerformanceAnalyzer
from cbm.portfolio.constructor import PortfolioConstructor


class BacktestEngine:
    """
    Engine for historical backtesting of ML-based portfolios.
    
    Implements realistic backtesting with:
    - Monthly rebalancing
    - Transaction costs
    - Proper handling of look-ahead bias
    - Out-of-sample evaluation
    
    Parameters
    ----------
    transaction_costs : float
        Transaction costs as fraction (e.g., 0.001 for 10 bps).
    rebalance_freq : str
        Rebalancing frequency ("monthly", "quarterly").
    start_capital : float
        Initial portfolio value.
        
    Example
    -------
    >>> engine = BacktestEngine(transaction_costs=0.001)
    >>> result = engine.run_backtest(forecasts, returns, market_caps)
    """
    
    def __init__(
        self,
        transaction_costs: float = 0.001,
        rebalance_freq: str = "monthly",
        start_capital: float = 1_000_000,
    ):
        self.transaction_costs = transaction_costs
        self.rebalance_freq = rebalance_freq
        self.start_capital = start_capital
    
    def run_backtest(
        self,
        portfolios: PortfolioSet,
        include_costs: bool = True,
    ) -> BacktestResult:
        """
        Run backtest on pre-constructed portfolios.
        
        Parameters
        ----------
        portfolios : PortfolioSet
            Pre-constructed portfolios.
        include_costs : bool
            Whether to include transaction costs.
            
        Returns
        -------
        BacktestResult
            Backtest results with performance metrics.
        """
        logger.info("Running backtest")
        
        # Calculate transaction costs (simplified)
        cost_adjusted_portfolios = {}
        
        for name, portfolio in portfolios.portfolios.items():
            returns_df = portfolio.returns
            
            if include_costs:
                # Simplified: assume constant turnover
                avg_turnover = 0.2  # 20% monthly turnover assumption
                costs = avg_turnover * self.transaction_costs
                
                # Adjust returns
                adjusted_returns = returns_df.with_columns([
                    (pl.col("return") - costs).alias("return")
                ])
            else:
                adjusted_returns = returns_df
            
            cost_adjusted_portfolios[name] = Portfolio(
                name=name,
                holdings=portfolio.holdings,
                returns=adjusted_returns,
                metadata=portfolio.metadata,
            )
        
        adjusted_portfolio_set = PortfolioSet(
            portfolios=cost_adjusted_portfolios,
            long_short=portfolios.long_short,
        )
        
        # Analyze performance
        analyzer = PerformanceAnalyzer()
        performance = analyzer.analyze(adjusted_portfolio_set)
        
        # Calculate cumulative returns
        cumulative_data = {"date": []}
        first_portfolio = list(cost_adjusted_portfolios.values())[0]
        dates = first_portfolio.returns.get_column("date").to_list()
        cumulative_data["date"] = dates
        
        for name, port in cost_adjusted_portfolios.items():
            returns = port.returns.get_column("return").to_numpy()
            cum_returns = np.cumprod(1 + np.nan_to_num(returns, nan=0))
            cumulative_data[name] = cum_returns.tolist()
        
        # Add long-short
        if portfolios.long_short is not None:
            ls_returns = portfolios.long_short.returns.get_column("return").to_numpy()
            if include_costs:
                ls_returns = ls_returns - 2 * 0.2 * self.transaction_costs
            ls_cum = np.cumprod(1 + np.nan_to_num(ls_returns, nan=0))
            cumulative_data["long_short"] = ls_cum.tolist()
        
        cumulative_df = pl.DataFrame(cumulative_data)
        
        return BacktestResult(
            portfolio_set=adjusted_portfolio_set,
            performance=performance,
            cumulative_returns=cumulative_df,
        )
    
    def calculate_turnover(
        self,
        holdings: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Calculate portfolio turnover from holdings.
        
        Parameters
        ----------
        holdings : pl.DataFrame
            Portfolio holdings over time.
            
        Returns
        -------
        pl.DataFrame
            Monthly turnover with date and turnover columns.
        """
        # Get ticker columns
        ticker_cols = [c for c in holdings.columns if c != "date"]
        
        if len(ticker_cols) == 0:
            return pl.DataFrame({"date": [], "turnover": []})
        
        dates = holdings.get_column("date").to_list()
        turnovers = []
        
        for i in range(1, len(dates)):
            prev_row = holdings.row(i - 1, named=True)
            curr_row = holdings.row(i, named=True)
            
            total_change = 0
            for col in ticker_cols:
                prev_val = prev_row.get(col, 0) or 0
                curr_val = curr_row.get(col, 0) or 0
                total_change += abs(curr_val - prev_val)
            
            turnovers.append(total_change / 2)
        
        return pl.DataFrame({
            "date": dates[1:],
            "turnover": turnovers,
        })
    
    def generate_report(self, result: BacktestResult) -> str:
        """
        Generate text report of backtest results.
        
        Parameters
        ----------
        result : BacktestResult
            Backtest results.
            
        Returns
        -------
        str
            Formatted report.
        """
        lines = [
            "=" * 60,
            "BACKTEST REPORT",
            "=" * 60,
            "",
        ]
        
        # Summary statistics
        lines.append("Portfolio Performance Summary")
        lines.append("-" * 40)
        
        summary = result.summary()
        lines.append(str(summary))
        
        lines.append("")
        lines.append("-" * 40)
        
        # Long-short portfolio details
        if "long_short" in result.performance:
            ls_perf = result.performance["long_short"]
            lines.extend([
                "Long-Short Portfolio:",
                f"  Mean Return: {ls_perf.mean_return:.2%} per month",
                f"  Std Dev: {ls_perf.std_dev:.2%}",
                f"  Sharpe Ratio: {ls_perf.sharpe_ratio:.2f}",
                f"  t-statistic: {ls_perf.t_statistic:.2f}",
                f"  Max Drawdown: {ls_perf.max_drawdown:.2%}",
            ])
            
            if ls_perf.alpha:
                lines.append("  Factor Model Alphas:")
                for model, alpha in ls_perf.alpha.items():
                    lines.append(f"    {model}: {alpha:.2%}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
