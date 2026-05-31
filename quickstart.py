import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 确保 torch DLL 在跨盘符工作目录下仍可加载（Windows WinError 1114 修复）
# 必须保持对返回值的引用，否则 AddedDllDirectory 被 GC 后目录会被移除
_dll_dirs = []
for _lib_path in [
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\lib",
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\bin",
]:
    if os.path.isdir(_lib_path) and hasattr(os, "add_dll_directory"):
        _dll_dirs.append(os.add_dll_directory(_lib_path))

import polars as pl

from cbm import CBMConfig, PortfolioEngine
from cbm.core.config import BacktestConfig, PortfolioConfig, TrainingConfig
from cbm.core.types import Architecture, LossFunction, WeightingScheme
from cbm.utils.results_io import save_pipeline_results
from spx_local import attach_stock_data, build_stock_data_from_spx_folder

# ---------------------------------------------------------------------------
# 与原先 quickstart 对齐的设定 + 本地 SPX 数据
# ---------------------------------------------------------------------------

# Wind 日频 CSV 根目录 → 经 spx_local 转为月度 StockData
SPX_DATA_ROOT = Path(r"E:\phd\LLM_trading\CNN_trading\SPX_volume_price")
# 全样本时间（readme：约 2011-04 至 2025-12；可按需收窄）
SPX_START_DATE = "2011-04-01"
SPX_END_DATE = "2025-12-31"
# 月度宽表缓存（v2：归一化到日历月末日，每月一行；与旧 multi-row 缓存不同名以强制重建）
MONTHLY_WIDE_CACHE = Path(__file__).resolve().parent / "data" / "spx_monthly_wide_v2.parquet"

# 训练 / 测试（月），须满足训练末月 < 测试首月，且落在 SPX 月度数据范围内
OPTIMIZATION_PERIOD = ("2012-01", "2020-12")
TEST_PERIOD = ("2021-01", "2025-12")

# 与原 quickstart 一致：MSE + EWPM（由 TrainingConfig 生效；见 engine.train_model 实现）；分五组；本地无市值 → 等权
LOSS_FUNCTION = "mse"
SAMPLE_WEIGHTING = WeightingScheme.EWPM
N_PORTFOLIOS = 5
PORTFOLIO_WEIGHTING = "equal"

# 对比四种结构（与 Architecture 枚举一致）
ARCHITECTURES = [
    Architecture.FNN.value,
    Architecture.CNN.value,
    Architecture.LSTM.value,
    Architecture.CNN_LSTM.value,
]

_result_root = Path(__file__).resolve().parent / "result"


def _forecast_table_diagnostics(fv: pl.DataFrame) -> dict:
    """预测宽表：缺失率、整体分布、按月的截面均值/标准差。"""
    tcols = [c for c in fv.columns if c != "date"]
    if not tcols:
        return {"error": "no_ticker_columns"}
    long_df = fv.unpivot(on=tcols, index="date", variable_name="ticker", value_name="pred")
    overall = long_df.select(
        pl.len().alias("n_cells"),
        pl.col("pred").is_null().sum().alias("n_null"),
        pl.col("pred").mean().alias("pred_mean"),
        pl.col("pred").std().alias("pred_std"),
        pl.col("pred").min().alias("pred_min"),
        pl.col("pred").max().alias("pred_max"),
    ).row(0, named=True)
    overall["null_fraction"] = overall["n_null"] / max(overall["n_cells"], 1)

    by_date = (
        long_df.group_by("date")
        .agg(
            pl.col("pred").mean().alias("cs_mean"),
            pl.col("pred").std().alias("cs_std"),
            pl.col("pred").is_null().mean().alias("cs_null_frac"),
        )
        .sort("date")
    )
    cs_mean_series = by_date.get_column("cs_mean").drop_nulls()
    return {
        "overall": {k: float(v) if isinstance(v, (float, int)) else v for k, v in overall.items() if k != "n_cells"},
        "n_months": fv.height,
        "n_forecast_tickers": len(tcols),
        "cross_section_mean_of_pred_mean": float(cs_mean_series.mean())
        if cs_mean_series.len() > 0
        else None,
        "cross_section_mean_of_pred_std": float(by_date.get_column("cs_std").drop_nulls().mean())
        if by_date.height > 0
        else None,
        "by_date_sample": by_date.head(6).to_dicts(),
        "by_date_tail_sample": by_date.tail(6).to_dicts(),
    }


def _features_diagnostics(engine: PortfolioEngine) -> dict:
    fs = engine.features
    if fs is None:
        return {}
    dates = fs.dates
    tickers = fs.tickers
    feat = fs.features
    fdim: Optional[int] = None
    if hasattr(feat, "shape") and len(feat.shape) >= 2:
        fdim = int(feat.shape[1])
    elif hasattr(feat, "shape") and len(feat.shape) == 1 and feat.shape[0] == 0:
        fdim = 0  # 无样本时避免 IndexError
    return {
        "n_samples": int(len(fs)),
        "feature_dim": fdim,
        "feature_shape": list(feat.shape) if hasattr(feat, "shape") else None,
        "target_shape": list(fs.targets.shape) if hasattr(fs.targets, "shape") else None,
        "n_unique_dates": int(len(set(dates.tolist() if hasattr(dates, "tolist") else dates))),
        "n_unique_tickers": int(len(set(tickers.tolist() if hasattr(tickers, "tolist") else tickers))),
    }


def _returns_panel_diagnostics(engine: PortfolioEngine) -> dict:
    r = engine.data.returns
    tcols = [c for c in r.columns if c != "date"]
    return {
        "returns_rows": r.height,
        "returns_ticker_columns": len(tcols),
        "date_min": str(r.get_column("date").min()),
        "date_max": str(r.get_column("date").max()),
    }


def _print_returns_sample(stock_data, n_dates: int = 5, n_tickers: int = 6) -> None:
    """打印收益宽表样例，便于核对是否符合 cbm（date + 多 ticker 列）。"""
    r = stock_data.returns.sort("date")
    tcols = [c for c in r.columns if c != "date"][:n_tickers]
    if not tcols:
        print("(无 ticker 列)")
        return
    sample = r.select(["date"] + tcols).head(n_dates)
    print(f"\n收益表样例（前 {n_dates} 月 × 前 {len(tcols)} 标的，与 FeatureEngineer 输入一致）:")
    print(sample)


def main() -> None:
    if not SPX_DATA_ROOT.is_dir():
        raise FileNotFoundError(f"SPX 数据目录不存在: {SPX_DATA_ROOT}")

    config = CBMConfig(
        training=TrainingConfig(
            optimization_period=OPTIMIZATION_PERIOD,
            weighting=SAMPLE_WEIGHTING,
            loss_function=LossFunction.MSE,
        ),
        portfolio=PortfolioConfig(
            n_portfolios=N_PORTFOLIOS,
            weighting=PORTFOLIO_WEIGHTING,
        ),
        backtest=BacktestConfig(test_period=TEST_PERIOD),
    )
    engine = PortfolioEngine(config)

    print("加载本地 SPX 日频并聚合为月度面板…")
    stock_data = build_stock_data_from_spx_folder(
        SPX_DATA_ROOT,
        start_date=SPX_START_DATE,
        end_date=SPX_END_DATE,
        tickers=None,
        monthly_wide_parquet_cache=MONTHLY_WIDE_CACHE,
        refresh_monthly_cache=False,
        # SPX 成分进出导致收益矩阵极稀疏；不填充时 12 月累积特征几乎全为 NaN → 0 条样本
        fill_missing_prices="forward",
    )
    attach_stock_data(engine, stock_data)
    print(
        f"月度数据: {len(stock_data.tickers)} 标的, "
        f"收益表 {stock_data.returns.height} 行, "
        f"区间 {stock_data.date_range[0]} ~ {stock_data.date_range[1]}"
    )
    print("数据 metadata:", json.dumps(stock_data.metadata, ensure_ascii=False, indent=2))
    _print_returns_sample(stock_data, n_dates=5, n_tickers=6)

    engine.prepare_features()
    base_diag = {
        "data_source": "spx_local_wind_csv",
        "spx_root": str(SPX_DATA_ROOT.resolve()),
        "optimization_period": list(OPTIMIZATION_PERIOD),
        "test_period": list(TEST_PERIOD),
        "loss_function": LOSS_FUNCTION,
        "sample_weighting": SAMPLE_WEIGHTING.value,
        "n_portfolios": N_PORTFOLIOS,
        "portfolio_weighting": PORTFOLIO_WEIGHTING,
        "returns_panel": _returns_panel_diagnostics(engine),
        "features": _features_diagnostics(engine),
    }

    comparison_rows: list[dict] = []

    for arch in ARCHITECTURES:
        print(f"\n========== 训练结构: {arch} ==========")
        model_id = engine.train_model(
            architecture=arch,
            loss_function=LOSS_FUNCTION,
            weighting=SAMPLE_WEIGHTING,
            optimization_period=OPTIMIZATION_PERIOD,
        )
        train_metrics = dict(engine._models[model_id]["metrics"])  # noqa: SLF001

        forecasts = engine.forecast(model_id=model_id, test_period=TEST_PERIOD)
        portfolios = engine.construct_portfolios(
            forecasts=forecasts,
            n_portfolios=N_PORTFOLIOS,
            weighting=PORTFOLIO_WEIGHTING,
        )
        performance = engine.analyze_performance(portfolios)

        for portfolio_name, metrics in performance.items():
            print(f"\n--- [{arch}] Portfolio {portfolio_name} ---")
            print(metrics.summary())

        diagnostics = {
            **base_diag,
            "architecture": arch,
            "training_metrics": {k: float(v) if hasattr(v, "item") else v for k, v in train_metrics.items()},
            "forecast_diagnostics": _forecast_table_diagnostics(forecasts.values),
        }

        run_dir = save_pipeline_results(
            engine=engine,
            model_id=model_id,
            performance=performance,
            forecasts=forecasts,
            portfolios=portfolios,
            result_dir=_result_root,
            diagnostics=diagnostics,
            experiment_tag=f"quickstart_spx_compare_{arch}",
        )

        # 直接使用 ModelRegistry 将集成子模型权重存入 run 目录下的 model/ 子目录
        from cbm.ml import ModelRegistry

        try:
            registry = ModelRegistry(path=str(run_dir / "model"))
            registry.save(model_id, engine._models[model_id])  # noqa: SLF001
            print(f"  模型权重已保存至: model/{model_id}/")
        except Exception as exc:  # noqa: BLE001
            print(f"  [警告] 模型权重保存失败（结果文件不受影响）: {exc}")

        row: dict = {
            "architecture": arch,
            "model_id": model_id,
            "run_dir": str(run_dir.resolve()),
        }
        for k, v in train_metrics.items():
            row[f"train_{k}"] = float(v) if hasattr(v, "item") else v
        ls = performance.get("long_short")
        if ls is not None:
            row["ls_mean_monthly"] = ls.mean_return
            row["ls_sharpe"] = ls.sharpe_ratio
            row["ls_tstat"] = ls.t_statistic
            row["ls_ann_return"] = ls.annualized_return
            row["ls_max_dd"] = ls.max_drawdown
            if ls.alpha:
                for model_name, a in ls.alpha.items():
                    row[f"ls_alpha_{model_name}"] = a
        for i in range(1, N_PORTFOLIOS + 1):
            key = str(i)
            if key in performance:
                m = performance[key]
                row[f"p{key}_mean_monthly"] = m.mean_return
                row[f"p{key}_sharpe"] = m.sharpe_ratio
        comparison_rows.append(row)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_json = _result_root / f"architecture_comparison_{stamp}.json"
    comp_csv = _result_root / f"architecture_comparison_{stamp}.csv"
    comp_json.write_text(
        json.dumps(comparison_rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    pl.DataFrame(comparison_rows).write_csv(comp_csv)
    print(f"\n多结构对比表: {comp_csv.resolve()}")
    print(f"多结构对比 JSON: {comp_json.resolve()}")
    print(f"各结构详细 run 目录: {_result_root.resolve()} / run_*")


if __name__ == "__main__":
    main()
