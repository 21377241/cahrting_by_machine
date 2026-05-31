"""CRSP 论文设定回测：加载扩展窗口 run 目录，运行主实验十分位 + Table 6 Top500。

支持：
  - 直接读取 ``forecasts.parquet``（默认）
  - ``--regenerate-forecasts`` 从 ``models/window_*/`` 重新生成预测（需超额收益特征）

用法::

    python crsp_paper_backtest.py --run-dir result/run_20260523_...
    python crsp_paper_backtest.py --run-dir ... --regenerate-forecasts
    python crsp_paper_backtest.py --run-dir ... --skip-main-decile
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date as dt_date
from pathlib import Path

import polars as pl
from loguru import logger

for _lib_path in [
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\lib",
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\bin",
]:
    if os.path.isdir(_lib_path) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_lib_path)

from cbm import CBMConfig, PortfolioEngine
from cbm.core.config import ModelConfig
from cbm.core.types import ReturnVariable
from cbm.data.crsp_sample_filter import apply_paper_sample_filter_to_features, panel_load_start
from cbm.data import FeatureEngineer
from cbm.data.french_rf import (
    align_rf_to_return_dates,
    apply_excess_returns,
    load_french_rf_monthly,
)
from cbm.ml import Forecaster, ModelRegistry
from cbm.ml.forecaster import merge_forecast_wides
from cbm.ml.paper_training import EXPANDING_WINDOWS, window_model_dir_name

from crsp_top500_backtest import (
    TEST_PERIOD as TOP500_TEST_PERIOD,
    compute_decile_portfolios,
    compute_top500_decile_portfolios,
    load_crsp_panel,
    load_forecasts_long,
    load_forecasts_long_from_wide,
)

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "result"
DATA_START = dt_date(1927, 1, 1)
DATA_END = dt_date(2022, 12, 31)
MAIN_TEST_PERIOD = ("1963-07", "2022-12")
N_PORTFOLIOS = 10


def _find_latest_paper_run() -> Path:
    candidates = sorted(RESULT_ROOT.glob("run_*_crsp_paper_*"))
    if not candidates:
        raise FileNotFoundError(f"未找到 crsp_paper run 目录: {RESULT_ROOT}")
    return candidates[-1]


def _load_windows_meta(run_dir: Path) -> list[dict]:
    path = run_dir / "windows.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("windows", [])
    # 回退：按 EXPANDING_WINDOWS 默认映射
    return [
        {
            "train_end": w["train_end"],
            "forecast_start": w["forecast_start"],
            "forecast_end": w["forecast_end"],
            "model_dir": window_model_dir_name(w["train_end"]),
        }
        for w in EXPANDING_WINDOWS
    ]


def _prepare_excess_features(engine: PortfolioEngine, run_dir: Path | None = None) -> None:
    rf = load_french_rf_monthly()
    rf_aligned = align_rf_to_return_dates(engine.data.returns, rf)
    rv = ReturnVariable.RET_RANK_NORM
    if run_dir is not None:
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rv = ReturnVariable(meta.get("return_variable", rv.value))

    engineer = FeatureEngineer(return_variable=rv)
    raw = engineer.create_features(
        engine.data,
        risk_free_rate=rf_aligned,
    )
    data_end = dt_date(2022, 12, 31)
    data_start = panel_load_start(dt_date(1927, 1, 1))
    engine._features = apply_paper_sample_filter_to_features(  # noqa: SLF001
        raw, data_start, data_end,
    )


def regenerate_expanding_forecasts(run_dir: Path) -> pl.DataFrame:
    """从各窗口 saved model 重新生成并合并预测。"""
    windows = _load_windows_meta(run_dir)
    models_base = run_dir / "models"
    if not models_base.exists():
        raise FileNotFoundError(f"未找到 models 目录: {models_base}")

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) if (run_dir / "meta.json").exists() else {}
    universe = meta.get("universe", "crsp_all")

    config = CBMConfig(model=ModelConfig(device="auto", batch_size=512))
    engine = PortfolioEngine(config)
    engine.load_data(
        source="crsp_local",
        universe=universe,
        start_date="1927-01-01",
        end_date="2022-12-31",
    )
    _prepare_excess_features(engine, run_dir)
    features = engine.features

    registry = ModelRegistry(path=str(models_base))
    wides: list[pl.DataFrame] = []

    for w in windows:
        model_dir = models_base / w.get("model_dir", window_model_dir_name(w["train_end"]))
        if not model_dir.exists():
            raise FileNotFoundError(f"窗口模型目录不存在: {model_dir}")
        _, model_data = registry.load(str(model_dir))
        fc = Forecaster(model=model_data["model"]).predict(
            features=features,
            test_period=(w["forecast_start"], w["forecast_end"]),
        )
        wides.append(fc.values)
        logger.info(
            f"重新预测 {w['forecast_start']}→{w['forecast_end']}: "
            f"{fc.values.height} 月"
        )

    combined = merge_forecast_wides(wides)
    combined.write_parquet(run_dir / "forecasts.parquet")
    return load_forecasts_long_from_wide(combined)


def run_main_decile_backtest(
    run_dir: Path,
    forecasts_long: pl.DataFrame,
    test_period: tuple[str, str],
    use_excess: bool = True,
) -> dict:
    """主实验：Table 3 十分位（NYSE 断点 + 全样本持仓，长表 + 论文筛选，市值加权）。"""
    t_start = dt_date(int(test_period[0][:4]), int(test_period[0][5:7]), 1)
    data_start = panel_load_start(t_start)
    returns, mcap, eligible, exchanges = load_crsp_panel(data_start, DATA_END)

    if use_excess:
        rf = load_french_rf_monthly()
        returns = apply_excess_returns(returns, rf, ym_col="ym")

    result = compute_decile_portfolios(
        forecasts=forecasts_long,
        returns=returns,
        mcap=mcap,
        test_period=test_period,
        top_n=None,
        n_portfolios=N_PORTFOLIOS,
        eligible=eligible,
        use_nyse_breakpoints=True,
        exchanges=exchanges,
    )

    out_dir = run_dir / "paper_backtest" / "main_decile"
    out_dir.mkdir(parents=True, exist_ok=True)

    perf = result["metrics"]
    summary = {
        "experiment": "main_decile_table3",
        "return_type": "excess" if use_excess else "total",
        "sample_filter": "paper_section_2_1",
        "breakpoints_universe": "eligible_nyse_only",
        "holdings_universe": "eligible_full_sample",
        "weighting": "value",
        "n_portfolios": N_PORTFOLIOS,
        "test_period": list(test_period),
        "performance": perf,
    }
    (out_dir / "performance.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    summary_lines = [f"主实验十分位 Table 3 ({test_period[0]} → {test_period[1]})"]
    summary_lines.append("  断点: NYSE-only | 持仓: 全合格样本")
    for key in [str(i) for i in range(1, N_PORTFOLIOS + 1)] + ["long_short"]:
        m = perf[key]
        label = f"P{key}" if key != "long_short" else "L/S (P10-P1)"
        summary_lines.append(
            f"  {label:12s}  mean={m['mean_return']*100:7.3f}%  "
            f"Sharpe={m['sharpe_ratio']:6.3f}  t={m['t_statistic']:6.2f}"
        )
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    if result["diagnostics"]:
        pl.DataFrame(result["diagnostics"]).write_csv(out_dir / "monthly_diagnostics.csv")

    logger.info(f"主实验十分位结果: {out_dir}")
    return perf


def run_top500_backtest(
    forecasts_long: pl.DataFrame,
    test_period: tuple[str, str],
    out_dir: Path,
    use_excess: bool = True,
) -> dict:
    """Table 6 Top500/Top500 十分位（超额收益）。"""
    t_start = dt_date(int(test_period[0][:4]), int(test_period[0][5:7]), 1)
    data_start = panel_load_start(t_start)
    returns, mcap, eligible, _ = load_crsp_panel(data_start, DATA_END)

    if use_excess:
        rf = load_french_rf_monthly()
        returns = apply_excess_returns(returns, rf, ym_col="ym")

    result = compute_top500_decile_portfolios(
        forecasts=forecasts_long,
        returns=returns,
        mcap=mcap,
        test_period=test_period,
        eligible=eligible,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    perf = result["metrics"]
    (out_dir / "performance.json").write_text(
        json.dumps(perf, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [f"Top500/Top500 十分位 ({test_period[0]} → {test_period[1]})"]
    for pname in sorted(perf.keys()):
        m = perf[pname]
        lines.append(
            f"  {pname:12s}  mean={m['mean_return']*100:7.3f}%  "
            f"t={m.get('t_statistic', float('nan')):6.2f}"
        )

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Top500 回测结果: {out_dir}")
    return perf


def main() -> None:
    parser = argparse.ArgumentParser(description="CRSP 论文设定回测")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--regenerate-forecasts", action="store_true")
    parser.add_argument("--skip-main-decile", action="store_true")
    parser.add_argument(
        "--test-period",
        default=",".join(MAIN_TEST_PERIOD),
        help="主实验测试期 YYYY-MM,YYYY-MM",
    )
    parser.add_argument(
        "--top500-test-period",
        default=",".join(TOP500_TEST_PERIOD),
    )
    args = parser.parse_args()

    run_dir = args.run_dir or _find_latest_paper_run()
    if not run_dir.exists():
        raise FileNotFoundError(f"run 目录不存在: {run_dir}")

    test_parts = args.test_period.split(",")
    main_test = (test_parts[0].strip(), test_parts[1].strip())
    top500_parts = args.top500_test_period.split(",")
    top500_test = (top500_parts[0].strip(), top500_parts[1].strip())

    print(f"\n回测 run 目录: {run_dir}")
    print(f"重新生成预测: {args.regenerate_forecasts}")

    if args.regenerate_forecasts:
        forecasts_long = regenerate_expanding_forecasts(run_dir)
    else:
        forecasts_long = load_forecasts_long(run_dir)

    if not args.skip_main_decile:
        print(f"\n主实验十分位 ({main_test[0]} → {main_test[1]})…")
        run_main_decile_backtest(run_dir, forecasts_long, main_test)

    print(f"\nTable 6 Top500 ({top500_test[0]} → {top500_test[1]})…")
    run_top500_backtest(
        forecasts_long,
        top500_test,
        run_dir / "paper_backtest" / "top500_decile_excess",
        use_excess=True,
    )

    print(f"\n回测完成。结果目录: {run_dir / 'paper_backtest'}")


if __name__ == "__main__":
    main()
