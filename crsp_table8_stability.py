"""论文 Section 5.1 / Table 8：子期间 MLER 交叉可预测性（稳定性检验）

Table 8（Subperiod-Based Forecasts and Returns）：
  - 在 6 个互不重叠子样本上各训练一套 ML（全月训练，非仅拟合月）
  - 对全测试期 1963-07 → 2022-12 生成预测
  - 用 Table 3 组合规则（NYSE 断点、市值加权）建 10−1 组合
  - 6×6 矩阵：行=收益子期间，列=排序用的 MLER^{t1,t2}

可选 ``--table7``：复现 Table 7（预测相关、10−1 共同持仓、收益 Pearson 相关），
  比较月份剔除任一侧训练区间（与论文 Table 7 脚注一致）。

用法（论文设计：各子期间独立重训）::

    python crsp_table8_stability.py --run-dir result/run_20260523_232834_crsp_paper_crsp_all_cnn_lstm_excess
    python crsp_table8_stability.py --run-dir ... --n-ensemble 5 --device cpu
    python crsp_table8_stability.py --run-dir ... --skip-train          # 已有 subperiod_* 预测
    python crsp_table8_stability.py --run-dir ... --subperiods 0,2    # 只训部分段
    python crsp_table8_stability.py --run-dir ... --table7

快捷（非论文 Table 8 设计，复用扩展窗口模型）::

    python crsp_table8_stability.py --use-expanding-windows --paper-run-dir result/run_...
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date as dt_date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import pearsonr, spearmanr

for _lib_path in [
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\lib",
    r"F:\envs\charting_by_machine\Lib\site-packages\torch\bin",
]:
    if os.path.isdir(_lib_path) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_lib_path)

from cbm import CBMConfig, PortfolioEngine
from cbm.core.config import ModelConfig, TrainingConfig
from cbm.core.types import Architecture, LossFunction, ReturnVariable, WeightingScheme
from cbm.data.crsp_sample_filter import apply_paper_sample_filter_to_features, panel_load_start
from cbm.data.french_rf import align_rf_to_return_dates, load_french_rf_monthly
from cbm.data import FeatureEngineer
from cbm.ml import Forecaster, ModelRegistry
from cbm.ml.forecaster import merge_forecast_wides
from cbm.ml.paper_training import EXPANDING_WINDOWS, PaperModelTrainer, window_model_dir_name
from cbm.ml.models.pytorch_impl import get_device

from crsp_top500_backtest import (
    NEWEY_WEST_LAGS,
    _add_ym,
    _assign_portfolios,
    _calc_breakpoints,
    _newey_west_tstat,
    _period_bounds,
    _prev_ym,
    _value_weighted_return,
    compute_decile_portfolios,
    load_crsp_panel,
    load_forecasts_long_from_wide,
)

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "result"

START_DATE = "1927-01-01"
END_DATE = "2022-12-31"
TEST_PERIOD = ("1963-07", "2022-12")
UNIVERSE = "crsp_all"
ARCHITECTURE = Architecture.CNN_LSTM
LOSS_FUNCTION = LossFunction.MSE
SAMPLE_WEIGHTING = WeightingScheme.EWPM
RETURN_VARIABLE = ReturnVariable.RET_RANK_NORM
N_PORTFOLIOS = 10

# Section 5.1：6 段训练子样本（上标 t1,t2）
SUBPERIOD_FITS: list[dict[str, str]] = [
    {"label": "192701_196306", "train_start": "1927-01", "train_end": "1963-06"},
    {"label": "196307_197412", "train_start": "1963-07", "train_end": "1974-12"},
    {"label": "197501_198412", "train_start": "1975-01", "train_end": "1984-12"},
    {"label": "198501_199412", "train_start": "1985-01", "train_end": "1994-12"},
    {"label": "199501_200412", "train_start": "1995-01", "train_end": "2004-12"},
    {"label": "200501_201412", "train_start": "2005-01", "train_end": "2014-12"},
]

# Table 8 行：收益考察子期间（与训练段对齐，196307 起为测试期）
RETURN_SUBPERIODS: list[dict[str, str]] = [
    {"label": "196307_197412", "start": "1963-07", "end": "1974-12"},
    {"label": "197501_198412", "start": "1975-01", "end": "1984-12"},
    {"label": "198501_199412", "start": "1985-01", "end": "1994-12"},
    {"label": "199501_200412", "start": "1995-01", "end": "2004-12"},
    {"label": "200501_201412", "start": "2005-01", "end": "2014-12"},
    {"label": "201501_202212", "start": "2015-01", "end": "2022-12"},
]

# 扩展窗口 train_end → Table 8 列名（与 SUBPERIOD_FITS 标签一致）
WINDOW_TRAIN_END_TO_LABEL: dict[str, str] = {
    spec["train_end"]: spec["label"] for spec in SUBPERIOD_FITS
}


def _check_parquet() -> None:
    parquet = ROOT / "crsp_data" / "crsp2525_monthly.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"未找到 CRSP 数据: {parquet}")


def _prepare_excess_features(engine: PortfolioEngine, paper_run_dir: Path | None = None) -> None:
    rf = load_french_rf_monthly()
    rf_aligned = align_rf_to_return_dates(engine.data.returns, rf)
    rv = RETURN_VARIABLE
    if paper_run_dir is not None:
        meta_path = paper_run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rv = ReturnVariable(meta.get("return_variable", rv.value))
    engineer = FeatureEngineer(return_variable=rv)
    raw = engineer.create_features(engine.data, risk_free_rate=rf_aligned)
    data_start = panel_load_start(dt_date(1927, 1, 1))
    engine._features = apply_paper_sample_filter_to_features(  # noqa: SLF001
        raw, data_start, dt_date(2022, 12, 31),
    )


def _load_paper_windows(paper_run_dir: Path) -> list[dict]:
    path = paper_run_dir / "windows.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("windows", [])
    return [
        {
            "train_end": w["train_end"],
            "forecast_start": w["forecast_start"],
            "forecast_end": w["forecast_end"],
            "model_dir": window_model_dir_name(w["train_end"]),
        }
        for w in EXPANDING_WINDOWS
    ]


def _expanding_forecast_cache_path(out_dir: Path, label: str) -> Path:
    return out_dir / "stability" / "expanding_window" / label / "forecasts.parquet"


def load_forecasts_from_paper_run(
    paper_run_dir: Path,
    out_dir: Path,
    features,
    fit_indices: list[int],
    regenerate: bool = False,
) -> dict[str, pl.DataFrame]:
    """从 crsp_paper_train 的六窗口模型生成全测试期预测，键为 Table 8 列名。"""
    windows = _load_paper_windows(paper_run_dir)
    models_base = paper_run_dir / "models"
    if not models_base.exists():
        raise FileNotFoundError(f"未找到 models 目录: {models_base}")

    registry = ModelRegistry(path=str(models_base))
    labels = [SUBPERIOD_FITS[i]["label"] for i in fit_indices]
    forecast_by_label: dict[str, pl.DataFrame] = {}

    for w in windows:
        train_end = w["train_end"]
        label = WINDOW_TRAIN_END_TO_LABEL.get(train_end)
        if label is None or label not in labels:
            continue

        cache = _expanding_forecast_cache_path(out_dir, label)
        if cache.exists() and not regenerate:
            logger.info(f"读取缓存预测: {cache}")
            wide = pl.read_parquet(cache)
            forecast_by_label[label] = load_forecasts_long_from_wide(wide)
            continue

        model_dir = models_base / w.get("model_dir", window_model_dir_name(train_end))
        if not model_dir.exists():
            raise FileNotFoundError(f"窗口模型目录不存在: {model_dir}")

        print(
            f"\n── 扩展窗口 {w.get('model_dir', train_end)} "
            f"(训练至 {train_end}) → 列 {label}，预测 {TEST_PERIOD[0]}→{TEST_PERIOD[1]}"
        )
        t0 = datetime.now()
        _, model_data = registry.load(str(model_dir))
        fc = Forecaster(model=model_data["model"]).predict(
            features=features,
            test_period=TEST_PERIOD,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        fc.values.write_parquet(cache)
        forecast_by_label[label] = load_forecasts_long_from_wide(fc.values)
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  完成 ({elapsed:.0f}s) | {fc.values.height} 月 → {cache}")

    missing = [lab for lab in labels if lab not in forecast_by_label]
    if missing:
        raise FileNotFoundError(
            f"未生成以下 Table 8 列的预测: {missing}。"
            f"请确认 paper run 含 6 个 window_* 目录。"
        )
    return forecast_by_label


def _parse_indices(spec: str, n: int) -> list[int]:
    return [int(p.strip()) for p in spec.split(",") if p.strip()]


def _period_to_ym_range(period: tuple[str, str]) -> tuple[int, int]:
    t0, t1 = _period_bounds(period)
    return t0.year * 100 + t0.month, t1.year * 100 + t1.month


def _ym_in_period(ym: int, period: tuple[str, str]) -> bool:
    lo, hi = _period_to_ym_range(period)
    return lo <= ym <= hi


def _subperiod_dir(run_dir: Path, label: str) -> Path:
    return run_dir / "stability" / f"subperiod_{label}"


def _forecasts_path(run_dir: Path, label: str) -> Path:
    return _subperiod_dir(run_dir, label) / "forecasts.parquet"


def _subperiod_models_dir(run_dir: Path) -> Path:
    return run_dir / "stability" / "subperiod_models"


def train_subperiod_models(
    engine: PortfolioEngine,
    run_dir: Path,
    indices: list[int],
    n_ensemble: int,
    trainer: PaperModelTrainer,
    registry: ModelRegistry,
    regenerate: bool = False,
) -> dict[str, pl.DataFrame]:
    """论文 Table 8：各子样本内独立训练 ML，对全测试期生成预测。"""
    features = engine.features
    forecast_by_label: dict[str, pl.DataFrame] = {}

    for idx in indices:
        spec = SUBPERIOD_FITS[idx]
        label = spec["label"]
        train_period = (spec["train_start"], spec["train_end"])
        out_dir = _subperiod_dir(run_dir, label)
        out_dir.mkdir(parents=True, exist_ok=True)
        fc_path = _forecasts_path(run_dir, label)

        if fc_path.exists() and not regenerate:
            print(f"\n── 子期间 {idx} ({label}): 读取已有预测 {fc_path}")
            forecast_by_label[label] = load_forecasts_long_from_wide(pl.read_parquet(fc_path))
            continue

        print(
            f"\n── 子期间 {idx} ({label}): "
            f"仅样本内训练 {train_period[0]} → {train_period[1]}（全月，非拟合月）"
        )
        t0 = datetime.now()

        ensemble, metrics = trainer.train(
            features=features,
            optimization_period=train_period,
            n_ensemble=n_ensemble,
            fitting_months_only=False,
        )

        model_id = f"subperiod_{label}"
        registry.save(
            model_id,
            {
                "model": ensemble,
                "metrics": metrics,
                "config": {
                    "experiment": "table8_subperiod_fit",
                    "train_period": list(train_period),
                    "fitting_months_only": False,
                    "architecture": ARCHITECTURE.value,
                },
            },
            metrics=metrics,
        )

        fc = Forecaster(model=ensemble).predict(features=features, test_period=TEST_PERIOD)
        fc.values.write_parquet(fc_path)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8",
        )
        forecast_by_label[label] = load_forecasts_long_from_wide(fc.values)

        elapsed = (datetime.now() - t0).total_seconds()
        print(
            f"  完成 ({elapsed:.0f}s) | val Spearman={metrics['val_spearman_corr']:.4f} | "
            f"预测月数={fc.values.height} → {fc_path}"
        )

    return forecast_by_label


def load_subperiod_forecasts(run_dir: Path, indices: list[int]) -> dict[str, pl.DataFrame]:
    out: dict[str, pl.DataFrame] = {}
    for idx in indices:
        label = SUBPERIOD_FITS[idx]["label"]
        path = _forecasts_path(run_dir, label)
        if not path.exists():
            raise FileNotFoundError(f"缺少子期间预测: {path}（去掉 --skip-train 先训练）")
        wide = pl.read_parquet(path)
        out[label] = load_forecasts_long_from_wide(wide)
    return out


def _monthly_long_short_sets(
    forecasts: pl.DataFrame,
    returns: pl.DataFrame,
    mcap: pl.DataFrame,
    eligible: pl.DataFrame,
    exchanges: pl.DataFrame,
    eval_period: tuple[str, str],
    exclude_train_periods: Iterable[tuple[str, str]] = (),
) -> tuple[list[int], list[set[str]], list[set[str]], list[float]]:
    """逐月计算 10−1 组合的多头/空头持仓集合与多空收益。"""
    from cbm.data.crsp_sample_filter import PAPER_NYSE_EXCHANGE

    ym_lo, ym_hi = _period_to_ym_range(eval_period)
    exclude = list(exclude_train_periods)

    fc = _add_ym(forecasts)
    ret = _add_ym(returns)
    mc = _add_ym(mcap)

    mcap_by_ym = {s["ym"][0]: s for s in mc.partition_by("ym", maintain_order=True)}
    fc_by_ym = {s["ym"][0]: s for s in fc.partition_by("ym", maintain_order=True)}
    ret_by_ym = {s["ym"][0]: s for s in ret.partition_by("ym", maintain_order=True)}

    eligible_by_ym: dict[int, set[str]] = {}
    for sub in eligible.partition_by("ym", maintain_order=True):
        eligible_by_ym[sub["ym"][0]] = set(sub["permno"].to_list())

    nyse_by_ym: dict[int, set[str]] = {}
    nyse = exchanges.filter(pl.col("primary_exch") == PAPER_NYSE_EXCHANGE)
    for sub in nyse.partition_by("ym", maintain_order=True):
        nyse_by_ym[sub["ym"][0]] = set(sub["permno"].to_list())

    yms = sorted(
        ym for ym in fc_by_ym
        if ym_lo <= ym <= ym_hi
        and not any(_ym_in_period(ym, p) for p in exclude)
    )

    months: list[int] = []
    long_sets: list[set[str]] = []
    short_sets: list[set[str]] = []
    ls_returns: list[float] = []

    for ym in yms:
        ym_prev = _prev_ym(ym)
        if ym_prev not in mcap_by_ym:
            continue
        elig = eligible_by_ym.get(ym, set())
        if not elig:
            continue

        mcap_prev = mcap_by_ym[ym_prev].filter(pl.col("permno").is_in(list(elig)))
        fc_t = fc_by_ym.get(ym)
        ret_t = ret_by_ym.get(ym)
        if fc_t is None or ret_t is None:
            continue

        pool = (
            mcap_prev.select("permno")
            .join(fc_t.select(["permno", "score"]), on="permno", how="inner")
            .join(mcap_prev.select(["permno", "mcap"]), on="permno", how="inner")
            .join(ret_t.select(["permno", "ret"]), on="permno", how="inner")
        )
        if pool.height < N_PORTFOLIOS:
            continue

        scores = dict(zip(pool["permno"].to_list(), pool["score"].to_list()))
        caps = dict(zip(pool["permno"].to_list(), pool["mcap"].to_list()))
        rets = dict(zip(pool["permno"].to_list(), pool["ret"].to_list()))

        nyse_set = nyse_by_ym.get(ym, set())
        bp_scores = [s for p, s in scores.items() if p in nyse_set]
        if len(bp_scores) < N_PORTFOLIOS:
            continue

        bps = _calc_breakpoints(bp_scores, N_PORTFOLIOS)
        assigns = _assign_portfolios(scores, bps)
        long_m = [p for p, g in assigns.items() if g == N_PORTFOLIOS]
        short_m = [p for p, g in assigns.items() if g == 1]
        if not long_m or not short_m:
            continue

        r_long = _value_weighted_return(long_m, rets, caps)
        r_short = _value_weighted_return(short_m, rets, caps)
        if np.isnan(r_long) or np.isnan(r_short):
            continue

        months.append(ym)
        long_sets.append(set(long_m))
        short_sets.append(set(short_m))
        ls_returns.append(r_long - r_short)

    return months, long_sets, short_sets, ls_returns


def build_table8(
    forecasts_by_label: dict[str, pl.DataFrame],
    returns: pl.DataFrame,
    mcap: pl.DataFrame,
    eligible: pl.DataFrame,
    exchanges: pl.DataFrame,
    fit_indices: list[int],
) -> pl.DataFrame:
    """构建 Table 8 矩阵（月均超额收益 % 与 NW t）。

    论文规则：每个 MLER^{t1,t2} 列在收益子期间内计算 10−1 时，
    剔除该列模型训练区间 [t1,t2] 内的月份。
    """
    labels = [SUBPERIOD_FITS[i]["label"] for i in fit_indices]
    label_to_train = {
        SUBPERIOD_FITS[i]["label"]: (
            SUBPERIOD_FITS[i]["train_start"],
            SUBPERIOD_FITS[i]["train_end"],
        )
        for i in fit_indices
    }
    rows: list[dict] = []

    for ret_spec in RETURN_SUBPERIODS:
        ret_period = (ret_spec["start"], ret_spec["end"])
        row: dict = {"return_period": ret_spec["label"]}
        for col_label in labels:
            train_period = label_to_train[col_label]
            fc = forecasts_by_label[col_label]
            res = compute_decile_portfolios(
                forecasts=fc,
                returns=returns,
                mcap=mcap,
                test_period=ret_period,
                top_n=None,
                n_portfolios=N_PORTFOLIOS,
                eligible=eligible,
                use_nyse_breakpoints=True,
                exchanges=exchanges,
                exclude_periods=[train_period],
            )
            m = res["metrics"]["long_short"]
            if m["n_months"] == 0:
                row[f"{col_label}_mean_pct"] = float("nan")
                row[f"{col_label}_t"] = float("nan")
            else:
                row[f"{col_label}_mean_pct"] = m["mean_return"] * 100
                row[f"{col_label}_t"] = m["t_statistic"]
        rows.append(row)

    return pl.DataFrame(rows)


def _monthly_forecast_spearman(
    fc_a: pl.DataFrame,
    fc_b: pl.DataFrame,
    exclude_periods: list[tuple[str, str]],
) -> float:
    a = _add_ym(fc_a)
    b = _add_ym(fc_b)
    joined = a.join(
        b.rename({"score": "score_b"}),
        on=["date", "permno", "ym"],
        how="inner",
    )
    lo, hi = _period_to_ym_range(TEST_PERIOD)
    corrs: list[float] = []
    for sub in joined.partition_by("ym", maintain_order=True):
        ym = sub["ym"][0]
        if ym < lo or ym > hi:
            continue
        if any(_ym_in_period(ym, p) for p in exclude_periods):
            continue
        if sub.height < 10:
            continue
        rho, _ = spearmanr(sub["score"].to_numpy(), sub["score_b"].to_numpy())
        if rho is not None and not np.isnan(rho):
            corrs.append(float(rho))
    return float(np.mean(corrs)) if corrs else float("nan")


def build_table7(
    forecasts_by_label: dict[str, pl.DataFrame],
    returns: pl.DataFrame,
    mcap: pl.DataFrame,
    eligible: pl.DataFrame,
    exchanges: pl.DataFrame,
    fit_indices: list[int],
) -> dict[str, pl.DataFrame]:
    """Table 7：两两预测相关、共同持仓、10−1 收益相关。"""
    labels = [SUBPERIOD_FITS[i]["label"] for i in fit_indices]
    train_periods = {
        lab: (SUBPERIOD_FITS[i]["train_start"], SUBPERIOD_FITS[i]["train_end"])
        for i, lab in zip(fit_indices, labels)
    }

    holdings_cache: dict[str, tuple[list[int], list[set[str]], list[set[str]], list[float]]] = {}
    for lab in labels:
        holdings_cache[lab] = _monthly_long_short_sets(
            forecasts_by_label[lab],
            returns,
            mcap,
            eligible,
            exchanges,
            TEST_PERIOD,
            exclude_train_periods=[train_periods[lab]],
        )

    n = len(labels)
    fc_corr = np.full((n, n), np.nan)
    hold_corr = np.full((n, n), np.nan)
    ret_corr = np.full((n, n), np.nan)

    for i, lab_i in enumerate(labels):
        fc_corr[i, i] = 1.0
        hold_corr[i, i] = 1.0
        ret_corr[i, i] = 1.0
        period_i = train_periods[lab_i]
        months_i, long_i, short_i, ret_i = holdings_cache[lab_i]
        ym_to_idx = {ym: k for k, ym in enumerate(months_i)}

        for j in range(i + 1, n):
            lab_j = labels[j]
            period_j = train_periods[lab_j]
            exclude = [period_i, period_j]
            fc_corr[i, j] = fc_corr[j, i] = _monthly_forecast_spearman(
                forecasts_by_label[lab_i],
                forecasts_by_label[lab_j],
                exclude,
            )

            months_j, long_j, short_j, ret_j = holdings_cache[lab_j]
            common_yms = sorted(set(months_i) & set(months_j))
            hold_pcts: list[float] = []
            ret_a: list[float] = []
            ret_b: list[float] = []
            for ym in common_yms:
                if any(_ym_in_period(ym, p) for p in exclude):
                    continue
                ki, kj = ym_to_idx[ym], months_j.index(ym)
                same_dir = len(long_i[ki] & long_j[kj]) + len(short_i[ki] & short_j[kj])
                n_i = len(long_i[ki]) + len(short_i[ki])
                n_j = len(long_j[kj]) + len(short_j[kj])
                avg_n = (n_i + n_j) / 2
                if avg_n > 0:
                    hold_pcts.append(same_dir / avg_n)
                ret_a.append(ret_i[ki])
                ret_b.append(ret_j[kj])

            hold_corr[i, j] = hold_corr[j, i] = (
                float(np.mean(hold_pcts)) if hold_pcts else float("nan")
            )
            if len(ret_a) >= 3:
                r, _ = pearsonr(ret_a, ret_b)
                ret_corr[i, j] = ret_corr[j, i] = float(r)

    def _matrix_df(mat: np.ndarray, title: str) -> pl.DataFrame:
        data = {"row": labels}
        for j, lab in enumerate(labels):
            data[lab] = [float(mat[i, j]) for i in range(n)]
        return pl.DataFrame(data).with_columns(pl.lit(title).alias("metric"))

    return {
        "forecast_correlations": _matrix_df(fc_corr, "forecast_correlations"),
        "common_holdings": _matrix_df(hold_corr, "common_holdings"),
        "return_correlations": _matrix_df(ret_corr, "return_correlations"),
    }


def _print_table8(df: pl.DataFrame, fit_indices: list[int]) -> None:
    labels = [SUBPERIOD_FITS[i]["label"] for i in fit_indices]
    print("\n" + "=" * 72)
    print("  Table 8 — Subperiod-Based Forecasts and Returns (10−1, %/month)")
    print("=" * 72)
    header = f"{'Return Period':<16}" + "".join(f"{lab:>14}" for lab in labels)
    print(header)
    print("-" * len(header))
    for row in df.iter_rows(named=True):
        parts = [f"{row['return_period']:<16}"]
        for lab in labels:
            mu = row.get(f"{lab}_mean_pct")
            t = row.get(f"{lab}_t")
            if mu is None or (isinstance(mu, float) and np.isnan(mu)):
                parts.append(f"{'—':>14}")
            else:
                parts.append(f"{mu:>6.2f}({t:>4.2f})"[:14].rjust(14))
        print("".join(parts))
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="CRSP 论文 Table 8 稳定性检验")
    parser.add_argument("--run-dir", type=Path, default=None, help="Table 8 结果输出目录（默认同 --paper-run-dir）")
    parser.add_argument(
        "--paper-run-dir",
        type=Path,
        default=None,
        help="与 --use-expanding-windows 联用：加载 crsp_paper_train 扩展窗口模型",
    )
    parser.add_argument(
        "--use-expanding-windows",
        action="store_true",
        help="非论文设计：复用扩展窗口模型（需 --paper-run-dir）",
    )
    parser.add_argument("--skip-train", action="store_true", help="跳过子期间训练，读取 stability/subperiod_* 预测")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="强制重新训练/预测（忽略已有 subperiod_* 缓存）",
    )
    parser.add_argument(
        "--regenerate-forecasts",
        action="store_true",
        help="同 --regenerate；与 --use-expanding-windows 联用",
    )
    parser.add_argument("--n-ensemble", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--subperiods", default=None, help="子期间索引 0-5，逗号分隔")
    parser.add_argument("--table7", action="store_true", help="同时输出 Table 7 两两矩阵")
    args = parser.parse_args()

    _check_parquet()
    fit_indices = (
        list(range(len(SUBPERIOD_FITS)))
        if args.subperiods is None
        else _parse_indices(args.subperiods, len(SUBPERIOD_FITS))
    )

    paper_run_dir = args.paper_run_dir
    if paper_run_dir is not None:
        paper_run_dir = paper_run_dir.resolve()
        if not paper_run_dir.exists():
            raise FileNotFoundError(f"paper run 不存在: {paper_run_dir}")

    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
    elif paper_run_dir is not None:
        run_dir = paper_run_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULT_ROOT / f"run_{stamp}_table8_stability"
    run_dir.mkdir(parents=True, exist_ok=True)

    regenerate = args.regenerate or args.regenerate_forecasts
    use_paper_windows = args.use_expanding_windows
    if use_paper_windows and paper_run_dir is None:
        raise ValueError("--use-expanding-windows 需要同时指定 --paper-run-dir")

    print("\n" + "=" * 64)
    print("  CRSP Table 8 — 子期间 MLER 稳定性（论文 Section 5.1）")
    print("=" * 64)
    print(f"  子期间索引   : {fit_indices}")
    print(f"  测试期预测   : {TEST_PERIOD[0]} → {TEST_PERIOD[1]}")
    if use_paper_windows:
        print(f"  模式         : 扩展窗口快捷 ({paper_run_dir})")
    elif args.skip_train:
        print(f"  模式         : 仅回测（读取 subperiod_* 预测）")
    else:
        print(f"  模式         : 论文设计 — 各子期间独立重训")
        print(f"  集成次数     : {args.n_ensemble}")
        print(f"  训练样本     : 子样本内全月（fitting_months_only=False）")
    print(f"  设备         : {get_device(args.device)}")
    print(f"  输出目录     : {run_dir}")
    print("=" * 64)

    config = CBMConfig(
        model=ModelConfig(device=args.device, batch_size=args.batch_size),
        training=TrainingConfig(
            loss_function=LOSS_FUNCTION,
            weighting=SAMPLE_WEIGHTING,
            return_variable=RETURN_VARIABLE,
            n_ensemble=args.n_ensemble,
            random_seed=42,
        ),
    )

    engine = PortfolioEngine(config)
    print("\n加载 CRSP 与特征…")
    engine.load_data(
        source="crsp_local",
        universe=UNIVERSE,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    _prepare_excess_features(engine, paper_run_dir=paper_run_dir)
    features = engine.features

    if use_paper_windows:
        forecasts_by_label = load_forecasts_from_paper_run(
            paper_run_dir=paper_run_dir,  # type: ignore[arg-type]
            out_dir=run_dir,
            features=features,
            fit_indices=fit_indices,
            regenerate=regenerate,
        )
    elif not args.skip_train:
        models_dir = _subperiod_models_dir(run_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        trainer = PaperModelTrainer(
            architecture=ARCHITECTURE,
            model_config=config.model,
            training_config=config.training,
        )
        registry = ModelRegistry(path=str(models_dir))
        forecasts_by_label = train_subperiod_models(
            engine,
            run_dir,
            fit_indices,
            args.n_ensemble,
            trainer,
            registry,
            regenerate=regenerate,
        )
    else:
        forecasts_by_label = load_subperiod_forecasts(run_dir, fit_indices)

    t_start = dt_date(1963, 7, 1)
    data_start = panel_load_start(t_start)
    data_end = dt_date(2022, 12, 31)
    returns, mcap, eligible, exchanges = load_crsp_panel(data_start, data_end)

    from cbm.data.french_rf import apply_excess_returns, load_french_rf_monthly

    rf = load_french_rf_monthly()
    returns = apply_excess_returns(returns, rf, ym_col="ym")

    print("\n构建 Table 8…")
    table8 = build_table8(
        forecasts_by_label, returns, mcap, eligible, exchanges, fit_indices,
    )
    out8 = run_dir / "stability" / "table8_subperiod_returns.csv"
    out8.parent.mkdir(parents=True, exist_ok=True)
    table8.write_csv(out8)
    _print_table8(table8, fit_indices)

    meta = {
        "experiment": "table8_subperiod_stability",
        "created_at": datetime.now().isoformat(),
        "test_period": list(TEST_PERIOD),
        "subperiod_fits": [SUBPERIOD_FITS[i] for i in fit_indices],
        "return_subperiods": RETURN_SUBPERIODS,
        "portfolio": "table3_decile_nyse_breakpoint_value_weighted",
    }
    if use_paper_windows:
        meta["forecast_source"] = "expanding_window_models"
        meta["paper_run_dir"] = str(paper_run_dir)
        meta["window_train_end_to_label"] = WINDOW_TRAIN_END_TO_LABEL
        meta["note"] = (
            "Forecasts from crsp_paper expanding-window models applied over full test period; "
            "Table 8 columns aligned by train_end. Differs from paper subperiod-only re-fit."
        )
    else:
        meta["forecast_source"] = "subperiod_only_refit"
        meta["n_ensemble"] = args.n_ensemble
        meta["fitting_months_only"] = False
        meta["note"] = (
            "Paper Table 8: ML trained only on each subperiod; "
            "portfolio months overlapping column train period excluded."
        )

    if args.table7:
        print("\n构建 Table 7…")
        t7 = build_table7(
            forecasts_by_label, returns, mcap, eligible, exchanges, fit_indices,
        )
        t7_dir = run_dir / "stability"
        for name, df in t7.items():
            path = t7_dir / f"table7_{name}.csv"
            df.write_csv(path)
            print(f"  已写 {path}")
        meta["table7"] = list(t7.keys())

    (run_dir / "stability" / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nTable 8 CSV: {out8}")
    print(f"元数据: {run_dir / 'stability' / 'meta.json'}")


if __name__ == "__main__":
    main()
