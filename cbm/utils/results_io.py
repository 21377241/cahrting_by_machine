"""将单次训练与回测产物写入 ``result`` 目录（JSON + Parquet + 文本摘要）。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import polars as pl
from loguru import logger

if TYPE_CHECKING:
    from cbm.core.engine import PortfolioEngine
    from cbm.core.types import Forecast, PerformanceMetrics, PortfolioSet


def _json_safe(obj: Any) -> Any:
    """将 numpy 标量、嵌套结构转为 JSON 可序列化类型。"""
    if obj is None:
        return None
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)):
        return obj
    return str(obj)


def _metrics_to_dict(m: "PerformanceMetrics") -> Dict[str, Any]:
    d = asdict(m)
    return _json_safe(d)


def save_pipeline_results(
    *,
    engine: "PortfolioEngine",
    model_id: str,
    performance: Dict[str, "PerformanceMetrics"],
    forecasts: Optional["Forecast"] = None,
    portfolios: Optional["PortfolioSet"] = None,
    cumulative_returns: Optional[pl.DataFrame] = None,
    result_dir: Union[str, Path] = "result",
    diagnostics: Optional[Dict[str, Any]] = None,
    experiment_tag: Optional[str] = None,
) -> Path:
    """
    在 ``result_dir/run_<时间戳>_<model_id 短前缀>/`` 下写入训练与回测结果。

    写入内容
    ----------
    - ``meta.json``：模型 ID、时间、数据 ``metadata``（若有）
    - ``training_metrics.json``、``training_config.json``
    - ``performance.json``：各组合绩效字段
    - ``summary.txt``：各组合 ``summary()`` 文本
    - ``forecasts.parquet``：预测宽表（若提供 ``forecasts``）
    - ``portfolio_<name>_returns.parquet``：各组合月度收益（若提供 ``portfolios``）
    - ``cumulative_returns.parquet``：净值累积曲线表（若提供 ``cumulative_returns``）
    - ``diagnostics.json``：可选结构化诊断（由调用方传入 ``diagnostics``）
    - ``meta.json`` 中可含 ``experiment_tag`` 字段

    Returns
    -------
    Path
        本次运行目录路径。
    """
    base = Path(result_dir)
    safe_id = model_id.replace(os.sep, "_").replace("/", "_").replace(":", "_")[:80]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{stamp}_{safe_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_entry = engine._models.get(model_id, {})  # noqa: SLF001
    metrics = model_entry.get("metrics", {})
    t_cfg = model_entry.get("config", {})

    meta = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": model_id,
    }
    if experiment_tag:
        meta["experiment_tag"] = experiment_tag
    if engine.data is not None and getattr(engine.data, "metadata", None):
        meta["data_metadata"] = _json_safe(dict(engine.data.metadata))

    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "training_metrics.json").write_text(
        json.dumps(_json_safe(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "training_config.json").write_text(
        json.dumps(_json_safe(t_cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    perf_out = {name: _metrics_to_dict(m) for name, m in performance.items()}
    (run_dir / "performance.json").write_text(
        json.dumps(perf_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [f"model_id: {model_id}", f"run_dir: {run_dir}", ""]
    for name, m in performance.items():
        lines.append(f"=== Portfolio {name} ===")
        lines.append(m.summary())
        lines.append("")
    (run_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    if diagnostics:
        (run_dir / "diagnostics.json").write_text(
            json.dumps(_json_safe(diagnostics), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if forecasts is not None and forecasts.values.height > 0:
        path_fc = run_dir / "forecasts.parquet"
        forecasts.values.write_parquet(path_fc)
        logger.info(f"已保存预测: {path_fc}")

    if portfolios is not None:
        for name, port in portfolios.portfolios.items():
            p = run_dir / f"portfolio_{name}_returns.parquet"
            port.returns.write_parquet(p)
        if portfolios.long_short is not None:
            p = run_dir / "portfolio_long_short_returns.parquet"
            portfolios.long_short.returns.write_parquet(p)
        logger.info(f"已保存组合收益序列至 {run_dir}")

    if cumulative_returns is not None and cumulative_returns.height > 0:
        p = run_dir / "cumulative_returns.parquet"
        cumulative_returns.write_parquet(p)
        logger.info(f"已保存 cumulative_returns: {p}")

    logger.info(f"训练与回测结果已写入: {run_dir.resolve()}")
    return run_dir
