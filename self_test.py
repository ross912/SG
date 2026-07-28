"""快速离线自检：架构、导入、五因子计算和四榜单。"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data.board_utils import is_main_board
from screening.ranking import rank
from screening.scanner import prepare_results_for_strategy, scan_stock


def _frame(rows: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 20 + np.cumsum(rng.normal(0.02, 0.2, rows))
    return pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=rows),
        "open": close - 0.1,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": rng.integers(100_000, 900_000, rows),
        "amount": rng.uniform(10_000_000, 90_000_000, rows),
        "turnover": 0.0,
        "pct_change": pd.Series(close).pct_change() * 100,
    })


def main() -> int:
    failures = []
    modules = [
        "config", "main", "dashboard", "data.fetcher", "data.storage",
        "screening.scanner", "screening.ranking", "screening.factor_eval",
        "feedback.recorder", "feedback.returns_tracker", "feedback.weight_optimizer",
        "services.deepseek", "services.news", "services.market_ai", "services.progress",
        "backtest.engine", "backtest.report",
    ]
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as error:
            failures.append(f"导入 {name}: {error}")

    root = Path(config.ROOT)
    removed = [
        root / "clustering", root / "data/sector_mapper.py",
        root / "screening/prescreener.py", root / "strategy/filters.py",
    ]
    failures.extend(f"应删除但仍存在: {path}" for path in removed if path.exists())

    if scan_stock("000001", "*ST离线样本", df=_frame()) is not None:
        failures.append("ST 样本未被排除")

    illiquid = _frame()
    illiquid.loc[illiquid.index[-20:], "amount"] = 1_000_000
    if scan_stock("000002", "低流动性离线样本", df=illiquid) is not None:
        failures.append("低流动性样本未被排除")

    result = scan_stock("000003", "普通离线样本", df=_frame())
    if result is None:
        failures.append("合成日线未生成扫描结果")
    else:
        if hasattr(result, "filter_pass"):
            failures.append("扫描结果仍含资格过滤字段")
        for strategy in config.STRATEGIES.values():
            prepared = prepare_results_for_strategy([result], strategy["directions"])
            if len(rank(prepared, max_pool=1)) != 1:
                failures.append(f"{strategy['label']} 未正常入榜")
    if not is_main_board("003001") or is_main_board("300001"):
        failures.append("主板代码范围不正确")

    if failures:
        print("自检失败:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("自检通过：核心模块、五因子、两策略、主板范围、ST 与流动性过滤均正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
