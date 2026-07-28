"""多因子排名与榜单表格转换。"""
import copy

import pandas as pd

from config import ALL_MARKET_POOL_SIZE
from screening.scanner import ScanResult


def rank(
    results: list[ScanResult],
    max_pool: int = ALL_MARKET_POOL_SIZE,
    min_score: float | None = None,
) -> list[ScanResult]:
    """按综合得分降序取前 N 名；资格已在扫描阶段完成，涨跌停不作过滤。"""
    candidates = [
        copy.copy(result)
        for result in results
        if min_score is None or result.score >= min_score
    ]
    candidates.sort(key=lambda result: (-result.score, result.code))
    return candidates[:max_pool]


def to_dataframe(pool: list[ScanResult]) -> pd.DataFrame:
    """将选股结果转为适合保存和展示的表格。"""
    columns = [
        "排名", "代码", "名称", "所属概念", "综合得分", "均线信号", "突破信号",
        "RSRS信号", "动量信号", "风险警告", "最新价", "涨跌幅",
        "换手率", "MA5", "MA20",
    ]
    rows = []
    for rank_index, result in enumerate(pool, 1):
        warnings = []
        if result.rsi_divergence:
            warnings.append("RSI背离")
        if result.momentum_exhaustion:
            warnings.append("动量衰竭")

        rows.append({
            "排名": rank_index,
            "代码": result.code,
            "名称": result.name,
            "所属概念": "",
            "综合得分": round(result.score, 4),
            "均线信号": _signal_text(result.ma_adx),
            "突破信号": _signal_text(result.donchian),
            "RSRS信号": _signal_text(result.rsrs),
            "动量信号": _signal_text(result.momentum),
            "风险警告": "、".join(warnings),
            "最新价": result.close,
            "涨跌幅": f"{result.pct_change:.2f}%",
            "换手率": f"{result.turnover:.2f}%" if result.turnover > 0 else "",
            "MA5": result.ma5 or "",
            "MA20": result.ma20 or "",
        })
    return pd.DataFrame(rows, columns=columns)


def _signal_text(value: int) -> str:
    if value == 1:
        return "偏强"
    if value == -1:
        return "偏弱"
    return "中性"
