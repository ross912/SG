"""统一管理不参与因子评分的入榜资格规则。"""
import pandas as pd

from config import (
    LIQUIDITY_LOOKBACK_DAYS,
    MIN_AVG_DAILY_AMOUNT,
    MIN_VALID_TRADING_DAYS,
)


def is_st_name(name: str) -> bool:
    """识别名称前缀为 ST、*ST、SST 或 S*ST 的风险警示股票。"""
    normalized = "".join(str(name or "").upper().split())
    return normalized.startswith(("ST", "*ST", "SST", "S*ST"))


def has_sufficient_liquidity(frame: pd.DataFrame) -> bool:
    """以最近 20 个交易日的成交额和有效成交日判断流动性是否达标。"""
    if "amount" not in frame.columns or len(frame) < LIQUIDITY_LOOKBACK_DAYS:
        return False

    amounts = pd.to_numeric(
        frame["amount"].tail(LIQUIDITY_LOOKBACK_DAYS), errors="coerce",
    ).fillna(0.0)
    valid_days = int((amounts > 0).sum())
    average_daily_amount = float(amounts.mean())
    return (
        valid_days >= MIN_VALID_TRADING_DAYS
        and average_daily_amount >= MIN_AVG_DAILY_AMOUNT
    )
