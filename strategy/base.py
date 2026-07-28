"""
策略基类：定义统一接口。
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


def forward_fill_signal(signals: np.ndarray,
                        stop_hit: np.ndarray | None = None) -> np.ndarray:
    """持仓维持：将买入(1)后直到卖出(-1)之间的0信号填充为1。

    Args:
        signals: 原始信号数组 (1=buy, 0=hold, -1=sell)
        stop_hit: 可选的止损数组，True处强制卖出

    Returns:
        填充后的信号数组
    """
    n = len(signals)
    result = signals.copy()
    in_position = False
    if stop_hit is not None:
        _stop = np.where(np.isnan(stop_hit.astype(float)), False, stop_hit).astype(bool)
    else:
        _stop = np.zeros(n, dtype=bool)
    for i in range(n):
        if result[i] == 1:
            in_position = True  # 新建仓，当日买入不受旧止损影响
        elif result[i] == -1 or (in_position and result[i] != 1 and _stop[i]):
            in_position = False
            result[i] = -1
        if in_position and result[i] == 0:
            result[i] = 1
    return result


class BaseStrategy(ABC):
    """所有策略的抽象基类。"""

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算策略指标，返回附加信号列的 DataFrame。"""
        ...
