"""
趋势跟踪策略：双均线+ADX、唐奇安通道突破。
"""
import pandas as pd
import numpy as np
from strategy.base import BaseStrategy, forward_fill_signal


class MaAdxStrategy(BaseStrategy):
    """
    双均线 + ADX 趋势过滤。

    - EMA(20) > EMA(60)：上升趋势
    - ADX(14) > 25：趋势有效
    - EMA(5) 上穿 EMA(10) 且在上升趋势中 → 买入信号
    - EMA(5) 下穿 EMA(10) 或 ADX 回落 → 卖出信号
    """

    def __init__(self, ema_short=5, ema_mid=20, ema_long=60,
                 adx_period=14, adx_threshold=25):
        self.ema_short = ema_short
        self.ema_mid = ema_mid
        self.ema_long = ema_long
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < self.ema_long + self.adx_period:
            df["signal"] = 0
            return df

        # 计算 EMA
        df["ema_s"] = df["close"].ewm(span=self.ema_short, adjust=False).mean()
        df["ema_m"] = df["close"].ewm(span=self.ema_mid, adjust=False).mean()
        df["ema_l"] = df["close"].ewm(span=self.ema_long, adjust=False).mean()

        # 趋势方向
        df["trend_up"] = df["ema_m"] > df["ema_l"]
        # 趋势转弱需要持续3日确认，避免单日噪音触发卖出
        df["trend_down_persistent"] = (~df["trend_up"]).rolling(3).min() == 1

        # ADX
        df = _add_adx(df, self.adx_period)
        df["trend_strong"] = df["adx"] > self.adx_threshold

        # EMA 交叉
        df["ema_cross_up"] = (df["ema_s"] > df["ema_m"]) & (df["ema_s"].shift(1) <= df["ema_m"].shift(1))
        df["ema_cross_down"] = (df["ema_s"] < df["ema_m"]) & (df["ema_s"].shift(1) >= df["ema_m"].shift(1))

        # 信号
        conditions = [
            df["ema_cross_up"] & df["trend_up"] & df["trend_strong"],
            df["ema_cross_down"] | df["trend_down_persistent"],
        ]
        choices = [1, -1]
        df["signal"] = np.select(conditions, choices, default=0)
        df["signal"] = forward_fill_signal(df["signal"].values)

        return df


class DonchianStrategy(BaseStrategy):
    """
    唐奇安通道突破策略。

    - 价格突破 N 日最高价 → 买入
    - 价格跌破 M 日最低价 → 卖出
    - ATR 动态止损
    """

    def __init__(self, entry_period=20, exit_period=10,
                 atr_period=14, atr_stop_multiple=2.0):
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < max(self.entry_period, self.exit_period) + self.atr_period:
            df["signal"] = 0
            return df

        # 通道
        df["upper"] = df["high"].rolling(self.entry_period).max()
        df["lower"] = df["low"].rolling(self.exit_period).min()

        # ATR
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1)),
            ),
        )
        df["atr"] = df["tr"].ewm(alpha=1/self.atr_period, adjust=False).mean()
        df["stop_price"] = df["close"] - self.atr_stop_multiple * df["atr"]

        # 突破信号
        df["breakout_up"] = df["close"] > df["upper"].shift(1)
        df["breakout_down"] = df["close"] < df["lower"].shift(1)

        conditions = [df["breakout_up"], df["breakout_down"]]
        choices = [1, -1]
        df["signal"] = np.select(conditions, choices, default=0)

        # 止损信号：价格跌破止损价
        df["stop_hit"] = df["close"] < df["stop_price"].shift(1)
        df["signal"] = forward_fill_signal(df["signal"].values, df["stop_hit"].values)

        return df


def _add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """给 DataFrame 添加 ADX 列。"""
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
    )

    atr = pd.Series(tr, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df["adx"] = dx.ewm(alpha=1/period, adjust=False).mean()
    return df
