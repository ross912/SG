"""
学术动量因子：Jegadeesh & Titman (1993) 12-1 月动量。
"""
import numpy as np
import pandas as pd
from strategy.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    经典学术动量：过去12个月（剔除最近1个月）的累计收益。
    参考：Jegadeesh & Titman (1993), Carhart (1997) 四因子模型。
    window=252 交易日（约12个月），skip=21 交易日（约1个月）。
    """

    def __init__(self, window: int = 252, skip: int = 21,
                 buy_threshold: float = 0.0, sell_threshold: float = -0.15):
        self.window = window
        self.skip = skip
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = pd.to_numeric(df["close"], errors="coerce")
        # t-skip 与 t-(window+skip) 的收益；shift 写法同时修复首个有效样本
        # 被多跳过一天的问题，并替代逐行 Python 循环。
        base = close.shift(self.window + self.skip)
        endpoint = close.shift(self.skip)
        raw_returns = endpoint.div(base.where(base > 0)).sub(1.0)
        df["momentum_ret"] = raw_returns
        df["signal"] = np.select(
            [raw_returns > self.buy_threshold, raw_returns < self.sell_threshold],
            [1, -1],
            default=0,
        ).astype(int)
        return df


class CandidateTrendFactors:
    """用于横截面选股研究的四类中期趋势候选因子。

    这里只计算单只股票能够独立得到的三个原始序列。行业中性残差必须在
    全市场截面完成后再计算，避免把行业中位数错误地当成单股时序指标。
    """

    def __init__(
        self,
        momentum_lookback: int = 252,
        skip: int = 21,
        high_window: int = 252,
        vol_window: int = 60,
        horizons: tuple[int, ...] = (20, 60, 120, 250),
    ):
        self.momentum_lookback = momentum_lookback
        self.skip = skip
        self.high_window = high_window
        self.vol_window = vol_window
        self.horizons = tuple(horizons)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        close = pd.to_numeric(output["close"], errors="coerce")

        # 标准 12-2 口径：从约 12 个月前到约 1 个月前，不把最近
        # 21 个交易日的短期反转混入信号。
        base = close.shift(self.momentum_lookback)
        endpoint = close.shift(self.skip)
        output["momentum_12_2"] = endpoint.div(base.where(base > 0)).sub(1.0)

        rolling_high = close.rolling(
            self.high_window,
            min_periods=max(2, int(self.high_window * 0.85)),
        ).max()
        output["high_52w"] = close.div(rolling_high.where(rolling_high > 0)).sub(1.0)

        daily_return = close.pct_change(fill_method=None)
        annual_vol = daily_return.rolling(
            self.vol_window,
            min_periods=min(
                self.vol_window,
                max(2, int(self.vol_window * 2 / 3)),
            ),
        ).std() * np.sqrt(252)
        components = []
        for horizon in self.horizons:
            horizon_vol = annual_vol * np.sqrt(horizon / 252)
            component = close.div(close.shift(horizon)).sub(1.0)
            component = component.div(horizon_vol.where(horizon_vol > 0)).clip(-5, 5)
            components.append(component)
        output["multi_horizon"] = pd.concat(components, axis=1).mean(
            axis=1, skipna=False,
        )
        return output


class RobustTrendFactors:
    """稳健趋势候选因子；通过独立样本外门槛前不进入生产评分。"""

    def __init__(
        self,
        momentum_window: int = 120,
        momentum_skip: int = 20,
        trend_window: int = 120,
        efficiency_window: int = 60,
        risk_window: int = 60,
        drawdown_window: int = 120,
        consistency_horizons: tuple[int, ...] = (20, 60, 120),
    ):
        self.momentum_window = momentum_window
        self.momentum_skip = momentum_skip
        self.trend_window = trend_window
        self.efficiency_window = efficiency_window
        self.risk_window = risk_window
        self.drawdown_window = drawdown_window
        self.consistency_horizons = tuple(consistency_horizons)

    @staticmethod
    def _rolling_trend_quality(
        close: pd.Series,
        window: int,
    ) -> pd.Series:
        """计算对数价格回归的年化斜率 × R²，避免逐窗口 Python 回调。"""
        log_price = np.log(close.where(close > 0))
        positions = pd.Series(
            np.arange(len(log_price), dtype=float),
            index=log_price.index,
        )
        count = log_price.rolling(window, min_periods=window).count()
        sum_y = log_price.rolling(window, min_periods=window).sum()
        sum_y2 = log_price.pow(2).rolling(window, min_periods=window).sum()
        sum_global_xy = (
            log_price.mul(positions)
            .rolling(window, min_periods=window)
            .sum()
        )
        start_position = positions - window + 1
        sum_xy = sum_global_xy - start_position * sum_y

        sum_x = window * (window - 1) / 2
        sum_x2 = window * (window - 1) * (2 * window - 1) / 6
        sxx = sum_x2 - sum_x * sum_x / window
        sxy = sum_xy - sum_x * sum_y / window
        syy = sum_y2 - sum_y.pow(2) / window
        slope = sxy / sxx
        r_squared = sxy.pow(2).div((sxx * syy).where(syy > 0)).clip(0, 1)
        quality = slope.mul(252).mul(r_squared)
        return quality.where(count.eq(window))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        close = pd.to_numeric(output["close"], errors="coerce")
        daily_return = close.pct_change(fill_method=None)

        endpoint = close.shift(self.momentum_skip)
        base = close.shift(self.momentum_window)
        output["medium_momentum"] = endpoint.div(
            base.where(base > 0),
        ).sub(1.0)

        output["trend_quality"] = self._rolling_trend_quality(
            close,
            self.trend_window,
        )

        path_length = close.diff().abs().rolling(
            self.efficiency_window,
            min_periods=self.efficiency_window,
        ).sum()
        net_move = close.sub(close.shift(self.efficiency_window))
        output["path_efficiency"] = net_move.div(
            path_length.where(path_length > 0),
        ).clip(-1, 1)

        downside_squared = daily_return.clip(upper=0).pow(2)
        downside_vol = downside_squared.rolling(
            self.risk_window,
            min_periods=self.risk_window,
        ).mean().pow(0.5).mul(np.sqrt(252))
        output["downside_risk_control"] = downside_vol.mul(-1)

        rolling_high = close.rolling(
            self.drawdown_window,
            min_periods=self.drawdown_window,
        ).max()
        output["drawdown_control"] = close.div(
            rolling_high.where(rolling_high > 0),
        ).sub(1.0)

        consistency_components = []
        for horizon in self.consistency_horizons:
            horizon_return = close.div(close.shift(horizon)).sub(1.0)
            scale = daily_return.rolling(
                max(20, min(horizon, self.risk_window)),
                min_periods=20,
            ).std().mul(np.sqrt(horizon))
            consistency_components.append(
                np.tanh(horizon_return.div(scale.where(scale > 0)))
            )
        output["multi_period_consistency"] = pd.concat(
            consistency_components,
            axis=1,
        ).mean(axis=1, skipna=False)
        return output


def detect_momentum_acceleration(df: pd.DataFrame,
                                  short_window: int = 5,
                                  mid_window: int = 20) -> dict:
    """
    检测动量加速度：5 日动量 vs 20 日动量的差值。

    正加速 = 趋势增强（利好趋势跟踪）
    负加速/减速 = 趋势衰竭风险（警告）

    返回:
        is_exhausted: bool  — 动量减速，趋势可能衰竭
        acceleration: float — 5d_ret - 20d_ret
        short_return: float — 5 日收益率
        mid_return: float   — 20 日收益率
        detail: str
    """
    closes = df["close"].values
    n = len(closes)
    needed = mid_window + 1

    if n < needed:
        return {"is_exhausted": False, "acceleration": 0.0,
                "short_return": 0.0, "mid_return": 0.0,
                "detail": "数据不足"}

    short_base = closes[-(short_window + 1)]
    mid_base = closes[-(mid_window + 1)]
    short_ret = (closes[-1] / short_base - 1) if short_base > 0 else 0.0
    mid_ret = (closes[-1] / mid_base - 1) if mid_base > 0 else 0.0
    acceleration = short_ret - mid_ret

    is_exhausted = acceleration < -0.03
    detail = ""
    if is_exhausted:
        detail = (f"动量衰竭: 5d={short_ret:.1%} 远弱于 20d={mid_ret:.1%}, "
                  f"加速度={acceleration:.1%}")
    elif acceleration > 0.05:
        detail = f"动量加速: 5d={short_ret:.1%} > 20d={mid_ret:.1%}"

    return {
        "is_exhausted": is_exhausted,
        "acceleration": round(acceleration, 4),
        "short_return": round(short_ret, 4),
        "mid_return": round(mid_ret, 4),
        "detail": detail,
    }
