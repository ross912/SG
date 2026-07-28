"""
RSRS（阻力支撑相对强度）策略 — 增强版。
参考：光大证券研报 + QuantsPlaybook RSRS 择时复现。

v2.0 增强：
- 右偏修正：RSRS_z × corr(returns, rsrs_beta)
- 自适应阈值：基于滚动百分位动态调整 buy/sell 阈值
- RSRS 动量：RSRS 自身的变化率作为辅助确认
"""
import pandas as pd
import numpy as np
from strategy.base import BaseStrategy, forward_fill_signal


class RsrsStrategy(BaseStrategy):
    """
    RSRS 斜率策略 — 增强版。

    参数：
        window: 滚动回归窗口（默认 18 日）
        z_score_window: 标准化窗口（默认 600 日）
        buy_threshold: 买入阈值（默认 0.7，牛市中可降低）
        sell_threshold: 卖出阈值（默认 -0.7）
        adaptive_threshold: 是否使用自适应阈值（基于滚动百分位）
    """

    def __init__(self, window=18, z_score_window=600,
                 buy_threshold=0.7, sell_threshold=-0.7,
                 adaptive_threshold=True):
        self.window = window
        self.z_score_window = z_score_window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.adaptive_threshold = adaptive_threshold

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        if n < self.window + 2:
            df["signal"] = 0
            return df

        # 1. 滚动 OLS 斜率
        beta_series = _rolling_beta(df["low"].values, df["high"].values, self.window)
        df["rsrs_beta"] = beta_series

        # 2. Z-score 标准化：每一行只使用当时及之前的数据。
        # 旧实现会在样本不足 600 日时用“整段数据”的均值和标准差，导致历史信号
        # 看到未来。扩展窗口 + 最大 600 日窗口既保留预热能力，也消除未来函数。
        min_periods = max(60, self.window * 3)
        roll_mean = df["rsrs_beta"].rolling(
            self.z_score_window, min_periods=min_periods
        ).mean()
        roll_std = df["rsrs_beta"].rolling(
            self.z_score_window, min_periods=min_periods
        ).std()
        df["rsrs_z"] = (df["rsrs_beta"] - roll_mean) / (roll_std + 1e-10)

        # 3. 右偏修正 v2：使用多周期相关性加权
        df["ret"] = df["close"].pct_change()
        if n >= self.window:
            # 短期相关性（快速反应）
            corr_short = df["ret"].rolling(max(self.window // 3, 5)).corr(df["rsrs_beta"])
            # 中期相关性（稳定性）
            corr_mid = df["ret"].rolling(self.window).corr(df["rsrs_beta"])
            # 加权平均：中期70% + 短期30%
            df["rsrs_corr"] = corr_mid.fillna(0) * 0.7 + corr_short.fillna(0) * 0.3
            df["rsrs_corrected"] = df["rsrs_z"] * (1 + df["rsrs_corr"].clip(-1, 1))
        else:
            df["rsrs_corrected"] = df["rsrs_z"]

        # 4. RSRS 动量（RSRS 自身的短期变化方向）
        df["rsrs_momentum"] = df["rsrs_corrected"].diff(3)

        # 5. 确定阈值
        buy_thresh = pd.Series(self.buy_threshold, index=df.index, dtype=float)
        sell_thresh = pd.Series(self.sell_threshold, index=df.index, dtype=float)

        if self.adaptive_threshold:
            # 阈值必须是逐行序列。旧实现取 iloc[-1] 后把“今天的阈值”应用到整段
            # 历史信号，回测会产生未来数据泄漏。
            buy_adaptive = (
                df["rsrs_corrected"]
                .rolling(120, min_periods=60)
                .quantile(0.70)
                .clip(0.3, 1.5)
            )
            sell_adaptive = (
                df["rsrs_corrected"]
                .rolling(120, min_periods=60)
                .quantile(0.30)
                .clip(-1.5, -0.3)
            )
            buy_thresh = buy_adaptive.fillna(self.buy_threshold)
            sell_thresh = sell_adaptive.fillna(self.sell_threshold)

        # 6. 信号生成
        val = df["rsrs_corrected"]
        mom = df["rsrs_momentum"]
        conditions = [
            (val > buy_thresh) & (mom > -0.1),   # 买入：RSRS强势 + 动量不跌
            (val < sell_thresh) | (mom < -0.5),  # 卖出：RSRS弱势 或 动量快速恶化
        ]
        choices = [1, -1]
        df["signal"] = np.select(conditions, choices, default=0)
        df["signal"] = forward_fill_signal(df["signal"].values)

        return df

def _rolling_beta(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """
    滚动窗口 OLS 斜率计算。
    y = alpha + beta * x 中的 beta。

    使用在线算法避免每窗口重复计算。
    """
    n = len(x)
    beta = np.full(n, np.nan)

    if n < window:
        return beta

    x_mean = pd.Series(x).rolling(window).mean().values
    y_mean = pd.Series(y).rolling(window).mean().values
    xy_mean = pd.Series(x * y).rolling(window).mean().values
    x2_mean = pd.Series(x * x).rolling(window).mean().values

    denominator = x2_mean - x_mean * x_mean
    valid = denominator > 1e-12
    beta[valid] = (xy_mean[valid] - x_mean[valid] * y_mean[valid]) / denominator[valid]

    return beta
