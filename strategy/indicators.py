"""榜单展示和风险提示使用的辅助指标。"""
import pandas as pd


def calc_ma_values(df: pd.DataFrame) -> dict:
    """计算最新 MA5 和 MA20。"""
    result = {}
    if len(df) >= 5:
        result["ma5"] = round(float(df["close"].tail(5).mean()), 2)
    if len(df) >= 20:
        result["ma20"] = round(float(df["close"].tail(20).mean()), 2)
    return result


def detect_rsi_divergence(
    df: pd.DataFrame,
    period: int = 14,
    price_high_window: int = 20,
    rsi_peak_window: int = 10,
) -> dict:
    """检测价格创新高但 RSI 未确认的顶背离；只作风险提示。"""
    if len(df) < period + 2:
        return {"has_divergence": False}
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    relative_strength = gain / (loss + 1e-10)
    rsi = 100 - 100 / (1 + relative_strength)
    latest = float(rsi.iloc[-1])
    price_new_high = float(df["close"].iloc[-1]) >= float(df["close"].tail(price_high_window).max())
    rsi_peak = float(rsi.tail(rsi_peak_window).max())
    return {"has_divergence": bool(price_new_high and latest < rsi_peak * 0.95)}
