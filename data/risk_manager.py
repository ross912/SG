"""依据大盘评分给出组合总仓位参考。"""
from config import POSITION_SIZING_MAP


def calc_total_position_ratio(regime_score: float) -> dict:
    """把 0~100 的市场评分线性映射到配置的总仓位区间。"""
    minimum = POSITION_SIZING_MAP["min_ratio"]
    maximum = POSITION_SIZING_MAP["max_ratio"]
    score = min(max(float(regime_score), 0.0), 100.0)
    ratio = round(minimum + score / 100 * (maximum - minimum), 2)
    if ratio > 0.70:
        label = "重仓"
    elif ratio > 0.40:
        label = "中等"
    elif ratio > 0.20:
        label = "轻仓"
    else:
        label = "极轻"
    return {
        "ratio": ratio,
        "ratio_pct": round(ratio * 100, 1),
        "label": label,
        "regime_score": score,
    }
