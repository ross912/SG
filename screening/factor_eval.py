"""生产与候选因子的横截面评估；动态权重仍只作用于生产五因子。"""

import copy

import numpy as np
import pandas as pd

from screening.scanner import ScanResult


_FACTOR_ATTRIBUTES = (
    "vol_price_score", "ma_adx_raw", "rsrs_raw",
    "donchian_raw", "momentum_raw",
    "momentum_12_2_raw", "high_52w_raw", "multi_horizon_raw",
    "industry_residual_momentum_raw",
)


def cross_sectional_standardize_factors(
    results: list[ScanResult],
) -> list[ScanResult]:
    """将生产与候选因子分别映射到 [-1, 1] 横截面百分位。"""
    standardized = [copy.copy(result) for result in results]
    if len(standardized) < 2:
        return standardized
    for attribute in _FACTOR_ATTRIBUTES:
        values = pd.Series(
            [getattr(result, attribute) for result in standardized],
            dtype=float,
        ).replace([np.inf, -np.inf], np.nan)
        valid = values.notna()
        if valid.sum() < 2 or values[valid].nunique() < 2:
            normalized = pd.Series(0.0, index=values.index)
        else:
            ranks = values[valid].rank(method="average", ascending=True)
            scaled = (ranks - 1.0) / (len(ranks) - 1.0) * 2.0 - 1.0
            normalized = pd.Series(0.0, index=values.index)
            normalized.loc[valid] = scaled
        for result, value in zip(standardized, normalized):
            setattr(result, attribute, round(float(value), 6))
    return standardized


def cross_sectional_normalize(results: list[ScanResult]) -> list[ScanResult]:
    """将综合得分映射为横截面百分位；所有扫描成功的股票都参与。"""
    if len(results) < 2:
        return results
    scores = pd.Series([result.score for result in results], dtype=float)
    ranks = scores.rank(method="average", ascending=True)
    normalized = (ranks - 1.0) / (len(scores) - 1.0)
    for result, value in zip(results, normalized):
        result.score = round(float(value), 4)
    return results


def compute_factor_dispersion(results: list[ScanResult]) -> dict[str, float]:
    """计算五个因子的横截面分散度，返回 0~1 的区分力指标。"""
    if len(results) < 5:
        return {}
    factors = {
        "volume_price": [r.vol_price_score for r in results],
        "ma_adx": [r.ma_adx_raw for r in results],
        "rsrs": [r.rsrs_raw for r in results],
        "donchian": [r.donchian_raw for r in results],
        "momentum": [r.momentum_raw for r in results],
    }
    return {
        name: round(min(float(np.nanstd(values)) / 0.3, 1.0), 4)
        for name, values in factors.items()
    }


def adjust_weights_by_dispersion(
    base_weights: dict[str, float],
    dispersion: dict[str, float],
    min_weight: float = 0.02,
) -> dict[str, float]:
    """降低低区分力因子的权重，并把五因子权重重新归一化。"""
    adjusted = dict(base_weights)
    for factor in ("volume_price", "ma_adx", "rsrs", "donchian", "momentum"):
        if factor not in adjusted or factor not in dispersion:
            continue
        value = dispersion[factor]
        multiplier = 0.3 if value < 0.2 else 0.7 if value < 0.4 else 1.0 if value < 0.6 else 1.2
        adjusted[factor] = max(min_weight, adjusted[factor] * multiplier)
    return _normalize_weights(adjusted)

def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return dict(weights)
    normalized = {name: round(value / total, 4) for name, value in weights.items()}
    largest = max(normalized, key=normalized.get)
    normalized[largest] = round(normalized[largest] + 1.0 - sum(normalized.values()), 4)
    return normalized
