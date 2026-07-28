"""全市场技术指标扫描：并行计算生产五因子、候选因子和风险提示。"""
import copy
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from config import (
    STRATEGY_PARAMS, SCORE_WEIGHTS, FACTOR_DIRECTION,
)
from data.storage import ensure_kline
from screening.eligibility import has_sufficient_liquidity, is_st_name
from strategy.trend_following import MaAdxStrategy, DonchianStrategy
from strategy.rsrs import RsrsStrategy
from strategy.indicators import calc_ma_values, detect_rsi_divergence
from strategy.momentum import (
    CandidateTrendFactors,
    MomentumStrategy,
    detect_momentum_acceleration,
)

_logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    code: str
    name: str = ""
    industry: str = ""
    # 离散信号（展示用）
    ma_adx: int = 0
    donchian: int = 0
    rsrs: int = 0
    momentum: int = 0
    # 连续强度值（评分用）
    ma_adx_raw: float = 0.0
    donchian_raw: float = 0.0
    rsrs_raw: float = 0.0
    momentum_raw: float = 0.0
    # 新趋势候选因子（研究/反馈用；通过样本外门槛前不进入生产综合分）
    momentum_12_2_raw: float = 0.0
    high_52w_raw: float = 0.0
    multi_horizon_raw: float = 0.0
    industry_residual_momentum_raw: float = 0.0
    score: float = 0.0
    vol_price_score: float = 0.0
    rsi_divergence: bool = False
    momentum_exhaustion: bool = False
    close: float = 0.0
    as_of_date: str = ""
    pct_change: float = 0.0
    turnover: float = 0.0
    ma5: float = 0.0
    ma20: float = 0.0


# 会话级内存缓存：避免同一次扫描中重复下载同一只股票
# LRU 容量上限 300 只，避免全市场扫描时内存无限增长
MAX_KLINE_CACHE = 300
SCAN_LOOKBACK_BARS = 700  # 最新截面计算只需覆盖 RSRS 600 日窗口及预热区间
_kline_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
_kline_lock: "threading.Lock | None" = None

# 当前市场状态（由 index_filter 模块设置）
_market_regime: str = "bull"


def _get_kline_lock() -> "threading.Lock":
    global _kline_lock
    if _kline_lock is None:
        _kline_lock = threading.Lock()
    return _kline_lock


def set_market_regime(regime: str) -> None:
    """设置当前市场状态：bull / sideways / bear。"""
    global _market_regime
    if regime in ("bull", "sideways", "bear"):
        _market_regime = regime


def _get_active_weights() -> dict[str, float]:
    """获取当前市场状态对应的五因子权重。"""
    return SCORE_WEIGHTS.get(_market_regime, SCORE_WEIGHTS["bull"])


def recalc_scores(results: list[ScanResult],
                  weights: dict[str, float] | None = None,
                  directions: dict[str, int] | None = None) -> list[ScanResult]:
    """用指定权重和方向重新计算纯五因子综合得分。"""
    w = weights or _get_active_weights()
    dirs = directions or FACTOR_DIRECTION

    for r in results:
        raw = (
            r.vol_price_score * w["volume_price"] * dirs["volume_price"] +
            r.ma_adx_raw * w["ma_adx"] * dirs["ma_adx"] +
            r.rsrs_raw * w["rsrs"] * dirs["rsrs"] +
            r.donchian_raw * w["donchian"] * dirs["donchian"] +
            r.momentum_raw * w["momentum"] * dirs["momentum"]
        )

        r.score = round(raw, 3) if not (np.isnan(raw) or np.isinf(raw)) else 0.0

    return results


def prepare_results_for_strategy(
    results: list[ScanResult],
    directions: dict[str, int],
    weights: dict[str, float] | None = None,
) -> list[ScanResult]:
    """复制一次因子扫描结果，并套用指定策略的方向和权重。"""
    prepared = [copy.copy(r) for r in results]
    return recalc_scores(prepared, weights=weights, directions=directions)


def scan_stock(code: str, name: str = "", industry: str = "",
               params: dict | None = None,
               df: pd.DataFrame | None = None) -> ScanResult | None:
    """
    扫描单只股票，返回 ScanResult 或 None（数据不足等）。
    K线数据优先级：传入df > 内存缓存 > 本地文件 > 云端实时下载。
    """
    if is_st_name(name):
        return None

    if df is None:
        with _get_kline_lock():
            df = _kline_cache.get(code)
            if df is not None:
                _kline_cache.move_to_end(code)
    if df is None:
        df = ensure_kline(code)  # 缺失→全量下载，过期→增量更新，最新→直接返回
        if df is not None:
            with _get_kline_lock():
                _kline_cache[code] = df
                while len(_kline_cache) > MAX_KLINE_CACHE:
                    _kline_cache.popitem(last=False)
    if df is None or len(df) < 120:
        return None
    if not has_sufficient_liquidity(df):
        return None

    result = ScanResult(code=code, name=name, industry=str(industry or "").strip())
    latest_date = pd.to_datetime(df["date"].iloc[-1]) if "date" in df.columns else None
    result.as_of_date = latest_date.strftime("%Y-%m-%d") if latest_date is not None else ""

    # 基础数据
    result.close = float(df["close"].iloc[-1])
    raw_pct = float(df["pct_change"].iloc[-1]) if "pct_change" in df.columns else float('nan')
    if pd.isna(raw_pct) and len(df) >= 2 and df["close"].iloc[-2] > 0:
        result.pct_change = round(float((df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100), 2)
    else:
        result.pct_change = raw_pct if not pd.isna(raw_pct) else 0.0

    # 换手率：akshare 返回 0-1 比例（0.24=24%），转为百分比
    if "turnover" in df.columns:
        raw_turnover = float(df["turnover"].iloc[-1])
        result.turnover = round(raw_turnover * 100, 2) if not pd.isna(raw_turnover) else 0.0

    # 日度选股只读取最新截面；限制计算窗口可显著减少多次 DataFrame copy/rolling。
    df_calc = df.tail(SCAN_LOOKBACK_BARS).copy() if len(df) > SCAN_LOOKBACK_BARS else df

    # === 策略信号 ===
    p = params or STRATEGY_PARAMS

    try:
        ma = MaAdxStrategy(**p["ma_adx"])
        df_ma = ma.calculate(df_calc)
        result.ma_adx = int(df_ma["signal"].iloc[-1]) if not df_ma.empty else 0
        ema_s = float(df_ma["ema_s"].iloc[-1])
        ema_m = float(df_ma["ema_m"].iloc[-1])
        adx_v = float(df_ma["adx"].iloc[-1])
        raw = (ema_s / ema_m - 1) * min(adx_v / 25, 1.0) * 10
        result.ma_adx_raw = raw if not (np.isnan(raw) or np.isinf(raw)) else 0.0
    except Exception as e:
        result.ma_adx = 0
        _logger.debug("(%s) ma_adx 计算失败: %s", code, e)

    try:
        dc = DonchianStrategy(**p["donchian"])
        df_dc = dc.calculate(df_calc)
        result.donchian = int(df_dc["signal"].iloc[-1]) if not df_dc.empty else 0
        close_v = float(df_dc["close"].iloc[-1])
        upper_v = float(df_dc["upper"].iloc[-1])
        lower_v = float(df_dc["lower"].iloc[-1])
        channel_w = upper_v - lower_v
        raw = (close_v - lower_v) / channel_w - 0.5 if channel_w > 0 else 0.0
        result.donchian_raw = raw if not (np.isnan(raw) or np.isinf(raw)) else 0.0
    except Exception as e:
        result.donchian = 0
        _logger.debug("(%s) donchian 计算失败: %s", code, e)

    try:
        rs = RsrsStrategy(**p["rsrs"])
        df_rs = rs.calculate(df_calc)
        result.rsrs = int(df_rs["signal"].iloc[-1]) if not df_rs.empty else 0
        rsrs_val = float(df_rs["rsrs_corrected"].iloc[-1])
        rsrs_mom = float(df_rs["rsrs_momentum"].iloc[-1])
        # 动量衰减折扣：水平×变化率，捕捉"支撑在弱化"的二阶信号
        if rsrs_mom > -0.1:
            dampen = 1.0
        elif rsrs_mom < -0.5:
            dampen = 0.7
        else:
            dampen = 1.0 - (abs(rsrs_mom) - 0.1) / 0.4 * 0.3
        raw = rsrs_val * dampen
        result.rsrs_raw = raw if not (np.isnan(raw) or np.isinf(raw)) else 0.0
    except Exception as e:
        result.rsrs = 0
        _logger.debug("(%s) rsrs 计算失败: %s", code, e)

    try:
        mom = MomentumStrategy(**p.get("momentum", {"window": 252, "skip": 21}))
        df_mom = mom.calculate(df_calc)
        result.momentum = int(df_mom["signal"].iloc[-1]) if not df_mom.empty else 0
        raw = float(df_mom["momentum_ret"].iloc[-1])
        result.momentum_raw = raw if not (np.isnan(raw) or np.isinf(raw)) else 0.0
    except Exception as e:
        result.momentum = 0
        _logger.debug("(%s) momentum 计算失败: %s", code, e)

    try:
        candidate = CandidateTrendFactors(**p.get("candidate_trend", {}))
        candidate_frame = candidate.calculate(df_calc)
        candidate_values = {
            "momentum_12_2_raw": candidate_frame["momentum_12_2"].iloc[-1],
            "high_52w_raw": candidate_frame["high_52w"].iloc[-1],
            "multi_horizon_raw": candidate_frame["multi_horizon"].iloc[-1],
        }
        for attribute, value in candidate_values.items():
            numeric = float(value)
            setattr(
                result,
                attribute,
                numeric if np.isfinite(numeric) else 0.0,
            )
    except Exception as e:
        _logger.debug("(%s) candidate_trend 计算失败: %s", code, e)

    # MA 值
    try:
        ma_vals = calc_ma_values(df_calc)
        result.ma5 = ma_vals.get("ma5", 0)
        result.ma20 = ma_vals.get("ma20", 0)
    except Exception:
        pass

    # 量价关系评分（优先级最高：量价>波动率>其他）
    vol_price_score = _calc_volume_price_score(df_calc)
    result.vol_price_score = vol_price_score

    # === 风险预警（不参与评分）===
    try:
        div_result = detect_rsi_divergence(df_calc)
        result.rsi_divergence = div_result.get("has_divergence", False)
    except Exception:
        result.rsi_divergence = False

    try:
        accel_result = detect_momentum_acceleration(df_calc)
        result.momentum_exhaustion = accel_result.get("is_exhausted", False)
    except Exception:
        result.momentum_exhaustion = False

    # 综合打分：使用 config 中的权重体系 + 因子方向
    weights = _get_active_weights()
    dirs = FACTOR_DIRECTION

    # 五因子加权：使用连续强度值，量价用原始 [0,1] 值，其余用 raw 字段
    # 乘以方向: +1=正向(高分好), -1=反向(低分好，均值回归)
    raw_score = (
        vol_price_score * weights["volume_price"] * dirs["volume_price"] +
        result.ma_adx_raw * weights["ma_adx"] * dirs["ma_adx"] +
        result.rsrs_raw * weights["rsrs"] * dirs["rsrs"] +
        result.donchian_raw * weights["donchian"] * dirs["donchian"] +
        result.momentum_raw * weights["momentum"] * dirs["momentum"]
    )

    result.score = round(raw_score, 3) if not (np.isnan(raw_score) or np.isinf(raw_score)) else 0.0
    return result


def _calc_volume_price_score(df: pd.DataFrame) -> float:
    """量价关系评分：放量上涨为佳，缩量下跌为弱。返回连续值 0~1。"""
    if len(df) < 20 or "volume" not in df.columns:
        return 0.5
    close = df["close"].values
    volume = df["volume"].values

    ret_5d = (close[-1] / close[-6] - 1) if close[-6] > 0 else 0
    vol_5d = volume[-5:].mean()
    vol_20d = volume[-20:].mean()
    vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1

    price_score = 0.5 + min(max(ret_5d * 10, -0.5), 0.5)
    vol_score = min(vol_ratio / 2, 1.0)
    return price_score * 0.6 + vol_score * 0.4


def scan_all(codes: list[str] | set[str],
             name_map: dict[str, str] | None = None,
             industry_map: dict[str, str] | None = None,
             max_workers: int = 12,
             progress_callback: Callable[[int, int, int], None] | None = None,
             ) -> list[ScanResult]:
    """扫描给定股票代码集合，返回技术指标结果。"""
    results = _parallel_scan(
        sorted(set(codes)), name_map, industry_map, max_workers, progress_callback,
    )
    return apply_industry_neutral_residual_momentum(results)


def _parallel_scan(codes: list[str], name_map: dict[str, str] | None,
                   industry_map: dict[str, str] | None,
                   max_workers: int,
                   progress_callback: Callable[[int, int, int], None] | None = None,
                   ) -> list[ScanResult]:
    """并行执行日线更新和因子计算，避免全市场数据重复读取或同时驻留内存。"""
    results: list[ScanResult] = []

    def _scan_one(code):
        name = name_map.get(code, "") if name_map else ""
        industry = industry_map.get(code, "") if industry_map else ""
        return scan_stock(code, name, industry)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_scan_one, code): code for code in codes}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                code = futures[future]
                print(f"  扫描 {code} 异常: {e}")
            if index % 200 == 0 or index == len(futures):
                print(f"  {index}/{len(futures)}  指标可用:{len(results)}")
            if progress_callback and (index % 50 == 0 or index == len(futures)):
                try:
                    progress_callback(index, len(futures), len(results))
                except Exception as error:
                    _logger.debug("进度回调失败: %s", error)

    return results


def apply_industry_neutral_residual_momentum(
    results: list[ScanResult],
    min_industry_size: int = 5,
    min_coverage: float = 0.60,
) -> list[ScanResult]:
    """以行业截面中位数为基准生成残差动量。

    行业覆盖不足时将该因子置零，而不是把全市场去均值伪装成行业中性。
    """
    if not results:
        return results
    covered = [result for result in results if result.industry]
    if len(covered) / len(results) < min_coverage:
        for result in results:
            result.industry_residual_momentum_raw = 0.0
        return results

    frame = pd.DataFrame({
        "industry": [result.industry for result in results],
        "momentum": [result.momentum_12_2_raw for result in results],
    })
    counts = frame.groupby("industry")["momentum"].transform("size")
    medians = frame.groupby("industry")["momentum"].transform("median")
    valid = frame["industry"].ne("") & counts.ge(min_industry_size)
    residual = frame["momentum"].sub(medians).where(valid, 0.0)
    for result, value in zip(results, residual):
        result.industry_residual_momentum_raw = (
            round(float(value), 8) if np.isfinite(value) else 0.0
        )
    return results
