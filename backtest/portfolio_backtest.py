"""
历史组合回测：在 2024-2026 年K线数据上运行扫描+排名流水线，
测试多组因子权重配置，比较累计收益。

用法:
    python -m backtest.portfolio_backtest              # 完整回测
    python -m backtest.portfolio_backtest --quick      # 快速模式（少股票、少日期）
    python -m backtest.portfolio_backtest --grid       # 网格搜索最优权重
"""
import argparse
import hashlib
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import BACKTEST_DIR, SCORE_WEIGHTS, STRATEGY_PARAMS
from data.storage import load_kline, list_local_codes

warnings.filterwarnings("ignore")

# ============================================================
#  配置
# ============================================================

FACTOR_NAMES = ["vol_price_score", "ma_adx", "donchian", "rsrs", "momentum"]
FACTOR_CONFIG_KEYS = ["volume_price", "ma_adx", "donchian", "rsrs", "momentum"]

DEFAULT_WEIGHTS = {
    "bull": {"volume_price": 0.28, "ma_adx": 0.26, "rsrs": 0.20, "donchian": 0.14, "momentum": 0.12},
    "sideways": {"volume_price": 0.32, "ma_adx": 0.18, "rsrs": 0.22, "donchian": 0.12, "momentum": 0.16},
    "bear": {"volume_price": 0.28, "ma_adx": 0.14, "rsrs": 0.28, "donchian": 0.12, "momentum": 0.18},
}

# 候选权重配置（config key 格式）
WEIGHT_CONFIGS: dict[str, dict[str, float]] = {
    "default_sideways": DEFAULT_WEIGHTS["sideways"],
    "default_bull": DEFAULT_WEIGHTS["bull"],
    "default_bear": DEFAULT_WEIGHTS["bear"],
    "equal": {"volume_price": 0.20, "ma_adx": 0.20, "rsrs": 0.20, "donchian": 0.20, "momentum": 0.20},
    "vol_price_heavy": {"volume_price": 0.50, "ma_adx": 0.15, "rsrs": 0.15, "donchian": 0.10, "momentum": 0.10},
    "trend_heavy": {"volume_price": 0.15, "ma_adx": 0.35, "rsrs": 0.15, "donchian": 0.25, "momentum": 0.10},
    "momentum_heavy": {"volume_price": 0.15, "ma_adx": 0.15, "rsrs": 0.25, "donchian": 0.10, "momentum": 0.35},
    "rsrs_focused": {"volume_price": 0.15, "ma_adx": 0.15, "rsrs": 0.40, "donchian": 0.15, "momentum": 0.15},
    "ic_optimized": {"volume_price": 0.3679, "ma_adx": 0.1985, "rsrs": 0.1832, "donchian": 0.0742, "momentum": 0.1762},
    "production_raw": dict(SCORE_WEIGHTS["sideways"]),
    "production_rank_standardized": dict(SCORE_WEIGHTS["sideways"]),
}
STANDARDIZED_CONFIGS = {"production_rank_standardized"}

FACTOR_CACHE_PATH = BACKTEST_DIR / "factor_cache.parquet"
FACTOR_CACHE_META_PATH = BACKTEST_DIR / "factor_cache.meta.json"
BACKTEST_PERIOD = ("2024-01-01", "2026-05-30")
TOP_N = 20
MIN_BARS = 200
WEEKLY_STEP = 5  # 每 N 个交易日回测一次


# ============================================================
#  因子预计算
# ============================================================

def _calc_volume_price_score(df: pd.DataFrame) -> pd.Series:
    """量价关系评分，返回与 df 对齐的 Series。"""
    close = df["close"].values.astype(float)
    volume = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(df))

    scores = np.full(len(df), 0.5)
    for i in range(20, len(df)):
        ret_5d = (close[i] / close[i - 5] - 1) if close[i - 5] > 0 else 0
        vol_5d = volume[i - 4:i + 1].mean()
        vol_20d = volume[i - 19:i + 1].mean()
        vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1
        price_score = 0.5 + min(max(ret_5d * 10, -0.5), 0.5)
        vol_score = min(vol_ratio / 2, 1.0)
        scores[i] = price_score * 0.6 + vol_score * 0.4
    return pd.Series(scores, index=df.index)


def _compute_ma_adx_raw(df: pd.DataFrame) -> pd.Series:
    """MA+ADX 连续强度值。"""
    from strategy.trend_following import MaAdxStrategy
    try:
        s = MaAdxStrategy(**STRATEGY_PARAMS["ma_adx"])
        result = s.calculate(df)
        ema_s = result["ema_s"].values.astype(float)
        ema_m = result["ema_m"].values.astype(float)
        adx = result["adx"].values.astype(float)
        raw = (ema_s / ema_m - 1) * np.minimum(adx / 25, 1.0) * 10
        return pd.Series(raw, index=df.index)
    except Exception:
        return pd.Series(0.0, index=df.index)


def _compute_donchian_raw(df: pd.DataFrame) -> pd.Series:
    """Donchian 通道位置。"""
    from strategy.trend_following import DonchianStrategy
    try:
        s = DonchianStrategy(**STRATEGY_PARAMS["donchian"])
        result = s.calculate(df)
        close = result["close"].values.astype(float)
        upper = result["upper"].values.astype(float)
        lower = result["lower"].values.astype(float)
        channel_w = upper - lower
        raw = np.where(channel_w > 0, (close - lower) / channel_w - 0.5, 0.0)
        return pd.Series(raw, index=df.index)
    except Exception:
        return pd.Series(0.0, index=df.index)


def _compute_rsrs_raw(df: pd.DataFrame) -> pd.Series:
    """RSRS 斜率 + 动量衰减。"""
    from strategy.rsrs import RsrsStrategy
    try:
        s = RsrsStrategy(**STRATEGY_PARAMS["rsrs"])
        result = s.calculate(df)
        rsrs_val = result["rsrs_corrected"].values.astype(float)
        rsrs_mom = result["rsrs_momentum"].values.astype(float)
        dampen = np.where(rsrs_mom > -0.1, 1.0,
                          np.where(rsrs_mom < -0.5, 0.7,
                                   1.0 - (np.abs(rsrs_mom) - 0.1) / 0.4 * 0.3))
        raw = rsrs_val * dampen
        return pd.Series(raw, index=df.index)
    except Exception:
        return pd.Series(0.0, index=df.index)


def _compute_momentum_raw(df: pd.DataFrame) -> pd.Series:
    """学术动量 (12-1月) 连续值。"""
    from strategy.momentum import MomentumStrategy
    try:
        s = MomentumStrategy(**STRATEGY_PARAMS["momentum"])
        result = s.calculate(df)
        raw = result["momentum_ret"].values.astype(float)
        return pd.Series(raw, index=df.index)
    except Exception:
        return pd.Series(0.0, index=df.index)


def precompute_factors(
    codes: list[str],
    start: str = "2023-06-01",
    end: str = "2026-05-30",
    force: bool = False,
) -> pd.DataFrame:
    """
    预计算所有股票的五个因子原始值，保存到 parquet。

    返回 DataFrame: [date, code, open, close, pct_change,
                     vol_price, ma_adx, rsrs, donchian, momentum]
    """
    code_hash = hashlib.sha256("\n".join(sorted(codes)).encode("utf-8")).hexdigest()
    expected_meta = {"code_hash": code_hash, "start": start, "end": end}
    if FACTOR_CACHE_PATH.exists() and FACTOR_CACHE_META_PATH.exists() and not force:
        try:
            with open(FACTOR_CACHE_META_PATH, "r", encoding="utf-8") as f:
                cache_meta = json.load(f)
            if all(cache_meta.get(k) == v for k, v in expected_meta.items()):
                print(f"[缓存] 加载已有因子数据: {FACTOR_CACHE_PATH}")
                df = pd.read_parquet(FACTOR_CACHE_PATH)
                df["date"] = pd.to_datetime(df["date"])
                print(f"  {df['code'].nunique()} 只股票, {df['date'].nunique()} 个交易日")
                return df
            print("[缓存] 股票池或日期范围已变化，重建因子缓存")
        except Exception:
            print("[缓存] 元数据损坏，重建因子缓存")

    print(f"[预计算] {len(codes)} 只股票的因子值...")
    print(f"  预计耗时: ~{len(codes) * 0.08 / 60:.0f} 分钟")

    all_rows = []
    t0 = time.time()
    failed = 0

    for i, code in enumerate(codes):
        df = load_kline(code)
        if df is None or len(df) < MIN_BARS:
            failed += 1
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        if len(df) < MIN_BARS:
            failed += 1
            continue

        close = df["close"].values.astype(float)
        open_price = df["open"].values.astype(float) if "open" in df.columns else close.copy()
        pct_change = df["pct_change"].values.astype(float) if "pct_change" in df.columns else np.full(len(df), np.nan)

        # 因子值
        vol_price = _calc_volume_price_score(df).values
        ma_adx = _compute_ma_adx_raw(df).values
        donchian = _compute_donchian_raw(df).values
        rsrs = _compute_rsrs_raw(df).values
        momentum = _compute_momentum_raw(df).values

        # 截断前 120 天（策略需要预热）
        warmup = 120
        for j in range(warmup, len(df)):
            all_rows.append({
                "date": df["date"].iloc[j],
                "code": code,
                "open": open_price[j],
                "close": close[j],
                "pct_change": pct_change[j] if not (np.isnan(pct_change[j])) else 0.0,
                # 因子值
                "vol_price": vol_price[j] if not np.isnan(vol_price[j]) else 0.5,
                "ma_adx": ma_adx[j] if not np.isnan(ma_adx[j]) else 0.0,
                "donchian": donchian[j] if not np.isnan(donchian[j]) else 0.0,
                "rsrs": rsrs[j] if not np.isnan(rsrs[j]) else 0.0,
                "momentum": momentum[j] if not np.isnan(momentum[j]) else 0.0,
            })

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(codes) - i - 1) / rate / 60
            print(f"  {i+1}/{len(codes)}  ({elapsed:.0f}s, 预计剩余 {remaining:.0f}min)")

    result = pd.DataFrame(all_rows)
    if result.empty:
        raise RuntimeError("没有可用于回测的因子数据，请先下载足够的 K 线历史")
    result["date"] = pd.to_datetime(result["date"])
    result.to_parquet(FACTOR_CACHE_PATH, index=False)
    with open(FACTOR_CACHE_META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            **expected_meta,
            "code_count": len(codes),
            "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"[预计算] 完成: {result['code'].nunique()} 只, {len(result)} 行, 耗时 {elapsed/60:.1f}min")
    print(f"  失败(数据不足): {failed}")
    return result


# ============================================================
#  市场状态检测
# ============================================================

def _detect_regime(df_000001: pd.DataFrame, date: pd.Timestamp) -> str:
    """根据上证指数 K 线判断当日市场状态。"""
    df = df_000001[df_000001["date"] <= date]
    if len(df) < 60:
        return "sideways"

    close = df["close"].values.astype(float)
    ma20 = pd.Series(close).rolling(20).mean().values
    ma60 = pd.Series(close).rolling(60).mean().values

    current = close[-1]
    ma20_val = ma20[-1]
    ma60_val = ma60[-1]
    ret_20d = (close[-1] / close[-20] - 1) if close[-20] > 0 else 0

    if current > ma20_val > ma60_val and ret_20d > 0.03:
        return "bull"
    elif current < ma20_val and ret_20d < -0.05:
        return "bear"
    return "sideways"


# ============================================================
#  回测引擎
# ============================================================

def _compute_scores(factor_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """计算综合得分。factor_df 需包含 vol_price, ma_adx, donchian, rsrs, momentum 列。"""
    score = np.zeros(len(factor_df))
    # config key → factor_df column name
    key_map = {
        "volume_price": "vol_price",
        "ma_adx": "ma_adx",
        "donchian": "donchian",
        "rsrs": "rsrs",
        "momentum": "momentum",
    }
    for config_key, weight in weights.items():
        col = key_map.get(config_key, config_key)
        if col in factor_df.columns:
            score += factor_df[col].fillna(0).values * weight
    return pd.Series(score, index=factor_df.index)


def _rank_standardize_day(factor_df: pd.DataFrame) -> pd.DataFrame:
    """按单日横截面把每个因子映射到 [-1, 1]，与生产候选算法保持一致。"""
    output = factor_df.copy()
    for column in ("vol_price", "ma_adx", "donchian", "rsrs", "momentum"):
        values = pd.to_numeric(output[column], errors="coerce")
        valid = values.notna()
        if valid.sum() < 2 or values[valid].nunique() < 2:
            output[column] = 0.0
            continue
        ranks = values[valid].rank(method="average", ascending=True)
        output[column] = 0.0
        output.loc[valid, column] = (
            (ranks - 1.0) / (len(ranks) - 1.0) * 2.0 - 1.0
        )
    return output


def run_backtest(
    factor_df: pd.DataFrame,
    weight_configs: dict[str, dict[str, float]],
    start_date: str = "2024-01-01",
    end_date: str = "2026-05-30",
    top_n: int = 20,
    weekly_step: int = 5,
    quick: bool = False,
    standardized_configs: set[str] | None = None,
) -> dict:
    """
    历史回测主循环。

    factor_df: [date, code, close, pct_change, vol_price, ma_adx, donchian, rsrs, momentum]
    返回: {config_name: {dates, cumulative_returns, metrics, ...}}
    """
    factor_df = factor_df.copy()
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    standardized_configs = standardized_configs or set()

    # 获取交易日历
    all_dates = [pd.Timestamp(d) for d in sorted(factor_df["date"].unique())]
    all_dates = [d for d in all_dates if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]
    if not all_dates:
        raise ValueError("指定日期范围内没有可回测数据")
    date_to_index = {d: i for i, d in enumerate(all_dates)}
    date_data = {
        pd.Timestamp(d): group.reset_index(drop=True)
        for d, group in factor_df.groupby("date", sort=False)
    }

    # 步进日期
    test_dates = all_dates[::weekly_step]
    if quick:
        test_dates = test_dates[:20]  # 快速模式只测 20 期

    print(f"\n[回测] {len(test_dates)} 个回测日期 (每 {weekly_step} 个交易日)")
    print(f"  日期范围: {test_dates[0].date()} ~ {test_dates[-1].date()}")
    print(f"  股票池: {factor_df['code'].nunique()} 只")
    print(f"  权重配置: {len(weight_configs)} 组")
    print(f"  每期选股: Top {top_n}")

    # 初始化结果
    results = {}
    for name in weight_configs:
        results[name] = {
            "dates": [],
            "picks": [],           # 每期选中的股票代码
            "daily_returns": [],   # 每期等权平均收益
            "cumulative_return": 1.0,
            "all_picks_returns": [],  # 所有选股的收益（用于统计）
        }

    t0 = time.time()
    for di, date in enumerate(test_dates):
        # 当天数据；不做资格或流动性筛选
        day_data = date_data.get(date, pd.DataFrame()).copy()
        if len(day_data) < top_n:
            continue
        standardized_day_data = (
            _rank_standardize_day(day_data)
            if standardized_configs else day_data
        )

        # 前向收益：找下一个交易日（或下N个）
        # T+1: 紧接的下一个交易日
        # T+5: 跳过 4 个的下一个
        next_idx = date_to_index[date] + 1
        if next_idx >= len(all_dates):
            continue
        fwd_indices = {
            horizon: next_idx + horizon - 1
            for horizon in (1, 5, 10, 20)
            if next_idx + horizon - 1 < len(all_dates)
        }

        # 信号在收盘后生成，统一按下一交易日开盘买入。
        entry_date = all_dates[next_idx]
        entry_source = date_data[entry_date]
        entry_col = "open" if "open" in entry_source.columns else "close"
        entry_prices = entry_source[["code", entry_col]].copy()
        entry_prices.columns = ["code", "entry_price"]

        # 获取前向价格
        fwd_prices = {}
        for horizon, idx in fwd_indices.items():
            fwd_date = all_dates[idx]
            fwd_data = date_data[fwd_date][["code", "close"]].copy()
            fwd_data.columns = ["code", f"close_{horizon}d"]
            fwd_prices[horizon] = fwd_data

        for config_name, weights in weight_configs.items():
            day_data_copy = (
                standardized_day_data.copy()
                if config_name in standardized_configs
                else day_data.copy()
            )
            day_data_copy["score"] = _compute_scores(day_data_copy, weights)

            # 取 top-N
            top = day_data_copy.nlargest(top_n, "score")

            # 计算各持有期的等权收益
            period_rets = {}
            for horizon, fwd_df in fwd_prices.items():
                merged = (
                    top[["code"]]
                    .merge(entry_prices, on="code", how="left")
                    .merge(fwd_df, on="code", how="left")
                )
                merged["ret"] = merged[f"close_{horizon}d"] / merged["entry_price"] - 1
                valid = merged["ret"].dropna()
                if len(valid) > 0:
                    period_rets[horizon] = float(valid.mean())

            # 只记录完整 T+5 持有期，避免样本尾部被 min() 截成 1~4 天。
            if 5 not in period_rets:
                continue
            main_ret = period_rets[5]

            results[config_name]["dates"].append(date)
            results[config_name]["daily_returns"].append(main_ret)
            results[config_name]["picks"].append(top["code"].tolist())
            if period_rets.get(1) is not None:
                one_day = (
                    top[["code"]]
                    .merge(entry_prices, on="code", how="left")
                    .merge(fwd_prices[1], on="code", how="left")
                )
                results[config_name]["all_picks_returns"].extend(
                    (one_day["close_1d"] / one_day["entry_price"] - 1).dropna().tolist()
                )

        if (di + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (di + 1) / elapsed
            remaining = (len(test_dates) - di - 1) / rate / 60
            print(f"  {di+1}/{len(test_dates)}  ({elapsed:.0f}s, 预计剩余 {remaining:.0f}min)")

    elapsed = time.time() - t0
    print(f"[回测] 完成, 耗时 {elapsed/60:.1f}min")

    # 计算累计收益
    for name in results:
        rets = results[name]["daily_returns"]
        if rets:
            cum = 1.0
            cum_series = [1.0]
            for r in rets:
                cum *= (1 + r)
                cum_series.append(cum)
            results[name]["cumulative_return"] = cum
            results[name]["cum_series"] = cum_series[1:]  # 跳过初始 1.0

    return results


# ============================================================
#  指标计算
# ============================================================

def compute_metrics(results: dict) -> pd.DataFrame:
    """计算各组配置的绩效指标。"""
    rows = []
    for name, data in results.items():
        rets = np.array(data["daily_returns"])
        if len(rets) < 10:
            continue

        n_periods = len(rets)
        win_rate = float((rets > 0).mean())

        # 年化（假设每期 ~1周，年化约50期）
        periods_per_year = 50
        total_return = data["cumulative_return"] - 1
        annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

        # 波动率
        annual_vol = float(np.std(rets, ddof=1) * np.sqrt(periods_per_year))

        # Sharpe (假设无风险利率 2%)
        rf = 0.02
        sharpe = (annual_return - rf) / annual_vol if annual_vol > 0 else 0

        # Max drawdown
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = float(dd.min())

        # Calmar
        calmar = annual_return / abs(max_dd) if max_dd < 0 else 0

        # Sortino (只考虑下行波动)
        down_rets = rets[rets < 0]
        down_vol = float(np.std(down_rets, ddof=1) * np.sqrt(periods_per_year)) if len(down_rets) > 1 else annual_vol
        sortino = (annual_return - rf) / down_vol if down_vol > 0 else 0

        # 平均收益
        avg_ret = float(np.mean(rets))
        picks = data.get("picks", [])
        retention = [
            len(set(previous) & set(current)) / max(len(previous), 1)
            for previous, current in zip(picks, picks[1:])
        ]
        average_retention = float(np.mean(retention)) if retention else 0.0

        rows.append({
            "配置": name,
            "累计收益": f"{total_return*100:+.1f}%",
            "年化收益": f"{annual_return*100:+.1f}%",
            "年化波动": f"{annual_vol*100:.1f}%",
            "Sharpe": round(sharpe, 2),
            "Sortino": round(sortino, 2),
            "最大回撤": f"{max_dd*100:.1f}%",
            "Calmar": round(calmar, 2),
            "胜率": f"{win_rate*100:.0f}%",
            "平均收益": f"{avg_ret*100:+.2f}%",
            "平均保留率": f"{average_retention*100:.1f}%",
            "样本数": n_periods,
            "_total_return": total_return,
            "_sharpe": sharpe,
            "_sortino": sortino,
            "_max_dd": max_dd,
        })

    df = pd.DataFrame(rows)
    return df


# ============================================================
#  网格搜索
# ============================================================

def grid_search_weights(
    factor_df: pd.DataFrame,
    start_date: str = "2024-06-01",
    end_date: str = "2026-05-30",
    top_n: int = 20,
    weekly_step: int = 5,
    n_samples: int = 500,
) -> pd.DataFrame:
    """
    随机采样权重空间，搜索最优组合。

    约束: 所有权重 ≥ 0.05, sum = 1.0
    """
    print(f"\n[网格搜索] 随机采样 {n_samples} 组权重...")

    np.random.seed(42)
    factor_df = factor_df.copy()
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    all_dates = [pd.Timestamp(d) for d in sorted(factor_df["date"].unique())]
    test_dates = [d for d in all_dates[::weekly_step]
                  if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]
    date_to_index = {d: i for i, d in enumerate(all_dates)}
    date_data = {
        pd.Timestamp(d): group.reset_index(drop=True)
        for d, group in factor_df.groupby("date", sort=False)
    }

    # 与正式回测使用相同的“次日开盘买入、第五日收盘卖出”口径。
    # 每个日期的因子矩阵/前向收益只构造一次，500 组权重只做矩阵乘法。
    period_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    factor_cols = ["vol_price", "ma_adx", "rsrs", "donchian", "momentum"]
    for date in test_dates:
        day_data = date_data.get(date, pd.DataFrame())
        if len(day_data) < top_n * 2:
            continue
        next_idx = date_to_index[date] + 1
        exit_idx = next_idx + 4
        if exit_idx >= len(all_dates):
            continue
        entry_source = date_data[all_dates[next_idx]]
        entry_col = "open" if "open" in entry_source.columns else "close"
        entry = entry_source[["code", entry_col]].copy()
        entry.columns = ["code", "entry_price"]
        exit_price = date_data[all_dates[exit_idx]][["code", "close"]].copy()
        exit_price.columns = ["code", "exit_price"]
        merged = day_data.merge(entry, on="code", how="inner").merge(
            exit_price, on="code", how="inner"
        )
        merged = merged[(merged["entry_price"] > 0) & merged["exit_price"].notna()]
        if len(merged) < top_n:
            continue
        matrix = merged[factor_cols].fillna(0).to_numpy(dtype=float)
        forward_returns = (merged["exit_price"] / merged["entry_price"] - 1).to_numpy(dtype=float)
        period_inputs.append((matrix, forward_returns))

    if not period_inputs:
        raise ValueError("没有满足过滤条件且具备完整 T+5 价格的回测样本")

    best_sharpe = -999
    best_weights = None
    best_return = 0
    results_list = []

    for si in range(n_samples):
        # 生成随机权重 (Dirichlet 分布)
        raw = np.random.exponential(1, 5)
        # 每个因子至少 5%，剩余 75% 按 Dirichlet 权重分配。
        w = 0.05 + 0.75 * raw / raw.sum()
        weights = {
            "volume_price": round(w[0], 4),
            "ma_adx": round(w[1], 4),
            "rsrs": round(w[2], 4),
            "donchian": round(w[3], 4),
            "momentum": round(w[4], 4),
        }

        period_rets = []
        weight_vector = np.array([
            weights["volume_price"], weights["ma_adx"], weights["rsrs"],
            weights["donchian"], weights["momentum"],
        ])
        for matrix, forward_returns in period_inputs:
            score = matrix @ weight_vector
            top_idx = np.argpartition(score, -top_n)[-top_n:]
            period_rets.append(float(np.mean(forward_returns[top_idx])))

        if len(period_rets) < 10:
            continue

        rets_arr = np.array(period_rets)
        annual_ret = (1 + np.mean(rets_arr)) ** 50 - 1
        annual_vol = float(np.std(rets_arr, ddof=1) * np.sqrt(50))
        sharpe = (annual_ret - 0.02) / annual_vol if annual_vol > 0 else 0
        total_ret = float(np.prod(1 + rets_arr) - 1)
        equity = np.cumprod(1 + rets_arr)
        peak = np.maximum.accumulate(equity)
        max_dd = float(np.min((equity - peak) / peak))

        results_list.append({
            **weights,
            "sharpe": round(sharpe, 3),
            "total_return": round(total_ret * 100, 1),
            "max_dd": round(max_dd * 100, 1),
            "n_periods": len(period_rets),
        })

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = dict(weights)
            best_return = total_ret

        if (si + 1) % 100 == 0:
            print(f"  {si+1}/{n_samples}  best Sharpe={best_sharpe:.2f}")

    if best_weights is None:
        return pd.DataFrame(results_list)

    print(f"\n  最优权重 (Sharpe={best_sharpe:.2f}, 收益={best_return*100:+.1f}%):")
    for k, v in best_weights.items():
        print(f"    {k}: {v:.4f}")

    results_df = pd.DataFrame(results_list).sort_values("sharpe", ascending=False)
    return results_df


# ============================================================
#  报告输出
# ============================================================

def print_comparison(metrics_df: pd.DataFrame) -> None:
    """打印配置对比表。"""
    display_cols = [
        "配置", "累计收益", "年化收益", "Sharpe", "Sortino",
        "最大回撤", "Calmar", "胜率", "平均保留率", "样本数",
    ]
    available = [c for c in display_cols if c in metrics_df.columns]
    print(f"\n{'='*100}")
    print("  因子权重配置对比 (T+5 持有期, Top 20)")
    print(f"{'='*100}")
    print(metrics_df[available].to_string(index=False))


def plot_comparison(results: dict, output_path: Path | None = None) -> None:
    """画累计收益曲线对比图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib 未安装，跳过绘图")
        return

    plt.figure(figsize=(14, 8))

    # 按最终累计收益排序
    sorted_configs = sorted(results.items(), key=lambda x: x[1].get("cumulative_return", 0), reverse=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_configs)))
    for (name, data), color in zip(sorted_configs, colors):
        if "cum_series" in data and len(data["cum_series"]) > 0:
            cum = np.array(data["cum_series"])
            plt.plot(range(len(cum)), cum, label=f"{name} ({data['cumulative_return']-1:+.1%})",
                     color=color, linewidth=1.5)

    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("回测期数 (每周一期)")
    plt.ylabel("累计净值")
    plt.title("因子权重配置回测对比 — Top 20, T+5 持有期")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path is None:
        output_path = BACKTEST_DIR / "weight_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n  图表已保存: {output_path}")


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="历史组合回测")
    parser.add_argument("--quick", action="store_true", help="快速模式")
    parser.add_argument("--grid", action="store_true", help="网格搜索最优权重")
    parser.add_argument("--start", type=str, default=BACKTEST_PERIOD[0])
    parser.add_argument("--end", type=str, default=BACKTEST_PERIOD[1])
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--step", type=int, default=WEEKLY_STEP)
    parser.add_argument("--force-cache", action="store_true", help="强制重建因子缓存")
    args = parser.parse_args()

    # 获取股票池
    all_codes = sorted(list_local_codes())
    print(f"本地K线: {len(all_codes)} 只")

    # 因子预计算
    factor_df = precompute_factors(all_codes, force=args.force_cache)

    if args.grid:
        # 网格搜索
        grid_results = grid_search_weights(
            factor_df,
            start_date=args.start,
            end_date=args.end,
            top_n=args.top_n,
            weekly_step=args.step,
            n_samples=500,
        )
        grid_path = BACKTEST_DIR / "grid_search_results.csv"
        grid_results.to_csv(grid_path, index=False, encoding="utf-8-sig")
        print(f"\n  网格搜索结果: {grid_path}")
        print("\n  Top 10 权重组合:")
        print(grid_results.head(10).to_string(index=False))

    # 运行回测
    results = run_backtest(
        factor_df,
        WEIGHT_CONFIGS,
        start_date=args.start,
        end_date=args.end,
        top_n=args.top_n,
        weekly_step=args.step,
        quick=args.quick,
        standardized_configs=STANDARDIZED_CONFIGS,
    )

    # 计算指标
    metrics = compute_metrics(results)

    # 按 Sharpe 排序
    if "_sharpe" in metrics.columns:
        metrics = metrics.sort_values("_sharpe", ascending=False)

    print_comparison(metrics)

    # 保存详细结果
    metrics_path = BACKTEST_DIR / "weight_comparison.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\n  指标已保存: {metrics_path}")

    # 画图
    plot_comparison(results)

    # 推荐最佳权重
    best_row = metrics.iloc[0]
    print(f"\n{'='*60}")
    print(f"  推荐: {best_row['配置']}")
    print(f"  年化收益: {best_row['年化收益']}")
    print(f"  Sharpe: {best_row['Sharpe']}")
    print(f"  最大回撤: {best_row['最大回撤']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
