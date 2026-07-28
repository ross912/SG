"""四类中期趋势候选因子的月度样本外研究。

本脚本与生产榜单隔离。它使用月度调仓、次日开盘成交、约 20 个交易日
持有，并扣除完整换手时约 36bp 的佣金、印花税和滑点。权重只在训练集
和验证集内选择，2026 年数据只用于最终测试和是否允许投产的门槛判断。

用法：
    .venv/bin/python -m backtest.candidate_factor_research
    .venv/bin/python -m backtest.candidate_factor_research --max-codes 500
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BACKTEST_DIR, OUTPUT_DIR, STRATEGY_PARAMS
from data.storage import list_local_codes, load_kline
from strategy.momentum import CandidateTrendFactors

FACTOR_COLUMNS = (
    "momentum_12_2",
    "high_52w",
    "multi_horizon",
    "industry_residual_momentum",
)
INDUSTRY_CACHE = OUTPUT_DIR / "stock_industry_cache.json"
REPORT_JSON = BACKTEST_DIR / "candidate_factor_research.json"
REPORT_CSV = BACKTEST_DIR / "candidate_factor_research.csv"


@dataclass
class PeriodInput:
    date: pd.Timestamp
    factors: np.ndarray
    returns: np.ndarray
    codes: np.ndarray


def _load_industries() -> dict[str, str]:
    if not INDUSTRY_CACHE.exists():
        return {}
    try:
        payload = json.loads(INDUSTRY_CACHE.read_text(encoding="utf-8"))
        return {
            str(code).zfill(6): str(industry).strip()
            for code, industry in payload.items()
            if str(industry).strip()
        }
    except Exception:
        return {}


def _trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
    reference = load_kline("000001")
    if reference is None or reference.empty:
        raise RuntimeError("缺少 000001 日线，无法构造交易日历")
    dates = pd.DatetimeIndex(
        pd.to_datetime(reference["date"]).sort_values().unique()
    )
    return dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def build_period_inputs(
    codes: list[str],
    start: str = "2021-01-01",
    end: str = "2099-12-31",
    holding_days: int = 20,
    top_n: int = 20,
) -> tuple[list[str], list[PeriodInput], dict]:
    calendar = _trading_calendar(start, end)
    warmup = max(STRATEGY_PARAMS["candidate_trend"]["momentum_lookback"], 260)
    signal_dates = calendar[warmup::holding_days]
    industries = _load_industries()
    rows: list[pd.DataFrame] = []
    started = time.time()

    for index, code in enumerate(codes, 1):
        frame = load_kline(code)
        if frame is None or len(frame) < warmup:
            continue
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = (
            frame.sort_values("date")
            .drop_duplicates("date")
            .set_index("date")
            .reindex(calendar)
        )
        candidate = CandidateTrendFactors(
            **STRATEGY_PARAMS["candidate_trend"],
        ).calculate(frame)
        amount = pd.to_numeric(
            frame.get("amount", pd.Series(index=frame.index, dtype=float)),
            errors="coerce",
        )
        liquid = (
            amount.rolling(20, min_periods=20).mean().ge(10_000_000)
            & amount.gt(0).rolling(20, min_periods=20).sum().ge(15)
        )
        positions = calendar.get_indexer(signal_dates)
        valid = positions + holding_days < len(calendar)
        positions = positions[valid]
        selected_dates = signal_dates[valid]
        open_price = pd.to_numeric(frame["open"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        output = pd.DataFrame({
            "date": selected_dates,
            "code": code,
            "industry": industries.get(code, ""),
            "momentum_12_2": candidate["momentum_12_2"].iloc[
                positions
            ].to_numpy(),
            "high_52w": candidate["high_52w"].iloc[positions].to_numpy(),
            "multi_horizon": candidate["multi_horizon"].iloc[
                positions
            ].to_numpy(),
            "entry": open_price.iloc[positions + 1].to_numpy(),
            "exit": close.iloc[positions + holding_days].to_numpy(),
            "liquid": liquid.iloc[positions].fillna(False).to_numpy(),
        })
        output = output[
            output["liquid"]
            & output["entry"].gt(0)
            & output["exit"].notna()
        ].drop(columns="liquid")
        if not output.empty:
            rows.append(output)
        if index % 500 == 0:
            print(
                f"  {index}/{len(codes)}，耗时 {time.time() - started:.0f}s",
                flush=True,
            )

    if not rows:
        raise RuntimeError("没有可用于候选因子研究的数据")
    panel = pd.concat(rows, ignore_index=True).dropna(
        subset=["momentum_12_2", "high_52w", "multi_horizon", "entry", "exit"],
    )
    industry_coverage = float(panel["industry"].ne("").mean())
    active_factors = list(FACTOR_COLUMNS[:3])
    residual_status = "disabled"
    if industry_coverage >= 0.60:
        sizes = panel.groupby(["date", "industry"])["momentum_12_2"].transform(
            "size",
        )
        medians = panel.groupby(["date", "industry"])["momentum_12_2"].transform(
            "median",
        )
        panel["industry_residual_momentum"] = (
            panel["momentum_12_2"].sub(medians).where(sizes.ge(5))
        )
        active_factors.append("industry_residual_momentum")
        residual_status = "enabled"
    else:
        panel["industry_residual_momentum"] = np.nan

    panel["forward_return"] = panel["exit"].div(panel["entry"]).sub(1.0)
    for factor in active_factors:
        panel[factor] = panel.groupby("date")[factor].rank(
            method="average", pct=True,
        ) * 2.0 - 1.0

    periods = []
    for date, day in panel.groupby("date", sort=True):
        valid_day = day.dropna(subset=active_factors + ["forward_return"])
        if len(valid_day) < max(100, top_n * 2):
            continue
        periods.append(PeriodInput(
            date=pd.Timestamp(date),
            factors=valid_day[active_factors].to_numpy(dtype=float),
            returns=valid_day["forward_return"].to_numpy(dtype=float),
            codes=valid_day["code"].astype(str).to_numpy(),
        ))
    metadata = {
        "industry_coverage": round(industry_coverage, 4),
        "industry_residual_status": residual_status,
        "stocks": int(panel["code"].nunique()),
        "periods": len(periods),
        "holding_days": holding_days,
        "top_n": top_n,
    }
    return active_factors, periods, metadata


def evaluate_weights(
    periods: list[PeriodInput],
    weights: np.ndarray,
    start: str,
    end: str,
    *,
    periods_per_year: float = 12.5,
    full_turnover_cost: float = 0.0036,
    top_n: int = 20,
) -> dict | None:
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    previous: set[str] = set()
    returns, excess_returns, turnovers = [], [], []
    for period in periods:
        if not start_date <= period.date <= end_date:
            continue
        score = period.factors @ weights
        selected = np.argpartition(score, -top_n)[-top_n:]
        current = set(period.codes[selected])
        turnover = (
            1.0 - len(previous & current) / top_n if previous else 1.0
        )
        gross_return = float(period.returns[selected].mean())
        cost = turnover * full_turnover_cost
        returns.append(gross_return - cost)
        excess_returns.append(
            gross_return - float(period.returns.mean()) - cost
        )
        turnovers.append(turnover)
        previous = current
    if len(returns) < 5:
        return None

    def metrics(values: list[float]) -> dict:
        array = np.asarray(values, dtype=float)
        equity = np.cumprod(1.0 + array)
        peak = np.maximum.accumulate(equity)
        annual_return = float(
            equity[-1] ** (periods_per_year / len(array)) - 1.0
        )
        annual_vol = float(array.std(ddof=1) * np.sqrt(periods_per_year))
        sharpe = (
            float(array.mean() * periods_per_year / annual_vol)
            if annual_vol > 0 else 0.0
        )
        return {
            "annual_return": round(annual_return, 6),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(float(((equity - peak) / peak).min()), 6),
        }

    return {
        "absolute": metrics(returns),
        "excess": metrics(excess_returns),
        "average_turnover": round(float(np.mean(turnovers)), 6),
        "samples": len(returns),
    }


def _candidate_weights(count: int, samples: int = 1000) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    candidates = [np.ones(count) / count]
    candidates.extend(np.eye(count))
    minimum = 0.05
    for _ in range(samples):
        raw = rng.dirichlet(np.full(count, 1.5))
        candidates.append(minimum + (1.0 - minimum * count) * raw)
    return candidates


def search_weights(
    factor_names: list[str],
    periods: list[PeriodInput],
    top_n: int,
) -> tuple[np.ndarray, dict]:
    ranked = []
    for weights in _candidate_weights(len(factor_names)):
        train = evaluate_weights(
            periods, weights, "2022-01-01", "2024-12-31", top_n=top_n,
        )
        if not train:
            continue
        objective = (
            train["excess"]["sharpe"]
            + 0.25 * train["absolute"]["sharpe"]
            + 0.20 * train["excess"]["max_drawdown"]
        )
        ranked.append((objective, weights, train))
    ranked.sort(key=lambda item: item[0], reverse=True)

    validation_ranked = []
    for _objective, weights, train in ranked[:100]:
        validation = evaluate_weights(
            periods, weights, "2025-01-01", "2025-12-31", top_n=top_n,
        )
        if validation:
            objective = (
                validation["excess"]["sharpe"]
                + 0.25 * validation["absolute"]["sharpe"]
                + 0.20 * validation["excess"]["max_drawdown"]
            )
            validation_ranked.append((objective, weights, train, validation))
    validation_ranked.sort(key=lambda item: item[0], reverse=True)
    if not validation_ranked:
        raise RuntimeError("验证集样本不足，无法选择候选权重")

    stable_weights = np.median(
        np.stack([item[1] for item in validation_ranked[:15]]),
        axis=0,
    )
    stable_weights = stable_weights / stable_weights.sum()
    metrics = {
        "train": evaluate_weights(
            periods, stable_weights, "2022-01-01", "2024-12-31", top_n=top_n,
        ),
        "validation": evaluate_weights(
            periods, stable_weights, "2025-01-01", "2025-12-31", top_n=top_n,
        ),
        "test": evaluate_weights(
            periods, stable_weights, "2026-01-01", "2099-12-31", top_n=top_n,
        ),
    }
    return stable_weights, metrics


def passes_production_gate(metrics: dict) -> tuple[bool, list[str]]:
    reasons = []
    for section in ("train", "validation", "test"):
        result = metrics.get(section)
        if not result:
            reasons.append(f"{section} 样本不足")
            continue
        if result["absolute"]["annual_return"] <= 0:
            reasons.append(f"{section} 绝对年化收益不为正")
        if result["excess"]["sharpe"] <= 0:
            reasons.append(f"{section} 超额 Sharpe 不为正")
        if result["absolute"]["max_drawdown"] < -0.35:
            reasons.append(f"{section} 最大回撤超过 35%")
    return not reasons, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="中期趋势候选因子样本外研究")
    parser.add_argument("--max-codes", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--holding-days", type=int, default=20)
    args = parser.parse_args()

    codes = sorted(list_local_codes())
    if args.max_codes > 0:
        codes = codes[:args.max_codes]
    factors, periods, metadata = build_period_inputs(
        codes,
        holding_days=args.holding_days,
        top_n=args.top_n,
    )
    weights, metrics = search_weights(factors, periods, args.top_n)
    passed, reasons = passes_production_gate(metrics)
    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "monthly_walk_forward_rank_standardized",
        "factors": factors,
        "weights": {
            name: round(float(value), 6)
            for name, value in zip(factors, weights)
        },
        "metadata": metadata,
        "metrics": metrics,
        "production_gate": {
            "passed": passed,
            "reasons": reasons,
        },
    }
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = []
    for section, result in metrics.items():
        rows.append({
            "阶段": section,
            "绝对年化": result["absolute"]["annual_return"],
            "绝对Sharpe": result["absolute"]["sharpe"],
            "最大回撤": result["absolute"]["max_drawdown"],
            "超额年化": result["excess"]["annual_return"],
            "超额Sharpe": result["excess"]["sharpe"],
            "平均换手": result["average_turnover"],
            "样本数": result["samples"],
        })
    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
