"""稳健趋势因子统一审计与滚动样本外研究。

本脚本只研究趋势榜，不修改均值回归模型。信号在调仓日收盘后生成，
次一交易日开盘买入，约 20 个交易日后收盘卖出，并按实际换手扣除
佣金、印花税和滑点。2026 年数据只用于最终测试，不参与因子筛选和
权重选择。

用法：
    .venv/bin/python -m backtest.robust_trend_research
    .venv/bin/python -m backtest.robust_trend_research --max-codes 500
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BACKTEST_DIR, OUTPUT_DIR
from data.storage import list_local_codes, load_kline
from screening.eligibility import is_st_name
from strategy.momentum import RobustTrendFactors

FACTOR_COLUMNS = (
    "medium_momentum",
    "trend_quality",
    "multi_period_consistency",
    "path_efficiency",
    "downside_risk_control",
    "drawdown_control",
)
RISK_FACTORS = {"downside_risk_control", "drawdown_control"}
NAMES_CACHE = OUTPUT_DIR / "stock_names_cache.json"
REPORT_JSON = BACKTEST_DIR / "robust_trend_research.json"
REPORT_CSV = BACKTEST_DIR / "robust_trend_factor_audit.csv"
PERIOD_CSV = BACKTEST_DIR / "robust_trend_period_returns.csv"


@dataclass
class PeriodInput:
    date: pd.Timestamp
    factors: np.ndarray
    returns: np.ndarray
    codes: np.ndarray
    eligible: np.ndarray


def _load_names() -> dict[str, str]:
    if not NAMES_CACHE.exists():
        return {}
    try:
        payload = json.loads(NAMES_CACHE.read_text(encoding="utf-8"))
        return {
            str(code).zfill(6): str(name)
            for code, name in payload.items()
        }
    except Exception:
        return {}


def _trading_calendar() -> pd.DatetimeIndex:
    reference = load_kline("000001")
    if reference is None or reference.empty:
        raise RuntimeError("缺少 000001 日线，无法构造交易日历")
    return pd.DatetimeIndex(
        pd.to_datetime(reference["date"]).sort_values().unique()
    )


def _price_limit(code: str) -> float:
    if code.startswith(("300", "301", "688")):
        return 0.20
    if code.startswith(("8", "9")):
        return 0.30
    return 0.10


def build_panel(
    codes: list[str],
    *,
    holding_days: int = 20,
) -> tuple[pd.DataFrame, dict]:
    calendar = _trading_calendar()
    research_calendar = calendar[
        (calendar >= pd.Timestamp("2021-01-01"))
        & (calendar <= pd.Timestamp("2099-12-31"))
    ]
    signal_calendar = research_calendar[
        research_calendar >= pd.Timestamp("2022-01-01")
    ][::holding_days]
    positions = research_calendar.get_indexer(signal_calendar)
    valid_signal = positions + holding_days < len(research_calendar)
    positions = positions[valid_signal]
    signal_calendar = signal_calendar[valid_signal]
    names = _load_names()
    factors = RobustTrendFactors()
    frames: list[pd.DataFrame] = []
    started = time.time()
    skipped_st = 0

    for index, code in enumerate(codes, 1):
        if is_st_name(names.get(code, "")):
            skipped_st += 1
            continue
        frame = load_kline(code)
        if frame is None or len(frame) < 260:
            continue
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = (
            frame.sort_values("date")
            .drop_duplicates("date")
            .set_index("date")
            .reindex(research_calendar)
        )
        calculated = factors.calculate(frame)
        close = pd.to_numeric(frame["close"], errors="coerce")
        open_price = pd.to_numeric(frame["open"], errors="coerce")
        amount = pd.to_numeric(
            frame.get("amount", pd.Series(index=frame.index, dtype=float)),
            errors="coerce",
        )
        liquid = (
            amount.rolling(20, min_periods=20).mean().ge(10_000_000)
            & amount.gt(0).rolling(20, min_periods=20).sum().ge(15)
        )
        entry_return = open_price.shift(-1).div(close).sub(1.0)
        can_buy = entry_return.lt(_price_limit(code) - 0.005)
        above_ma120 = close.gt(close.rolling(120, min_periods=120).mean())
        positive_trend = close.div(close.shift(60)).sub(1.0).gt(0)

        selected = pd.DataFrame({
            "date": signal_calendar,
            "code": code,
            **{
                factor: calculated[factor].iloc[positions].to_numpy()
                for factor in FACTOR_COLUMNS
            },
            "entry": open_price.iloc[positions + 1].to_numpy(),
            "exit": close.iloc[positions + holding_days].to_numpy(),
            "liquid": liquid.iloc[positions].fillna(False).to_numpy(),
            "can_buy": can_buy.iloc[positions].fillna(False).to_numpy(),
            "eligible": (
                above_ma120 & positive_trend
            ).iloc[positions].fillna(False).to_numpy(),
        })
        selected = selected[
            selected["liquid"]
            & selected["can_buy"]
            & selected["entry"].gt(0)
            & selected["exit"].gt(0)
        ]
        if not selected.empty:
            frames.append(selected.drop(columns=["liquid", "can_buy"]))
        if index % 500 == 0:
            print(
                f"  {index}/{len(codes)}，面板构建耗时 {time.time() - started:.0f}s",
                flush=True,
            )

    if not frames:
        raise RuntimeError("没有可用于稳健趋势研究的数据")
    panel = pd.concat(frames, ignore_index=True)
    panel["forward_return"] = panel["exit"].div(panel["entry"]).sub(1.0)
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(FACTOR_COLUMNS) + ["forward_return"],
    )
    for factor in FACTOR_COLUMNS:
        panel[factor] = panel.groupby("date")[factor].rank(
            method="average",
            pct=True,
        ) * 2.0 - 1.0
    metadata = {
        "stocks": int(panel["code"].nunique()),
        "periods": int(panel["date"].nunique()),
        "rows": len(panel),
        "holding_days": holding_days,
        "skipped_current_st": skipped_st,
        "execution": "signal_close_to_next_open_then_holding_close",
        "limit_rule": "next_open_at_limit_up_is_not_buyable",
        "survivorship_bias": (
            "当前股票池缺少历史退市股票，结果仍存在幸存者偏差"
        ),
    }
    return panel, metadata


def build_periods(
    panel: pd.DataFrame,
    factors: list[str],
    *,
    top_n: int,
) -> list[PeriodInput]:
    periods = []
    for date, day in panel.groupby("date", sort=True):
        valid = day.dropna(subset=factors + ["forward_return"])
        if len(valid) < max(100, top_n * 2):
            continue
        periods.append(PeriodInput(
            date=pd.Timestamp(date),
            factors=valid[factors].to_numpy(dtype=float),
            returns=valid["forward_return"].to_numpy(dtype=float),
            codes=valid["code"].astype(str).to_numpy(),
            eligible=valid["eligible"].astype(bool).to_numpy(),
        ))
    return periods


def evaluate_weights(
    periods: list[PeriodInput],
    weights: np.ndarray,
    start: str,
    end: str,
    *,
    top_n: int,
    full_turnover_cost: float = 0.0036,
    return_periods: bool = False,
) -> dict | None:
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    previous: set[str] = set()
    net_returns = []
    excess_returns = []
    turnovers = []
    selected_dates = []
    period_rows = []
    for period in periods:
        if not start_date <= period.date <= end_date:
            continue
        eligible_indices = np.flatnonzero(period.eligible)
        if len(eligible_indices) < top_n:
            continue
        scores = period.factors[eligible_indices] @ weights
        selected_local = np.argpartition(scores, -top_n)[-top_n:]
        selected = eligible_indices[selected_local]
        current = set(period.codes[selected])
        denominator = max(len(previous), len(current), 1)
        turnover = (
            1.0 - len(previous & current) / denominator
            if previous else 1.0
        )
        gross_return = float(period.returns[selected].mean())
        benchmark = float(period.returns.mean())
        cost = turnover * full_turnover_cost
        net_return = gross_return - cost
        net_returns.append(net_return)
        excess_returns.append(gross_return - benchmark - cost)
        turnovers.append(turnover)
        selected_dates.append(period.date)
        period_rows.append({
            "date": period.date.strftime("%Y-%m-%d"),
            "net_return": net_return,
            "excess_return": excess_returns[-1],
            "turnover": turnover,
            "selected": ",".join(sorted(current)),
        })
        previous = current
    if len(net_returns) < 5:
        return None

    def metrics(values: list[float]) -> dict:
        array = np.asarray(values, dtype=float)
        equity = np.cumprod(1.0 + array)
        peak = np.maximum.accumulate(equity)
        calendar_gaps = np.diff(
            np.asarray(selected_dates, dtype="datetime64[D]"),
        ).astype(float)
        median_gap = float(np.median(calendar_gaps)) if len(calendar_gaps) else 29.0
        periods_per_year = 365.25 / max(median_gap, 1.0)
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
            "max_drawdown": round(
                float(((equity - peak) / peak).min()),
                6,
            ),
        }

    result = {
        "absolute": metrics(net_returns),
        "excess": metrics(excess_returns),
        "average_turnover": round(float(np.mean(turnovers)), 6),
        "samples": len(net_returns),
    }
    if return_periods:
        result["period_returns"] = period_rows
    return result


def _split_mask(dates: pd.Series, split: str) -> pd.Series:
    years = pd.to_datetime(dates).dt.year
    if split == "train":
        return years.between(2022, 2024)
    if split == "validation":
        return years.eq(2025)
    return years.ge(2026)


def factor_audit(
    panel: pd.DataFrame,
    periods: list[PeriodInput],
    *,
    top_n: int,
) -> dict[str, dict]:
    audit: dict[str, dict] = {}
    for factor_index, factor in enumerate(FACTOR_COLUMNS):
        factor_result = {}
        for split, start, end in (
            ("train", "2022-01-01", "2024-12-31"),
            ("validation", "2025-01-01", "2025-12-31"),
            ("test", "2026-01-01", "2099-12-31"),
        ):
            subset = panel[
                _split_mask(panel["date"], split)
                & panel["eligible"].astype(bool)
            ]
            daily_ic = subset.groupby("date", sort=True).apply(
                lambda day: day[factor].corr(
                    day["forward_return"],
                    method="spearman",
                ),
                include_groups=False,
            ).dropna()
            ic_std = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else 0.0
            weights = np.zeros(len(FACTOR_COLUMNS), dtype=float)
            weights[factor_index] = 1.0
            portfolio = evaluate_weights(
                periods,
                weights,
                start,
                end,
                top_n=top_n,
            )
            factor_result[split] = {
                "mean_rank_ic": round(float(daily_ic.mean()), 6),
                "ic_ir": round(
                    float(daily_ic.mean() / ic_std),
                    4,
                ) if ic_std > 0 else 0.0,
                "ic_periods": len(daily_ic),
                "portfolio": portfolio,
            }
        audit[factor] = factor_result
    return audit


def select_factors(audit: dict[str, dict], panel: pd.DataFrame) -> tuple[list[str], dict]:
    eligible = []
    rejected = {}
    for factor in FACTOR_COLUMNS:
        train = audit[factor]["train"]
        validation = audit[factor]["validation"]
        reasons = []
        if train["mean_rank_ic"] <= 0:
            reasons.append("训练期 Rank IC 不为正")
        if validation["mean_rank_ic"] <= 0:
            reasons.append("验证期 Rank IC 不为正")
        validation_portfolio = validation["portfolio"]
        if (
            not validation_portfolio
            or validation_portfolio["excess"]["sharpe"] <= 0
        ):
            reasons.append("验证期单因子超额 Sharpe 不为正")
        if reasons:
            rejected[factor] = reasons
        else:
            eligible.append(factor)

    train_panel = panel[_split_mask(panel["date"], "train")]
    correlations = train_panel[list(FACTOR_COLUMNS)].corr(
        method="spearman",
    ).round(4)
    eligible.sort(
        key=lambda name: audit[name]["validation"]["mean_rank_ic"],
        reverse=True,
    )
    selected = []
    for factor in eligible:
        if any(abs(correlations.loc[factor, other]) > 0.80 for other in selected):
            rejected[factor] = ["与更稳定候选因子的训练期相关系数超过 0.80"]
            continue
        selected.append(factor)
    return selected, {
        "rejected": rejected,
        "train_spearman_correlation": correlations.to_dict(),
    }


def _candidate_weights(
    factor_names: list[str],
    *,
    samples: int = 3000,
) -> list[np.ndarray]:
    count = len(factor_names)
    if count == 1:
        return [np.ones(1)]
    maximum = 0.35 if count >= 3 else 0.65
    minimum_risk = 0.20 if RISK_FACTORS.intersection(factor_names) else 0.0
    risk_indices = [
        index for index, name in enumerate(factor_names)
        if name in RISK_FACTORS
    ]
    rng = np.random.default_rng(42)
    candidates = [np.ones(count) / count]
    while len(candidates) < samples:
        weights = rng.dirichlet(np.full(count, 1.5))
        if float(weights.max()) > maximum:
            continue
        if (
            risk_indices
            and float(weights[risk_indices].sum()) < minimum_risk
        ):
            continue
        candidates.append(weights)
    return candidates


def search_weights(
    periods: list[PeriodInput],
    factor_names: list[str],
    *,
    top_n: int,
) -> tuple[np.ndarray, dict]:
    candidates = []
    for weights in _candidate_weights(factor_names):
        train = evaluate_weights(
            periods,
            weights,
            "2022-01-01",
            "2024-12-31",
            top_n=top_n,
        )
        if not train:
            continue
        objective = (
            train["excess"]["sharpe"]
            + 0.20 * train["absolute"]["sharpe"]
            + 0.25 * train["excess"]["max_drawdown"]
            - 0.15 * train["average_turnover"]
        )
        candidates.append((objective, weights, train))
    candidates.sort(key=lambda item: item[0], reverse=True)

    validation_candidates = []
    for _objective, weights, train in candidates[:150]:
        validation = evaluate_weights(
            periods,
            weights,
            "2025-01-01",
            "2025-12-31",
            top_n=top_n,
        )
        if not validation:
            continue
        objective = (
            validation["excess"]["sharpe"]
            + 0.20 * validation["absolute"]["sharpe"]
            + 0.25 * validation["excess"]["max_drawdown"]
            - 0.15 * validation["average_turnover"]
        )
        validation_candidates.append(
            (objective, weights, train, validation),
        )
    validation_candidates.sort(key=lambda item: item[0], reverse=True)
    if not validation_candidates:
        raise RuntimeError("验证期样本不足，无法选择稳健趋势权重")

    stable = np.median(
        np.stack([item[1] for item in validation_candidates[:20]]),
        axis=0,
    )
    stable = stable / stable.sum()
    metrics = {
        "train": evaluate_weights(
            periods, stable, "2022-01-01", "2024-12-31", top_n=top_n,
        ),
        "validation": evaluate_weights(
            periods, stable, "2025-01-01", "2025-12-31", top_n=top_n,
        ),
        "test": evaluate_weights(
            periods,
            stable,
            "2026-01-01",
            "2099-12-31",
            top_n=top_n,
            return_periods=True,
        ),
    }
    return stable, metrics


def production_gate(metrics: dict) -> tuple[bool, list[str]]:
    reasons = []
    for split in ("train", "validation"):
        result = metrics.get(split)
        if not result:
            reasons.append(f"{split} 样本不足")
            continue
        if result["absolute"]["annual_return"] <= 0:
            reasons.append(f"{split} 绝对年化收益不为正")
        if result["excess"]["annual_return"] <= 0:
            reasons.append(f"{split} 超额年化收益不为正")
        if result["excess"]["sharpe"] <= 0:
            reasons.append(f"{split} 超额 Sharpe 不为正")
        if result["absolute"]["max_drawdown"] < -0.35:
            reasons.append(f"{split} 最大回撤超过 35%")
    test = metrics.get("test")
    if not test or test["samples"] < 5:
        reasons.append("最终测试期样本少于 5 期")
    elif test["excess"]["annual_return"] <= 0:
        reasons.append("最终测试期超额年化收益不为正")
    return not reasons, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="稳健趋势因子统一审计")
    parser.add_argument("--max-codes", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--holding-days", type=int, default=20)
    args = parser.parse_args()

    codes = sorted(list_local_codes())
    if args.max_codes > 0:
        codes = codes[:args.max_codes]
    panel, metadata = build_panel(codes, holding_days=args.holding_days)
    all_periods = build_periods(panel, list(FACTOR_COLUMNS), top_n=args.top_n)
    audit = factor_audit(panel, all_periods, top_n=args.top_n)
    selected, selection = select_factors(audit, panel)

    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "fixed_horizon_point_in_time_rank_audit",
        "metadata": metadata,
        "factor_audit": audit,
        "selection": {
            "selected": selected,
            **selection,
        },
    }
    exit_code = 2
    if selected:
        selected_periods = build_periods(
            panel,
            selected,
            top_n=args.top_n,
        )
        weights, metrics = search_weights(
            selected_periods,
            selected,
            top_n=args.top_n,
        )
        passed, reasons = production_gate(metrics)
        payload.update({
            "weights": {
                name: round(float(value), 6)
                for name, value in zip(selected, weights)
            },
            "metrics": metrics,
            "production_gate": {
                "passed": passed,
                "reasons": reasons,
            },
        })
        period_rows = metrics["test"].pop("period_returns", [])
        pd.DataFrame(period_rows).to_csv(
            PERIOD_CSV,
            index=False,
            encoding="utf-8-sig",
        )
        exit_code = 0 if passed else 2
    else:
        payload["production_gate"] = {
            "passed": False,
            "reasons": ["没有因子同时通过训练期和验证期单因子门槛"],
        }

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = []
    for factor, sections in audit.items():
        for split, result in sections.items():
            portfolio = result["portfolio"] or {}
            rows.append({
                "因子": factor,
                "阶段": split,
                "RankIC": result["mean_rank_ic"],
                "ICIR": result["ic_ir"],
                "单因子超额年化": (
                    portfolio.get("excess", {}).get("annual_return")
                ),
                "单因子超额Sharpe": (
                    portfolio.get("excess", {}).get("sharpe")
                ),
                "单因子绝对年化": (
                    portfolio.get("absolute", {}).get("annual_return")
                ),
                "单因子最大回撤": (
                    portfolio.get("absolute", {}).get("max_drawdown")
                ),
            })
    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
