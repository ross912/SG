"""SG Quant 日度入口：生成基本面价值榜与四份技术研究榜单。"""
import argparse
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from config import (
    ALL_MARKET_POOL_SIZE,
    FUNDAMENTAL_POOL_SIZE,
    KLINE_DIR,
    MAIN_BOARD_POOL_SIZE,
    OUTPUT_DIR,
    SCORE_WEIGHTS,
    STRATEGIES,
)
from data.board_utils import is_main_board
from data.concept_tags import apply_concept_tags, get_concept_tags
from data.fetcher import (
    get_all_a_stocks,
    get_market_daily_snapshot,
    get_stock_missing_daily,
    get_trading_codes,
)
from data.fundamentals import (
    refresh_fundamental_cache,
    refresh_valuation_cache,
)
from data.forward_outlook import refresh_forward_cache
from data.index_filter import (
    assess_market_overview,
    default_market_regime,
    load_cached_market_regime,
    print_market_regime,
    save_market_overview,
    save_market_regime,
)
from data.risk_manager import calc_total_position_ratio
from data.storage import (
    expected_market_date,
    append_market_daily_snapshot,
    latest_kline_dates,
    list_local_codes,
    set_active_trading_codes,
    set_offline_mode,
)
from feedback.recorder import init_db, record_ranking
from feedback.returns_tracker import backfill_forward_returns
from feedback.weight_optimizer import auto_weekly_optimize, load_optimized_weights
from output.stock_pool import print_stock_pool, save_stock_pool
from screening.eligibility import is_st_name
from screening.factor_eval import (
    adjust_weights_by_dispersion,
    compute_factor_dispersion,
    cross_sectional_normalize,
    cross_sectional_standardize_factors,
)
from screening.fundamental import build_fundamental_ranking
from screening.ranking import rank, to_dataframe
from screening.scanner import prepare_results_for_strategy, scan_all, set_market_regime
from services.market_ai import generate_daily_market_summary, save_market_snapshot
from services.news import refresh_news
from services.progress import complete as complete_progress
from services.progress import add_warning as add_progress_warning
from services.progress import fail as fail_progress
from services.progress import start as start_progress
from services.progress import update as update_progress
from services.storage import write_json

_NAMES_CACHE = OUTPUT_DIR / "stock_names_cache.json"
_INDUSTRY_CACHE = OUTPUT_DIR / "stock_industry_cache.json"
_STOCK_META_CACHE = OUTPUT_DIR / "stock_metadata_cache.csv"


def load_universe(
    offline: bool,
    full_market: bool,
) -> tuple[set[str], dict[str, str], dict[str, str]]:
    """加载代码与名称；全市场模式优先在线股票列表，失败时退回本地缓存。"""
    names = _load_name_cache()
    industries = _load_industry_cache()
    local_codes = set(list_local_codes())
    if offline or not full_market:
        return local_codes, names, industries

    try:
        stocks = get_all_a_stocks()
        codes = set(stocks["code"].astype(str).str.zfill(6))
        fresh_names = dict(zip(
            stocks["code"].astype(str).str.zfill(6),
            stocks["name"].astype(str),
        ))
        names.update(fresh_names)
        _save_name_cache(names)
        if "industry" in stocks.columns:
            fresh_industries = {
                str(code).zfill(6): str(industry).strip()
                for code, industry in zip(stocks["code"], stocks["industry"])
                if pd.notna(industry) and str(industry).strip()
            }
            industries.update(fresh_industries)
            if fresh_industries:
                _save_industry_cache(industries)
        _save_stock_metadata(stocks)
        return codes, names, industries
    except Exception as error:
        print(f"股票列表在线获取失败，退回本地缓存: {error}")
        return local_codes, names, industries


def build_strategy_pools(
    base_results,
    regime_name: str,
    *,
    standardize_factors: bool = False,
) -> dict[str, dict]:
    """用同一份五因子扫描结果构造四个榜单。"""
    pools = {}
    base_weights = dict(SCORE_WEIGHTS.get(regime_name, SCORE_WEIGHTS["sideways"]))
    dispersion = compute_factor_dispersion(base_results)
    adjusted_weights = adjust_weights_by_dispersion(base_weights, dispersion)
    scoring_results = (
        cross_sectional_standardize_factors(base_results)
        if standardize_factors else base_results
    )
    for strategy_name, strategy in STRATEGIES.items():
        prepared = prepare_results_for_strategy(
            scoring_results,
            directions=strategy["directions"],
            weights=adjusted_weights,
        )
        prepared = cross_sectional_normalize(prepared)
        main_results = [result for result in prepared if is_main_board(result.code)]
        pools[strategy["csv_main"]] = {
            "strategy": strategy_name,
            "label": f"{strategy['label']} 主板 Top {MAIN_BOARD_POOL_SIZE}",
            "items": rank(main_results, max_pool=MAIN_BOARD_POOL_SIZE),
        }
        pools[strategy["csv_all"]] = {
            "strategy": strategy_name,
            "label": f"{strategy['label']} 全市场 Top {ALL_MARKET_POOL_SIZE}",
            "items": rank(prepared, max_pool=ALL_MARKET_POOL_SIZE),
        }
        compact = ", ".join(f"{name}={value:.3f}" for name, value in adjusted_weights.items())
        print(f"[{strategy['label']}] 动态权重: {compact}")
    return pools


def write_outputs(
    pools: dict[str, dict],
    regime_name: str,
    date_str: str,
    signal_date: str | None = None,
    allow_concept_network: bool = True,
) -> list:
    """补充概念标签后保存四个 CSV，并写入反馈数据库。"""
    selected_codes = {
        result.code for item in pools.values() for result in item["items"]
    }
    concepts = get_concept_tags(selected_codes, allow_network=allow_concept_network)
    paths = []
    for suffix, item in pools.items():
        frame = apply_concept_tags(to_dataframe(item["items"]), concepts)
        print_stock_pool(frame, date_str, title=item["label"])
        paths.append(save_stock_pool(frame, date_str, suffix=suffix))
        record_ranking(
            item["items"],
            date_str=signal_date or date_str,
            list_type=suffix,
            market_regime=regime_name,
            strategy=item["strategy"],
        )
    return paths


def _load_name_cache() -> dict[str, str]:
    if not _NAMES_CACHE.exists():
        return {}
    try:
        with _NAMES_CACHE.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _save_name_cache(names: dict[str, str]) -> None:
    with _NAMES_CACHE.open("w", encoding="utf-8") as handle:
        json.dump(names, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _load_industry_cache() -> dict[str, str]:
    if not _INDUSTRY_CACHE.exists():
        return {}
    try:
        with _INDUSTRY_CACHE.open("r", encoding="utf-8") as handle:
            return {
                str(code).zfill(6): str(industry).strip()
                for code, industry in json.load(handle).items()
                if str(industry).strip()
            }
    except Exception:
        return {}


def _save_industry_cache(industries: dict[str, str]) -> None:
    with _INDUSTRY_CACHE.open("w", encoding="utf-8") as handle:
        json.dump(industries, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _load_stock_metadata() -> pd.DataFrame:
    if not _STOCK_META_CACHE.exists():
        return pd.DataFrame(columns=["code", "name", "industry", "list_date", "market"])
    try:
        frame = pd.read_csv(_STOCK_META_CACHE, dtype={"code": str, "list_date": str})
    except Exception:
        return pd.DataFrame(columns=["code", "name", "industry", "list_date", "market"])
    for column in ("name", "industry", "list_date", "market"):
        frame[column] = frame[column].astype(object)
    updated = False
    industry_map = _load_industry_cache()
    if industry_map:
        missing_industry = (
            frame["industry"].fillna("").astype(str).str.strip().isin({"", "nan"})
        )
        mapped = frame.loc[missing_industry, "code"].astype(str).str.zfill(6).map(
            industry_map,
        )
        available = mapped.fillna("").astype(str).str.strip().ne("")
        if available.any():
            frame.loc[mapped.index[available], "industry"] = mapped.loc[available]
            updated = True
    missing = frame["list_date"].fillna("").astype(str).str.strip().isin({"", "nan"})
    for index, code in frame.loc[missing, "code"].items():
        path = KLINE_DIR / f"{str(code).zfill(6)}.parquet"
        if not path.exists():
            continue
        try:
            dates = pd.read_parquet(path, columns=["date"])["date"]
            first_date = pd.to_datetime(dates, errors="coerce").min()
        except Exception:
            continue
        if pd.notna(first_date):
            frame.at[index, "list_date"] = first_date.strftime("%Y%m%d")
            updated = True
    if updated:
        frame.to_csv(_STOCK_META_CACHE, index=False, encoding="utf-8-sig")
    return frame


def _save_stock_metadata(stocks: pd.DataFrame) -> None:
    frame = stocks.copy()
    if "symbol" in frame.columns and "code" not in frame.columns:
        frame = frame.rename(columns={"symbol": "code"})
    for column in ("code", "name", "industry", "list_date", "market"):
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].astype(object)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    if _STOCK_META_CACHE.exists():
        try:
            existing = pd.read_csv(
                _STOCK_META_CACHE, dtype={"code": str, "list_date": str},
            ).drop_duplicates("code").set_index("code")
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty:
            for column in ("industry", "list_date", "market"):
                if column not in existing:
                    continue
                old_values = frame["code"].map(existing[column])
                missing = (
                    frame[column].fillna("").astype(str).str.strip().isin({"", "nan"})
                )
                frame.loc[missing, column] = old_values.loc[missing]
    industry_map = _load_industry_cache()
    if industry_map:
        missing = frame["industry"].fillna("").astype(str).str.strip().isin({"", "nan"})
        frame.loc[missing, "industry"] = frame.loc[missing, "code"].map(industry_map)
    frame[["code", "name", "industry", "list_date", "market"]].drop_duplicates(
        "code",
    ).to_csv(_STOCK_META_CACHE, index=False, encoding="utf-8-sig")


def _load_saved_weights() -> None:
    saved = load_optimized_weights()
    if not saved:
        return
    allowed = set(next(iter(SCORE_WEIGHTS.values())))
    loaded = []
    for regime_name, weights in saved.items():
        if regime_name not in SCORE_WEIGHTS:
            continue
        valid = {
            name: float(value)
            for name, value in weights.items()
            if name in allowed and float(value) >= 0
        }
        total = sum(valid.values())
        if set(valid) != allowed or total <= 0:
            continue
        SCORE_WEIGHTS[regime_name] = {
            name: value / total for name, value in valid.items()
        }
        loaded.append(regime_name)
    if loaded:
        print(f"已加载历史优化权重: {', '.join(loaded)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A股基本面价值与技术研究系统")
    parser.add_argument("--full", action="store_true", help="在线获取全市场股票列表并扫描")
    parser.add_argument("--offline", action="store_true", help="只使用本地日线缓存")
    parser.add_argument("--clean", action="store_true", help="清空日线缓存后重新执行全市场更新")
    parser.add_argument(
        "--refresh-fundamentals",
        action="store_true",
        help="忽略低频缓存，强制刷新公开财务报表与质押数据",
    )
    return parser.parse_args()


def resolve_market_date(results: list, fallback: str) -> str:
    """以多数股票的实际行情截面作为榜单日期，不能由单只异常日期决定。"""
    dates = []
    for result in results:
        value = str(getattr(result, "as_of_date", "") or "")
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            continue
        dates.append(value)
    if not dates:
        return fallback
    counts = Counter(dates)
    highest_count = max(counts.values())
    return max(date for date, count in counts.items() if count == highest_count)


@dataclass(frozen=True)
class MarketDataFreshness:
    snapshot_date: str
    excluded_codes: frozenset[str] = frozenset()
    warning: str = ""
    checked_count: int = 0


def validate_market_data_freshness(
    codes: set[str],
    names: dict[str, str],
    expected_date: str,
    trading_codes: set[str] | None = None,
) -> MarketDataFreshness:
    """校验当日行情；异常股票仅告警并从榜单剔除，绝不中断流程。"""
    trading_codes = (trading_codes or get_trading_codes(expected_date)) & codes
    if not trading_codes:
        return MarketDataFreshness(
            snapshot_date=expected_date,
            warning=(
                f"行情数据错误（流程继续）：{expected_date} 未识别到有成交股票，"
                "无法验证行情完整性"
            ),
        )
    actual_dates = latest_kline_dates(trading_codes)
    stale_codes = sorted(
        code for code in trading_codes
        if actual_dates.get(code) != expected_date
    )
    warning = ""
    if 0 < len(stale_codes) < 10:
        details = "、".join(
            f"{code} {names.get(code, '')}({actual_dates.get(code, '缺失')})".strip()
            for code in stale_codes
        )
        warning = (
            f"行情数据错误（流程继续）：{len(stale_codes)} 只股票不是 "
            f"{expected_date} 数据，已从榜单剔除：{details}"
        )
    elif stale_codes:
        warning = (
            f"行情数据错误（流程继续）：共有 {len(stale_codes)} 只股票不是 "
            f"{expected_date} 数据，已全部从榜单剔除"
        )
    return MarketDataFreshness(
        snapshot_date=expected_date,
        excluded_codes=frozenset(stale_codes),
        warning=warning,
        checked_count=len(trading_codes),
    )


def _run_pipeline(args: argparse.Namespace, run_id: str) -> int:
    started_date = datetime.now().strftime("%Y-%m-%d")
    started = time.time()
    print(f"\n{'=' * 60}\n  SG Quant 基本面价值与量化研究  |  {started_date}\n{'=' * 60}")
    update_progress(
        run_id, stage="initializing", percent=2,
        message="正在初始化数据库与历史权重",
    )

    if args.offline:
        set_offline_mode(True)
    if args.clean:
        if KLINE_DIR.exists():
            shutil.rmtree(KLINE_DIR)
        KLINE_DIR.mkdir(parents=True, exist_ok=True)
        print("日线缓存已清空")

    init_db()
    _load_saved_weights()
    full_market = args.full or args.clean
    codes, names, industries = load_universe(args.offline, full_market)
    st_codes = {code for code in codes if is_st_name(names.get(code, ""))}
    codes.difference_update(st_codes)
    print(
        f"股票范围: {len(codes)} 只；名称: {len(names)} 条；"
        f"行业分类: {len(industries)} 条；已排除 ST: {len(st_codes)} 只"
    )
    update_progress(
        run_id, stage="universe", percent=5,
        message=f"股票范围 {len(codes)} 只，已排除 ST {len(st_codes)} 只",
        total=len(codes),
    )
    if not codes:
        print("没有可扫描的股票代码")
        return 1

    target_date = expected_market_date(datetime.now()).strftime("%Y-%m-%d")
    active_codes: set[str] | None = None
    daily_snapshot: pd.DataFrame | None = None
    if full_market and not args.offline:
        update_progress(
            run_id, stage="daily_increment", percent=6,
            message=f"正在获取 {target_date} 全市场单日增量",
            total=len(codes),
        )
        try:
            daily_snapshot = get_market_daily_snapshot(target_date)
            volume = daily_snapshot["volume"].fillna(0)
            amount = daily_snapshot["amount"].fillna(0)
            active_codes = set(
                daily_snapshot.loc[
                    (volume > 0) | (amount > 0), "code",
                ].astype(str)
            )
            set_active_trading_codes(target_date, active_codes)
            increment_stats = append_market_daily_snapshot(daily_snapshot, codes)
            repaired_gaps = 0
            for code in increment_stats["gap_codes"]:
                current_date = latest_kline_dates({code}).get(code, "")
                if not current_date:
                    continue
                start_date = (
                    datetime.strptime(current_date, "%Y-%m-%d")
                    + pd.Timedelta(days=1)
                ).strftime("%Y-%m-%d")
                missing_rows = get_stock_missing_daily(
                    code, start_date, target_date,
                )
                repair_stats = append_market_daily_snapshot(
                    missing_rows, {code}, allow_date_gap=True,
                )
                repaired_gaps += repair_stats["updated"]
            print(
                "全市场单日增量: "
                f"追加 {increment_stats['updated']}，"
                f"已是最新 {increment_stats['already_current']}，"
                f"缺口区间补齐 {repaired_gaps} 根，"
                f"无历史缓存 {increment_stats['missing_cache']}"
            )
        except Exception as error:
            set_active_trading_codes(target_date, None)
            print(
                "全市场单日增量失败，降级为逐股增量: "
                f"{type(error).__name__}: {error}"
            )

    if args.offline:
        regime = load_cached_market_regime() or default_market_regime()
    else:
        regime, market_overview = assess_market_overview()
        save_market_regime(regime)
        save_market_overview(market_overview)
    set_market_regime(regime.regime)
    print_market_regime(regime)
    position = calc_total_position_ratio(regime.score)
    print(f"建议总仓位: {position['ratio_pct']:.1f}%（{position['label']}）")
    update_progress(
        run_id, stage="market_regime", percent=8,
        message=f"市场状态：{regime.regime}；建议仓位 {position['ratio_pct']:.1f}%",
    )

    fundamental_frame = pd.DataFrame()
    fundamental_diagnostics: dict = {}
    update_progress(
        run_id, stage="fundamentals", percent=9,
        message="正在更新财报、前瞻证据与当日估值；高风险和前瞻数据不足股票将剔除",
        total=len(codes),
    )
    try:
        fundamentals, fundamental_meta = refresh_fundamental_cache(
            universe_count=len(codes),
            allow_network=not args.offline,
            force=args.refresh_fundamentals,
        )
        valuations, valuation_meta = refresh_valuation_cache(
            target_date,
            allow_network=not args.offline,
        )
        outlook, forward_meta = refresh_forward_cache(
            allow_network=not args.offline,
            force=args.refresh_fundamentals,
        )
        fundamentals = fundamentals.loc[
            fundamentals["code"].astype(str).str.zfill(6).isin(codes)
        ].copy()
        valuations = valuations.loc[
            valuations["code"].astype(str).str.zfill(6).isin(codes)
        ].copy()
        if not outlook.empty:
            outlook = outlook.loc[
                outlook["code"].astype(str).str.zfill(6).isin(codes)
            ].copy()
        fundamental_frame, fundamental_diagnostics = build_fundamental_ranking(
            fundamentals,
            valuations,
            _load_stock_metadata(),
            outlook,
            valuation_date=valuation_meta.get("trade_date", target_date),
            top_n=FUNDAMENTAL_POOL_SIZE,
        )
        fundamental_diagnostics["fundamental_source"] = fundamental_meta
        fundamental_diagnostics["valuation_source"] = valuation_meta
        fundamental_diagnostics["forward_source"] = forward_meta
        print(
            "基本面价值榜候选: "
            f"输入 {fundamental_diagnostics.get('input_count', 0)}，"
            f"高风险剔除 {fundamental_diagnostics.get('high_risk_excluded', 0)}，"
            f"数据不足 {fundamental_diagnostics.get('data_missing_excluded', 0)}，"
            f"其中前瞻不足 {fundamental_diagnostics.get('forward_data_missing_excluded', 0)}，"
            f"达到门槛 {fundamental_diagnostics.get('eligible_count', 0)}，"
            f"入榜 {len(fundamental_frame)}"
        )
    except Exception as error:
        warning = (
            "基本面价值榜生成失败（原四榜单继续）："
            f"{type(error).__name__}: {error}"
        )
        print(warning)
        add_progress_warning(run_id, warning)

    print("\n更新日线并计算生产五因子与趋势候选因子（每只股票只走一次）...")
    scan_workers = max(1, int(os.getenv("SG_QUANT_SCAN_WORKERS", "6")))

    def report_scan(completed: int, total: int, valid: int) -> None:
        ratio = completed / total if total else 0
        update_progress(
            run_id,
            stage="scanning",
            percent=12 + ratio * 66,
            message=(
                f"正在更新日线并计算生产/候选因子："
                f"{completed}/{total}，有效 {valid}"
            ),
            current=completed,
            total=total,
            valid=valid,
        )

    results = scan_all(
        codes,
        name_map=names,
        industry_map=industries,
        max_workers=scan_workers,
        progress_callback=report_scan,
    )
    print(f"生产/候选因子计算完成: {len(results)}/{len(codes)} 只")
    if not results:
        return 3

    if full_market and not args.offline:
        update_progress(
            run_id, stage="validating", percent=79,
            message=f"正在核验 {target_date} 当日有成交股票的数据日期",
            current=len(codes), total=len(codes), valid=len(results),
        )
        freshness = validate_market_data_freshness(
            codes, names, target_date, trading_codes=active_codes,
        )
        snapshot_date = freshness.snapshot_date
        if freshness.warning:
            print(f"[数据警告] {freshness.warning}")
            add_progress_warning(run_id, freshness.warning)
        if freshness.excluded_codes:
            before_count = len(results)
            results = [
                result for result in results
                if result.code not in freshness.excluded_codes
            ]
            print(
                f"已剔除日期异常股票: {before_count - len(results)} 只；"
                f"榜单候选剩余 {len(results)} 只"
            )
    else:
        snapshot_date = resolve_market_date(results, fallback=started_date)
    print(f"市场数据截面: {snapshot_date}")
    if snapshot_date != started_date:
        print(
            f"任务启动日 {started_date} 与行情截面 {snapshot_date} 不同；"
            "榜单、信号与日报统一使用行情截面日期"
        )
    save_market_snapshot(
        results,
        snapshot_date=snapshot_date,
        regime=regime,
        position=position,
        market_daily_snapshot=daily_snapshot,
    )
    update_progress(
        run_id, stage="ranking", percent=82,
        message="正在保存基本面价值榜并构造趋势与均值回归四榜单",
        current=len(codes), total=len(codes), valid=len(results),
    )
    pools = build_strategy_pools(results, regime.regime)
    paths = []
    if not fundamental_frame.empty:
        print(
            "\n[基本面价值 · 全市场 Top 30]\n"
            + fundamental_frame[
                ["排名", "代码", "名称", "综合得分", "暴雷风险", "入选理由", "风险提示"]
            ].to_string(index=False)
        )
        paths.append(save_stock_pool(
            fundamental_frame,
            snapshot_date,
            suffix="fundamental30",
        ))
        write_json(
            OUTPUT_DIR / f"fundamental_diagnostics_{snapshot_date}.json",
            fundamental_diagnostics,
        )
        write_json(
            OUTPUT_DIR / "fundamental_diagnostics_latest.json",
            fundamental_diagnostics,
        )
    paths.extend(write_outputs(
        pools,
        regime.regime,
        snapshot_date,
        signal_date=snapshot_date,
        allow_concept_network=not args.offline,
    ))
    update_progress(
        run_id, stage="feedback", percent=90,
        message="五榜单已生成，正在回填技术榜单收益与更新动态权重",
    )

    returns_stats = backfill_forward_returns()
    print(f"前向收益回填: 检查 {returns_stats['checked']}，新增 {returns_stats['filled']}")
    auto_weekly_optimize(verbose=True)

    if full_market and not args.offline:
        update_progress(
            run_id, stage="news", percent=94,
            message="正在抓取最近 24 小时财经新闻",
        )
        try:
            news_payload = refresh_news(hours=24, limit=50)
            print(f"财经新闻: {news_payload.get('item_count', 0)} 条")
        except Exception as error:
            news_payload = {}
            print(f"财经新闻更新失败，榜单不受影响: {type(error).__name__}: {error}")

        update_progress(
            run_id, stage="daily_summary", percent=97,
            message="正在调用 DeepSeek 生成每日市场总结",
        )
        summary_payload = generate_daily_market_summary(
            date_str=snapshot_date,
            snapshot_date=snapshot_date,
            news_payload=news_payload,
        )
        if summary_payload.get("status") == "completed":
            print("每日市场总结已生成")
        else:
            print(f"每日市场总结失败，榜单不受影响: {summary_payload.get('error', '未知错误')}")

    update_progress(
        run_id, stage="finalizing", percent=99,
        message="正在整理输出文件",
    )
    print("输出文件:")
    for path in paths:
        print(f"  {path}")
    print(f"流程完成，用时 {(time.time() - started) / 60:.1f} 分钟")
    return 0


def main() -> int:
    args = parse_args()
    mode = "offline" if args.offline else "full" if (args.full or args.clean) else "local"
    run_id = start_progress(mode)
    try:
        exit_code = _run_pipeline(args, run_id)
    except Exception as error:
        fail_progress(run_id, f"流程异常：{type(error).__name__}: {error}")
        raise
    if exit_code == 0:
        complete_progress(run_id)
    else:
        fail_progress(run_id, f"流程提前结束，退出码 {exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
