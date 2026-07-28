"""
本地 Parquet 存储：按股票代码分文件，支持增量更新。
"""
import os
from datetime import datetime

import pandas as pd
from pathlib import Path
from config import KLINE_DIR

_OFFLINE = False
_ACTIVE_CODES_DATE = ""
_ACTIVE_CODES: set[str] | None = None


def set_offline_mode(flag: bool) -> None:
    global _OFFLINE
    _OFFLINE = flag


def set_active_trading_codes(date_value: str, codes: set[str] | None) -> None:
    """记录本轮有成交股票；已确认停牌的旧缓存无需逐只请求。"""
    global _ACTIVE_CODES_DATE, _ACTIVE_CODES
    _ACTIVE_CODES_DATE = str(date_value)
    _ACTIVE_CODES = set(codes) if codes is not None else None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名为英文，避免 akshare 中英文列名混用导致 KeyError。"""
    col_map = {}
    turnover_is_percent = False
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ("date", "日期", "时间", "time"):
            col_map[c] = "date"
        elif cl in ("open", "开盘"):
            col_map[c] = "open"
        elif cl in ("high", "最高"):
            col_map[c] = "high"
        elif cl in ("low", "最低"):
            col_map[c] = "low"
        elif cl in ("close", "收盘"):
            col_map[c] = "close"
        elif cl in ("volume", "成交量"):
            col_map[c] = "volume"
        elif cl in ("amount", "成交额"):
            col_map[c] = "amount"
        elif cl in ("outstanding_share", "流通股"):
            col_map[c] = "outstanding_share"
        elif cl in ("turnover", "换手率"):
            col_map[c] = "turnover"
            if cl == "换手率":
                turnover_is_percent = True
        elif cl in ("pct_change", "涨跌幅"):
            col_map[c] = "pct_change"
        elif cl in ("amplitude", "振幅"):
            col_map[c] = "amplitude"
    if col_map:
        df = df.rename(columns=col_map)
    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
        if turnover_is_percent:
            df["turnover"] = df["turnover"] / 100.0
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def kline_path(code: str) -> Path:
    return KLINE_DIR / f"{code}.parquet"


def load_kline(code: str) -> pd.DataFrame | None:
    """加载单只股票本地K线数据。"""
    p = kline_path(code)
    if p.exists():
        df = pd.read_parquet(p)
        return _normalize_columns(df)
    return None


def save_kline(code: str, df: pd.DataFrame) -> None:
    """保存单只股票K线到本地。"""
    df.to_parquet(kline_path(code), index=False)


def update_kline(code: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """增量更新：合并新数据到已有数据，去重排序。"""
    new_df = _normalize_columns(new_df)
    existing = load_kline(code)
    if existing is not None:
        existing = _align_existing_qfq(existing, new_df)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    else:
        combined = new_df.sort_values("date")
    combined = combined.reset_index(drop=True)
    if "close" in combined.columns:
        combined["pct_change"] = combined["close"].pct_change() * 100
    save_kline(code, combined)
    return combined


def _align_existing_qfq(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """用重叠区间识别前复权基准变化，并同比例校正旧历史价格。"""
    if "close" not in existing.columns or "close" not in new_df.columns:
        return existing
    overlap = existing[["date", "close"]].merge(
        new_df[["date", "close"]], on="date", suffixes=("_old", "_new")
    ).dropna()
    overlap = overlap[(overlap["close_old"] > 0) & (overlap["close_new"] > 0)]
    if len(overlap) < 3:
        return existing
    ratios = overlap["close_new"] / overlap["close_old"]
    ratio = float(ratios.median())
    if abs(ratio - 1.0) <= 0.001:
        return existing
    if float((ratios / ratio - 1.0).abs().max()) > 0.002:
        return existing
    aligned = existing.copy()
    for column in ("open", "high", "low", "close", "pre_close"):
        if column in aligned.columns:
            aligned[column] = aligned[column] * ratio
    return aligned


def list_local_codes() -> list[str]:
    """列出本地已缓存的所有股票代码。"""
    return [p.stem for p in KLINE_DIR.glob("*.parquet")]


def _data_ready_minutes() -> int:
    """读取行情就绪时间，优先使用 HH:MM，兼容旧的整点配置。"""
    raw = os.getenv("SG_QUANT_DATA_READY_TIME", "15:35").strip()
    try:
        parsed = datetime.strptime(raw, "%H:%M")
        return parsed.hour * 60 + parsed.minute
    except ValueError:
        legacy_raw = os.getenv("SG_QUANT_DATA_READY_HOUR", "").strip()
        if not legacy_raw:
            return 15 * 60 + 35
        try:
            legacy_hour = int(legacy_raw)
        except ValueError:
            return 15 * 60 + 35
        return min(max(legacy_hour, 0), 23) * 60


def expected_market_date(now) -> pd.Timestamp:
    """返回本次运行必须具备的最近市场日期；默认 15:35 后要求当天数据。"""
    today = pd.Timestamp(now.date())
    current_minutes = now.hour * 60 + now.minute
    if today.weekday() >= 5 or current_minutes < _data_ready_minutes():
        return today - pd.offsets.BDay(1)
    return today


def _expected_market_date(now) -> pd.Timestamp:
    """兼容旧维护脚本；新代码请调用 expected_market_date。"""
    return expected_market_date(now)


def latest_kline_dates(codes: set[str]) -> dict[str, str]:
    """只读取日期列，返回指定股票本地日线的最后交易日。"""
    output: dict[str, str] = {}
    for code in codes:
        path = kline_path(code)
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path, columns=["date"])
            if frame.empty:
                continue
            latest = pd.to_datetime(frame["date"], errors="coerce").max()
            if pd.notna(latest):
                output[code] = latest.strftime("%Y-%m-%d")
        except (OSError, ValueError, KeyError):
            continue
    return output


def append_market_daily_snapshot(
    snapshot: pd.DataFrame,
    codes: set[str],
    *,
    allow_date_gap: bool = False,
) -> dict:
    """只给缺少当天且前一工作日连续的缓存追加一根全市场日线。"""
    stats = {
        "updated": 0, "already_current": 0,
        "missing_cache": 0, "gap": 0, "gap_codes": [],
    }
    if snapshot is None or snapshot.empty:
        return stats
    required = {
        "code", "date", "open", "high", "low", "close",
        "pre_close", "volume", "amount",
    }
    if not required.issubset(snapshot.columns):
        raise ValueError("全市场单日行情字段不完整")

    for row in snapshot.to_dict("records"):
        code = str(row.get("code", "")).zfill(6)
        if code not in codes:
            continue
        existing = load_kline(code)
        if existing is None or existing.empty:
            stats["missing_cache"] += 1
            continue
        snapshot_date = pd.Timestamp(row["date"]).normalize()
        latest_date = pd.Timestamp(existing["date"].max()).normalize()
        if latest_date >= snapshot_date:
            stats["already_current"] += 1
            continue
        previous_business_day = (snapshot_date - pd.offsets.BDay(1)).normalize()
        if not allow_date_gap and latest_date != previous_business_day:
            stats["gap"] += 1
            stats["gap_codes"].append(code)
            continue

        aligned = existing.copy()
        old_close = pd.to_numeric(aligned["close"], errors="coerce").iloc[-1]
        pre_close = pd.to_numeric(pd.Series([row["pre_close"]]), errors="coerce").iloc[0]
        if pd.notna(old_close) and old_close > 0 and pd.notna(pre_close) and pre_close > 0:
            ratio = float(pre_close / old_close)
            if 0.01 <= ratio <= 100:
                for column in ("open", "high", "low", "close", "pre_close"):
                    if column in aligned.columns:
                        aligned[column] = pd.to_numeric(
                            aligned[column], errors="coerce",
                        ) * ratio

        new_row = {
            column: aligned[column].iloc[-1]
            for column in aligned.columns
        }
        new_row["date"] = snapshot_date
        for column in (
            "open", "high", "low", "close", "pre_close",
            "volume", "amount", "pct_change",
        ):
            if column in new_row:
                new_row[column] = row.get(column)
        if "amplitude" in new_row and pre_close > 0:
            new_row["amplitude"] = (
                (float(row["high"]) - float(row["low"])) / float(pre_close) * 100
            )
        if "turnover" in new_row and "outstanding_share" in new_row:
            outstanding = pd.to_numeric(
                pd.Series([new_row["outstanding_share"]]), errors="coerce",
            ).iloc[0]
            if pd.notna(outstanding) and outstanding > 0:
                new_row["turnover"] = float(row["volume"]) / float(outstanding)
            else:
                new_row["turnover"] = float("nan")
        combined = pd.concat(
            [aligned, pd.DataFrame([new_row])],
            ignore_index=True,
        )
        combined = combined.drop_duplicates("date", keep="last").sort_values("date")
        combined["pct_change"] = combined["close"].pct_change() * 100
        save_kline(code, combined.reset_index(drop=True))
        stats["updated"] += 1
    return stats


def ensure_kline(code: str) -> pd.DataFrame | None:
    """确保K线数据为最新：缺失→全量下载，过期→增量更新，最新→直接返回。
    离线模式下仅从本地缓存加载，不做任何网络请求。"""
    if _OFFLINE:
        return load_kline(code)

    from datetime import datetime, timedelta

    df = load_kline(code)
    now = datetime.now()

    if df is not None:
        latest = pd.Timestamp(df["date"].max()).normalize()
        required_date = expected_market_date(now)
        # 盘前/盘中允许使用上一工作日；收盘数据稳定后要求更新到当天。
        # 中国法定节假日不是普通工作日，接口返回空时下方逻辑会保留缓存。
        if latest >= required_date:
            return df
        if (
            _ACTIVE_CODES is not None
            and _ACTIVE_CODES_DATE == required_date.strftime("%Y-%m-%d")
            and code not in _ACTIVE_CODES
        ):
            return df
        try:
            from data.fetcher import get_kline_daily
            # 保留两周重叠区间，用于识别公司行为导致的前复权基准变化。
            start = (latest - timedelta(days=14)).strftime("%Y%m%d")
            end = now.strftime("%Y%m%d")
            new_df = get_kline_daily(code, start_date=start, end_date=end)
            if new_df is not None and not new_df.empty:
                df = update_kline(code, new_df)
                return df
        except Exception:
            return df

        # 增量接口返回空数据（周末、节假日、盘中尚未更新）时保留现有缓存。
        # 旧逻辑会继续走下面的“全量下载”，导致每只股票重复拉取 2020 年以来数据。
        return df

    try:
        from data.fetcher import get_kline_daily
        df = get_kline_daily(code)
        if df is not None and not df.empty:
            save_kline(code, df)
            return df
    except Exception:
        pass

    return None
