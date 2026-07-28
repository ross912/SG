"""
回填前向收益率到 SQLite，基于交易日天数（跳过周末和节假日）。
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from data.storage import load_kline

_logger = logging.getLogger(__name__)

FORWARD_PERIODS = [1, 5, 10, 20]


def _count_trading_days(df: pd.DataFrame, signal_date_str: str,
                         days_forward: int) -> float | None:
    """从 signal_date 往后数 days_forward 个交易日，返回 close / signal_close - 1。"""
    signal_dt = pd.Timestamp(signal_date_str)
    df_sorted = df.sort_values("date").reset_index(drop=True)
    positions = df_sorted.index[df_sorted["date"].dt.normalize() == signal_dt.normalize()]
    if len(positions) == 0:
        return None
    pos = int(positions[0])
    target = pos + days_forward
    if target >= len(df_sorted):
        return None

    signal_close = float(df_sorted.iloc[pos]["close"])
    forward_close = float(df_sorted.iloc[target]["close"])
    if signal_close <= 0:
        return None
    return round((forward_close / signal_close - 1), 6)


def backfill_forward_returns(db_path: Path | None = None,
                              start_date: str | None = None,
                              end_date: str | None = None) -> dict[str, int]:
    """
    回填 signals 表中缺少的 forward_returns 行。

    返回 {"checked": N, "filled": M} 统计。
    """
    import sqlite3

    if db_path is None:
        db_path = OUTPUT_DIR / "feedback.db"

    if not db_path.exists():
        return {"checked": 0, "filled": 0}

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

    stats = {"checked": 0, "filled": 0}
    try:
        where = "WHERE 1=1"
        params: list = []
        if start_date:
            where += " AND date >= ?"
            params.append(start_date)
        if end_date:
            where += " AND date <= ?"
            params.append(end_date)

        rows = conn.execute(
            f"SELECT id, date, code, close FROM signals {where} ORDER BY date", params
        ).fetchall()

        existing_pairs = {
            (r["signal_id"], r["days_forward"])
            for r in conn.execute(
                "SELECT signal_id, days_forward FROM forward_returns"
            ).fetchall()
        }
        rows_by_code: dict[str, list] = {}
        for row in rows:
            rows_by_code.setdefault(row["code"], []).append(row)

        # 同一股票可能同时出现在多个榜单；每个代码只读取一次 Parquet。
        for code, code_rows in rows_by_code.items():
            kdf = load_kline(code)
            if kdf is None:
                continue
            kdf["date"] = pd.to_datetime(kdf["date"])

            for row in code_rows:
                signal_id = row["id"]
                date_str = row["date"]
                signal_rows = kdf[
                    kdf["date"].dt.normalize() == pd.Timestamp(date_str).normalize()
                ]
                if signal_rows.empty:
                    continue
                signal_close = float(signal_rows.iloc[0]["close"])

                for days in FORWARD_PERIODS:
                    if (signal_id, days) in existing_pairs:
                        continue

                    fwd = _count_trading_days(kdf, date_str, days)
                    if fwd is None:
                        continue

                    forward_close = signal_close * (1 + fwd)
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    conn.execute(
                        """INSERT OR IGNORE INTO forward_returns
                           (signal_id, days_forward, forward_return, close_signal,
                            close_forward, calculated_date)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (signal_id, days, fwd, signal_close, round(forward_close, 2), today_str),
                    )
                    existing_pairs.add((signal_id, days))
                    stats["filled"] += 1

                stats["checked"] += 1

        conn.commit()
    except Exception:
        conn.rollback()
        _logger.warning("回填前向收益率失败", exc_info=True)
    finally:
        conn.close()

    return stats


def compute_ic_for_regime(db_path: Path | None, factor_name: str,
                           regime: str, lookback_days: int = 60,
                           forward_days: int = 5,
                           min_samples: int = 20) -> dict | None:
    """
    计算单个因子在指定市场状态下、指定前向天数下的 Spearman rank IC。

    返回 {"mean_ic": float, "ic_ir": float, "samples": int} 或 None。
    """
    import sqlite3
    from scipy.stats import spearmanr

    if db_path is None:
        db_path = OUTPUT_DIR / "feedback.db"
    if not db_path.exists():
        return None

    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute("""
            SELECT s.date, s.code,
                   AVG(fv.factor_value) AS factor_value,
                   AVG(fr.forward_return) AS forward_return
            FROM factor_values fv
            JOIN signals s ON fv.signal_id = s.id
            JOIN forward_returns fr ON s.id = fr.signal_id
            WHERE fv.factor_name = ?
              AND s.market_regime = ?
              AND s.date >= ?
              AND fr.days_forward = ?
              AND fv.factor_value IS NOT NULL
              AND fr.forward_return IS NOT NULL
            GROUP BY s.date, s.code
        """, (factor_name, regime, cutoff, forward_days)).fetchall()

        if len(rows) < min_samples:
            return None

        frame = pd.DataFrame([dict(row) for row in rows])
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["factor_value", "forward_return"]
        )
        if len(frame) < min_samples:
            return None
        daily_ic = []
        for _, day in frame.groupby("date"):
            if len(day) < 5:
                continue
            ic, _ = spearmanr(day["factor_value"], day["forward_return"])
            if np.isfinite(ic):
                daily_ic.append(float(ic))
        if not daily_ic:
            return None
        mean_ic = float(np.mean(daily_ic))
        standard_deviation = float(np.std(daily_ic, ddof=1)) if len(daily_ic) > 1 else 0.0
        ic_ir = mean_ic / standard_deviation if standard_deviation > 0 else 0.0
        return {
            "mean_ic": round(mean_ic, 4),
            "ic_ir": round(ic_ir, 4),
            "samples": len(frame),
        }
    finally:
        conn.close()
