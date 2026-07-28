"""把四个榜单、生产因子和候选研究因子追加写入 SQLite。"""
import logging
import sqlite3
from pathlib import Path

from config import OUTPUT_DIR

_logger = logging.getLogger(__name__)
DB_FILENAME = "feedback.db"


def get_db_path() -> Path:
    return OUTPUT_DIR / DB_FILENAME


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def init_db(db_path: Path | None = None) -> None:
    """创建反馈数据库；已有旧数据库时只补充缺失的 strategy 列。"""
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(path)
    try:
        connection.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            code TEXT NOT NULL,
            name TEXT,
            list_type TEXT NOT NULL,
            rank_position INTEGER,
            score REAL,
            close REAL,
            pct_change REAL,
            turnover REAL,
            ma5 REAL,
            ma20 REAL,
            market_regime TEXT NOT NULL,
            strategy TEXT NOT NULL DEFAULT 'trend',
            recorded_at TEXT DEFAULT (datetime('now','localtime')),
            UNIQUE(date, code, list_type)
        );

        CREATE TABLE IF NOT EXISTS factor_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER NOT NULL,
            factor_name TEXT NOT NULL,
            factor_value REAL,
            UNIQUE(signal_id, factor_name),
            FOREIGN KEY (signal_id) REFERENCES signals(id)
        );

        CREATE TABLE IF NOT EXISTS forward_returns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER NOT NULL,
            days_forward INTEGER NOT NULL,
            forward_return REAL,
            close_signal REAL,
            close_forward REAL,
            calculated_date TEXT,
            UNIQUE(signal_id, days_forward),
            FOREIGN KEY (signal_id) REFERENCES signals(id)
        );

        CREATE TABLE IF NOT EXISTS weight_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            market_regime TEXT NOT NULL,
            factors_json TEXT,
            weights_json TEXT,
            method TEXT,
            sample_count INTEGER,
            note TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date);
        CREATE INDEX IF NOT EXISTS idx_signals_code_date ON signals(code, date);
        CREATE INDEX IF NOT EXISTS idx_signals_list_date ON signals(list_type, date);
        CREATE INDEX IF NOT EXISTS idx_signals_regime_date ON signals(market_regime, date);
        CREATE INDEX IF NOT EXISTS idx_factor_signal ON factor_values(signal_id);
        CREATE INDEX IF NOT EXISTS idx_factor_name_signal ON factor_values(factor_name, signal_id);
        CREATE INDEX IF NOT EXISTS idx_fr_signal ON forward_returns(signal_id);
        """)
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(signals)").fetchall()
        }
        if "strategy" not in columns:
            connection.execute(
                "ALTER TABLE signals ADD COLUMN strategy TEXT NOT NULL DEFAULT 'trend'"
            )
        connection.commit()
    finally:
        connection.close()


def record_ranking(
    pool: list,
    date_str: str,
    list_type: str,
    market_regime: str,
    strategy: str = "trend",
    db_path: Path | None = None,
) -> int:
    """记录一个榜单，同时保存生产因子与尚未启用的候选因子。"""
    if not pool:
        return 0
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(path)
    count = 0
    try:
        # 同一次横截面排名必须使用同一个市场数据截面日期。停牌股自己的
        # 最后交易日可能更早，若逐股采用会把今天的信号误写进历史截面。
        signal_date = str(date_str)
        for position, result in enumerate(pool, 1):
            values = {
                "volume_price": float(getattr(result, "vol_price_score", 0.0)),
                "ma_adx": float(getattr(result, "ma_adx_raw", 0.0)),
                "rsrs": float(getattr(result, "rsrs_raw", 0.0)),
                "donchian": float(getattr(result, "donchian_raw", 0.0)),
                "momentum": float(getattr(result, "momentum_raw", 0.0)),
                "momentum_12_2": float(
                    getattr(result, "momentum_12_2_raw", 0.0)
                ),
                "high_52w": float(getattr(result, "high_52w_raw", 0.0)),
                "multi_horizon": float(
                    getattr(result, "multi_horizon_raw", 0.0)
                ),
                "industry_residual_momentum": float(
                    getattr(result, "industry_residual_momentum_raw", 0.0)
                ),
            }
            connection.execute(
                """INSERT INTO signals
                   (date, code, name, list_type, rank_position, score, close, pct_change,
                    turnover, ma5, ma20, market_regime, strategy)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(date, code, list_type) DO UPDATE SET
                       name=excluded.name,
                       rank_position=excluded.rank_position,
                       score=excluded.score,
                       close=excluded.close,
                       pct_change=excluded.pct_change,
                       turnover=excluded.turnover,
                       ma5=excluded.ma5,
                       ma20=excluded.ma20,
                       market_regime=excluded.market_regime,
                       strategy=excluded.strategy""",
                (
                    signal_date, str(result.code), str(result.name), list_type, position,
                    float(getattr(result, "score", 0.0)), float(getattr(result, "close", 0.0)),
                    float(getattr(result, "pct_change", 0.0)), float(getattr(result, "turnover", 0.0)),
                    float(getattr(result, "ma5", 0.0)), float(getattr(result, "ma20", 0.0)),
                    market_regime, strategy,
                ),
            )
            row = connection.execute(
                "SELECT id FROM signals WHERE date=? AND code=? AND list_type=?",
                (signal_date, str(result.code), list_type),
            ).fetchone()
            if not row:
                continue
            signal_id = row[0]
            connection.executemany(
                """INSERT INTO factor_values (signal_id, factor_name, factor_value)
                   VALUES (?, ?, ?)
                   ON CONFLICT(signal_id, factor_name) DO UPDATE SET
                       factor_value=excluded.factor_value""",
                [(signal_id, name, value) for name, value in values.items()],
            )
            count += 1
        connection.commit()
    except Exception:
        connection.rollback()
        _logger.warning("记录榜单 %s 失败", list_type, exc_info=True)
    finally:
        connection.close()
    return count
