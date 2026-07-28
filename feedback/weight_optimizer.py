"""固定五因子的周度 IC 调权；不会增删因子或自动翻转策略方向。"""
import json
import sqlite3
from datetime import datetime

from config import FACTOR_NAMES, OUTPUT_DIR, SCORE_WEIGHTS
from feedback.returns_tracker import compute_ic_for_regime

DB_PATH = OUTPUT_DIR / "feedback.db"
WEIGHTS_CACHE_PATH = OUTPUT_DIR / "optimized_weights.json"


def load_optimized_weights() -> dict[str, dict[str, float]] | None:
    """加载已保存的分市场状态权重；旧格式中的非五因子字段会被忽略。"""
    if not WEIGHTS_CACHE_PATH.exists():
        return None
    try:
        with WEIGHTS_CACHE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        weights_by_regime = {}
        for regime, data in payload.get("regimes", {}).items():
            weights_by_regime[regime] = {
                name: float(value)
                for name, value in data.get("weights", {}).items()
                if name in FACTOR_NAMES
            }
        return weights_by_regime
    except Exception:
        return None


def auto_weekly_optimize(
    min_samples: int = 50,
    min_interval_days: int = 5,
    lookback: int = 60,
    smoothing: float = 0.5,
    verbose: bool = True,
) -> dict | None:
    """在样本充足且距上次调权满五天时，按五日 Rank IC 平滑调整权重。"""
    if not DB_PATH.exists():
        return None
    updated = {}
    for regime in _available_regimes():
        days_since = _days_since_last_update(regime)
        if days_since is not None and days_since < min_interval_days:
            if verbose:
                print(f"五因子调权跳过 {regime}：距上次仅 {days_since} 天")
            continue
        ic_values = {}
        samples = []
        for factor in FACTOR_NAMES:
            result = compute_ic_for_regime(
                DB_PATH,
                factor,
                regime,
                lookback_days=lookback,
                forward_days=5,
                min_samples=min_samples,
            )
            if result:
                ic_values[factor] = float(result["mean_ic"])
                samples.append(int(result["samples"]))
        if len(ic_values) != len(FACTOR_NAMES):
            if verbose:
                print(f"五因子调权跳过 {regime}：样本不足")
            continue

        absolute_total = sum(abs(value) for value in ic_values.values())
        if absolute_total <= 0:
            continue
        target = {name: abs(value) / absolute_total for name, value in ic_values.items()}
        base = SCORE_WEIGHTS.get(regime, SCORE_WEIGHTS["sideways"])
        blended = {
            name: (1 - smoothing) * base[name] + smoothing * target[name]
            for name in FACTOR_NAMES
        }
        normalized = _normalize(blended)
        SCORE_WEIGHTS[regime] = normalized
        updated[regime] = normalized
        _record_history(regime, normalized, min(samples), lookback)
        if verbose:
            compact = ", ".join(f"{name}={weight:.3f}" for name, weight in normalized.items())
            print(f"五因子调权完成 {regime}: {compact}")

    if not updated:
        return None
    _save_optimized_weights(updated)
    return {"regimes": list(updated), "weights_by_regime": updated, "updated": True}


def _available_regimes() -> list[str]:
    connection = sqlite3.connect(DB_PATH)
    try:
        return [
            row[0] for row in connection.execute(
                "SELECT DISTINCT market_regime FROM signals WHERE market_regime IS NOT NULL"
            ).fetchall()
        ]
    finally:
        connection.close()


def _days_since_last_update(regime: str) -> int | None:
    connection = sqlite3.connect(DB_PATH)
    try:
        row = connection.execute(
            """SELECT date FROM weight_history
               WHERE market_regime = ? ORDER BY date DESC LIMIT 1""",
            (regime,),
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    return (datetime.now() - datetime.strptime(row[0], "%Y-%m-%d")).days


def _record_history(regime: str, weights: dict[str, float], samples: int, lookback: int) -> None:
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.execute(
            """INSERT INTO weight_history
               (date, market_regime, factors_json, weights_json, method, sample_count, note)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().strftime("%Y-%m-%d"),
                regime,
                json.dumps(list(FACTOR_NAMES), ensure_ascii=False),
                json.dumps(weights, ensure_ascii=False),
                "fixed-five-weekly-ic",
                samples,
                f"lookback={lookback}",
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _save_optimized_weights(weights_by_regime: dict[str, dict[str, float]]) -> None:
    merged = load_optimized_weights() or {}
    merged.update(weights_by_regime)
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "factors": list(FACTOR_NAMES),
        "regimes": {
            regime: {
                "weights": weights,
            }
            for regime, weights in merged.items()
        },
    }
    with WEIGHTS_CACHE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    normalized = {name: value / total for name, value in weights.items()}
    # 让浮点和严格等于 1，差额放到当前最大权重因子。
    rounded = {name: round(value, 6) for name, value in normalized.items()}
    largest = max(rounded, key=rounded.get)
    rounded[largest] = round(rounded[largest] + 1.0 - sum(rounded.values()), 6)
    return rounded
