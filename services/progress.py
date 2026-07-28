"""全市场流程的持久化进度状态。"""
import threading
import uuid
from datetime import datetime
from typing import Any

from config import OUTPUT_DIR
from services.storage import read_json, write_json


STATUS_PATH = OUTPUT_DIR / "run_status.json"
_lock = threading.Lock()


def start(mode: str) -> str:
    run_id = uuid.uuid4().hex
    now = _now()
    _write({
        "run_id": run_id,
        "mode": mode,
        "state": "running",
        "stage": "initializing",
        "percent": 1,
        "message": "正在初始化全市场流程",
        "current": 0,
        "total": 0,
        "valid": 0,
        "warnings": [],
        "warning_count": 0,
        "started_at": now,
        "updated_at": now,
        "finished_at": "",
    })
    return run_id


def update(
    run_id: str,
    *,
    stage: str,
    percent: float,
    message: str,
    current: int | None = None,
    total: int | None = None,
    valid: int | None = None,
) -> None:
    with _lock:
        payload = read_json(STATUS_PATH, {}) or {}
        if payload.get("run_id") != run_id:
            return
        payload.update({
            "state": "running",
            "stage": stage,
            "percent": round(min(max(float(percent), 0.0), 100.0), 1),
            "message": str(message),
            "updated_at": _now(),
        })
        if current is not None:
            payload["current"] = int(current)
        if total is not None:
            payload["total"] = int(total)
        if valid is not None:
            payload["valid"] = int(valid)
        write_json(STATUS_PATH, payload)


def complete(run_id: str, message: str = "全市场流程已完成") -> None:
    payload = read_json(STATUS_PATH, {}) or {}
    warnings = payload.get("warnings") or []
    if warnings:
        _finish(
            run_id,
            "completed_with_warnings",
            100,
            f"{message}（有 {len(warnings)} 项数据警告）",
        )
    else:
        _finish(run_id, "completed", 100, message)


def add_warning(run_id: str, message: str) -> None:
    """追加非致命警告；流程保持 running，最终以有警告状态完成。"""
    warning = str(message).strip()[:1200]
    if not warning:
        return
    with _lock:
        payload = read_json(STATUS_PATH, {}) or {}
        if payload.get("run_id") != run_id:
            return
        warnings = list(payload.get("warnings") or [])
        if warning not in warnings:
            warnings.append(warning)
        payload.update({
            "warnings": warnings,
            "warning_count": len(warnings),
            "updated_at": _now(),
        })
        write_json(STATUS_PATH, payload)


def fail(run_id: str, message: str) -> None:
    payload = read_json(STATUS_PATH, {}) or {}
    percent = float(payload.get("percent", 0))
    _finish(run_id, "failed", percent, str(message)[:300])


def read() -> dict[str, Any]:
    payload = read_json(STATUS_PATH, None)
    if not isinstance(payload, dict):
        return {
            "state": "idle", "stage": "idle", "percent": 0,
            "message": "尚未执行全市场流程", "current": 0,
            "total": 0, "valid": 0,
        }
    return payload


def _finish(run_id: str, state: str, percent: float, message: str) -> None:
    with _lock:
        payload = read_json(STATUS_PATH, {}) or {}
        if payload.get("run_id") != run_id:
            return
        now = _now()
        payload.update({
            "state": state,
            "stage": state,
            "percent": round(float(percent), 1),
            "message": message,
            "updated_at": now,
            "finished_at": now,
        })
        write_json(STATUS_PATH, payload)


def _write(payload: dict[str, Any]) -> None:
    with _lock:
        write_json(STATUS_PATH, payload)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
