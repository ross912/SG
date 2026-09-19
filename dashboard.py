"""SG Quant 登录保护仪表盘：榜单、进度、新闻、每日总结与数据问答。"""
import hmac
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    stream_with_context,
    url_for,
)

from config import OUTPUT_DIR, ROOT
from data.index_filter import load_cached_market_overview
from data.storage import load_kline
from screening.scanner import scan_stock
from services.deepseek import DeepSeekError, is_configured
from services.market_ai import (
    generate_daily_market_summary,
    load_latest_snapshot,
    list_summary_history,
    load_latest_summary,
    load_summary_for_date,
    load_summary_status,
    stream_market_chat,
)
from services.news import load_latest_news, refresh_news
from services.progress import complete as complete_progress
from services.progress import fail as fail_progress
from services.progress import read as read_progress
from services.web_search import search_latest_news, should_search_web


app = Flask(__name__)
app.secret_key = os.getenv("SG_QUANT_SESSION_SECRET", "").strip() or secrets.token_hex(32)
app.config.update(
    DASHBOARD_PASSWORD=os.getenv("SG_QUANT_DASHBOARD_PASSWORD", "").strip(),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("SG_QUANT_COOKIE_SECURE", "0") == "1",
    PERMANENT_SESSION_LIFETIME=12 * 60 * 60,
    MAX_CONTENT_LENGTH=64 * 1024,
)

LISTS = {
    "fundamental30": "基本面价值 · 全市场 Top 30",
    "all30": "趋势跟踪 · 全市场 Top 30",
    "main10": "趋势跟踪 · 主板 Top 10",
    "mr_all30": "均值回归 · 全市场 Top 30",
    "mr_main10": "均值回归 · 主板 Top 10",
}
_FILE_PATTERN = re.compile(
    r"^stock_pool_(\d{4}-\d{2}-\d{2})_"
    r"(fundamental30|main10|all30|mr_main10|mr_all30)\.csv$"
)
_run_process: subprocess.Popen | None = None
_news_thread: threading.Thread | None = None
_summary_thread: threading.Thread | None = None
_task_lock = threading.Lock()
_chat_lock = threading.Lock()
_chat_requests: dict[str, deque[float]] = defaultdict(deque)
_login_lock = threading.Lock()
_login_attempts: dict[str, deque[float]] = defaultdict(deque)


@app.before_request
def require_authentication():
    if app.config.get("TESTING") and app.config.get("BYPASS_AUTH_FOR_TESTS", True):
        return None
    if request.endpoint in {"login", "static"}:
        return None
    if not app.config["DASHBOARD_PASSWORD"]:
        if request.path.startswith("/api/"):
            return jsonify({"error": "仪表盘密码未配置"}), 503
        return render_template("login.html", error="服务器尚未配置仪表盘密码"), 503
    if not session.get("authenticated"):
        if request.path.startswith("/api/"):
            return jsonify({"error": "请先登录"}), 401
        return redirect(url_for("login", next=request.full_path))
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.path.startswith("/api/"):
        supplied = request.headers.get("X-CSRF-Token", "")
        expected = session.get("csrf_token", "")
        if not expected or not hmac.compare_digest(supplied, expected):
            return jsonify({"error": "请求校验失败，请刷新页面重试"}), 403
    return None


@app.context_processor
def inject_page_context():
    return {"csrf_token": session.get("csrf_token", "")}


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        if not _allow_login_attempt(check_only=True):
            return render_template(
                "login.html", error="尝试次数过多，请稍后再试",
            ), 429
        expected = app.config["DASHBOARD_PASSWORD"]
        supplied = request.form.get("password", "")
        if expected and hmac.compare_digest(supplied, expected):
            _clear_login_attempts()
            session.clear()
            session["authenticated"] = True
            session["csrf_token"] = secrets.token_urlsafe(24)
            session.permanent = True
            target = request.args.get("next", "/")
            if not target.startswith("/") or target.startswith("//"):
                target = "/"
            return redirect(target)
        _allow_login_attempt(check_only=False)
        error = "密码不正确"
    return render_template("login.html", error=error)


@app.post("/logout")
def logout():
    supplied = request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    if expected and hmac.compare_digest(supplied, expected):
        session.clear()
    return redirect(url_for("login"))


@app.get("/")
def index():
    return render_template(
        "dashboard.html",
        page="dashboard",
        lists=[{"id": key, "label": label} for key, label in LISTS.items()],
    )


@app.get("/news")
def news_page():
    return render_template("news.html", page="news")


@app.get("/summary")
def summary_page():
    return render_template("summary.html", page="summary")


@app.get("/chat")
def chat_page():
    return render_template("chat.html", page="chat")


@app.get("/api/health")
def health():
    progress = _current_progress()
    return jsonify({
        "ok": True,
        "running": progress.get("state") == "running",
        "deepseek_configured": is_configured(),
        "progress": progress,
    })


@app.get("/api/market_overview")
def market_overview_api():
    payload = load_cached_market_overview()
    if payload:
        return jsonify(payload)
    return jsonify({
        "regime": "unknown", "label": "待更新", "score": 50,
        "color_hue": 48, "market_date": "", "generated_at": "", "indices": [],
        "details": "尚无四指数缓存，请先执行全市场流程",
    })


@app.get("/api/run/status")
def run_status():
    progress = _current_progress()
    return jsonify({
        "running": progress.get("state") == "running",
        "progress": progress,
        "log_tail": _tail_log(40),
    })


@app.post("/api/run")
def run_full():
    global _run_process
    progress = _current_progress()
    if (_run_process is not None and _run_process.poll() is None) or _recently_running(progress):
        return jsonify({"message": "全市场流程已在运行"}), 409
    log_path = OUTPUT_DIR / "online_run.log"
    log_handle = log_path.open("w", encoding="utf-8")
    _run_process = subprocess.Popen(
        [sys.executable, "-u", "main.py", "--full"],
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    log_handle.close()
    return jsonify({"message": "全市场流程已启动，可在本页查看实时进度"})


@app.get("/api/stock_pool/<list_id>")
def stock_pool(list_id: str):
    if list_id not in LISTS:
        return jsonify({"error": "未知榜单"}), 404
    path, date_value = _latest_pool(list_id)
    if path is None:
        return jsonify({"label": LISTS[list_id], "date": "", "rows": []})
    frame = pd.read_csv(path, dtype={"代码": str}).fillna("")
    return jsonify({"label": LISTS[list_id], "date": date_value, "rows": frame.to_dict("records")})


@app.get("/api/news")
def news_api():
    payload = load_latest_news()
    payload["refreshing"] = _news_thread is not None and _news_thread.is_alive()
    return jsonify(payload)


@app.post("/api/news/refresh")
def refresh_news_api():
    global _news_thread
    if _current_progress().get("state") == "running":
        return jsonify({"message": "全市场流程正在运行，新闻将在流程后段自动更新"}), 409
    with _task_lock:
        if _news_thread is not None and _news_thread.is_alive():
            return jsonify({"message": "新闻正在更新"}), 409
        _news_thread = threading.Thread(target=_refresh_news_task, daemon=True)
        _news_thread.start()
    return jsonify({"message": "新闻更新已启动"})


@app.get("/api/summary")
def summary_api():
    date_value = request.args.get("date", "").strip()
    payload = load_summary_for_date(date_value) if date_value else load_latest_summary()
    summary_progress = load_summary_status()
    pipeline_progress = _current_progress()
    return jsonify({
        "summary": payload,
        "history": list_summary_history(),
        "generating": _summary_is_running(summary_progress, pipeline_progress),
        "summary_progress": summary_progress,
        "pipeline_progress": pipeline_progress,
    })


@app.post("/api/summary/generate")
def generate_summary_api():
    global _summary_thread
    pipeline_progress = _current_progress()
    if pipeline_progress.get("state") == "running":
        return jsonify({"message": "全市场流程正在运行，完成后会自动生成每日总结"}), 409
    with _task_lock:
        if _summary_is_running(load_summary_status(), pipeline_progress):
            return jsonify({"message": "每日总结正在生成"}), 409
        _summary_thread = threading.Thread(target=_summary_task, daemon=True)
        _summary_thread.start()
    return jsonify({"message": "每日总结生成已启动"})


@app.post("/api/chat")
def chat_api():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    history = payload.get("history", [])
    web_search_mode = str(payload.get("web_search", "auto")).strip().lower()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    if len(question) > 1200:
        return jsonify({"error": "单次问题不能超过 1200 字"}), 400
    if not isinstance(history, list):
        return jsonify({"error": "对话历史格式错误"}), 400
    if web_search_mode not in {"auto", "on", "off"}:
        return jsonify({"error": "联网模式无效"}), 400
    if not is_configured():
        return jsonify({"error": "DeepSeek API 未配置"}), 503
    if not _allow_chat_request():
        return jsonify({"error": "请求过于频繁，请稍后再试"}), 429

    @stream_with_context
    def event_stream():
        try:
            web_payload = None
            triggered = should_search_web(question, web_search_mode)
            if triggered:
                yield _sse({"status": "正在联网检索近期资讯…", "searching": True})
                web_payload = search_latest_news(question)
                public_results = [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "source": item.get("source", ""),
                        "published_at": item.get("published_at", ""),
                    }
                    for item in web_payload.get("results", [])[:8]
                ]
                yield _sse({
                    "searching": False,
                    "search": {
                        "mode": web_search_mode,
                        "triggered": True,
                        "count": len(public_results),
                        "providers": web_payload.get("providers", []),
                        "searched_at": web_payload.get("searched_at", ""),
                        "cached": bool(web_payload.get("cached")),
                        "results": public_results,
                    },
                    "status": (
                        f"已获取 {len(public_results)} 条近期资讯"
                        if public_results else "联网检索未返回结果，继续使用本地数据"
                    ),
                })
            else:
                yield _sse({
                    "searching": False,
                    "search": {
                        "mode": web_search_mode, "triggered": False, "count": 0,
                        "providers": [], "results": [],
                    },
                    "status": "本次问题使用平台本地数据",
                })
            for chunk in stream_market_chat(
                question, history, web_search_payload=web_payload,
            ):
                yield _sse({"delta": chunk})
            yield _sse({"done": True})
        except DeepSeekError as error:
            yield _sse({"error": str(error), "done": True})
        except Exception as error:
            yield _sse({"error": f"问答失败：{type(error).__name__}", "done": True})

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/stock/<code>")
def stock_detail(code: str):
    normalized = str(code).strip().zfill(6)
    result = scan_stock(normalized)
    if result is None:
        return jsonify({"error": "日线数据不足或未通过资格过滤"}), 404
    return jsonify({
        "code": result.code, "name": result.name, "date": result.as_of_date,
        "close": result.close, "pct_change": result.pct_change, "turnover": result.turnover,
        "score": result.score, "volume_price": result.vol_price_score,
        "ma_adx": result.ma_adx_raw, "rsrs": result.rsrs_raw,
        "donchian": result.donchian_raw, "momentum": result.momentum_raw,
    })


@app.get("/api/kline/<code>")
def kline(code: str):
    frame = load_kline(str(code).strip().zfill(6))
    if frame is None:
        return jsonify({"error": "没有本地日线"}), 404
    data = frame.tail(240).copy()
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    keep = [column for column in ("date", "open", "high", "low", "close", "volume") if column in data]
    return jsonify(data[keep].to_dict("records"))


def _latest_pool(list_id: str) -> tuple[Path | None, str]:
    matches = []
    for path in OUTPUT_DIR.glob("stock_pool_*.csv"):
        match = _FILE_PATTERN.match(path.name)
        if match and match.group(2) == list_id:
            matches.append((match.group(1), path))
    if not matches:
        return None, ""
    date_value, path = max(matches, key=lambda item: item[0])
    return path, date_value


def _current_progress() -> dict:
    global _run_process
    progress = read_progress()
    if _run_process is not None:
        return_code = _run_process.poll()
        if return_code is not None and progress.get("state") == "running":
            run_id = progress.get("run_id", "")
            if return_code == 0:
                complete_progress(run_id)
            else:
                fail_progress(run_id, f"全市场子进程退出，代码 {return_code}")
            progress = read_progress()
    return progress


def _recently_running(progress: dict) -> bool:
    if progress.get("state") != "running":
        return False
    try:
        updated = datetime.fromisoformat(progress.get("updated_at", ""))
        return (datetime.now().astimezone() - updated).total_seconds() < 15 * 60
    except (TypeError, ValueError):
        return False


def _tail_log(line_count: int) -> list[str]:
    path = OUTPUT_DIR / "online_run.log"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return list(deque(handle, maxlen=line_count))
    except OSError:
        return []


def _refresh_news_task() -> None:
    try:
        refresh_news(hours=24, limit=50)
    except Exception:
        app.logger.exception("新闻刷新失败")


def _summary_task() -> None:
    try:
        snapshot = load_latest_snapshot()
        snapshot_date = str(snapshot.get("snapshot_date", ""))
        latest = load_latest_summary()
        if not snapshot_date:
            snapshot_date = latest.get("snapshot_date", datetime.now().strftime("%Y-%m-%d"))
        generate_daily_market_summary(
            date_str=datetime.now().strftime("%Y-%m-%d"),
            snapshot_date=snapshot_date,
            news_payload=load_latest_news(),
        )
    except Exception:
        app.logger.exception("每日总结生成失败")


def _allow_chat_request() -> bool:
    limit = max(1, int(os.getenv("SG_QUANT_CHAT_RATE_LIMIT", "30")))
    now = time.time()
    key = str(session.get("csrf_token") or request.remote_addr or "anonymous")
    with _chat_lock:
        bucket = _chat_requests[key]
        while bucket and now - bucket[0] > 3600:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
    return True


def _summary_is_running(summary_progress: dict, pipeline_progress: dict) -> bool:
    if _summary_thread is not None and _summary_thread.is_alive():
        return True
    if (
        pipeline_progress.get("state") == "running"
        and pipeline_progress.get("stage") in {"news", "daily_summary", "finalizing"}
    ):
        return True
    if summary_progress.get("state") != "running":
        return False
    try:
        updated = datetime.fromisoformat(summary_progress.get("updated_at", ""))
        return (datetime.now().astimezone() - updated).total_seconds() < 10 * 60
    except (TypeError, ValueError):
        return False


def _login_key() -> str:
    return str(request.remote_addr or "unknown")


def _allow_login_attempt(*, check_only: bool) -> bool:
    limit = max(1, int(os.getenv("SG_QUANT_LOGIN_RATE_LIMIT", "10")))
    now = time.time()
    key = _login_key()
    with _login_lock:
        bucket = _login_attempts[key]
        while bucket and now - bucket[0] > 3600:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        if not check_only:
            bucket.append(now)
    return True


def _clear_login_attempts() -> None:
    with _login_lock:
        _login_attempts.pop(_login_key(), None)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


if __name__ == "__main__":
    host = os.getenv("SG_QUANT_HOST", "127.0.0.1")
    port = int(os.getenv("SG_QUANT_PORT", "5000"))
    if os.getenv("SG_QUANT_OPEN_BROWSER", "1") == "1" and host in {"127.0.0.1", "localhost"}:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host=host, port=port, debug=False)
