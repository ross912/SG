"""新闻、DeepSeek、每日总结、进度与鉴权的离线测试。"""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from dashboard import app
from screening.scanner import ScanResult
from services import deepseek, market_ai, news, progress, web_search


class DeepSeekClientTests(unittest.TestCase):
    def test_completion_uses_configured_key_without_returning_it(self):
        response = Mock()
        response.ok = True
        response.json.return_value = {
            "model": "deepseek-v4-flash",
            "choices": [{"message": {"content": "测试结果"}}],
            "usage": {"total_tokens": 9},
        }
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret"}), patch(
            "services.deepseek.requests.post", return_value=response,
        ) as request_mock:
            content, metadata = deepseek.complete([
                {"role": "user", "content": "测试"},
            ])
        self.assertEqual(content, "测试结果")
        self.assertEqual(metadata["usage"]["total_tokens"], 9)
        headers = request_mock.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-secret")
        self.assertNotIn("test-secret", str((content, metadata)))
        request_payload = request_mock.call_args.kwargs["json"]
        self.assertEqual(request_payload["model"], "deepseek-v4-flash")
        self.assertEqual(request_payload["thinking"], {"type": "disabled"})


class NewsServiceTests(unittest.TestCase):
    def test_news_keeps_original_title_without_deepseek(self):
        now = pd.Timestamp.now().isoformat()
        item = {
            "id": "abc", "fingerprint": "fingerprint", "source": "东方财富",
            "original_title": "原始标题", "content": "公司披露重要公告。",
            "published_at": now, "url": "https://example.com",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            intelligence_dir = Path(temp_dir)
            with patch.object(news, "INTELLIGENCE_DIR", intelligence_dir), patch.object(
                news, "LATEST_NEWS_PATH", intelligence_dir / "latest_news.json",
            ), patch.object(
                news, "fetch_latest_news", return_value=([dict(item)], []),
            ):
                payload = news.refresh_news(hours=24, limit=50)
                saved = news.load_latest_news()
        self.assertEqual(payload["item_count"], 1)
        self.assertEqual(payload["items"][0]["title"], "原始标题")
        self.assertNotIn("ai_title", payload["items"][0])
        self.assertNotIn("summary_status", payload["items"][0])
        self.assertEqual(saved["items"][0]["id"], "abc")

    def test_news_empty_refresh_keeps_previous_cache(self):
        item = {
            "id": "abc", "fingerprint": "fingerprint", "source": "新浪财经",
            "original_title": "原始标题", "content": "原始内容",
            "published_at": pd.Timestamp.now().isoformat(), "url": "https://example.com",
            "title": "原始标题",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            intelligence_dir = Path(temp_dir)
            news_path = intelligence_dir / "latest_news.json"
            news_path.parent.mkdir(parents=True, exist_ok=True)
            news.write_json(news_path, {"items": [item], "generated_at": "2026-07-22T09:00:00+08:00"})
            with patch.object(news, "INTELLIGENCE_DIR", intelligence_dir), patch.object(
                news, "LATEST_NEWS_PATH", intelligence_dir / "latest_news.json",
            ), patch.object(
                news, "fetch_latest_news", return_value=([], ["新闻源不可用"]),
            ):
                payload = news.refresh_news()
        self.assertEqual(payload["items"][0]["title"], "原始标题")
        self.assertEqual(payload["refresh_errors"], ["新闻源不可用"])


class WebSearchTests(unittest.TestCase):
    def setUp(self):
        web_search._CACHE.clear()

    def test_auto_mode_only_triggers_for_fresh_or_event_driven_questions(self):
        self.assertTrue(web_search.should_search_web("今天有哪些最新财经新闻？", "auto"))
        self.assertTrue(web_search.should_search_web("600000大跌的原因是什么？", "auto"))
        self.assertFalse(web_search.should_search_web("解释一下RSRS指标", "auto"))
        self.assertTrue(web_search.should_search_web("解释一下RSRS指标", "on"))
        self.assertFalse(web_search.should_search_web("最新消息", "off"))
        self.assertEqual(
            web_search._build_query("今天A股市场有哪些最新政策与重要公司消息？"),
            "A股 (政策 OR 公告 OR 公司)",
        )
        self.assertEqual(
            web_search._build_tavily_query("今天A股市场有哪些最新政策与重要公司消息？"),
            "中国A股市场最近一周的重要政策、监管动态和上市公司公告",
        )

    def test_rss_parser_keeps_source_date_and_safe_url(self):
        sample = """<?xml version="1.0" encoding="UTF-8"?>
        <rss><channel><item>
          <title>测试公司发布公告 - 测试财经</title>
          <link>https://example.com/story</link>
          <pubDate>Wed, 22 Jul 2026 09:30:00 GMT</pubDate>
          <source>测试财经</source>
          <description><![CDATA[<p>公告摘要</p>]]></description>
        </item></channel></rss>""".encode()
        rows = web_search._parse_rss(sample, provider="测试搜索")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "测试财经")
        self.assertEqual(rows[0]["snippet"], "公告摘要")
        self.assertTrue(rows[0]["published_at"].startswith("2026-07-22"))

    def test_chat_messages_include_numbered_live_sources(self):
        live = {
            "query": "测试",
            "searched_at": "2026-07-23T12:00:00+08:00",
            "providers": ["Google News"],
            "results": [{
                "title": "测试新闻", "source": "测试财经",
                "published_at": "2026-07-23T10:00:00+08:00",
                "snippet": "测试摘要", "url": "https://example.com/news",
            }],
        }
        with patch.object(market_ai, "build_market_context", return_value={}), patch.object(
            market_ai, "load_latest_summary", return_value={},
        ), patch.object(market_ai, "_question_stock_context", return_value=[]):
            messages = market_ai.build_chat_messages("最新消息", [], web_search_payload=live)
        self.assertIn('"citation": "联网1"', messages[1]["content"])
        self.assertIn("https://example.com/news", messages[1]["content"])

    def test_tavily_is_primary_and_skips_rss_when_results_are_sufficient(self):
        rows = [
            {
                "title": f"测试新闻{index}", "url": f"https://example.com/{index}",
                "source": "测试财经", "published_at": "2026-07-23T10:00:00+08:00",
                "snippet": "摘要", "provider": "Tavily",
            }
            for index in range(4)
        ]
        with patch.dict(os.environ, {
            "TAVILY_API_KEY": "test-key",
            "SG_QUANT_WEB_SEARCH_PROVIDER": "auto",
        }), patch.object(
            web_search, "_fetch_tavily", return_value=rows,
        ) as tavily_mock, patch.object(
            web_search, "_fetch_rss_fallbacks",
        ) as rss_mock:
            payload = web_search.search_latest_news("最新政策", max_results=6)
        self.assertEqual(payload["providers"], ["Tavily"])
        self.assertEqual(len(payload["results"]), 4)
        tavily_mock.assert_called_once()
        rss_mock.assert_not_called()

    def test_tavily_failure_falls_back_to_rss(self):
        rss_row = {
            "title": "RSS 新闻", "url": "https://example.com/rss",
            "source": "测试财经", "published_at": "2026-07-23T10:00:00+08:00",
            "snippet": "摘要", "provider": "Google News",
        }
        with patch.dict(os.environ, {
            "TAVILY_API_KEY": "test-key",
            "SG_QUANT_WEB_SEARCH_PROVIDER": "auto",
        }), patch.object(
            web_search, "_fetch_tavily", side_effect=TimeoutError,
        ), patch.object(
            web_search, "_fetch_rss_fallbacks",
            return_value=([rss_row], ["Google News"], []),
        ) as rss_mock:
            payload = web_search.search_latest_news("最新政策", max_results=6)
        self.assertEqual(payload["providers"], ["Google News"])
        self.assertEqual(payload["results"][0]["provider"], "Google News")
        self.assertIn("Tavily: TimeoutError", payload["errors"])
        rss_mock.assert_called_once()

    def test_tavily_request_uses_one_credit_basic_news_search(self):
        response = Mock()
        response.json.return_value = {
            "results": [{
                "title": "测试新闻", "url": "https://example.com/news",
                "content": "新闻摘要", "published_date": "2026-07-23T08:00:00Z",
            }],
        }
        response.raise_for_status.return_value = None
        with patch("services.web_search.requests.post", return_value=response) as request_mock:
            rows = web_search._fetch_tavily(
                "A股 市场", api_key="test-key", timeout=8, max_results=6,
            )
        request_payload = request_mock.call_args.kwargs["json"]
        self.assertEqual(request_payload["search_depth"], "basic")
        self.assertEqual(request_payload["topic"], "finance")
        self.assertEqual(request_payload["time_range"], "week")
        self.assertFalse(request_payload["include_answer"])
        self.assertIn("eastmoney.com", request_payload["include_domains"])
        self.assertEqual(rows[0]["provider"], "Tavily")
        self.assertNotIn("test-key", str(rows))


class MarketAiTests(unittest.TestCase):
    def test_market_snapshot_contains_breadth_and_extremes(self):
        results = [
            ScanResult(code="000001", name="甲", pct_change=2.0, turnover=3.0, close=10.0),
            ScanResult(code="000002", name="乙", pct_change=-1.0, turnover=5.0, close=20.0),
            ScanResult(code="000003", name="丙", pct_change=0.0, turnover=1.0, close=30.0),
        ]
        regime = SimpleNamespace(regime="sideways", score=50, reason="测试")
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            with patch.object(market_ai, "INTELLIGENCE_DIR", directory), patch.object(
                market_ai, "LATEST_SNAPSHOT_PATH", directory / "latest_market_snapshot.json",
            ):
                payload = market_ai.save_market_snapshot(
                    results,
                    snapshot_date="2026-07-22",
                    regime=regime,
                    position={"ratio_pct": 50.0, "label": "中等"},
                )
        self.assertEqual(payload["breadth"]["advancers"], 1)
        self.assertEqual(payload["breadth"]["decliners"], 1)
        self.assertEqual(payload["top_gainers"][0]["code"], "000001")
        self.assertEqual(payload["highest_turnover"][0]["code"], "000002")

    def test_market_snapshot_breadth_uses_unfiltered_full_market_daily_data(self):
        results = [
            ScanResult(code="000001", name="候选股", pct_change=-1.0, close=10.0),
        ]
        full_market = pd.DataFrame({
            "pct_change": [2.0, 1.0, -0.5, 0.0],
        })
        regime = SimpleNamespace(regime="sideways", score=50)
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            with patch.object(market_ai, "INTELLIGENCE_DIR", directory), patch.object(
                market_ai, "LATEST_SNAPSHOT_PATH", directory / "latest_market_snapshot.json",
            ):
                payload = market_ai.save_market_snapshot(
                    results,
                    snapshot_date="2026-07-23",
                    regime=regime,
                    position={"ratio_pct": 50.0, "label": "中等"},
                    market_daily_snapshot=full_market,
                )
        self.assertEqual(payload["breadth_scope"], "full_market_traded")
        self.assertEqual(payload["market_count"], 4)
        self.assertEqual(payload["breadth"]["advancers"], 2)
        self.assertEqual(payload["breadth"]["decliners"], 1)

    def test_daily_summary_persists_completed_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            with patch.object(market_ai, "INTELLIGENCE_DIR", directory), patch.object(
                market_ai, "LATEST_SUMMARY_PATH", directory / "latest_market_summary.json",
            ), patch.object(
                market_ai, "SUMMARY_STATUS_PATH", directory / "summary_status.json",
            ), patch.object(
                market_ai, "build_market_context", return_value={"market_snapshot": {}},
            ), patch.object(
                market_ai, "complete", return_value=("测试总结", {"model": "deepseek-v4-flash"}),
            ):
                result = market_ai.generate_daily_market_summary(
                    date_str="2026-07-22", snapshot_date="2026-07-22",
                )
                status = market_ai.load_summary_status()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["percent"], 100.0)


class ProgressTests(unittest.TestCase):
    def test_progress_lifecycle_is_persisted(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            progress, "STATUS_PATH", Path(temp_dir) / "status.json",
        ):
            run_id = progress.start("full")
            progress.update(
                run_id, stage="scanning", percent=45,
                message="扫描中", current=100, total=200, valid=95,
            )
            middle = progress.read()
            progress.complete(run_id)
            finished = progress.read()
        self.assertEqual(middle["current"], 100)
        self.assertEqual(middle["valid"], 95)
        self.assertEqual(finished["state"], "completed")
        self.assertEqual(finished["percent"], 100.0)

    def test_nonfatal_warning_finishes_as_completed_with_warnings(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            progress, "STATUS_PATH", Path(temp_dir) / "status.json",
        ):
            run_id = progress.start("full")
            progress.add_warning(run_id, "1只股票缺少当日数据")
            progress.complete(run_id)
            finished = progress.read()
        self.assertEqual(finished["state"], "completed_with_warnings")
        self.assertEqual(finished["warning_count"], 1)
        self.assertEqual(finished["warnings"], ["1只股票缺少当日数据"])


class DashboardAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.previous = {
            "TESTING": app.config.get("TESTING"),
            "BYPASS_AUTH_FOR_TESTS": app.config.get("BYPASS_AUTH_FOR_TESTS"),
            "DASHBOARD_PASSWORD": app.config.get("DASHBOARD_PASSWORD"),
        }
        app.config.update(
            TESTING=True,
            BYPASS_AUTH_FOR_TESTS=False,
            DASHBOARD_PASSWORD="test-password",
        )
        self.client = app.test_client()

    def tearDown(self):
        app.config.update(self.previous)

    def test_login_protects_pages_and_api(self):
        self.assertEqual(self.client.get("/").status_code, 302)
        self.assertEqual(self.client.get("/api/news").status_code, 401)
        response = self.client.post("/login", data={"password": "test-password"})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/news").status_code, 200)
        self.assertEqual(self.client.get("/summary").status_code, 200)
        self.assertEqual(self.client.get("/chat").status_code, 200)

    def test_mutating_api_requires_csrf_token(self):
        self.client.post("/login", data={"password": "test-password"})
        self.assertEqual(self.client.post("/api/news/refresh").status_code, 403)
        with self.client.session_transaction() as flask_session:
            token = flask_session["csrf_token"]
        with patch("dashboard.threading.Thread") as thread_mock, patch(
            "dashboard._current_progress", return_value={"state": "completed"},
        ):
            response = self.client.post(
                "/api/news/refresh", headers={"X-CSRF-Token": token},
            )
        self.assertEqual(response.status_code, 200)
        thread_mock.assert_called_once()


class DashboardWebChatTests(unittest.TestCase):
    def test_chat_stream_reports_search_results_before_answer(self):
        app.config.update(TESTING=True, BYPASS_AUTH_FOR_TESTS=True)
        client = app.test_client()
        search_payload = {
            "query": "最新消息", "searched_at": "2026-07-23T12:00:00+08:00",
            "providers": ["Google News"], "cached": False, "errors": [],
            "results": [{
                "title": "测试新闻", "url": "https://example.com/news",
                "source": "测试财经", "published_at": "2026-07-23T10:00:00+08:00",
                "snippet": "摘要",
            }],
        }
        with patch("dashboard.is_configured", return_value=True), patch(
            "dashboard._allow_chat_request", return_value=True,
        ), patch("dashboard.search_latest_news", return_value=search_payload), patch(
            "dashboard.stream_market_chat", return_value=iter(["回答"]),
        ) as stream_mock:
            response = client.post("/api/chat", json={
                "question": "今天最新消息是什么？", "history": [], "web_search": "auto",
            })
            body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('"search"', body)
        self.assertIn("https://example.com/news", body)
        self.assertIn('"delta": "回答"', body)
        self.assertIsNotNone(stream_mock.call_args.kwargs["web_search_payload"])


if __name__ == "__main__":
    unittest.main()
