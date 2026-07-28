"""四榜单和仪表盘的离线冒烟测试。"""
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from data.storage import append_market_daily_snapshot, expected_market_date
from dashboard import app
from main import (
    build_strategy_pools,
    resolve_market_date,
    validate_market_data_freshness,
)
from screening.scanner import ScanResult


class PipelineSmokeTests(unittest.TestCase):
    def test_data_ready_cutoff_is_1535(self):
        with patch.dict("os.environ", {"SG_QUANT_DATA_READY_TIME": "15:35"}, clear=False):
            before = expected_market_date(datetime(2026, 7, 23, 15, 34))
            ready = expected_market_date(datetime(2026, 7, 23, 15, 35))
        self.assertEqual(str(before.date()), "2026-07-22")
        self.assertEqual(str(ready.date()), "2026-07-23")

    def test_cross_midnight_run_uses_latest_market_snapshot_date(self):
        results = [
            SimpleNamespace(as_of_date="2026-07-22"),
            SimpleNamespace(as_of_date="2026-07-23"),
            SimpleNamespace(as_of_date=""),
        ]
        self.assertEqual(resolve_market_date(results, "2026-07-22"), "2026-07-23")

    def test_market_date_uses_majority_instead_of_single_newer_stock(self):
        results = [
            SimpleNamespace(as_of_date="2026-07-22"),
            SimpleNamespace(as_of_date="2026-07-22"),
            SimpleNamespace(as_of_date="2026-07-23"),
        ]
        self.assertEqual(resolve_market_date(results, "2026-07-23"), "2026-07-22")

    def test_trading_stock_with_stale_kline_warns_without_stopping(self):
        with patch("main.get_trading_codes", return_value={"000001", "000002"}), patch(
            "main.latest_kline_dates",
            return_value={"000001": "2026-07-23", "000002": "2026-07-22"},
        ):
            freshness = validate_market_data_freshness(
                {"000001", "000002"},
                {"000001": "平安银行", "000002": "万科A"},
                "2026-07-23",
            )
        self.assertEqual(freshness.snapshot_date, "2026-07-23")
        self.assertEqual(freshness.excluded_codes, frozenset({"000002"}))
        self.assertIn("000002 万科A(2026-07-22)", freshness.warning)
        self.assertIn("流程继续", freshness.warning)

    def test_ten_or_more_stale_stocks_only_reports_count(self):
        codes = {f"{index:06d}" for index in range(12)}
        dates = {code: "2026-07-22" for code in codes}
        names = {code: f"样本{index}" for index, code in enumerate(sorted(codes))}
        with patch("main.latest_kline_dates", return_value=dates):
            freshness = validate_market_data_freshness(
                codes,
                names,
                "2026-07-23",
                trading_codes=codes,
            )
        self.assertEqual(len(freshness.excluded_codes), 12)
        self.assertIn("共有 12 只股票", freshness.warning)
        self.assertNotIn("样本0", freshness.warning)

    def test_suspended_stock_is_not_required_to_have_today_kline(self):
        with patch("main.get_trading_codes", return_value={"000001"}), patch(
            "main.latest_kline_dates", return_value={"000001": "2026-07-23"},
        ):
            freshness = validate_market_data_freshness(
                {"000001", "000002"},
                {"000001": "平安银行", "000002": "停牌样本"},
                "2026-07-23",
            )
        self.assertEqual(freshness.snapshot_date, "2026-07-23")
        self.assertFalse(freshness.warning)

    def test_single_day_snapshot_only_appends_missing_bar_and_aligns_qfq_history(self):
        existing = pd.DataFrame({
            "date": pd.to_datetime(["2026-07-21", "2026-07-22"]),
            "open": [9.5, 10.0], "high": [10.2, 10.3],
            "low": [9.4, 9.8], "close": [10.0, 10.0],
            "pre_close": [9.4, 10.0], "volume": [1000, 1200],
            "amount": [9500, 12000], "pct_change": [1.0, 0.0],
            "outstanding_share": [100_000, 100_000],
            "turnover": [0.01, 0.012], "amplitude": [8.0, 5.0],
        })
        snapshot = pd.DataFrame([{
            "code": "000001", "date": pd.Timestamp("2026-07-23"),
            "open": 9.0, "high": 10.0, "low": 8.9, "close": 9.9,
            "pre_close": 9.0, "volume": 2_000, "amount": 19_000,
            "pct_change": 10.0,
        }])
        with patch("data.storage.load_kline", return_value=existing), patch(
            "data.storage.save_kline",
        ) as save_mock:
            stats = append_market_daily_snapshot(snapshot, {"000001"})
        saved = save_mock.call_args.args[1]
        self.assertEqual(stats["updated"], 1)
        self.assertEqual(saved["date"].max(), pd.Timestamp("2026-07-23"))
        self.assertAlmostEqual(float(saved.iloc[-2]["close"]), 9.0)
        self.assertAlmostEqual(float(saved.iloc[-1]["pct_change"]), 10.0)

    def test_market_date_falls_back_when_results_have_no_valid_date(self):
        results = [SimpleNamespace(as_of_date=""), SimpleNamespace(as_of_date="invalid")]
        self.assertEqual(resolve_market_date(results, "2026-07-22"), "2026-07-22")

    def test_one_scan_builds_exactly_four_full_size_lists(self):
        results = []
        for index in range(50):
            code = f"600{index:03d}" if index < 25 else f"300{index:03d}"
            value = index / 50
            results.append(ScanResult(
                code=code,
                vol_price_score=value,
                ma_adx_raw=value - 0.5,
                rsrs_raw=value * 0.8 - 0.4,
                donchian_raw=value - 0.5,
                momentum_raw=value * 0.6 - 0.3,
                ma_adx=1 if value > 0.5 else -1,
                rsrs=1 if value > 0.5 else -1,
                donchian=1 if value > 0.5 else -1,
                momentum=1 if value > 0.5 else -1,
            ))
        pools = build_strategy_pools(results, "sideways")
        self.assertEqual(set(pools), {"main10", "all30", "mr_main10", "mr_all30"})
        self.assertEqual(len(pools["main10"]["items"]), 10)
        self.assertEqual(len(pools["all30"]["items"]), 30)
        self.assertEqual(len(pools["mr_main10"]["items"]), 10)
        self.assertEqual(len(pools["mr_all30"]["items"]), 30)
        self.assertNotEqual(
            pools["all30"]["items"][0].code,
            pools["mr_all30"]["items"][0].code,
        )

    def test_dashboard_core_routes(self):
        app.config.update(TESTING=True, BYPASS_AUTH_FOR_TESTS=True)
        client = app.test_client()
        dashboard_response = client.get("/")
        self.assertEqual(dashboard_response.status_code, 200)
        self.assertIn(b'id="marketBanner"', dashboard_response.data)
        self.assertIn(b"completed_with_warnings", dashboard_response.data)
        self.assertIn(b'id="progressWarnings"', dashboard_response.data)
        self.assertEqual(client.get("/api/health").status_code, 200)
        self.assertEqual(client.get("/api/market_overview").status_code, 200)
        self.assertEqual(client.get("/news").status_code, 200)
        summary_response = client.get("/summary")
        chat_response = client.get("/chat")
        self.assertEqual(summary_response.status_code, 200)
        self.assertEqual(chat_response.status_code, 200)
        self.assertIn(b"SGQ.renderMarkdown(item.summary)", summary_response.data)
        self.assertIn(b"renderAnswerMarkdown(answerNode, answer)", chat_response.data)
        self.assertIn(b"renderMarkdown(markdown)", chat_response.data)
        self.assertIn(b"this.escape(protectedText)", chat_response.data)
        self.assertNotIn(b"inlineSummaryMarkup", summary_response.data)
        self.assertEqual(client.get("/api/run/status").status_code, 200)
        self.assertEqual(client.get("/api/stock_pool/unknown").status_code, 404)


if __name__ == "__main__":
    unittest.main()
