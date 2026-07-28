"""不依赖网络的核心回归测试。"""
import importlib
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
from backtest.engine import run_backtest
from data.board_utils import is_main_board
from data.concept_tags import apply_concept_tags
from data import index_filter
from screening.eligibility import has_sufficient_liquidity, is_st_name
from screening.factor_eval import (
    adjust_weights_by_dispersion,
    cross_sectional_normalize,
    cross_sectional_standardize_factors,
)
from screening.ranking import rank, to_dataframe
from screening.scanner import (
    ScanResult,
    apply_industry_neutral_residual_momentum,
    prepare_results_for_strategy,
    scan_stock,
)
from strategy.momentum import (
    CandidateTrendFactors,
    MomentumStrategy,
    RobustTrendFactors,
)
from strategy.rsrs import RsrsStrategy


def _ohlcv_frame(n: int = 700, latest_return: float | None = None, turnover: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 20 + np.cumsum(rng.normal(0.03, 0.25, n))
    if latest_return is not None:
        close[-1] = close[-2] * (1 + latest_return)
    spread = rng.uniform(0.1, 0.6, n)
    frame = pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=n),
        "open": close - 0.05,
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": rng.integers(100_000, 1_000_000, n),
        "amount": rng.uniform(20_000_000, 80_000_000, n),
        "turnover": turnover,
    })
    frame["pct_change"] = frame["close"].pct_change() * 100
    return frame


class ConfigTests(unittest.TestCase):
    def test_score_weights_sum_to_one(self):
        for regime, weights in config.SCORE_WEIGHTS.items():
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=6, msg=regime)

    def test_exact_four_lists_and_two_strategies(self):
        self.assertEqual(set(config.STRATEGIES), {"trend", "mean_reversion"})
        suffixes = {
            value for strategy in config.STRATEGIES.values()
            for key, value in strategy.items() if key.startswith("csv_")
        }
        self.assertEqual(suffixes, {"main10", "all30", "mr_main10", "mr_all30"})

    def test_token_is_not_hardcoded(self):
        self.assertEqual(config.TUSHARE_TOKEN, os.getenv("TUSHARE_TOKEN", ""))

    def test_unvalidated_candidate_factors_are_not_enabled_in_production(self):
        self.assertFalse(config.CANDIDATE_FACTOR_STATUS["production_enabled"])
        self.assertTrue(set(config.CANDIDATE_FACTOR_NAMES).isdisjoint(
            config.FACTOR_NAMES,
        ))
        self.assertEqual(
            config.TREND_MODEL_STATUS["replacement_status"],
            "rejected",
        )
        self.assertTrue(set(config.ROBUST_TREND_FACTOR_NAMES).isdisjoint(
            config.FACTOR_NAMES,
        ))


class MarketOverviewTests(unittest.TestCase):
    @staticmethod
    def _index_frame(start: float, end: float) -> pd.DataFrame:
        return pd.DataFrame({
            "date": pd.bdate_range(end="2026-07-23", periods=160),
            "close": np.linspace(start, end, 160),
        })

    def test_primary_regime_uses_only_shanghai_while_all_indices_are_reported(self):
        frames = {
            "000001": self._index_frame(100, 220),
            "399001": self._index_frame(220, 100),
            "399006": self._index_frame(220, 100),
            "000688": self._index_frame(220, 100),
        }
        with patch.object(
            index_filter, "fetch_index_kline", side_effect=lambda code: frames[code],
        ), patch.object(
            index_filter,
            "expected_market_date",
            return_value=pd.Timestamp("2026-07-23"),
        ):
            regime, overview = index_filter.assess_market_overview()
        self.assertEqual(regime.regime, "bull")
        self.assertEqual(overview["regime"], "bull")
        self.assertEqual([item["key"] for item in overview["indices"]], [
            "shanghai", "shenzhen", "chinext", "star50",
        ])
        self.assertTrue(all(
            item["regime"] == "bear" for item in overview["indices"][1:]
        ))
        self.assertLess(overview["color_hue"], 30)

    def test_color_rule_maps_bear_to_green_and_bull_to_red(self):
        self.assertGreater(index_filter._score_to_hue(15), 100)
        self.assertEqual(index_filter._score_to_hue(50), 48)
        self.assertLess(index_filter._score_to_hue(90), 15)

    def test_stale_index_history_is_completed_from_one_spot_snapshot(self):
        dates = pd.bdate_range(end="2026-07-22", periods=160)
        frames = {
            code: pd.DataFrame({
                "date": dates,
                "close": np.linspace(100, 200, len(dates)),
            })
            for _key, code, _name in index_filter.DISPLAY_INDICES
        }
        spot = pd.DataFrame([
            {
                "symbol": index_filter._code_to_index_symbol(code),
                "open": 200.0, "high": 205.0, "low": 198.0,
                "close": 203.0, "pre_close": 200.0,
                "volume": 1_000, "amount": 2_000,
            }
            for _key, code, _name in index_filter.DISPLAY_INDICES
        ])
        with patch.object(
            index_filter, "fetch_index_kline", side_effect=lambda code: frames[code],
        ), patch.object(
            index_filter, "fetch_index_spot_snapshot", return_value=spot,
        ), patch.object(
            index_filter, "expected_market_date", return_value=pd.Timestamp("2026-07-23"),
        ):
            _regime, overview = index_filter.assess_market_overview()
        self.assertEqual(overview["market_date"], "2026-07-23")
        self.assertTrue(all(
            item["date"] == "2026-07-23" for item in overview["indices"]
        ))

    def test_index_date_mismatch_stops_market_assessment(self):
        dates = pd.bdate_range(end="2026-07-22", periods=160)
        stale = pd.DataFrame({"date": dates, "close": np.linspace(100, 200, len(dates))})
        with patch.object(
            index_filter, "fetch_index_kline", return_value=stale,
        ), patch.object(
            index_filter, "fetch_index_spot_snapshot", return_value=pd.DataFrame(),
        ), patch.object(
            index_filter, "expected_market_date", return_value=pd.Timestamp("2026-07-23"),
        ):
            with self.assertRaisesRegex(RuntimeError, "四指数日期不一致"):
                index_filter.assess_market_overview()


class UniverseTests(unittest.TestCase):
    def test_main_board_is_only_no_extra_qualification_codes(self):
        for code in ("600000", "601318", "603000", "605001", "000001", "001234", "002001", "003001"):
            self.assertTrue(is_main_board(code), code)
        for code in ("300001", "301001", "688001", "830001", "920001", "900901"):
            self.assertFalse(is_main_board(code), code)

    def test_st_and_extremely_illiquid_are_filtered_but_price_limits_are_not(self):
        liquid_frame = _ohlcv_frame(latest_return=0.20)
        illiquid_frame = liquid_frame.copy()
        illiquid_frame.loc[illiquid_frame.index[-20:], "amount"] = 1_000_000
        suspended_frame = liquid_frame.copy()
        suspended_frame.loc[suspended_frame.index[-20:-5], "amount"] = 0

        self.assertTrue(is_st_name("*ST样本"))
        self.assertTrue(is_st_name("S*ST样本"))
        self.assertFalse(is_st_name("普通样本"))
        self.assertFalse(has_sufficient_liquidity(illiquid_frame))
        self.assertFalse(has_sufficient_liquidity(suspended_frame))
        self.assertTrue(has_sufficient_liquidity(liquid_frame))
        self.assertIsNone(scan_stock("000001", "*ST样本", df=liquid_frame))
        self.assertIsNone(scan_stock("000002", "普通样本", df=illiquid_frame))
        result = scan_stock("000003", "普通样本", df=liquid_frame)
        self.assertIsNotNone(result)
        self.assertEqual(len(rank([result], max_pool=10)), 1)


class StrategyTests(unittest.TestCase):
    def test_score_is_exactly_the_dynamic_five_factor_weighted_sum(self):
        result = ScanResult(
            code="000001",
            vol_price_score=1.0,
            ma_adx_raw=1.0,
            rsrs_raw=1.0,
            donchian_raw=1.0,
            momentum_raw=1.0,
            ma_adx=1,
            rsrs=1,
            donchian=1,
            momentum=1,
        )
        scored = prepare_results_for_strategy(
            [result], config.STRATEGIES["trend"]["directions"],
            weights=config.SCORE_WEIGHTS["sideways"],
        )[0]
        self.assertEqual(scored.score, 1.0)

    def test_daily_dispersion_dynamically_changes_only_five_weights(self):
        base = config.SCORE_WEIGHTS["sideways"]
        adjusted = adjust_weights_by_dispersion(base, {
            "volume_price": 0.8, "ma_adx": 0.05, "rsrs": 0.8,
            "donchian": 0.8, "momentum": 0.8,
        })
        self.assertEqual(set(adjusted), set(config.FACTOR_DIRECTION))
        self.assertLess(adjusted["ma_adx"], base["ma_adx"])
        low_volume = adjust_weights_by_dispersion(base, {
            "volume_price": 0.05, "ma_adx": 0.8, "rsrs": 0.8,
            "donchian": 0.8, "momentum": 0.8,
        })
        self.assertLess(low_volume["volume_price"], base["volume_price"])
        self.assertEqual(sum(adjusted.values()), 1.0)

    def test_factor_standardization_equalizes_scales_and_preserves_order(self):
        results = [
            ScanResult(
                code=f"{index:06d}",
                vol_price_score=float(index),
                ma_adx_raw=float(index * 10),
                rsrs_raw=float(index * 100),
                donchian_raw=float(index * 1000),
                momentum_raw=float(index * 10000),
            )
            for index in range(5)
        ]
        standardized = cross_sectional_standardize_factors(results)
        for attribute in (
            "vol_price_score", "ma_adx_raw", "rsrs_raw",
            "donchian_raw", "momentum_raw",
        ):
            values = [getattr(result, attribute) for result in standardized]
            self.assertEqual(values, [-1.0, -0.5, 0.0, 0.5, 1.0])
        self.assertEqual(results[-1].momentum_raw, 40000.0)

    def test_momentum_first_valid_bar_and_value(self):
        frame = pd.DataFrame({"close": np.arange(1.0, 13.0)})
        output = MomentumStrategy(window=5, skip=2).calculate(frame)
        self.assertTrue(pd.isna(output.loc[6, "momentum_ret"]))
        self.assertAlmostEqual(output.loc[7, "momentum_ret"], 5.0)

    def test_candidate_12_2_uses_lookback_endpoint_and_skips_recent_bars(self):
        frame = pd.DataFrame({"close": np.arange(1.0, 21.0)})
        output = CandidateTrendFactors(
            momentum_lookback=5,
            skip=2,
            high_window=5,
            vol_window=3,
            horizons=(2, 3),
        ).calculate(frame)
        self.assertAlmostEqual(output.loc[5, "momentum_12_2"], 3.0)
        self.assertAlmostEqual(output.loc[5, "high_52w"], 0.0)
        self.assertTrue(np.isfinite(output.loc[10, "multi_horizon"]))

    def test_robust_trend_factors_reward_smooth_uptrend_and_control_risk(self):
        close = pd.Series(np.linspace(10.0, 30.0, 180))
        frame = pd.DataFrame({"close": close})
        output = RobustTrendFactors().calculate(frame)
        latest = output.iloc[-1]
        self.assertGreater(latest["medium_momentum"], 0)
        self.assertGreater(latest["trend_quality"], 0)
        self.assertAlmostEqual(latest["path_efficiency"], 1.0)
        self.assertAlmostEqual(latest["downside_risk_control"], 0.0)
        self.assertAlmostEqual(latest["drawdown_control"], 0.0)
        self.assertGreater(latest["multi_period_consistency"], 0)

    def test_robust_trend_quality_is_negative_for_smooth_downtrend(self):
        frame = pd.DataFrame({"close": np.linspace(30.0, 10.0, 180)})
        latest = RobustTrendFactors().calculate(frame).iloc[-1]
        self.assertLess(latest["trend_quality"], 0)
        self.assertLess(latest["path_efficiency"], 0)

    def test_industry_residual_requires_coverage_and_neutralizes_groups(self):
        covered = [
            ScanResult(
                code=f"{index:06d}",
                industry="行业甲" if index < 5 else "行业乙",
                momentum_12_2_raw=float(index),
            )
            for index in range(10)
        ]
        apply_industry_neutral_residual_momentum(covered)
        first_group = [
            result.industry_residual_momentum_raw for result in covered[:5]
        ]
        second_group = [
            result.industry_residual_momentum_raw for result in covered[5:]
        ]
        self.assertEqual(first_group, [-2.0, -1.0, 0.0, 1.0, 2.0])
        self.assertEqual(second_group, [-2.0, -1.0, 0.0, 1.0, 2.0])

        uncovered = [
            ScanResult(code=f"{index:06d}", momentum_12_2_raw=float(index))
            for index in range(10)
        ]
        apply_industry_neutral_residual_momentum(uncovered)
        self.assertTrue(all(
            result.industry_residual_momentum_raw == 0.0
            for result in uncovered
        ))

    def test_rsrs_prefix_is_invariant_to_future_rows(self):
        frame = _ohlcv_frame(260)
        strategy = RsrsStrategy(window=18, z_score_window=120, adaptive_threshold=True)
        prefix = strategy.calculate(frame.iloc[:200].copy())
        full = strategy.calculate(frame.copy())
        for column in ("rsrs_z", "rsrs_corrected", "signal"):
            left = prefix[column].iloc[-1]
            right = full[column].iloc[199]
            if pd.isna(left):
                self.assertTrue(pd.isna(right), column)
            else:
                self.assertAlmostEqual(float(left), float(right), places=10, msg=column)

    def test_mean_reversion_reverses_factor_direction_without_eligibility_filter(self):
        base = ScanResult(
            code="000001", vol_price_score=0.2, ma_adx_raw=-0.2,
            donchian_raw=-0.2, rsrs_raw=-0.2, momentum_raw=-0.2,
        )
        trend = prepare_results_for_strategy([base], {name: 1 for name in config.FACTOR_DIRECTION})[0]
        mean = prepare_results_for_strategy([base], {name: -1 for name in config.FACTOR_DIRECTION})[0]
        self.assertLess(trend.score, mean.score)


class RankingTests(unittest.TestCase):
    def test_equal_scores_get_equal_percentiles(self):
        results = [
            ScanResult(code="1", score=2.0),
            ScanResult(code="2", score=2.0),
            ScanResult(code="3", score=9.0),
            ScanResult(code="4", score=4.0),
        ]
        cross_sectional_normalize(results)
        self.assertEqual(results[0].score, results[1].score)
        self.assertEqual(results[2].score, 1.0)

    def test_concept_enrichment_does_not_change_score_or_order(self):
        pool = [ScanResult(code="000001", score=0.9), ScanResult(code="000002", score=0.8)]
        frame = to_dataframe(pool)
        enriched = apply_concept_tags(frame, {"000001": ["概念甲", "概念乙"]})
        self.assertEqual(enriched["代码"].tolist(), frame["代码"].tolist())
        self.assertEqual(enriched["综合得分"].tolist(), frame["综合得分"].tolist())
        self.assertEqual(enriched.loc[0, "所属概念"], "概念甲、概念乙")

    def test_offline_concept_lookup_uses_cache_without_network(self):
        import data.concept_tags as concept_tags

        with patch.object(
            concept_tags,
            "_load_cache",
            return_value={"000001": ["缓存概念"]},
        ), patch.object(
            concept_tags,
            "_get_tushare_pro",
            side_effect=AssertionError("离线模式不应创建网络客户端"),
        ):
            output = concept_tags.get_concept_tags(
                {"000001", "000002"},
                allow_network=False,
            )
        self.assertEqual(output["000001"], ["缓存概念"])
        self.assertEqual(output["000002"], [])


class FeedbackTests(unittest.TestCase):
    def test_rank_ic_collapses_duplicate_lists_and_averages_daily_cross_sections(self):
        from feedback.recorder import init_db
        from feedback.returns_tracker import compute_ic_for_regime

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "feedback.db"
            init_db(db_path)
            connection = sqlite3.connect(db_path)
            for date_value in ("2026-06-01", "2026-06-02"):
                for index in range(25):
                    for list_type in ("all30", "mr_all30"):
                        cursor = connection.execute(
                            """INSERT INTO signals
                               (date, code, list_type, market_regime, strategy)
                               VALUES (?, ?, ?, 'sideways', 'trend')""",
                            (date_value, f"{index:06d}", list_type),
                        )
                        signal_id = cursor.lastrowid
                        connection.execute(
                            """INSERT INTO factor_values
                               (signal_id, factor_name, factor_value)
                               VALUES (?, 'volume_price', ?)""",
                            (signal_id, float(index)),
                        )
                        connection.execute(
                            """INSERT INTO forward_returns
                               (signal_id, days_forward, forward_return)
                               VALUES (?, 5, ?)""",
                            (signal_id, float(index) / 100),
                        )
            connection.commit()
            connection.close()
            result = compute_ic_for_regime(
                db_path,
                "volume_price",
                "sideways",
                lookback_days=365,
                forward_days=5,
                min_samples=50,
            )
        self.assertEqual(result["samples"], 50)
        self.assertEqual(result["mean_ic"], 1.0)

    def test_feedback_records_production_and_candidate_factors(self):
        from feedback.recorder import init_db, record_ranking

        result = ScanResult(
            code="000001", as_of_date="2026-01-05", ma_adx_raw=0.1234,
            donchian_raw=-0.2345, rsrs_raw=0.3456, momentum_raw=0.4567,
            vol_price_score=0.6789, momentum_12_2_raw=0.12,
            high_52w_raw=-0.03, multi_horizon_raw=0.45,
            industry_residual_momentum_raw=0.08,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "feedback.db"
            init_db(db_path)
            record_ranking(
                [result], "2026-01-06", "main10", "sideways", db_path=db_path,
            )
            result.vol_price_score = 0.1111
            result.score = 0.2222
            record_ranking(
                [result], "2026-01-06", "main10", "sideways", db_path=db_path,
            )
            connection = sqlite3.connect(db_path)
            values = dict(connection.execute(
                "SELECT factor_name, factor_value FROM factor_values"
            ).fetchall())
            signal_date = connection.execute("SELECT date FROM signals").fetchone()[0]
            updated_score = connection.execute("SELECT score FROM signals").fetchone()[0]
            columns = {row[1] for row in connection.execute("PRAGMA table_info(signals)")}
            connection.close()
        self.assertEqual(set(values), {
            "volume_price", "ma_adx", "rsrs", "donchian", "momentum",
            "momentum_12_2", "high_52w", "multi_horizon",
            "industry_residual_momentum",
        })
        self.assertEqual(signal_date, "2026-01-06")
        self.assertAlmostEqual(values["volume_price"], 0.1111)
        self.assertAlmostEqual(updated_score, 0.2222)
        self.assertNotIn("sector", columns)

    def test_suspended_stock_does_not_start_forward_return_at_future_resume(self):
        from feedback.returns_tracker import _count_trading_days

        frame = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05", "2026-01-08", "2026-01-09"]),
            "close": [10.0, 11.0, 12.0],
        })
        self.assertIsNone(_count_trading_days(frame, "2026-01-06", 1))
        self.assertAlmostEqual(
            _count_trading_days(frame, "2026-01-08", 1),
            1 / 11,
            places=6,
        )

    def test_weekly_interval_is_tracked_per_market_regime(self):
        from datetime import datetime, timedelta
        from feedback.recorder import init_db
        import feedback.weight_optimizer as optimizer

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "feedback.db"
            init_db(db_path)
            connection = sqlite3.connect(db_path)
            today = datetime.now().strftime("%Y-%m-%d")
            old = (datetime.now() - timedelta(days=9)).strftime("%Y-%m-%d")
            connection.executemany(
                """INSERT INTO weight_history (date, market_regime)
                   VALUES (?, ?)""",
                [(today, "bear"), (old, "bull")],
            )
            connection.commit()
            connection.close()
            with patch.object(optimizer, "DB_PATH", db_path):
                self.assertEqual(optimizer._days_since_last_update("bear"), 0)
                self.assertGreaterEqual(optimizer._days_since_last_update("bull"), 9)
                self.assertIsNone(optimizer._days_since_last_update("sideways"))


class BacktestTests(unittest.TestCase):
    def test_signal_executes_next_day_and_commission_does_not_cancel_trade(self):
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-05", periods=4),
            "open": [10.0, 10.0, 11.0, 11.0],
            "high": [10.2, 10.2, 11.2, 11.2],
            "low": [9.8, 9.8, 10.8, 10.8],
            "close": [10.0, 10.0, 11.0, 11.0],
            "pct_change": [0.0, 0.0, 0.0, 0.0],
        })
        result = run_backtest(
            frame, pd.Series([1, -1, 0, 0]), initial_capital=10_000,
            commission_rate=0.0, stamp_tax_rate=0.0, slippage=0.0,
            code="000001", execute_on_next_bar=True,
        )
        self.assertEqual(result.trades.iloc[0]["date"], frame.loc[1, "date"])
        self.assertEqual(result.trades.iloc[0]["shares"], 900)
        self.assertGreater(result.trades.iloc[1]["pnl"], 0)


class StorageTests(unittest.TestCase):
    def test_market_freshness_requires_today_after_data_ready_hour(self):
        from datetime import datetime
        from data.storage import _expected_market_date

        self.assertEqual(
            _expected_market_date(datetime(2026, 7, 21, 10, 0)),
            pd.Timestamp("2026-07-20"),
        )
        self.assertEqual(
            _expected_market_date(datetime(2026, 7, 21, 18, 0)),
            pd.Timestamp("2026-07-21"),
        )
        self.assertEqual(
            _expected_market_date(datetime(2026, 7, 20, 10, 0)),
            pd.Timestamp("2026-07-17"),
        )

    def test_short_new_listing_is_cached_for_future_incremental_updates(self):
        import data.storage as storage

        short_history = _ohlcv_frame(20)
        fake_fetcher = types.ModuleType("data.fetcher")
        fake_fetcher.get_kline_daily = lambda *args, **kwargs: short_history
        with patch.object(storage, "load_kline", return_value=None), patch.object(
            storage, "save_kline",
        ) as save_mock, patch.dict(sys.modules, {"data.fetcher": fake_fetcher}):
            output = storage.ensure_kline("001999")
        self.assertIs(output, short_history)
        save_mock.assert_called_once()

    def test_qfq_overlap_rescales_old_history_and_prefers_fresh_rows(self):
        import data.storage as storage

        existing = _ohlcv_frame(20)
        fresh = existing.tail(5).copy()
        for column in ("open", "high", "low", "close"):
            fresh[column] = fresh[column] * 2
        fresh.loc[fresh.index[-1], "amount"] += 123.0
        with patch.object(storage, "load_kline", return_value=existing), patch.object(
            storage, "save_kline",
        ):
            output = storage.update_kline("000001", fresh)
        self.assertAlmostEqual(output.iloc[0]["close"], existing.iloc[0]["close"] * 2)
        self.assertAlmostEqual(output.iloc[-1]["close"], fresh.iloc[-1]["close"])
        self.assertAlmostEqual(output.iloc[-1]["amount"], fresh.iloc[-1]["amount"])
        self.assertAlmostEqual(
            output.iloc[-1]["pct_change"],
            (output.iloc[-1]["close"] / output.iloc[-2]["close"] - 1) * 100,
        )

    def test_empty_increment_does_not_fall_through_to_full_download(self):
        import data.storage as storage

        stale = _ohlcv_frame(130)
        stale["date"] = pd.bdate_range("2023-01-02", periods=130)
        calls = []
        fake_fetcher = types.ModuleType("data.fetcher")

        def fake_get_kline_daily(code, start_date="20200101", end_date=None, adjust="qfq"):
            calls.append((start_date, end_date))
            return None

        fake_fetcher.get_kline_daily = fake_get_kline_daily
        with patch.object(storage, "load_kline", return_value=stale), patch.dict(
            sys.modules, {"data.fetcher": fake_fetcher},
        ):
            output = storage.ensure_kline("000001")
        self.assertIs(output, stale)
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(calls[0][0], "20200101")


class FetcherTests(unittest.TestCase):
    def test_qfq_public_entry_does_not_fall_back_to_unadjusted_tushare(self):
        import data.fetcher as fetcher

        adjusted = _ohlcv_frame(130)
        with patch.object(
            fetcher, "_fetch_kline_akshare", return_value=adjusted,
        ), patch.object(
            fetcher, "_fetch_kline_tushare",
            side_effect=AssertionError("复权入口不应调用未复权 Tushare"),
        ):
            output = fetcher.get_kline_daily("000001", adjust="qfq")
        self.assertEqual(len(output), len(adjusted))

    def test_tushare_adjust_requires_an_actual_factor_series(self):
        import data.fetcher as fetcher

        class FakePro:
            def daily(self, **kwargs):
                return pd.DataFrame({
                    "ts_code": ["000001.SZ"], "trade_date": ["20260102"],
                    "open": [11.0], "high": [12.5], "low": [10.5],
                    "close": [12.0], "pre_close": [10.0], "change": [2.0],
                    "pct_chg": [20.0], "vol": [100.0], "amount": [10.0],
                })

            def adj_factor(self, **kwargs):
                return pd.DataFrame()

        with patch.object(fetcher, "TUSHARE_TOKEN", "test-token"), patch.object(
            fetcher, "_get_tushare_pro", return_value=FakePro(),
        ), patch.object(fetcher, "_tushare_wait_token"):
            output = fetcher._fetch_kline_tushare("000001", "20260101", "20260102")
        self.assertIsNone(output)

    def test_tushare_qfq_uses_latest_factor_and_date_order(self):
        fake_akshare = types.ModuleType("akshare")
        with patch.dict(sys.modules, {"akshare": fake_akshare}):
            sys.modules.pop("data.fetcher", None)
            fetcher = importlib.import_module("data.fetcher")

        class FakePro:
            def daily(self, **kwargs):
                return pd.DataFrame({
                    "ts_code": ["000001.SZ", "000001.SZ"], "trade_date": ["20260102", "20260101"],
                    "open": [11.0, 9.0], "high": [12.5, 10.5], "low": [10.5, 8.5],
                    "close": [12.0, 10.0], "pre_close": [10.0, 9.0], "change": [2.0, 1.0],
                    "pct_chg": [20.0, 11.11], "vol": [100.0, 100.0], "amount": [10.0, 10.0],
                })

            def adj_factor(self, **kwargs):
                return pd.DataFrame({"trade_date": ["20260102", "20260101"], "adj_factor": [2.0, 1.0]})

        waits = []
        with patch.object(fetcher, "TUSHARE_TOKEN", "test-token"), patch.object(
            fetcher, "_get_tushare_pro", return_value=FakePro(),
        ), patch.object(fetcher, "_tushare_wait_token", side_effect=lambda: waits.append(1)):
            output = fetcher._normalize_dataframe(
                fetcher._fetch_kline_tushare("000001", "20260101", "20260102")
            )
        self.assertEqual(output["date"].dt.strftime("%Y%m%d").tolist(), ["20260101", "20260102"])
        self.assertAlmostEqual(output["close"].iloc[0], 5.0)
        self.assertAlmostEqual(output["close"].iloc[1], 12.0)
        self.assertAlmostEqual(output["pct_change"].iloc[1], 140.0)
        self.assertEqual(len(waits), 2)


class ArchitectureTests(unittest.TestCase):
    def test_removed_modules_are_absent(self):
        root = Path(config.ROOT)
        removed = [
            "clustering", "data/sector_mapper.py", "data/sector_strength.py",
            "screening/prescreener.py", "screening/limit_up_detector.py",
            "strategy/filters.py", "strategy/sector_allocation.py",
        ]
        for relative in removed:
            self.assertFalse((root / relative).exists(), relative)


if __name__ == "__main__":
    unittest.main()
