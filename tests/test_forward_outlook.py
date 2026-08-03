"""前瞻证据规范化测试。"""
import unittest

import pandas as pd

from data.forward_outlook import (
    _forecast_period_candidates,
    _normalize_analyst_forecast,
    _normalize_performance_forecast,
)


class ForwardOutlookTests(unittest.TestCase):
    def test_official_forecast_prioritizes_attributable_profit(self):
        frame = pd.DataFrame([
            {
                "股票代码": "000001",
                "预测指标": "扣除非经常性损益后的净利润",
                "预测数值": 80,
                "业绩变动幅度": 10,
                "业绩变动原因": "研发费用下降",
                "预告类型": "预增",
                "公告日期": "2026-07-20",
            },
            {
                "股票代码": "000001",
                "预测指标": "归属于上市公司股东的净利润",
                "预测数值": 100,
                "业绩变动幅度": 20,
                "业绩变动原因": "在手订单增长并加大研发投入",
                "预告类型": "预增",
                "公告日期": "2026-07-19",
            },
        ])
        result = _normalize_performance_forecast(frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["official_profit_forecast"], 100)
        self.assertEqual(result.iloc[0]["rd_evidence_score"], 1.0)
        self.assertEqual(result.iloc[0]["order_evidence_score"], 1.0)

    def test_generic_innovation_is_not_mislabelled_as_rd(self):
        frame = pd.DataFrame([{
            "股票代码": "000001",
            "预测指标": "归属于上市公司股东的净利润",
            "预测数值": 100,
            "业绩变动幅度": 20,
            "业绩变动原因": "服务科技产业创新，提升综合实力",
            "预告类型": "预增",
            "公告日期": "2026-07-19",
        }])
        result = _normalize_performance_forecast(frame)
        self.assertTrue(pd.isna(result.iloc[0]["rd_evidence_score"]))
        self.assertEqual(result.iloc[0]["rd_evidence"], "")

    def test_analyst_forecast_uses_current_and_next_year(self):
        frame = pd.DataFrame([{
            "代码": "600000",
            "研报数": 10,
            "机构投资评级(近六个月)-买入": 6,
            "机构投资评级(近六个月)-增持": 3,
            "机构投资评级(近六个月)-中性": 1,
            "机构投资评级(近六个月)-减持": 0,
            "机构投资评级(近六个月)-卖出": 0,
            "2025预测每股收益": 1.0,
            "2026预测每股收益": 2.0,
            "2027预测每股收益": 2.4,
        }])
        result = _normalize_analyst_forecast(frame, current_year=2026)
        self.assertEqual(result.iloc[0]["analyst_eps_base_year"], 2026)
        self.assertEqual(result.iloc[0]["analyst_eps_next_year"], 2027)
        self.assertAlmostEqual(result.iloc[0]["analyst_eps_growth"], 20.0)
        self.assertAlmostEqual(result.iloc[0]["positive_rating_ratio"], 0.9)

    def test_july_prefers_half_year_forecast(self):
        periods = _forecast_period_candidates(pd.Timestamp("2026-07-29").date())
        self.assertEqual(periods[0], "20260630")


if __name__ == "__main__":
    unittest.main()
