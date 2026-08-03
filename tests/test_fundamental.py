"""基本面价值榜评分、分模型与风险门禁测试。"""
import unittest

import pandas as pd

from screening.fundamental import _valuation_score, build_fundamental_ranking


class FundamentalRankingTests(unittest.TestCase):
    def setUp(self):
        self.fundamentals = pd.DataFrame([
            self._fundamental("000001", pledge=2, debt=48, profit_yoy=35),
            self._fundamental("000002", pledge=18, debt=60, profit_yoy=20),
            self._fundamental("000003", pledge=60, debt=50, profit_yoy=25),
            self._fundamental("000004", pledge=0, debt=88, profit_yoy=15),
        ])
        self.valuations = pd.DataFrame([
            self._valuation("000001", pe=8, pb=1.1, ps=1.2, dividend=3.0),
            self._valuation("000002", pe=12, pb=1.6, ps=1.8, dividend=2.0),
            self._valuation("000003", pe=6, pb=0.9, ps=1.0, dividend=4.0),
            self._valuation("000004", pe=10, pb=0.8, ps=0.8, dividend=1.0),
        ])
        self.metadata = pd.DataFrame([
            {"code": "000001", "name": "稳健制造", "industry": "机械", "list_date": "20100101"},
            {"code": "000002", "name": "现金流样本", "industry": "机械", "list_date": "20120101"},
            {"code": "000003", "name": "高质押样本", "industry": "机械", "list_date": "20130101"},
            {"code": "000004", "name": "银行样本", "industry": "银行", "list_date": "20000101"},
        ])
        self.outlook = pd.DataFrame([
            self._outlook("000001", profit_yoy=35, eps_growth=20),
            self._outlook("000002", profit_yoy=20, eps_growth=15),
            self._outlook("000003", profit_yoy=40, eps_growth=25),
            self._outlook("000004", profit_yoy=15, eps_growth=12),
        ])

    def test_high_pledge_is_hard_excluded_and_low_risk_still_has_note(self):
        result, diagnostics = build_fundamental_ranking(
            self.fundamentals,
            self.valuations,
            self.metadata,
            self.outlook,
            valuation_date="20260727",
            top_n=30,
        )
        self.assertNotIn("000003", set(result["代码"]))
        self.assertGreaterEqual(diagnostics["high_risk_excluded"], 1)
        self.assertTrue(result["风险提示"].astype(str).str.len().gt(0).all())
        self.assertNotIn("高", set(result["暴雷风险"]))

    def test_financial_model_does_not_use_industrial_debt_gate(self):
        result, _ = build_fundamental_ranking(
            self.fundamentals,
            self.valuations,
            self.metadata,
            self.outlook,
            valuation_date="20260727",
            top_n=30,
        )
        bank = result.loc[result["代码"] == "000004"]
        self.assertFalse(bank.empty)
        self.assertEqual(bank.iloc[0]["模型"], "金融")

    def test_fundamental_columns_are_explainable(self):
        result, diagnostics = build_fundamental_ranking(
            self.fundamentals,
            self.valuations,
            self.metadata,
            self.outlook,
            valuation_date="20260727",
            top_n=2,
        )
        self.assertLessEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["排名"], 1)
        for column in (
            "质量分", "估值分", "历史成长分", "未来空间分",
            "风险控制分", "未来数据置信度", "未来空间依据",
            "入选理由", "风险提示", "财报期", "估值日期",
        ):
            self.assertIn(column, result.columns)
        self.assertEqual(diagnostics["report_period"], "20260331")

    def test_missing_ps_and_dividend_are_reweighted_within_valuation_dimension(self):
        row = pd.Series({
            "model_type": "一般行业",
            "pe_ttm": 8.0,
            "pb": 1.0,
            "ps_ttm": float("nan"),
            "dv_ttm": float("nan"),
            "pe_ttm_value": 1.0,
            "pb_value": 1.0,
            "ps_ttm_value": 0.0,
            "dv_ttm_value": 0.0,
        })
        self.assertEqual(_valuation_score(row), 25.0)

    def test_predicted_loss_is_hard_excluded(self):
        outlook = self.outlook.copy()
        outlook.loc[outlook["code"] == "000001", "official_profit_forecast"] = -1
        result, diagnostics = build_fundamental_ranking(
            self.fundamentals,
            self.valuations,
            self.metadata,
            outlook,
            valuation_date="20260727",
            top_n=30,
        )
        self.assertNotIn("000001", set(result["代码"]))
        reasons = str(diagnostics["high_risk_examples"])
        self.assertIn("最新业绩预告预计亏损", reasons)

    def test_missing_forward_core_data_is_excluded(self):
        result, diagnostics = build_fundamental_ranking(
            self.fundamentals,
            self.valuations,
            self.metadata,
            pd.DataFrame(),
            valuation_date="20260727",
            top_n=30,
        )
        self.assertTrue(result.empty)
        self.assertEqual(diagnostics["forward_data_missing_excluded"], 4)

    @staticmethod
    def _fundamental(code, *, pledge, debt, profit_yoy):
        return {
            "code": code,
            "report_period": "20260331",
            "net_profit": 200_000_000,
            "net_profit_yoy": profit_yoy,
            "revenue": 1_000_000_000,
            "revenue_yoy": 18,
            "operating_cost": 500_000_000,
            "operating_profit": 240_000_000,
            "total_assets": 4_000_000_000,
            "total_assets_yoy": 12,
            "total_liabilities": 2_000_000_000,
            "debt_to_assets": debt,
            "equity": 2_000_000_000,
            "operating_cashflow": 240_000_000,
            "pledge_ratio": pledge,
        }

    @staticmethod
    def _valuation(code, *, pe, pb, ps, dividend):
        return {
            "code": code,
            "trade_date": "20260727",
            "close": 10,
            "turnover_rate": 1.2,
            "pe_ttm": pe,
            "pb": pb,
            "ps_ttm": ps,
            "dv_ttm": dividend,
            "total_mv": 10_000_000,
            "circ_mv": 8_000_000,
        }

    @staticmethod
    def _outlook(code, *, profit_yoy, eps_growth):
        return {
            "code": code,
            "official_profit_forecast": 250_000_000,
            "official_profit_yoy": profit_yoy,
            "forecast_type": "预增",
            "forecast_reason": "在手订单增长，加大研发投入，新产品逐步落地",
            "forecast_notice_date": "2026-07-20",
            "rd_evidence_score": 1.0,
            "order_evidence_score": 1.0,
            "rd_evidence": "加大研发投入，新产品逐步落地",
            "order_evidence": "在手订单增长",
            "research_report_count": 8,
            "analyst_eps_growth": eps_growth,
            "positive_rating_ratio": 0.9,
        }


if __name__ == "__main__":
    unittest.main()
