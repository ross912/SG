"""基本面价值榜：质量、估值、历史成长、未来空间与风险门禁。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FUNDAMENTAL_WEIGHTS = {
    "质量分": 25.0,
    "估值分": 25.0,
    "历史成长分": 15.0,
    "未来空间分": 20.0,
    "风险控制分": 15.0,
}

FINANCIAL_KEYWORDS = ("银行", "保险", "证券", "金融", "信托", "期货")
REAL_ESTATE_KEYWORDS = ("房地产", "地产", "房产", "园区开发")


def build_fundamental_ranking(
    fundamentals: pd.DataFrame,
    valuations: pd.DataFrame,
    stock_metadata: pd.DataFrame,
    outlook: pd.DataFrame | None = None,
    *,
    valuation_date: str,
    top_n: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """构造基本面价值榜及可解释诊断。

    高风险股票、核心财务字段缺失股票和未达到最低质量门槛的股票均不进入
    最终榜单。低风险不等于无风险，因此所有入选项都会保留风险提示。
    """
    if fundamentals.empty or valuations.empty:
        return _empty_output(), {
            "input_count": 0, "eligible_count": 0, "selected_count": 0,
            "high_risk_excluded": 0, "data_missing_excluded": 0,
            "threshold_excluded": 0,
        }

    frame = fundamentals.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    value_frame = valuations.copy()
    value_frame["code"] = value_frame["code"].astype(str).str.zfill(6)
    meta = _normalize_metadata(stock_metadata)
    frame = frame.merge(value_frame, on="code", how="inner", suffixes=("", "_valuation"))
    frame = frame.merge(meta, on="code", how="left", suffixes=("", "_meta"))
    if outlook is not None and not outlook.empty:
        outlook_frame = outlook.copy()
        outlook_frame["code"] = outlook_frame["code"].astype(str).str.zfill(6)
        frame = frame.merge(
            outlook_frame.drop_duplicates("code"),
            on="code",
            how="left",
        )
    frame["name"] = frame["name"].fillna(frame.get("statement_name", "")).fillna("")
    frame["industry"] = frame["industry"].fillna("").astype(str)
    frame["list_date"] = frame["list_date"].fillna("").astype(str)
    frame["model_type"] = frame.apply(_model_type, axis=1)
    frame["comparison_group"] = _comparison_groups(frame)

    report_period = _normalize_period(
        frame.get("report_period", pd.Series([""])).dropna().iloc[0]
        if frame.get("report_period", pd.Series(dtype=object)).notna().any()
        else ""
    )
    annualizer = _annualizer(report_period)
    frame["roe"] = _safe_div(frame["net_profit"] * annualizer, frame["equity"]) * 100
    frame["roa"] = _safe_div(frame["net_profit"] * annualizer, frame["total_assets"]) * 100
    frame["gross_margin"] = (
        _safe_div(frame["revenue"] - frame["operating_cost"], frame["revenue"]) * 100
    )
    frame["net_margin"] = _safe_div(frame["net_profit"], frame["revenue"]) * 100
    frame["cash_conversion"] = _safe_div(frame["operating_cashflow"], frame["net_profit"])
    frame["asset_turnover"] = _safe_div(
        frame["revenue"] * annualizer, frame["total_assets"],
    )
    frame = _prepare_forward_features(frame)

    for metric in ("pe_ttm", "pb", "ps_ttm"):
        frame[f"{metric}_value"] = _low_is_good_percentile(
            frame, metric, "comparison_group",
        )
    frame["dv_ttm_value"] = _high_is_good_percentile(
        frame, "dv_ttm", "comparison_group",
    )

    quality_scores: list[float] = []
    valuation_scores: list[float] = []
    historical_growth_scores: list[float] = []
    future_space_scores: list[float] = []
    risk_scores: list[float] = []
    risk_levels: list[str] = []
    risk_notes: list[str] = []
    hard_reasons: list[str] = []
    entry_reasons: list[str] = []

    for _, row in frame.iterrows():
        quality = _quality_score(row)
        valuation = _valuation_score(row)
        historical_growth = _historical_growth_score(row)
        future_space = _future_space_score(row)
        risk, hard, warnings = _risk_assessment(row)
        level = "高" if hard or risk < 6 else "低" if risk >= 12 else "中"
        if not warnings:
            warnings = ["低风险不等于无风险，未覆盖突发事件与审计意见"]
        elif level == "低":
            warnings.append("仍可能存在未披露或突发风险")
        quality_scores.append(quality)
        valuation_scores.append(valuation)
        historical_growth_scores.append(historical_growth)
        future_space_scores.append(future_space)
        risk_scores.append(risk)
        risk_levels.append(level)
        risk_notes.append("；".join(dict.fromkeys(warnings)))
        hard_reasons.append("；".join(hard))
        entry_reasons.append(
            _entry_reason(
                row, quality, valuation, historical_growth, future_space,
            ),
        )

    frame["质量分"] = quality_scores
    frame["估值分"] = valuation_scores
    frame["历史成长分"] = historical_growth_scores
    frame["未来空间分"] = future_space_scores
    frame["风险控制分"] = risk_scores
    frame["暴雷风险"] = risk_levels
    frame["风险提示"] = risk_notes
    frame["高风险原因"] = hard_reasons
    frame["入选理由"] = entry_reasons
    frame["综合得分"] = (
        frame["质量分"] + frame["估值分"] + frame["历史成长分"]
        + frame["未来空间分"] + frame["风险控制分"]
    )

    core_columns = [
        "net_profit", "revenue", "total_assets", "total_liabilities",
        "equity", "operating_cashflow", "pe_ttm", "pb",
    ]
    financial_missing = frame[core_columns].isna().sum(axis=1) >= 3
    forward_missing = frame["forward_core_coverage"] < 3
    data_missing = financial_missing | forward_missing
    hard_risk = frame["暴雷风险"].eq("高")
    threshold_pass = (
        (frame["质量分"] >= 11.5)
        & (frame["估值分"] >= 8)
        & (frame["历史成长分"] >= 4)
        & (frame["未来空间分"] >= 8)
        & (frame["综合得分"] >= 55)
    )
    eligible = frame.loc[~data_missing & ~hard_risk & threshold_pass].copy()
    eligible = eligible.sort_values(
        ["综合得分", "质量分", "估值分", "code"],
        ascending=[False, False, False, True],
    ).head(top_n)

    output = _format_output(
        eligible,
        valuation_date=valuation_date,
        report_period=report_period,
    )
    diagnostics = {
        "input_count": int(len(frame)),
        "eligible_count": int((~data_missing & ~hard_risk & threshold_pass).sum()),
        "selected_count": int(len(output)),
        "high_risk_excluded": int((hard_risk & ~data_missing).sum()),
        "data_missing_excluded": int(data_missing.sum()),
        "forward_data_missing_excluded": int(forward_missing.sum()),
        "threshold_excluded": int((~data_missing & ~hard_risk & ~threshold_pass).sum()),
        "report_period": report_period,
        "valuation_date": str(valuation_date),
        "high_risk_examples": frame.loc[
            hard_risk & ~data_missing, ["code", "name", "高风险原因"]
        ].head(10).to_dict("records"),
        "limitations": (
            "未来空间基于行业财务景气、市场份额变化、公司业绩预告、机构盈利"
            "预测及公告中的研发/订单证据，不是市场规模预测。机构覆盖存在偏差；"
            "风险门禁仍不覆盖尚未披露事件、审计意见和监管调查。"
        ),
    }
    return output, diagnostics


def _quality_score(row: pd.Series) -> float:
    model = row["model_type"]
    if model == "金融":
        score = (
            15 * _scale(row["roe"], 5, 18)
            + 5 * _scale(row["roa"], 0.3, 2.0)
            + 5 * _scale(row["net_margin"], 5, 35)
            + 5 * _scale(row.get("total_assets_yoy"), -5, 20)
        )
    elif model == "房地产":
        score = (
            10 * _scale(row["roe"], 3, 15)
            + 5 * _scale(row["net_margin"], 2, 18)
            + 8 * _scale(row["cash_conversion"], 0, 1.5)
            + 7 * _scale(row["debt_to_assets"], 90, 55, higher=False)
        )
    elif model == "上市不足三年":
        score = (
            10 * _scale(row["roe"], 4, 18)
            + 7 * _scale(row["gross_margin"], 12, 55)
            + 5 * _scale(row["net_margin"], 2, 20)
            + 5 * _scale(row["cash_conversion"], 0, 1.5)
            + 3 * _scale(row["debt_to_assets"], 85, 30, higher=False)
        )
    else:
        score = (
            12 * _scale(row["roe"], 5, 20)
            + 5 * _scale(row["gross_margin"], 10, 50)
            + 3 * _scale(row["net_margin"], 3, 20)
            + 6 * _scale(row["cash_conversion"], 0, 1.5)
            + 4 * _scale(row["debt_to_assets"], 85, 30, higher=False)
        )
    return round(float(score * 25.0 / 30.0), 2)


def _valuation_score(row: pd.Series) -> float:
    model = row["model_type"]
    if model == "金融":
        weights = {"pe_ttm": 10.0, "pb": 15.0, "dv_ttm": 5.0}
    elif model == "房地产":
        weights = {"pe_ttm": 5.0, "pb": 12.0, "ps_ttm": 10.0, "dv_ttm": 3.0}
    elif model == "上市不足三年":
        weights = {"pe_ttm": 8.0, "pb": 7.0, "ps_ttm": 12.0, "dv_ttm": 3.0}
    else:
        weights = {"pe_ttm": 12.0, "pb": 8.0, "ps_ttm": 6.0, "dv_ttm": 4.0}
    available = {
        metric: weight
        for metric, weight in weights.items()
        if _valuation_metric_available(row, metric)
    }
    if not available:
        return 0.0
    weighted = sum(
        weight * _finite_or_zero(row.get(f"{metric}_value"))
        for metric, weight in available.items()
    )
    return round(float(weighted * 25.0 / sum(available.values())), 2)


def _valuation_metric_available(row: pd.Series, metric: str) -> bool:
    value = _number(row.get(metric))
    if value is None:
        return False
    if metric == "dv_ttm":
        return value >= 0
    return value > 0


def _historical_growth_score(row: pd.Series) -> float:
    score = (
        6 * _scale(row.get("revenue_yoy"), -10, 30)
        + 6 * _scale(row.get("net_profit_yoy"), -20, 50)
        + 3 * _scale(row.get("total_assets_yoy"), -5, 20)
    )
    return round(float(score), 2)


def _future_space_score(row: pd.Series) -> float:
    model = row["model_type"]
    if model == "金融":
        weights = {
            "industry_outlook_value": 5.0,
            "market_share_strength": 5.0,
            "earnings_outlook_value": 10.0,
        }
    elif model == "房地产":
        weights = {
            "industry_outlook_value": 5.0,
            "market_share_strength": 4.0,
            "earnings_outlook_value": 8.0,
            "order_evidence_score": 3.0,
        }
    else:
        weights = {
            "industry_outlook_value": 4.0,
            "market_share_strength": 3.0,
            "earnings_outlook_value": 7.0,
            "rd_evidence_score": 3.0,
            "order_evidence_score": 3.0,
        }
    return round(float(sum(
        weight * _finite_or_zero(row.get(metric))
        for metric, weight in weights.items()
    )), 2)


def _prepare_forward_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    numeric_columns = [
        "official_profit_forecast", "official_profit_yoy",
        "research_report_count", "analyst_eps_growth", "positive_rating_ratio",
        "rd_evidence_score", "order_evidence_score",
    ]
    text_columns = [
        "forecast_type", "forecast_reason", "forecast_notice_date",
        "rd_evidence", "order_evidence",
    ]
    for column in numeric_columns:
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in text_columns:
        if column not in data:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str)

    industries = data["industry"].replace("", np.nan)
    industry_counts = industries.value_counts()
    data["forward_group"] = [
        industry if pd.notna(industry) and industry_counts.get(industry, 0) >= 5
        else model
        for industry, model in zip(industries, data["model_type"])
    ]
    revenue_yoy = pd.to_numeric(data.get("revenue_yoy"), errors="coerce")
    revenue = pd.to_numeric(data.get("revenue"), errors="coerce").where(lambda s: s > 0)
    data["industry_revenue_growth"] = revenue_yoy.groupby(
        data["forward_group"],
    ).transform("median")
    positive_growth = revenue_yoy.gt(0).astype(float).where(revenue_yoy.notna())
    data["industry_growth_breadth"] = positive_growth.groupby(
        data["forward_group"],
    ).transform("mean")
    data["industry_outlook_value"] = (
        0.6 * data["industry_revenue_growth"].map(lambda value: _scale(value, -10, 30))
        + 0.4 * data["industry_growth_breadth"].map(
            lambda value: _scale(value, 0.35, 0.75),
        )
    )

    prior_revenue = revenue / (1 + revenue_yoy / 100).where(
        (1 + revenue_yoy / 100) > 0.05,
    )
    current_group_revenue = revenue.groupby(data["forward_group"]).transform("sum")
    prior_group_revenue = prior_revenue.groupby(data["forward_group"]).transform("sum")
    data["market_share_pct"] = revenue / current_group_revenue * 100
    previous_share = prior_revenue / prior_group_revenue * 100
    data["market_share_change_pp"] = data["market_share_pct"] - previous_share
    share_change_rank = data["market_share_change_pp"].groupby(
        data["forward_group"],
    ).rank(pct=True)
    share_level_rank = data["market_share_pct"].groupby(
        data["forward_group"],
    ).rank(pct=True)
    data["market_share_strength"] = (
        0.7 * share_change_rank + 0.3 * share_level_rank
    )

    official_available = (
        data["official_profit_forecast"].notna()
        | data["official_profit_yoy"].notna()
        | data["forecast_type"].str.strip().ne("")
    )
    analyst_available = (
        data["research_report_count"].fillna(0).gt(0)
        & (
            data["analyst_eps_growth"].notna()
            | data["positive_rating_ratio"].notna()
        )
    )
    official_value = data["official_profit_yoy"].map(
        lambda value: _scale(value, -20, 50),
    )
    official_value = official_value.where(
        data["official_profit_forecast"].fillna(1) > 0,
        0.0,
    )
    analyst_raw_value = (
        0.7 * data["analyst_eps_growth"].map(lambda value: _scale(value, -10, 30))
        + 0.3 * data["positive_rating_ratio"].map(
            lambda value: _scale(value, 0.5, 1.0),
        )
    )
    analyst_coverage = data["research_report_count"].map(
        lambda value: _scale(value, 1, 10),
    )
    analyst_value = 0.5 + (analyst_raw_value - 0.5) * analyst_coverage
    data["earnings_outlook_value"] = np.select(
        [official_available & analyst_available, official_available, analyst_available],
        [0.6 * official_value + 0.4 * analyst_value, official_value, analyst_value],
        default=np.nan,
    )

    industry_available = (
        data["industry_revenue_growth"].notna()
        & data["industry_growth_breadth"].notna()
    )
    share_available = (
        data["market_share_pct"].notna()
        & data["market_share_change_pp"].notna()
    )
    forecast_available = official_available | analyst_available
    data["forward_core_coverage"] = (
        industry_available.astype(int)
        + share_available.astype(int)
        + forecast_available.astype(int)
    )
    data["forward_evidence_count"] = (
        data["forward_core_coverage"]
        + data["rd_evidence_score"].notna().astype(int)
        + data["order_evidence_score"].notna().astype(int)
    )
    data["未来数据置信度"] = np.select(
        [
            (data["forward_core_coverage"] == 3)
            & (data["forward_evidence_count"] >= 4),
            data["forward_core_coverage"] == 3,
        ],
        ["高", "中"],
        default="低",
    )
    data["行业景气"] = data.apply(_industry_outlook_text, axis=1)
    data["市占变化"] = data.apply(_market_share_text, axis=1)
    data["盈利预期"] = data.apply(_earnings_outlook_text, axis=1)
    data["未来空间依据"] = data.apply(_future_space_reason, axis=1)
    return data


def _risk_assessment(row: pd.Series) -> tuple[float, list[str], list[str]]:
    hard: list[str] = []
    warnings: list[str] = []
    score = 15.0
    model = row["model_type"]
    equity = _number(row.get("equity"))
    profit = _number(row.get("net_profit"))
    revenue = _number(row.get("revenue"))
    debt = _number(row.get("debt_to_assets"))
    pledge = _number(row.get("pledge_ratio"))
    cash_conversion = _number(row.get("cash_conversion"))
    profit_yoy = _number(row.get("net_profit_yoy"))
    official_profit_forecast = _number(row.get("official_profit_forecast"))

    if equity is not None and equity <= 0:
        hard.append("净资产为负")
    if profit is not None and profit <= 0:
        hard.append("最新报告期净利润为负")
    if revenue is not None and revenue <= 0:
        hard.append("最新报告期营业收入异常")
    if pledge is not None and pledge >= 50:
        hard.append(f"股权质押比例 {pledge:.1f}%")
    if model == "房地产" and debt is not None and debt >= 92:
        hard.append(f"房地产模型资产负债率 {debt:.1f}%")
    if model not in {"金融", "房地产"} and debt is not None and debt >= 85:
        hard.append(f"资产负债率 {debt:.1f}%")
    if (
        profit_yoy is not None and profit_yoy <= -80
        and cash_conversion is not None and cash_conversion < 0
    ):
        hard.append("利润同比大幅恶化且经营现金流为负")
    if official_profit_forecast is not None and official_profit_forecast < 0:
        hard.append("最新业绩预告预计亏损")

    if pledge is None:
        warnings.append("股权质押数据未覆盖")
        score -= 1
    elif pledge >= 30:
        warnings.append(f"股权质押偏高 {pledge:.1f}%")
        score -= 8
    elif pledge >= 15:
        warnings.append(f"股权质押 {pledge:.1f}%")
        score -= 4
    elif pledge > 0:
        warnings.append(f"存在股权质押 {pledge:.1f}%")
        score -= 2

    if model != "金融" and debt is not None:
        medium_debt = 85 if model == "房地产" else 75
        watch_debt = 75 if model == "房地产" else 65
        if debt >= medium_debt:
            warnings.append(f"资产负债率偏高 {debt:.1f}%")
            score -= 6
        elif debt >= watch_debt:
            warnings.append(f"资产负债率需关注 {debt:.1f}%")
            score -= 3

    if model != "金融" and cash_conversion is not None:
        if cash_conversion < 0:
            warnings.append("经营现金流与利润背离")
            score -= 5
        elif cash_conversion < 0.5:
            warnings.append("利润现金含量偏低")
            score -= 3

    revenue_yoy = _number(row.get("revenue_yoy"))
    if revenue_yoy is not None and revenue_yoy < -20:
        warnings.append(f"营收同比下降 {abs(revenue_yoy):.1f}%")
        score -= 3
    if profit_yoy is not None and profit_yoy < -30:
        warnings.append(f"净利润同比下降 {abs(profit_yoy):.1f}%")
        score -= 3
    if model == "上市不足三年":
        warnings.append("上市历史不足三年，跨周期数据有限")
        score -= 2

    return round(max(0.0, score), 2), hard, warnings


def _entry_reason(
    row: pd.Series,
    quality: float,
    valuation: float,
    historical_growth: float,
    future_space: float,
) -> str:
    dimensions = sorted(
        [
            ("质量", quality / 25),
            ("估值", valuation / 25),
            ("历史成长", historical_growth / 15),
            ("未来空间", future_space / 20),
        ],
        key=lambda item: (-item[1], item[0]),
    )
    strengths = "、".join(item[0] for item in dimensions[:2])
    valuation_bits = []
    pe_label = str(row.get("pe_basis") or "PE(TTM)")
    for label, key in ((pe_label, "pe_ttm"), ("PB", "pb"), ("PS(TTM)", "ps_ttm")):
        value = _number(row.get(key))
        if value is not None and value > 0:
            valuation_bits.append(f"{label} {value:.2f}")
    valuation_text = "，".join(valuation_bits[:2]) or "估值数据有限"
    return (
        f"{strengths}得分靠前；{valuation_text}；"
        f"{row.get('未来空间依据', '')}；"
        f"采用{row['model_type']}模型与{row['comparison_group']}同组比较"
    )


def _industry_outlook_text(row: pd.Series) -> str:
    growth = _number(row.get("industry_revenue_growth"))
    breadth = _number(row.get("industry_growth_breadth"))
    if growth is None or breadth is None:
        return "行业财务样本不足"
    return f"行业营收中位增速 {growth:.1f}%，增长公司占比 {breadth * 100:.0f}%"


def _market_share_text(row: pd.Series) -> str:
    share = _number(row.get("market_share_pct"))
    change = _number(row.get("market_share_change_pp"))
    if share is None or change is None:
        return "市占变化数据不足"
    direction = "提升" if change > 0.005 else "下降" if change < -0.005 else "基本持平"
    return f"财报收入份额约 {share:.2f}%，同比{direction} {abs(change):.2f} 个百分点"


def _earnings_outlook_text(row: pd.Series) -> str:
    parts: list[str] = []
    official_yoy = _number(row.get("official_profit_yoy"))
    forecast_type = str(row.get("forecast_type") or "").strip()
    if official_yoy is not None:
        parts.append(f"业绩预告同比 {official_yoy:.1f}%")
    elif forecast_type:
        parts.append(f"业绩预告：{forecast_type}")
    analyst_growth = _number(row.get("analyst_eps_growth"))
    reports = _number(row.get("research_report_count"))
    if analyst_growth is not None:
        suffix = f"（{int(reports)}份研报）" if reports else ""
        parts.append(f"机构下一年 EPS 增速 {analyst_growth:.1f}%{suffix}")
    return "；".join(parts) if parts else "缺少公司预告和机构盈利预测"


def _future_space_reason(row: pd.Series) -> str:
    pieces = [
        str(row.get("行业景气") or ""),
        str(row.get("市占变化") or ""),
        str(row.get("盈利预期") or ""),
    ]
    rd = str(row.get("rd_evidence") or "").strip()
    order = str(row.get("order_evidence") or "").strip()
    if rd:
        pieces.append(f"研发证据：{rd}")
    if order:
        pieces.append(f"订单/产能证据：{order}")
    return "；".join(piece for piece in pieces if piece)


def _format_output(
    frame: pd.DataFrame,
    *,
    valuation_date: str,
    report_period: str,
) -> pd.DataFrame:
    columns = [
        "排名", "代码", "名称", "所属行业", "模型", "综合得分",
        "质量分", "估值分", "历史成长分", "未来空间分", "风险控制分",
        "未来数据置信度", "暴雷风险", "入选理由", "未来空间依据",
        "风险提示", "行业景气", "市占变化", "盈利预期", "研发证据",
        "订单/产能证据", "ROE", "营收同比", "净利润同比", "PE",
        "PE口径", "PB", "股息率(TTM)", "质押比例", "最新价", "财报期",
        "估值日期",
    ]
    rows = []
    for rank_index, (_, row) in enumerate(frame.iterrows(), 1):
        rows.append({
            "排名": rank_index,
            "代码": str(row["code"]).zfill(6),
            "名称": row.get("name", ""),
            "所属行业": row.get("industry", ""),
            "模型": row.get("model_type", ""),
            "综合得分": round(float(row["综合得分"]), 2),
            "质量分": round(float(row["质量分"]), 2),
            "估值分": round(float(row["估值分"]), 2),
            "历史成长分": round(float(row["历史成长分"]), 2),
            "未来空间分": round(float(row["未来空间分"]), 2),
            "风险控制分": round(float(row["风险控制分"]), 2),
            "未来数据置信度": row.get("未来数据置信度", ""),
            "暴雷风险": row.get("暴雷风险", ""),
            "入选理由": row.get("入选理由", ""),
            "未来空间依据": row.get("未来空间依据", ""),
            "风险提示": row.get("风险提示", ""),
            "行业景气": row.get("行业景气", ""),
            "市占变化": row.get("市占变化", ""),
            "盈利预期": row.get("盈利预期", ""),
            "研发证据": row.get("rd_evidence", ""),
            "订单/产能证据": row.get("order_evidence", ""),
            "ROE": _percent(row.get("roe")),
            "营收同比": _percent(row.get("revenue_yoy")),
            "净利润同比": _percent(row.get("net_profit_yoy")),
            "PE": _display_number(row.get("pe_ttm")),
            "PE口径": row.get("pe_basis", "PE(TTM)"),
            "PB": _display_number(row.get("pb")),
            "股息率(TTM)": _percent(row.get("dv_ttm")),
            "质押比例": _percent(row.get("pledge_ratio")),
            "最新价": _display_number(row.get("close")),
            "财报期": _date_text(report_period),
            "估值日期": _date_text(str(valuation_date)),
        })
    return pd.DataFrame(rows, columns=columns)


def _normalize_metadata(stock_metadata: pd.DataFrame) -> pd.DataFrame:
    if stock_metadata is None or stock_metadata.empty:
        return pd.DataFrame(columns=["code", "name", "industry", "list_date"])
    frame = stock_metadata.copy()
    rename = {
        "symbol": "code", "股票代码": "code", "股票简称": "name",
        "上市日期": "list_date",
    }
    frame = frame.rename(columns=rename)
    for column in ("code", "name", "industry", "list_date"):
        if column not in frame:
            frame[column] = ""
    frame["code"] = frame["code"].astype(str).str.split(".").str[0].str.zfill(6)
    return frame[["code", "name", "industry", "list_date"]].drop_duplicates("code")


def _model_type(row: pd.Series) -> str:
    list_date = pd.to_datetime(row.get("list_date"), errors="coerce")
    if pd.notna(list_date):
        years = (pd.Timestamp.today().normalize() - list_date).days / 365.25
        if years < 3:
            return "上市不足三年"
    industry = str(row.get("industry", ""))
    if any(keyword in industry for keyword in FINANCIAL_KEYWORDS):
        return "金融"
    if any(keyword in industry for keyword in REAL_ESTATE_KEYWORDS):
        return "房地产"
    return "一般行业"


def _comparison_groups(frame: pd.DataFrame) -> pd.Series:
    industries = frame["industry"].replace("", np.nan)
    counts = industries.value_counts()
    return pd.Series(
        [
            industry if pd.notna(industry) and counts.get(industry, 0) >= 15 else model
            for industry, model in zip(industries, frame["model_type"])
        ],
        index=frame.index,
    )


def _low_is_good_percentile(
    frame: pd.DataFrame,
    metric: str,
    group: str,
) -> pd.Series:
    values = pd.to_numeric(frame.get(metric), errors="coerce")
    valid = values.where(values > 0)
    ranks = valid.groupby(frame[group]).rank(pct=True, ascending=True)
    group_sizes = valid.groupby(frame[group]).transform("count")
    global_ranks = valid.rank(pct=True, ascending=True)
    ranks = ranks.where(group_sizes >= 5, global_ranks)
    return (1.0 - ranks + 0.01).clip(0, 1).fillna(0)


def _high_is_good_percentile(
    frame: pd.DataFrame,
    metric: str,
    group: str,
) -> pd.Series:
    values = pd.to_numeric(frame.get(metric), errors="coerce")
    valid = values.where(values >= 0)
    ranks = valid.groupby(frame[group]).rank(pct=True, ascending=True)
    group_sizes = valid.groupby(frame[group]).transform("count")
    global_ranks = valid.rank(pct=True, ascending=True)
    return ranks.where(group_sizes >= 5, global_ranks).fillna(0)


def _scale(value: Any, low: float, high: float, *, higher: bool = True) -> float:
    number = _number(value)
    if number is None or high == low:
        return 0.0
    result = (number - low) / (high - low)
    if not higher:
        result = 1.0 - result
    return float(np.clip(result, 0, 1))


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    left = pd.to_numeric(numerator, errors="coerce")
    right = pd.to_numeric(denominator, errors="coerce")
    return left / right.replace(0, np.nan)


def _annualizer(report_period: str) -> float:
    month = str(report_period)[4:6]
    return {"03": 4.0, "06": 2.0, "09": 4.0 / 3.0, "12": 1.0}.get(month, 1.0)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _finite_or_zero(value: Any) -> float:
    number = _number(value)
    return number if number is not None else 0.0


def _display_number(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.2f}"


def _percent(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.2f}%"


def _date_text(value: str) -> str:
    digits = str(value).replace("-", "")
    if len(digits) == 8 and digits.isdigit():
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    return str(value)


def _normalize_period(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _empty_output() -> pd.DataFrame:
    return _format_output(
        pd.DataFrame(),
        valuation_date="",
        report_period="",
    )
