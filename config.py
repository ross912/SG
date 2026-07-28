"""
全局配置 — 集中管理技术指标、策略参数和路径。
"""
import os
from pathlib import Path

# === 路径 ===
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data_files"
KLINE_DIR = DATA_DIR / "kline"
OUTPUT_DIR = ROOT / "output_files"
BACKTEST_DIR = ROOT / "backtest_results"

for d in [DATA_DIR, KLINE_DIR, OUTPUT_DIR, BACKTEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Tushare Pro API ===
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# === 策略参数 ===
STRATEGY_PARAMS = {
    "ma_adx": {
        "ema_short": 5,
        "ema_mid": 20,
        "ema_long": 60,
        "adx_period": 14,
        "adx_threshold": 25,
    },
    "donchian": {
        "entry_period": 20,
        "exit_period": 10,
        "atr_period": 14,
        "atr_stop_multiple": 2.0,
    },
    "rsrs": {
        "window": 18,
        "buy_threshold": 0.7,
        "sell_threshold": -0.7,
        "adaptive_threshold": True,
    },
    "momentum": {
        "window": 252,           # 12个月交易日
        "skip": 21,              # 跳过最近1个月（避免短期反转干扰）
        "buy_threshold": 0.0,    # 累计收益 > 0 → 买入
        "sell_threshold": -0.15, # 累计收益 < -15% → 卖出
    },
    "candidate_trend": {
        "momentum_lookback": 252,
        "skip": 21,
        "high_window": 252,
        "vol_window": 60,
        "horizons": (20, 60, 120, 250),
    },
    "robust_trend": {
        "momentum_window": 120,
        "momentum_skip": 20,
        "trend_window": 120,
        "efficiency_window": 60,
        "risk_window": 60,
        "drawdown_window": 120,
        "consistency_horizons": (20, 60, 120),
    },
}

# 新趋势候选因子已进入扫描、反馈记录和独立样本外研究流程。
# 在通过训练/验证/最终测试三段稳定性门槛前，不参与生产综合分。
CANDIDATE_FACTOR_NAMES = (
    "momentum_12_2",
    "high_52w",
    "multi_horizon",
    "industry_residual_momentum",
)
CANDIDATE_FACTOR_STATUS = {
    "production_enabled": False,
    "reason": (
        "2022-2026 月度样本外研究未达到跨阶段稳定性门槛；"
        "行业残差因子等待可靠行业分类覆盖后再验证"
    ),
}
ROBUST_TREND_FACTOR_NAMES = (
    "medium_momentum",
    "trend_quality",
    "multi_period_consistency",
    "path_efficiency",
    "downside_risk_control",
    "drawdown_control",
)
TREND_MODEL_STATUS = {
    "active_version": "legacy-five-factor-2026-05-31",
    "active_label": "旧五因子趋势模型",
    "replacement_status": "rejected",
    "replacement_label": "稳健趋势重构未通过样本外验证",
    "replacement_report": "backtest_results/robust_trend_research.json",
    "message": (
        "现有趋势榜仍使用旧五因子模型；新候选在5/10/20日持有期均未通过"
        "训练期与2025验证期门槛，因此没有写入生产权重。"
    ),
}

# === 综合打分权重（市场状态自适应）===
# 五因子模型：量价关系、均线趋势(MA+ADX)、RSRS斜率、通道突破(Donchian)、
#             学术动量(12-1月)
# RSI 和短期反转已从评分中移除，升级为风险预警工具（背离/衰竭检测）
#
# 权重基于 2025-2026 历史回测 IC 优化 (2026-05-31 更新)。
# 原始比例合计 1.01，此处按原比例归一化为 1.00，避免综合分整体偏移。
_BASE_SCORE_WEIGHTS = {
    "volume_price": 0.3069,
    "ma_adx": 0.3762,
    "rsrs": 0.0495,
    "donchian": 0.1188,
    "momentum": 0.1486,
}
SCORE_WEIGHTS = {
    regime: dict(_BASE_SCORE_WEIGHTS)
    for regime in ("bull", "sideways", "bear")
}

# 因子方向: +1=正向(高分好), -1=反向(低分好)
# 趋势追踪系统：所有因子方向为正向
# 动态优化只调整权重；策略方向固定，避免自动翻转口径。
FACTOR_DIRECTION = {
    "volume_price":  +1,
    "ma_adx":        +1,
    "rsrs":          +1,
    "donchian":      +1,
    "momentum":      +1,
}
FACTOR_NAMES = tuple(FACTOR_DIRECTION)

# 策略注册表：并行运行两套策略
STRATEGIES = {
    "trend": {
        "label": "趋势跟踪",
        "directions": dict(FACTOR_DIRECTION),
        "csv_main": "main10",
        "csv_all":  "all30",
    },
    "mean_reversion": {
        "label": "均值回归",
        "directions": {name: -direction for name, direction in FACTOR_DIRECTION.items()},
        "csv_main": "mr_main10",
        "csv_all":  "mr_all30",
    },
}

MAIN_BOARD_POOL_SIZE = 10
ALL_MARKET_POOL_SIZE = 30

# === 入榜资格：风险警示与流动性 ===
# 最近 20 个交易日中，至少 15 日有正成交额，且 20 日平均成交额不低于 1,000 万元。
LIQUIDITY_LOOKBACK_DAYS = 20
MIN_VALID_TRADING_DAYS = 15
MIN_AVG_DAILY_AMOUNT = 10_000_000

# === 大盘仓位建议 ===
POSITION_SIZING_MAP = {"min_ratio": 0.10, "max_ratio": 0.95}

# === 回测默认参数 ===
BACKTEST_DEFAULT = {
    "start": "2024-01-01",
    "end": "2026-05-23",
    "initial_capital": 1_000_000,
    "commission_rate": 0.0003,   # 佣金
    "stamp_tax_rate": 0.001,     # 印花税（卖出）
    "slippage": 0.001,           # 滑点
}
