"""
回测报告生成：权益曲线图 + 绩效指标表。
"""
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"),
)

import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import BACKTEST_DIR


def generate_report(result, strategy_name: str = "Strategy",
                    stock_code: str = "", output_dir: Path | None = None) -> Path:
    """
    生成回测报告（PNG 图表）。

    参数：
        result: BacktestResult 实例
        strategy_name: 策略名称
        stock_code: 股票代码
        output_dir: 输出目录

    返回输出文件路径。
    """
    if output_dir is None:
        output_dir = BACKTEST_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = result.df
    metrics = result.metrics

    # 英文标签（避免中文字体缺失）
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2, 1, 1]})
    fig.suptitle(f"{strategy_name} - {stock_code} Backtest Report", fontsize=14, fontweight="bold")

    # === 子图1：权益曲线 + 回撤 ===
    ax1 = axes[0]
    equity = df["equity"].values
    dates = df["date"].values if "date" in df.columns else range(len(df))

    ax1.plot(dates, equity, color="#1f77b4", linewidth=1.2, label="Equity")
    ax1.fill_between(range(len(equity)), equity, equity[0], alpha=0.08, color="#1f77b4")

    # 标记买卖点
    if not result.trades.empty:
        for _, t in result.trades.iterrows():
            t_date = t["date"]
            t_type = t["type"]
            if "date" in df.columns:
                idx = df[df["date"] == t_date].index
                if len(idx) > 0:
                    idx = idx[0]
                else:
                    continue
            else:
                idx = t_date if isinstance(t_date, (int, float)) else 0

            if idx < len(equity):
                color = "red" if t_type == "buy" else "green"
                marker = "^" if t_type == "buy" else "v"
                ax1.scatter(idx, equity[idx], c=color, marker=marker, s=40, alpha=0.8, zorder=5)

    ax1.set_ylabel("Equity (CNY)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 回撤子图
    ax1b = ax1.twinx()
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-10) * 100
    ax1b.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color="red", label="Drawdown %")
    ax1b.set_ylabel("Drawdown (%)", color="red")
    ax1b.tick_params(axis="y", labelcolor="red")
    ax1b.set_ylim(min(drawdown.min() * 1.2, -1), 5)

    # === 子图2：日收益率分布 ===
    ax2 = axes[1]
    daily_rets = result.daily_returns.dropna() if hasattr(result, "daily_returns") else pd.Series(dtype=float)
    if len(daily_rets) > 0:
        daily_rets = daily_rets[daily_rets != 0]
        ax2.hist(daily_rets * 100, bins=60, color="#2ca02c", alpha=0.7, edgecolor="white")
        ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax2.axvline(x=daily_rets.mean() * 100, color="blue", linestyle="--", alpha=0.7,
                    label=f"Mean: {daily_rets.mean()*100:.3f}%")
        ax2.set_xlabel("Daily Return (%)")
        ax2.set_ylabel("Frequency")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

    # === 子图3：绩效指标表 ===
    ax3 = axes[2]
    ax3.axis("off")

    if metrics and "error" not in metrics:
        text_lines = [
            f"Initial Capital: {metrics.get('initial_capital', '-')}",
            f"Final Equity: {metrics.get('final_equity', '-')}",
            f"Total Return: {metrics.get('total_return', '-')}",
            f"Annual Return: {metrics.get('annual_return', '-')}",
            f"Max Drawdown: {metrics.get('max_drawdown', '-')}",
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', '-')}",
            f"Calmar Ratio: {metrics.get('calmar_ratio', '-')}",
            f"Volatility: {metrics.get('volatility', '-')}",
            f"Win Rate: {metrics.get('win_rate', '-')}",
            f"Total Trades: {metrics.get('total_trades', '-')}",
        ]
        text = "\n".join(text_lines)
        ax3.text(0.05, 0.9, text, transform=ax3.transAxes, fontsize=11,
                 fontfamily="monospace", verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    filename = f"{strategy_name}_{stock_code}_report.png".replace(" ", "_")
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filepath


def print_metrics(metrics: dict, strategy_name: str = "") -> None:
    """控制台打印绩效指标。"""
    if strategy_name:
        print(f"\n{'='*50}")
        print(f"  {strategy_name} 回测绩效")
        print(f"{'='*50}")

    if "error" in metrics:
        print(f"  错误: {metrics['error']}")
        return

    for k, v in metrics.items():
        label = {
            "initial_capital": "Initial Cap",
            "final_equity": "Final Equity",
            "total_return": "Total Return",
            "annual_return": "Annual Return",
            "max_drawdown": "Max Drawdown",
            "sharpe_ratio": "Sharpe Ratio",
            "calmar_ratio": "Calmar Ratio",
            "volatility": "Volatility",
            "win_rate": "Win Rate",
            "total_trades": "Total Trades",
            "buy_count": "Buy Count",
            "sell_count": "Sell Count",
        }.get(k, k)
        print(f"  {label:12s}: {v}")
