"""
向量化回测引擎 — A 股规则完整实现。
支持 T+1、涨跌停、印花税（卖出）、佣金、滑点。
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """回测结果容器。"""
    df: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    daily_returns: pd.Series


def run_backtest(df: pd.DataFrame, signals: pd.Series,
                 initial_capital: float = 1_000_000,
                 commission_rate: float = 0.0003,
                 stamp_tax_rate: float = 0.001,
                 slippage: float = 0.001,
                 code: str = "",
                 execute_on_next_bar: bool = True) -> BacktestResult:
    """
    向量化回测引擎。

    参数：
        df: 含 OHLCV 的日线数据，索引为 0..N-1
        signals: 信号序列，1=买入/持有, -1=卖出/空仓, 0=保持现状
        initial_capital: 初始资金
        commission_rate: 佣金费率
        stamp_tax_rate: 印花税（仅卖出）
        slippage: 滑点比例
        code: 股票代码，用于判断 10%/20%/30% 涨跌停
        execute_on_next_bar: 默认 True，信号在下一交易日开盘执行，避免同日收盘未来函数

    返回 BacktestResult，含逐日权益曲线、交易记录、绩效指标。
    """
    n = len(df)
    if n == 0:
        return BacktestResult(df=df, trades=pd.DataFrame(), metrics={},
                              daily_returns=pd.Series(dtype=float))

    close = df["close"].values.astype(float)
    open_price = (
        df["open"].values.astype(float)
        if "open" in df.columns else close.copy()
    )
    pct_change = df["pct_change"].fillna(0).values.astype(float) / 100.0 if "pct_change" in df.columns else np.zeros(n)

    sig = np.array(signals.values, dtype=int) if hasattr(signals, "values") else np.array(signals, dtype=int)
    if len(sig) != n:
        raise ValueError(f"signals 长度 {len(sig)} 与行情长度 {n} 不一致")

    position = np.zeros(n, dtype=int)     # 持仓状态: 0=空仓, 1=持仓
    shares = np.zeros(n, dtype=float)     # 持有股数
    cash = np.zeros(n, dtype=float)       # 现金
    equity = np.zeros(n, dtype=float)     # 总权益
    trades_list: list[dict] = []

    cash[0] = initial_capital
    equity[0] = initial_capital
    last_buy_total_cost = 0.0
    last_buy_index = -1

    for i in range(n):
        if i > 0:
            position[i] = position[i - 1]
            shares[i] = shares[i - 1]
            cash[i] = cash[i - 1]

        code_str = str(code or df.iloc[i].get("code", ""))
        if code_str.startswith(("8", "9")):
            price_limit = 0.30
        elif code_str.startswith("3") or code_str.startswith("688"):
            price_limit = 0.20
        else:
            price_limit = 0.10
        limit_up = pct_change[i] >= price_limit - 0.005
        limit_down = pct_change[i] <= -price_limit + 0.005
        action = sig[i - 1] if execute_on_next_bar and i > 0 else (sig[i] if not execute_on_next_bar else 0)
        base_exec_price = open_price[i] if np.isfinite(open_price[i]) and open_price[i] > 0 else close[i]

        # 先处理卖出；last_buy_index 明确落实 A 股 T+1。
        if action == -1 and position[i] == 1 and i > last_buy_index and not limit_down:
            sell_price = base_exec_price * (1 - slippage)
            revenue = shares[i] * sell_price
            commission = max(revenue * commission_rate, 5)
            stamp_tax = revenue * stamp_tax_rate
            net_revenue = revenue - commission - stamp_tax
            trade_pnl = net_revenue - last_buy_total_cost
            cash[i] += net_revenue
            trades_list.append({
                "date": df["date"].iloc[i] if "date" in df.columns else i,
                "type": "sell",
                "price": sell_price,
                "shares": int(shares[i]),
                "amount": net_revenue,
                "pnl": trade_pnl,
            })
            shares[i] = 0
            position[i] = 0
            last_buy_total_cost = 0.0
            last_buy_index = -1

        # 再处理买入
        if action == 1 and position[i] == 0 and not limit_up:
            buy_price = base_exec_price * (1 + slippage)
            max_shares = int(cash[i] / (buy_price * (1 + commission_rate) * 100)) * 100
            # 佣金最低 5 元时，上式仍可能多买一手；逐手回退而不是放弃整笔交易。
            while max_shares >= 100:
                estimated_cost = max_shares * buy_price
                estimated_commission = max(estimated_cost * commission_rate, 5)
                if estimated_cost + estimated_commission <= cash[i]:
                    break
                max_shares -= 100
            if max_shares >= 100:
                cost = max_shares * buy_price
                commission = max(cost * commission_rate, 5)
                total_cost = cost + commission
                if total_cost <= cash[i]:
                    cash[i] -= total_cost
                    shares[i] = max_shares
                    position[i] = 1
                    last_buy_total_cost = total_cost
                    last_buy_index = i
                    trades_list.append({
                        "date": df["date"].iloc[i] if "date" in df.columns else i,
                        "type": "buy",
                        "price": buy_price,
                        "shares": max_shares,
                        "amount": -total_cost,
                    })

        equity[i] = cash[i] + shares[i] * close[i]

    # 构建结果 DataFrame
    result_df = df.copy()
    result_df["position"] = position
    result_df["shares"] = shares
    result_df["cash"] = cash
    result_df["equity"] = equity

    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame(
        columns=["date", "type", "price", "shares", "amount"]
    )

    # 日收益率: r_i = (equity_i - equity_{i-1}) / equity_{i-1}
    daily_returns = pd.Series(np.zeros(n), index=df.index)
    if n > 1:
        rets = (equity[1:] - equity[:-1]) / np.where(equity[:-1] > 0, equity[:-1], initial_capital)
        daily_returns.iloc[0] = 0.0
        daily_returns.iloc[1:] = rets

    # 绩效指标
    metrics = _calculate_metrics(equity, daily_returns, trades_df, initial_capital)

    return BacktestResult(
        df=result_df,
        trades=trades_df,
        metrics=metrics,
        daily_returns=daily_returns,
    )


def _calculate_metrics(equity: np.ndarray, daily_returns: pd.Series,
                       trades: pd.DataFrame, initial_capital: float) -> dict:
    """计算回测绩效指标。"""
    if len(equity) < 2:
        return {"error": "数据不足"}

    final_equity = equity[-1]
    total_return = (final_equity / initial_capital - 1) * 100

    # 年化收益率（交易日约 250 天）
    n_days = len(equity)
    years = n_days / 250
    annual_return = (final_equity / initial_capital) ** (1 / max(years, 0.01)) - 1

    # 最大回撤
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-10)
    max_drawdown = drawdown.min() * 100

    # 最大回撤区间
    # Sharpe ratio
    ann_return_daily = daily_returns.mean() * 250 if len(daily_returns) > 0 else 0
    ann_vol_daily = daily_returns.std() * np.sqrt(250) if len(daily_returns) > 0 else 1
    sharpe = ann_return_daily / (ann_vol_daily + 1e-10)

    # Calmar ratio
    calmar = annual_return / (abs(max_drawdown) / 100 + 1e-10)

    # 胜率：基于每笔卖出的 PnL
    if not trades.empty:
        sells = trades[trades["type"] == "sell"]
        if len(sells) > 0:
            win_count = sum(1 for _, t in sells.iterrows() if t.get("pnl", 0) > 0)
            win_rate = win_count / len(sells) * 100
        else:
            win_rate = 0
    else:
        win_rate = 0

    # 交易次数
    buy_count = len(trades[trades["type"] == "buy"]) if not trades.empty else 0
    sell_count = len(trades[trades["type"] == "sell"]) if not trades.empty else 0

    return {
        "initial_capital": f"{initial_capital:,.0f}",
        "final_equity": f"{final_equity:,.0f}",
        "total_return": f"{total_return:.2f}%",
        "annual_return": f"{annual_return*100:.2f}%",
        "max_drawdown": f"{max_drawdown:.2f}%",
        "sharpe_ratio": f"{sharpe:.3f}",
        "calmar_ratio": f"{calmar:.3f}",
        "volatility": f"{ann_vol_daily*100:.2f}%",
        "win_rate": f"{win_rate:.1f}%",
        "total_trades": buy_count,
        "buy_count": buy_count,
        "sell_count": sell_count,
    }
