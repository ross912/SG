"""单股技术策略回测入口。"""
import argparse
from datetime import datetime

import pandas as pd

from backtest.engine import run_backtest
from backtest.report import generate_report, print_metrics
from config import BACKTEST_DEFAULT, STRATEGY_PARAMS
from data.fetcher import get_kline_daily
from data.storage import load_kline, save_kline
from strategy.rsrs import RsrsStrategy
from strategy.trend_following import DonchianStrategy, MaAdxStrategy


def backtest_stock(
    code: str,
    strategy_name: str = "ma_adx",
    start: str | None = None,
    end: str | None = None,
) -> bool:
    """回测一只股票；交易引擎保留真实涨跌停无法成交约束。"""
    start = start or BACKTEST_DEFAULT["start"]
    end = end or datetime.now().strftime("%Y-%m-%d")
    frame = load_kline(code)
    if frame is None:
        frame = get_kline_daily(
            code,
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
        )
        if frame is not None:
            save_kline(code, frame)
    if frame is None or len(frame) < 120:
        print(f"{code} 数据不足，跳过")
        return False

    frame = frame[
        (pd.to_datetime(frame["date"]) >= pd.Timestamp(start))
        & (pd.to_datetime(frame["date"]) <= pd.Timestamp(end))
    ].copy()
    if len(frame) < 60:
        print(f"{code} 在指定日期内数据不足，跳过")
        return False

    constructors = {
        "ma_adx": lambda: MaAdxStrategy(**STRATEGY_PARAMS["ma_adx"]),
        "donchian": lambda: DonchianStrategy(**STRATEGY_PARAMS["donchian"]),
        "rsrs": lambda: RsrsStrategy(**STRATEGY_PARAMS["rsrs"]),
    }
    calculated = constructors[strategy_name]().calculate(frame)
    result = run_backtest(
        calculated,
        calculated["signal"],
        initial_capital=BACKTEST_DEFAULT["initial_capital"],
        commission_rate=BACKTEST_DEFAULT["commission_rate"],
        stamp_tax_rate=BACKTEST_DEFAULT["stamp_tax_rate"],
        slippage=BACKTEST_DEFAULT["slippage"],
        code=code,
        execute_on_next_bar=True,
    )
    print_metrics(result.metrics, f"{code} {strategy_name}")
    print(f"图表已保存: {generate_report(result, strategy_name, code)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="单股技术策略回测")
    parser.add_argument("--stock", action="append", required=True, help="股票代码；可重复传入")
    parser.add_argument("--strategy", choices=["ma_adx", "donchian", "rsrs"], default="ma_adx")
    parser.add_argument("--start", default=BACKTEST_DEFAULT["start"])
    parser.add_argument("--end", default=BACKTEST_DEFAULT["end"])
    args = parser.parse_args()
    success = 0
    for raw_code in args.stock:
        code = str(raw_code).strip().zfill(6)
        success += int(backtest_stock(code, args.strategy, args.start, args.end))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
