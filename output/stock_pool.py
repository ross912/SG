"""
每日选股池输出：CSV + 控制台打印。
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from config import OUTPUT_DIR


def save_stock_pool(df: pd.DataFrame, date_str: str = "",
                    output_dir: Path | None = None,
                    suffix: str = "") -> Path:
    """
    保存选股池到 CSV 文件。
    suffix: 文件名后缀，如 "main10", "all30", "mr_main10", "mr_all30"
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not date_str:
        date_str = datetime.now().strftime("%Y%m%d")

    name = f"stock_pool_{date_str}_{suffix}.csv" if suffix else f"stock_pool_{date_str}.csv"
    filepath = output_dir / name
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    return filepath


def print_stock_pool(df: pd.DataFrame, date_str: str = "",
                     title: str = "Stock Pool") -> None:
    """控制台美化打印选股池。"""
    if df.empty:
        print(f"\n  [{title}] No stocks meet criteria.")
        return

    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*80}")
    print(f"  {title}  |  {date_str}")
    print(f"{'='*80}")

    d = df.copy()
    show_cols = ['排名','代码','名称','所属概念','综合得分','均线信号','突破信号','RSRS信号','动量信号','最新价','涨跌幅','换手率','MA5','MA20']
    d = d[[c for c in show_cols if c in d.columns]]
    d.columns = ['Rank','Code','Name','Concepts','Score','MA','Break','RSRS','Momentum','Price',
                 'PctChg','Turnover','MA5','MA20'][:len(d.columns)]
    print(d.to_string(index=False))

    print(f"\n  Total: {len(df)} stocks | Saved to output_files/")
