"""
交易资格范围识别：主板 / 科创板 / 创业板 / 北交所。
"""
def classify_board(code: str) -> str:
    """根据股票代码判断交易市场层级。"""
    code = str(code).strip()
    if not code:
        return "未知"

    # 北交所：常见 8/9 开头；4 开头多为老三板/退市整理，不归入北交所。
    if code.startswith(("8", "9")):
        return "北交所"

    # 科创板：688xxx
    if code.startswith("688"):
        return "科创板"

    # 创业板：300xxx, 301xxx
    if code.startswith(("300", "301")):
        return "创业板"

    # 主板：600/601/603/605 (上交所), 000/001/002/003 (深交所)
    if code.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return "主板"

    return "未知"


def is_main_board(code: str) -> bool:
    """是否为主板股票（无交易门槛）。"""
    return classify_board(code) == "主板"
