# SG Quant 基本面价值与量化研究

当前版本将基本面价值榜置于所有技术榜单之前。基本面榜不使用均线、动量、
突破等日 K 信号决定入选；最新价格只用于计算当日估值。原趋势与均值回归榜
继续保留为独立技术研究视角，不能与基本面结论混为一谈。

## 输出榜单

- 基本面价值：全市场 Top 30
- 趋势跟踪：主板 Top 10、全市场 Top 30
- 均值回归：主板 Top 10、全市场 Top 30

主板范围是沪深两市无需额外投资者资格的 A 股代码段。全市场包含主板、创业板、科创板和北交所。名称前缀为 ST、*ST、SST 或 S*ST 的股票会被排除；另要求最近 20 个交易日中至少 15 日有正成交额，且 20 日平均成交额不低于 1,000 万元。涨停和跌停不作为选股过滤条件。

## 基本面价值榜

基本面榜采用五维满分 100 分：

- 质量 25 分：ROE、ROA、利润率、经营现金流含金量、负债结构；
- 估值 25 分：PE(TTM)、PB、PS(TTM)、股息率，按同行业或同模型组比较；
- 历史成长 15 分：营业收入、净利润和总资产同比变化；
- 未来空间 20 分：行业财务景气、上市公司收入份额变化、公司业绩预告、
  机构 EPS 预测，以及公告中明确出现的研发、订单或扩产证据；
- 风险控制 15 分：股权质押、负债、现金流背离、利润恶化和数据覆盖。

“未来空间”不是由大模型阅读新闻后主观打分。行业景气使用同行业营收增速
中位数与正增长公司占比；竞争地位使用公司收入占同一上市公司行业样本收入的
比例及同比变化，因此它是“上市公司财报收入份额”，不是整个社会市场份额；
盈利预期结合最新公司业绩预告和东方财富汇总的机构 EPS 预测。研发、订单与
扩产只有在业绩预告原因中出现明确文字时才计分，未提及只表示没有可验证证据，
不能解释为公司没有研发或订单。

未来空间的三个关键模块是行业景气、财报收入份额和盈利预期。任何股票缺少
其中一个模块，未来数据置信度即为低并取消入榜资格；研发、订单证据会把
置信度从中提高到高，但不是所有行业都强制要求存在这些证据。金融模型把研发
与订单权重重新分配给行业、份额和盈利预期；房地产模型保留项目/产能证据；
一般行业和上市不足三年模型均保留研发与订单证据。

金融、房地产、上市不足三年的公司使用不同评分模型。金融公司不套用一般
工业企业资产负债率门槛；房地产使用更高但仍有限制的杠杆门槛；上市不足
三年的公司会因跨周期样本有限受到风险扣分。行业样本不足时退回同模型组，
样本组仍不足时使用全市场分位，避免单只股票在只有自己的组内自动获得高分。

以下情况视为高暴雷风险并硬剔除：净资产为负、最新报告期亏损、异常营业
收入、一般企业资产负债率达到 85%、房地产达到 92%、股权质押达到 50%，
利润同比暴跌同时经营现金流为负，以及最新业绩预告预计亏损。风险控制分
低于 6 分的股票同样不入榜。其余股票仍必须满足质量、估值、历史成长、
未来空间与总分最低门槛，因此不会单纯因为低 PE 入选。

所有入选股票都显示“暴雷风险”和“风险提示”。即使标记为低风险，也会明确
注明低风险不等于无风险。当前免费数据源无法稳定覆盖财务审计意见、监管调查
和尚未披露事件，平台不得把缺失信息解释成安全结论。

数据与更新频率：

- 财务报表和质押统计：AkShare 对应的东方财富公开全市场接口，默认缓存 7 天；
- 当前季度覆盖不足全市场 75% 时，自动退回最近完整报告期；
- 前瞻证据：东方财富业绩预告与机构盈利预测，默认每天至多刷新一次；
- 行业景气与财报收入份额跟随财报缓存更新，日常不会重复下载历史财报；
- 估值：Tushare `daily_basic`，每个交易日刷新；
- Tushare 暂时限流或不可用时，依次降级到东方财富或新浪全市场快照；
  若降级源仅有 PE、PB，估值维度会在可用指标内部重新归一到 25 分，并在
  诊断文件和运行进度中明确记录口径，缺失的 PS、股息率不会被当成零分；
- 财报缓存：`data_files/fundamentals/fundamentals_latest.csv`；
- 估值缓存：`data_files/fundamentals/valuation_latest.csv`；
- 前瞻缓存：`data_files/fundamentals/forward_outlook_latest.csv`；
- 诊断：`output_files/fundamental_diagnostics_latest.json`。

需要立即检查新财报时可执行：

```bash
.venv/bin/python main.py --full --refresh-fundamentals
```

缓存天数可通过 `SG_QUANT_FUNDAMENTAL_CACHE_DAYS=7` 和
`SG_QUANT_FORWARD_CACHE_DAYS=1` 调整。外部财务接口
失败时保留原四份技术榜单，并在进度区显示基本面榜降级警告；不会生成一份
看似正常但实际缺数据的基本面榜。

## 五个评分因子

1. 量价关系
2. MA + ADX
3. RSRS
4. Donchian 通道位置
5. 12-1 月动量

用于排序的原始综合值严格由这五项的动态权重加总得到，不再叠加日周月共振、共线性折扣或固定奖励系数；榜单中的“综合得分”再单调映射为全市场横截面百分位，方便阅读且不改变顺序。日内根据五因子的横截面分散度调权；周度根据去重后的每日 Rank IC 平滑调权，各市场状态独立计算调权间隔，但不自动新增因子或翻转策略方向。

“所属概念”只在最终榜单生成后补充，用于阅读，不参与评分、排序或过滤。概念接口不可用时该列留空，榜单仍正常输出。

### 四个趋势候选因子

以下因子已接入单股扫描、SQLite 反馈记录和独立研究脚本，但尚未加入生产
综合分：

1. `momentum_12_2`：约 12 个月前至约 1 个月前的累计收益；
2. `high_52w`：收盘价相对过去 252 个交易日最高收盘价的位置；
3. `multi_horizon`：20/60/120/250 日收益按实现波动率缩放后的组合；
4. `industry_residual_momentum`：12-2 动量减去同行业当日截面中位数。

全市场在线股票列表会同时缓存 Tushare `industry` 字段到
`output_files/stock_industry_cache.json`。行业覆盖率低于 60% 时，行业残差
因子统一置零并在研究报告中标记为 `disabled`，不能退化成全市场去均值后
仍宣称“行业中性”。行业字段只采用当前分类，研究报告必须同时披露它不具备
历史时点行业变更记录这一限制。

独立研究命令：

```bash
.venv/bin/python -m backtest.candidate_factor_research
```

研究采用月度调仓、次日开盘成交、约 20 个交易日持有、Top 20，并按完整
换手约 36bp 扣除佣金、印花税和滑点。2022–2024 为训练集，2025 为验证集，
2026 至今只作最终测试。权重先在训练集筛选，再在验证集做稳定性选择，最终
测试集不参与选权。只有三个阶段的绝对收益、超额 Sharpe 和最大回撤同时通过
门槛，才允许把候选权重写入 `SCORE_WEIGHTS`。

2026-07-23 的本地全市场研究覆盖 5379 只股票、54 个非重叠月度持有期。
由于行业缓存尚无有效覆盖，行业残差因子未参与本轮选权；其余三因子组合未
通过生产门槛：训练期年化 -30.6%、最大回撤 -70.2%，验证期虽然绝对年化
+11.8%，但超额年化 -26.9%。因此候选因子只进入观察和反馈记录，没有为了
增加榜单变化而强行上线。完整结果位于
`backtest_results/candidate_factor_research.json` 和
`backtest_results/candidate_factor_research.csv`。

### 稳健趋势重构审计

`strategy.momentum.RobustTrendFactors` 和
`backtest.robust_trend_research` 实现第二轮趋势重构研究，覆盖以下六个互补
候选：120–20 日中期动量、对数价格回归斜率 × R²、多周期一致性、价格路径
效率、下行波动控制和距 120 日高点回撤控制。所有因子都固定为“数值越大越
好”，研究脚本不允许为了回测好看而自动翻转成均值回归。

```bash
.venv/bin/python -m backtest.robust_trend_research --holding-days 20
.venv/bin/python -m backtest.robust_trend_research --holding-days 10
.venv/bin/python -m backtest.robust_trend_research --holding-days 5
```

执行口径为信号日收盘后选股、次日开盘成交、Top 20，扣除完整换手约 36bp
成本；次日开盘封涨停的股票视为不可买入。股票还必须位于 120 日均线上方且
60 日收益为正，避免把低波动但没有上涨趋势的股票混入趋势榜。训练期为
2022–2024，2025 为验证期，2026 至今只作最终测试。

2026-07-23 的全市场审计覆盖 5178 只非当前 ST 股票。在 5、10、20 日三个
持有周期上，中期动量、趋势质量、多周期一致性和路径效率的训练期与 2025
验证期 Rank IC 均为负；下行波动控制的 Rank IC 为正，但 2025 单因子超额
收益和超额 Sharpe 均为负。没有候选同时通过训练与验证门槛，因此本次重构
被拒绝，未修改生产趋势权重。默认报告和分周期报告位于：

- `backtest_results/robust_trend_research.json`
- `backtest_results/robust_trend_research_5d.json`
- `backtest_results/robust_trend_research_10d.json`
- `backtest_results/robust_trend_research_20d.json`
- `backtest_results/robust_trend_factor_audit*.csv`

仪表盘会明确显示当前趋势榜仍运行旧五因子模型，以及稳健趋势替换方案未通过
验证。不能把“未替换”误读为旧模型已经有效；旧模型此前同样未通过严格回测。
在取得可靠的历史时点行业分类、财务公告日数据和退市股票数据前，不应宣称
当前结果不存在幸存者偏差或可以代表实盘收益。

### 因子尺度与回测约束

`screening.factor_eval.cross_sectional_standardize_factors()` 提供生产及候选因子的横截面百分位标准化，将每个因子统一映射到 `[-1, 1]`，用于研究权重真实贡献。该版本目前没有启用到生产榜单：2024-01 至 2026-05、每周调仓、次日开盘买入并持有 5 日的对比回测中，标准化趋势组合虽优于原始趋势组合，但年化收益仍为负，不能仅因榜单换手增加就上线。回测结果保存在 `backtest_results/weight_comparison.csv`；任何因子尺度、方向或周期变更都应先重新运行回测，并同时检查收益、回撤、Sharpe 和平均榜单保留率。

## 本地运行

```bash
cp .env.example .env
# 在 .env 中填写 TUSHARE_TOKEN
./run_online.sh
./run_local_mac.sh
```

也可以直接运行：

```bash
.venv/bin/python main.py --full
.venv/bin/python main.py --offline
.venv/bin/python self_test.py
.venv/bin/python -m unittest discover -s tests -v
```

## 财经新闻、DeepSeek 总结与数据问答

Dashboard 新增三个页面：

- **全局牛熊状态栏**：固定在所有功能页顶部。主状态严格只根据上证指数的 MA20/MA60 位置和 MA20 方向判定；小字分别显示上证指数、深证成指、创业板指和科创50的状态、点位与当日涨跌。配色按平台约定为绿色偏熊、黄色震荡、红色偏牛。新浪历史日线尚未落库当天数据时，系统用一次新浪批量指数收盘快照补齐四指数最后一根 K 线；15:35 后四指数日期必须全部等于要求日期，否则全市场流程报错，不保存混合日期状态。
- **财经新闻**：抓取最近 24 小时的东方财富、新浪财经和财联社快讯，最多 50 条。直接展示来源原标题，不调用 AI 二次概括；单个新闻源失败不影响其他来源或选股榜单。
- **每日总结**：全市场流程成功生成五榜单后，自动将当日行情快照、四指数状态、基本面价值榜、四份技术榜单和新闻交给 DeepSeek，保存当日总结与历史记录。涨跌家数使用未经过 ST、流动性或入榜资格过滤的全市场单日行情；选股有效数继续单独记录，不能用候选股口径冒充全市场宽度。页面同时显示全量流程和总结生成进度，并将存档的 Markdown 安全渲染成标题、段落与列表，不直接显示格式符号。
- **数据问答**：DeepSeek 综合平台已缓存的行情截面、五榜单、每日总结、新闻和问题中匹配到的个股日线回答。涉及“今天、最新、消息、政策、公告”等时效性问题时默认智能联网；配置 Key 后优先使用 Tavily，失败或结果不足才补 Google News 与 Bing News RSS。页面也可选择“始终联网”或“仅使用本地数据”。回答保持 SSE 流式输出，并在接收过程中实时渲染标题、列表、引用、粗体及代码片段；联网结果附带来源链接，搜索失败时自动降级到本地数据，不会伪装拥有实时价格。

在 `.env` 中至少设置：

```dotenv
DEEPSEEK_API_KEY=你的新密钥
DEEPSEEK_MODEL=deepseek-v4-flash
SG_QUANT_DASHBOARD_PASSWORD=强密码
SG_QUANT_SESSION_SECRET=长随机字符串
TAVILY_API_KEY=你的Tavily密钥
SG_QUANT_WEB_SEARCH_PROVIDER=auto
SG_QUANT_WEB_SEARCH_TIMEOUT=8
SG_QUANT_WEB_SEARCH_CACHE_SECONDS=600
```

API Key 只从环境变量或 `.env` 读取；`.env` 已加入 `.gitignore`。登录、CSRF 校验、对话频率限制和 HTTP-only 会话 Cookie 默认启用。公网部署必须使用 HTTPS，并设置 `SG_QUANT_COOKIE_SECURE=1`。

`TAVILY_API_KEY` 可选但建议配置；未配置、额度耗尽、请求失败或结果不足时，系统自动使用免密钥 RSS。公开 RSS 不是正式 SLA API，可能随服务方调整而变化。检索层已与 DeepSeek 问答逻辑分离，后续替换搜索服务不需要改动页面协议。

## 联网问答维护指南

### 请求链路与代码入口

一次问答按以下顺序执行：

1. `templates/chat.html` 将问题、最近 10 条历史消息和 `web_search` 模式提交到 `POST /api/chat`。
2. `dashboard.py::chat_api()` 完成登录、CSRF、频率限制和参数校验，并通过 SSE 返回检索状态、来源和 DeepSeek 文本片段。
3. `services/web_search.py::should_search_web()` 根据模式和问题时效性决定是否联网。
4. `services/web_search.py::search_latest_news()` 优先查询 Tavily；主源失败或少于 4 条时才并行补充 Google News 与 Bing News RSS，最后统一排序、去重并缓存。
5. `services/market_ai.py::build_chat_messages()` 将联网结果与本地行情、五榜单、每日总结、平台新闻和相关个股日线合并成 DeepSeek 上下文。
6. `services/market_ai.py::stream_market_chat()` 调用 DeepSeek 流式接口；前端在回答后展示本次实际使用的联网来源。

```text
浏览器问题
  -> /api/chat
  -> 是否需要联网
       -> 否：本地行情上下文
       -> 是：Tavily 主源
               -> 充足：直接使用
               -> 失败/不足：Google News + Bing News 补充
             -> 去重/缓存 -> 联网资讯上下文
  -> DeepSeek 流式回答
  -> 回答正文 + 可点击来源
```

关键文件：

- `services/web_search.py`：触发规则、查询改写、Tavily/RSS 请求、主备切换、解析、去重和内存缓存。
- `services/market_ai.py`：本地数据与联网资讯的提示词拼装。
- `dashboard.py`：问答接口、SSE 事件和失败降级。
- `templates/chat.html`：三种联网模式、检索状态和来源链接。
- `tests/test_intelligence.py::WebSearchTests`：触发规则、RSS 解析和联网上下文测试。
- `tests/test_intelligence.py::DashboardWebChatTests`：问答 SSE 中的检索结果与流式回答测试。

### 三种联网模式

| 页面选项 | 请求值 | 行为 |
| --- | --- | --- |
| 智能触发 | `auto` | 默认模式。时效性问题联网，纯指标解释等稳定问题使用本地数据。 |
| 本次始终联网 | `on` | 无论问题内容如何，本次都执行资讯检索。 |
| 仅使用本地数据 | `off` | 完全跳过外部检索，只使用平台已有数据。 |

智能触发规则集中在 `services/web_search.py`：

- 包含“最新、近期、今天、新闻、消息、公告、政策、财报、舆情、利好、利空”等词时联网。
- 个股代码或公司问题同时包含“原因、影响、异动、大涨、大跌、停复牌”等事件词时联网。
- 修改规则时只调整 `_AUTO_SEARCH_PATTERN`、`_MARKET_EVENT_PATTERN` 和 `should_search_web()`，同时补充对应单元测试。
- 修改查询改写时分别检查 `_build_tavily_query()` 与 `_build_query()`：前者使用自然语言，后者为 Google/Bing RSS 保留布尔关键词。

### 检索源与数据边界

- `SG_QUANT_WEB_SEARCH_PROVIDER=auto` 且存在 `TAVILY_API_KEY` 时，先调用 Tavily Search。
- Tavily 使用独立的自然语言查询改写，不复用面向 RSS 的布尔查询；固定采用 `topic=finance`、`search_depth=basic`、`time_range=week`，并限定交易所及国内主流财经域名。不请求 Tavily 生成答案或返回全文；基础搜索每次只消耗一次基础搜索额度。
- Tavily 返回至少 4 条时不再调用 RSS；失败或少于 4 条时，并行补充 Google News 与 Bing News。
- 未配置 Tavily Key 或将 provider 设为 `rss` 时，直接使用两个 RSS 源。
- Google News RSS 查询最近 7 天；Bing News RSS 作为并行补充。两个 RSS 源相互独立，单源失败不会阻断另一个源或 DeepSeek 回答。
- 默认最多向页面和模型提供 8 条结果；标题规范化后去重，并优先保留发布时间较新的结果。
- 只使用搜索源提供的标题、来源、发布时间、摘要和链接，不抓取文章全文。模型不得根据标题补写正文中不存在的事实。
- RSS 响应上限为 2 MB；仅接受 `http://` 或 `https://` 结果链接。
- Google/Bing RSS 都不是带 SLA 的正式 API，接口格式可能变化。解析异常应先更新 `_parse_rss()` 及测试样例，不要在前端临时兼容。

### 环境变量

| 变量 | 默认值 | 约束与说明 |
| --- | ---: | --- |
| `TAVILY_API_KEY` | 空 | Tavily API 密钥；只允许写入 `.env` 或服务器密钥环境，不能进入源码和普通部署包。 |
| `SG_QUANT_WEB_SEARCH_PROVIDER` | `auto` | `auto` 优先 Tavily 并保留 RSS 降级；`rss` 完全跳过 Tavily。其他值会回退为 `auto`。 |
| `SG_QUANT_WEB_SEARCH_TIMEOUT` | `8` | 单个资讯源的 HTTP 超时秒数，代码限制在 2–20 秒。 |
| `SG_QUANT_WEB_SEARCH_CACHE_SECONDS` | `600` | 相同查询的进程内缓存时间，代码限制在 30–3600 秒。 |
| `SG_QUANT_CHAT_RATE_LIMIT` | `30` | 单个登录会话或客户端每小时允许的问答请求数，检索与 DeepSeek 共用该限制。 |
| `DEEPSEEK_TIMEOUT` | `120` | DeepSeek 请求超时；资讯检索超时不使用此变量。 |

缓存位于 Dashboard 进程内，服务重启后自动清空。当前服务器建议保持一个 Gunicorn worker；若未来增加 worker，应把检索缓存、任务状态和限流状态一并迁移到 Redis。

### SSE 协议

`POST /api/chat` 会按顺序返回以下事件字段：

- `status`：面向用户的阶段文字，例如“正在联网检索近期资讯…”。
- `searching`：是否仍在检索。
- `search`：包含 `mode`、`triggered`、`count`、`providers`、`searched_at`、`cached` 和 `results`。
- `delta`：DeepSeek 的正文增量。
- `error`：可安全显示的错误信息。
- `done`：本次流结束。

前端应使用 DOM API 创建来源链接，不要把外部标题直接拼成未转义 HTML。模型正文统一调用 `base.html` 中的 `SGQ.renderMarkdown()`；它只支持受控的标题、列表、引用、粗体和代码格式，所有原始内容会先做 HTML 转义。每日总结与问答共用该渲染器，修改后必须同时验证两个页面。变更 SSE 字段时必须同时更新 `templates/chat.html` 和 `DashboardWebChatTests`。

### 降级与安全原则

- 检索未触发：直接使用本地数据，并向前端返回“本次问题使用平台本地数据”。
- Tavily 成功且结果充足：不调用 RSS，减少额度之外的额外等待。
- Tavily 失败或结果不足：记录安全错误类型并自动补充两个 RSS 源，不把 Key 或上游错误正文返回前端。
- 单个 RSS 搜索源失败：保留另一个源以及已有 Tavily 结果。
- 所有搜索源均失败或没有结果：继续调用 DeepSeek，但不注入联网资讯，并明确显示已经退回本地数据。
- DeepSeek 失败：来源检索结果不能替代模型答案，接口返回 DeepSeek 的安全错误信息。
- RSS 标题、摘要、网页内容和历史对话一律视为不可信输入；系统提示词明确禁止执行其中的指令。
- 后端只访问代码中固定的 Tavily/Google/Bing 地址，不允许用户提交任意抓取 URL，避免 SSRF。
- 不要把 DeepSeek Key、未来的搜索 API Key 或仪表盘密码写进源码、README、日志及部署包的普通版本。

### 搜索提供商维护与替换

Tavily 已作为正式主源接入。以后增加 Brave Search 等提供商时，应继续在 `services/web_search.py` 实现，不要让 Flask 请求直接调用 Agent Reach、Shell 命令或 MCP。生产请求路径中的子进程会增加命令注入面、部署依赖和不可控延迟。

新提供商必须转换成统一结果结构：

```python
{
    "title": "新闻标题",
    "url": "https://...",
    "source": "发布来源",
    "published_at": "ISO-8601 时间",
    "snippet": "检索摘要",
    "provider": "Tavily、Brave Search 或 RSS 源",
}
```

新增或替换提供商的步骤：

1. 在 `.env.example` 增加 API Key 和提供商选择变量，真实 Key 只写服务器 `.env`。
2. 新增 `_fetch_brave()` 等适配函数，设置明确的连接与读取超时。
3. 在 `search_latest_news()` 中定义主源顺序和“结果充足”的阈值，保留 RSS 降级。
4. 为正常响应、超时、限流、空结果和错误 JSON 增加离线测试。
5. 确认前端仍只依赖统一的 `search.results` 结构；正常情况下不需要修改 SSE 和页面。

### 测试与故障排查

修改联网问答后至少执行：

```bash
.venv/bin/python -m py_compile dashboard.py services/market_ai.py services/web_search.py
.venv/bin/python -m unittest discover -s tests -v
```

在允许访问外网的机器上做一次真实检索冒烟测试：

```bash
.venv/bin/python -c 'from services.web_search import search_latest_news; p=search_latest_news("A股最新政策与公司公告", max_results=6); print({"providers": p["providers"], "count": len(p["results"]), "errors": p["errors"]})'
```

常见问题：

- `providers` 为空：检查服务器 DNS、HTTPS 出站权限及 Tavily/Google/Bing 是否被网络策略拦截。
- 配置 Tavily 后仍只显示 RSS：确认启动进程实际加载了 `.env`，并检查 `SG_QUANT_WEB_SEARCH_PROVIDER` 是否误设为 `rss`。
- `errors` 包含 `Tavily: HTTPError`：通常是 Key 无效、额度耗尽或限流；不要把上游响应正文直接输出到日志或页面。
- 只有一个 provider：另一个源失败时属于正常降级；查看返回的 `errors` 判断超时还是 XML 格式变化。
- 智能模式没有联网：先确认问题是否命中触发词；需要强制验证时在页面选择“本次始终联网”。
- 联网结果很多但回答没有 `[联网N]`：检查 `build_chat_messages()` 是否注入 `live_web_news`，以及系统提示词是否仍要求编号引用。
- 页面没有来源链接：检查 SSE 的 `search.results`，再检查浏览器控制台；不要只检查 DeepSeek 正文。
- 搜索延迟过高：降低单源超时或缩短上游连接链路，不要提高 Flask/Gunicorn worker 数来掩盖阻塞问题。

维护完成后还应验证移动端联网模式控件、流式光标、来源链接，以及“仅本地数据”模式确实不发起外部请求。

`--full` 在线获取全部上市 A 股并增量更新日线；`--offline` 只读取本地日线、最近一次 online 保存的市场状态和概念缓存，不发起网络请求。`--clean` 会清空日线缓存后重新下载，通常不需要使用。程序从 15:35 起要求取得当天行情，可用 `SG_QUANT_DATA_READY_TIME=HH:MM` 调整。全量流程先用一次 Tushare 全市场单日请求，只给缺少当天数据且日期连续的本地缓存追加一根日线；已经更新的股票和确认停牌的股票不再逐只请求。极少数存在多日断档的股票再按代码请求从最后缓存日到当天之间的缺失区间，不重抓完整历史；没有历史缓存的新股才走首次全量下载。追加时通过每根日线的当日昨收校正历史前复权基准，避免除权日价格断层。扫描完成后会根据当日实际成交股票再次校验；日期缺失或仍为旧日期的股票会从当日榜单候选中剔除，但无论数量多少都不终止流程。异常少于 10 只时，日志和前端逐只列出代码、名称与实际日期；达到 10 只时只显示总数。流程最终状态为“已完成（有警告）”，不会把异常股票的旧数据混入新榜单。

复权日线使用 AkShare 新浪接口，东财接口作为备用。旧 Tushare Token 的 `daily` 虽可读取原始日线，但 `adj_factor` 实测只有 1 次/小时，因此程序不会把缺少复权因子的原始价冒充前复权价。Tushare 继续用于股票列表等有权限的接口；1.4.x 客户端默认旧域名不可用时自动切换到 `api.tushare.pro`。
日线默认 6 并发，可通过 `SG_QUANT_SCAN_WORKERS` 调整；并发过高会提高新浪接口断连和限流概率。第三方库未声明超时时，程序默认补上 15 秒 HTTP 上限（`SG_QUANT_HTTP_TIMEOUT`），防止 DNS 波动后工作线程永久卡住。

## 服务器运行

服务器使用 Python 3.12：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
SG_QUANT_HOST=0.0.0.0 SG_QUANT_OPEN_BROWSER=0 ./run_server.sh
```

定时选股任务执行 `./run_online.sh`，建议安排在交易日 17:00 以后。仪表盘默认端口是 `5000`，公网部署时应由 Nginx/Caddy 反向代理并配置访问控制，不应直接暴露 Flask/Gunicorn 端口。
仪表盘默认使用 1 个 Gunicorn worker，因为“执行全市场”任务状态保存在进程内；如改为多 worker，应先把任务状态迁移到 Redis/数据库。

### macOS 本地网页运行

直接执行 `./run_server.sh` 属于前台运行，终端或任务会话结束后网页也会停止。
本地使用时应在独立 Terminal 窗口中运行并保持该窗口开启：

```bash
cd "/Users/midongkeji/Documents/自媒体/SG_Quant_Mac本地调试版/quant"
./run_server.sh
```

不要直接从位于“文稿”目录的项目注册 LaunchAgent。macOS 隐私机制默认禁止
后台 LaunchAgent 访问该目录，会出现 `Operation not permitted` 并反复启动
失败。如需登录后自动启动，应先把项目迁到不受隐私保护的服务目录，再配置
LaunchAgent；或者在系统设置中明确授予相应后台程序文件访问权限。

## 主要目录

- `main.py`：生成基本面价值榜与四份技术榜单
- `screening/`：基本面评分、风险门禁、生产五因子、趋势候选因子和排名
- `data/`：财务报表、估值、日线、股票列表、市场状态与概念标签缓存
- `feedback/`：榜单记录、前向收益和固定五因子调权
- `services/`：新闻、联网检索、DeepSeek 客户端、每日总结和进度状态
- `dashboard.py`：榜单、新闻、总结、联网问答、登录与流式接口
- `backtest/`：单策略、生产五因子组合回测、候选趋势因子和稳健趋势统一审计

真实 Token 只放在 `.env` 或服务器环境变量中，不写入源码。
