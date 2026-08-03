# SG 代码文件功能说明（中文备注）

> 本文件逐一说明仓库中每个文件/目录的职责，方便日后维护和修改。
> 最后更新：2026-08-03（与云端部署版本一致）

## 一、根目录：入口与配置

| 文件 | 功能 |
|---|---|
| `main.py` | **每日投研主流程入口**。负责全市场任务编排：增量行情更新 → 数据日期一致性质检 → 因子计算与横截面排序 → 输出四份候选榜单 → 触发每日总结。支持 `--full`（在线全市场）、`--offline`（只用本地缓存）、`--clean`（清空缓存重跑）三种模式 |
| `dashboard.py` | **网页工作台服务**（Flask）。负责页面路由、密码登录、会话限流、SSE 流式问答、任务进度接口 |
| `config.py` | **全局配置中心**。定义数据目录路径（`data_files/`）、因子权重、策略参数、市场状态阈值等，所有模块从这里读配置 |
| `backtest_main.py` | **回测入口**。单独运行历史回测，验证策略/因子表现（研究用，不参与每日流程） |
| `self_test.py` | **环境自检脚本**。部署或改动后运行，验证核心模块、五因子、两策略、ST 与流动性过滤是否正常 |
| `requirements.txt` | Python 依赖清单（akshare / tushare / pandas / flask / gunicorn 等） |
| `run_online.sh` | **每日定时任务调用的脚本**：加载 `.env` 后执行 `main.py --full`（服务器 cron 每天 15:42 跑它） |
| `run_server.sh` | 启动 Dashboard 网页服务（gunicorn 常驻，服务器上由 systemd 托管） |
| `run_local_mac.sh` | Mac 本地启动脚本（本地调试用，服务器上不用） |
| `.env` | **真实密钥配置（高危！）**：Tushare / DeepSeek / Tavily 密钥、Dashboard 密码、会话密钥。仓库已转私有才存放，**绝不可转公开** |
| `.env.example` | `.env` 的空模板，说明每个配置项的含义 |
| `README.md` | 项目总说明（面向作品集的公开版介绍） |
| `LICENSE` / `SECURITY.md` | 版权声明 / 安全策略说明 |

## 二、`data/`：数据层（行情与基本面）

| 文件 | 功能 |
|---|---|
| `fetcher.py` | **行情抓取核心**。Tushare 为主、akshare 兜底；内置限频令牌桶与熔断机制；负责日线增量更新和全市场单日批量补齐 |
| `storage.py` | 行情 parquet 文件的读写、每只股票最新日期查询 |
| `rate_guard.py` | 请求速率控制，防止打爆数据源接口额度 |
| `fundamentals.py` | **基本面与估值数据缓存**（财报、质押等低频数据，默认缓存 7 天） |
| `forward_outlook.py` | **前瞻证据缓存**：业绩预告与机构盈利预测（默认缓存 1 天） |
| `index_filter.py` | 指数成分与市场范围过滤（主板/全市场口径） |
| `board_utils.py` | 板块工具函数 |
| `concept_tags.py` | 为榜单补充概念标签（仅展示，不参与评分过滤） |
| `risk_manager.py` | 依据大盘评分给出组合总仓位参考（如熊市轻仓建议） |

## 三、`screening/`：选股层（资格、因子、排名）

| 文件 | 功能 |
|---|---|
| `eligibility.py` | **入榜资格规则**：统一管理 ST 剔除、流动性门槛等不参与评分的硬性过滤 |
| `fundamental.py` | **基本面价值榜**：质量、估值、历史成长、未来空间与风险门禁 |
| `factor_eval.py` | 生产五因子与候选因子的横截面评估；动态权重只作用于生产因子 |
| `ranking.py` | 因子标准化与横截面排名 |
| `scanner.py` | 全市场并行扫描引擎（多进程计算每只股票的指标与得分） |

## 四、`strategy/`：策略层

| 文件 | 功能 |
|---|---|
| `base.py` | 策略基类与统一接口 |
| `indicators.py` | 技术指标计算（均线、动量等） |
| `trend_following.py` | 趋势跟踪策略（生产策略之一） |
| `momentum.py` | 动量/均值回归策略（生产策略之一） |
| `rsrs.py` | RSRS 市场择时指标（牛熊判断输入之一） |

## 五、`services/`：服务层（AI 与资讯）

| 文件 | 功能 |
|---|---|
| `market_ai.py` | **每日市场快照 + DeepSeek 每日总结 + 问答上下文组装**（把本地行情/榜单/新闻喂给大模型） |
| `deepseek.py` | DeepSeek API 封装（流式输出、超时与降级处理） |
| `news.py` | 多源财经快讯聚合、去重、本地缓存（保留原标题与链接） |
| `web_search.py` | 问答联网检索：Tavily 主源，Google/Bing RSS 兜底，全部失败退回本地数据 |
| `progress.py` | 全市场流程的持久化进度状态（前端进度条的数据来源，`output_files/run_status.json`） |
| `storage.py` | 服务层轻量 JSON 持久化工具 |

## 六、`feedback/`：反馈与迭代

| 文件 | 功能 |
|---|---|
| `recorder.py` | 把每日四份榜单、生产因子、候选因子追加写入 SQLite（用于事后验证） |
| `returns_tracker.py` | 跟踪入榜股票的后续收益表现 |
| `weight_optimizer.py` | 生产五因子的周度 IC 调权（只调权重，不增删因子、不翻转策略方向） |

## 七、`backtest/`：回测与因子研究

| 文件 | 功能 |
|---|---|
| `engine.py` | 回测引擎 |
| `report.py` | 回测结果报告生成 |
| `portfolio_backtest.py` | 组合级回测 |
| `candidate_factor_research.py` | 候选趋势因子的月度样本外研究（训练/验证/测试隔离） |
| `robust_trend_research.py` | 稳健趋势因子审计与滚动样本外研究 |

## 八、前端与测试

| 文件/目录 | 功能 |
|---|---|
| `templates/` | 网页模板：`dashboard.html`（主工作台）、`login.html`（登录）、`chat.html`（问答）、`news.html`（新闻）、`summary.html`（每日总结）、`base.html`（公共骨架） |
| `static/app.css` | 工作台设计系统样式 |
| `tests/` | 回归测试：核心口径（`test_core_regressions.py`）、流程冒烟（`test_pipeline_smoke.py`）、AI 上下文（`test_intelligence.py`）、基本面与前瞻（`test_fundamental.py`、`test_forward_outlook.py`） |
| `output/stock_pool.py` | 榜单股票池输出工具 |
| `docs/` | 作品集材料（架构、PRD、演示脚本、复盘等，与运行无关） |
| `evidence/` | 因子研究证据存档（含未通过门禁的失败实验，诚实性材料） |
| `sample_data/` | 示例数据说明 |

## 九、不在仓库中的内容（刻意排除）

| 内容 | 位置 | 原因 |
|---|---|---|
| `data_files/`（516MB 行情缓存） | 服务器 `/opt/sg-quant/` + Mac 本地 | 体积大、每日变化，不适合 git 管理 |
| `backtest_results/`（149MB 回测产出） | Mac 本地 | 研究产物，关键结论已存档于 `evidence/` |
| `output_files/`（每日榜单/总结产出） | 服务器 | 运行产物，每天生成 |
| `logs/` | 服务器 | 运行日志 |

## 十、部署拓扑速查

- **云服务器**：阿里云轻量 116.62.118.251，`/opt/sg-quant/`，cron 每天 15:42（周一至周五）跑 `run_online.sh`，systemd 托管 `sg-quant-dashboard`（端口 5000）
- **本仓库**：代码与密钥的权威备份；改代码后需手动同步到服务器（见 README 部署章节）
- **Mac 本地**：`Documents/自媒体/SG_Quant_Mac本地调试版/quant` 为日常编辑副本
