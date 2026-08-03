# 产品架构与用户流程

## 一、产品架构图

```mermaid
flowchart TB
    subgraph Experience[体验层]
        DASH[榜单与进度]
        NEWSUI[财经新闻]
        SUMMARY[每日总结]
        CHAT[数据问答]
        AUTH[登录与会话]
    end

    subgraph Orchestration[编排与产品规则层]
        PIPE[全市场任务编排]
        PROGRESS[阶段 / 进度 / 警告]
        CONTEXT[AI 上下文构建]
        POLICY[联网触发与降级策略]
    end

    subgraph Intelligence[智能层]
        FUND[基本面评分 / 风险门禁]
        FACTOR[技术因子计算与排序]
        REGIME[牛熊状态与市场宽度]
        DEEPSEEK[DeepSeek 总结 / 问答]
        SEARCH[Tavily + RSS]
    end

    subgraph Data[数据与证据层]
        QUOTE[个股与指数行情]
        FIN[财报 / 估值 / 质押]
        FORWARD[业绩预告 / 机构预测]
        NEWS[新闻标题 / 来源 / 时间]
        CACHE[(本地缓存)]
        REPORT[(榜单 / 日报 / 实验报告)]
    end

    DASH --> PIPE
    PIPE --> PROGRESS
    PIPE --> QUOTE
    PIPE --> NEWS
    QUOTE --> CACHE
    NEWS --> CACHE
    FIN --> CACHE
    FORWARD --> CACHE
    CACHE --> FUND
    CACHE --> FACTOR
    CACHE --> REGIME
    FUND --> REPORT
    FACTOR --> REPORT
    REGIME --> REPORT
    REPORT --> DASH
    REPORT --> CONTEXT
    CACHE --> CONTEXT
    CHAT --> POLICY
    POLICY --> SEARCH
    SEARCH --> CONTEXT
    CONTEXT --> DEEPSEEK
    DEEPSEEK --> CHAT
    DEEPSEEK --> SUMMARY
    AUTH --> DASH
    AUTH --> CHAT
```

## 二、每日全市场用户流程

```mermaid
flowchart LR
    A[用户点击执行全市场] --> B[确认交易日与 15:35 就绪时间]
    B --> C[只补齐缺失行情]
    C --> D{交易股票日期一致?}
    D -- 是 --> E[计算市场宽度与四指数状态]
    D -- 否 --> W[记录警告并排除异常股票]
    W --> E
    E --> F[加载财报 / 估值 / 前瞻证据]
    F --> R{高风险或关键数据缺失?}
    R -- 是 --> X[剔除并记录理由]
    R -- 否 --> V[生成基本面价值 Top 30]
    X --> T[生成四份技术榜单]
    V --> T
    T --> G[抓取最近 24 小时新闻]
    G --> H{DeepSeek 可用?}
    H -- 是 --> I[生成并存档每日总结]
    H -- 否 --> J[保留榜单并显示总结失败]
    I --> K[完成或带警告完成]
    J --> K
```

## 三、问答用户流程

```mermaid
flowchart LR
    Q[用户提问] --> MODE{联网模式}
    MODE -- 仅本地 --> LOCAL[装配本地数据]
    MODE -- 始终联网 --> WEB[执行资讯检索]
    MODE -- 智能触发 --> INTENT{问题是否时效敏感?}
    INTENT -- 否 --> LOCAL
    INTENT -- 是 --> WEB
    WEB --> T{Tavily 结果充足?}
    T -- 是 --> MERGE[合并来源与本地证据]
    T -- 否 --> RSS[Google / Bing RSS 补充]
    RSS --> MERGE
    LOCAL --> LLM[DeepSeek]
    MERGE --> LLM
    LLM --> STREAM[SSE 流式正文]
    STREAM --> SOURCE[展示本次依据与降级状态]
```

## 四、关键产品边界

| 边界 | 产品约束 |
| --- | --- |
| 模型与事实 | 行情、榜单、新闻由数据层提供；模型只解释，不生成缺失事实 |
| 基本面与技术 | 基本面榜不以日 K 信号决定入选；四份技术榜保留为独立视角 |
| 发展空间 | 使用可追溯的财务与前瞻证据；关键模块缺失时不入榜 |
| 核心与增强 | 五份榜单是核心产物；新闻、搜索、AI 总结是可降级增强能力 |
| 数据异常 | 不终止整批任务，但异常股票不参与当日排名 |
| 实时性 | 明确数据日期和搜索时间，不将缓存数据描述为实时 |
| 策略实验 | 通过生产门禁才允许替换权重；失败实验保留证据 |
| 交易风险 | 不下单、不承诺收益、不将研究榜单包装为投资建议 |
