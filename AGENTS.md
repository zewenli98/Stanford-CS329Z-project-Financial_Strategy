# Agent guide — Financial Strategy

本文件给 **coding agent / 协作者** 用：说明仓库在做什么、主调用链、各路径职责，以及「改某类功能应先读哪些文件」。人类用户的安装与 quick start 仍以 [README.md](README.md) 为准。

## 项目目标（一句话）

在严格的 **point-in-time（时点）** 约束下，对美股普通股做研究流水线：确定性特征 + 粗筛 + LangGraph 多角色研究 + 校验，输出按 **预期当日开盘→收盘绝对波动** 排序的 watchlist。**不下单**；启发式分数 **不是概率**。

## 三条命令

| 命令 | 入口 | 作用 |
|------|------|------|
| `screen` | `src/financial_strategy/cli.py` → `service.run_screening` | 新建一次不可变 screening run（写入 `runs/`） |
| `replay` | `cli.py` → `audit/replay.replay_run` | 校验并确定性重建已有成功 run |
| `evaluate` | `cli.py` → `evaluation/runner.run_evaluation` | 历史任务：冻结预测再对标签算指标 |

```bash
uv sync --extra dev
uv run pytest
uv run financial-strategy screen --config configs/screening_2026-10-14.yaml
```

- **无 API key 开发**：测试里把 `agents.mode` 设为 `scripted`（见 `tests/test_end_to_end.py`）。
- **Live LLM**：`.env` 中 `OPENROUTER_API_KEY`；配置见 YAML 的 `agents.*`（示例模型 `openai/gpt-5.6-luna`）。

## 主调用链（先读这条线）

```
cli.py
  └─ service.py :: run_screening
        ├─ config.load_mvp_config
        ├─ task/validator.validate_and_freeze_task
        ├─ data/gateway.PointInTimeGateway  (+ providers, calendar)
        ├─ features/engine.compute_features
        ├─ ranking/screen.coarse_screen
        ├─ ranking/ranker.market_components
        ├─ audit/writer.RunWriter  (落盘 snapshot / features / screening)
        ├─ llm.make_client + agents.AgentSuite
        └─ workflow/graph.build_screening_graph → invoke
              ├─ macro 节点 (agents.roles)
              ├─ Send → workflow/symbol_graph (每标的)
              └─ rank 节点 (ranking/ranker.rank_cards)
```

评估路径：`evaluation/runner.py` 会多次调用 `run_screening`（可带 task 覆盖），再读标签与 `evaluation/metrics.py`。

## 顶层目录

| 路径 | 用途 |
|------|------|
| `configs/` | 可运行 YAML：`screening_*.yaml`、`evaluation_*.yaml`、`sector_exposures.json` |
| `data/2026-10-01_2026-10-14/` | **冻结的真实数据快照**（universe、行情 parquet、证据 jsonl、raw SEC/GDELT、`provenance.json`） |
| `scripts/fetch_real_data.py` | 拉取/更新快照数据（非每次 screen 必跑） |
| `src/financial_strategy/` | 全部应用代码 |
| `tests/` | pytest；`test_end_to_end.py` 是整条 pipeline 的「活文档」 |
| `runs/` | `screen` 输出（运行时生成，通常不入库） |
| `cache/` | LLM 缓存路径（配置项，本地） |

**不要**在未理解时点语义的情况下改 `data/gateway.py` 的过滤逻辑，或把 cutoff 之后的证据喂给模型。

## `src/financial_strategy/` 模块地图

| 模块 / 文件 | 职责 |
|-------------|------|
| `cli.py` | Typer CLI：`screen` / `replay` / `evaluate` |
| `service.py` | **主编排**：确定性准备 → 写 snapshot → LangGraph → audit 发布 |
| `config.py` | YAML → `MVPConfig`；`content_hash` 等 |
| `errors.py` | `StrategyError` 及错误码（配置、任务、审计等） |
| `schemas/models.py` | **领域契约**（Pydantic，`extra=forbid`）：Task、Bar、Evidence、ResearchCard、Verdict… |
| `task/validator.py` | 校验并冻结 `ScreeningTask`、敞口、警告 |
| `data/calendar.py` | 交易所日历（如 XNYS） |
| `data/providers.py` | 本地 universe / 行情 / 公司&宏观证据 |
| `data/gateway.py` | **时点网关**：只暴露 cutoff 前可用的 bars 与 evidence |
| `features/engine.py` | 从 bars 算 per-symbol 特征（完整 session 要求见 config） |
| `ranking/screen.py` | 粗筛候选（volatility + anomaly 权重） |
| `ranking/ranker.py` | 市场分量、与 research card 合并、`rank_cards` 产出 watchlist |
| `retrieval/bm25.py` | 确定性 BM25；agent **不**直接读原始文件树 |
| `agents/roles.py` | macro / company / synthesis / verifier；`scripted_responder` 供 CI |
| `llm/client.py` | OpenRouter 结构化输出、限流与尝试记录 |
| `llm/cache.py` | 响应缓存 |
| `verification/deterministic.py` | ResearchCard 硬规则校验 |
| `workflow/state.py` | LangGraph state 类型 |
| `workflow/graph.py` | 顶层图：macro → 并行 symbol → rank |
| `workflow/symbol_graph.py` | 单标的：company → synthesis → verify → 修订循环 |
| `audit/writer.py` | Write-once run 目录与 hash |
| `audit/replay.py` | 从 run 目录复现确定性部分 |
| `evaluation/runner.py` | 多日评估编排 |
| `evaluation/metrics.py` | 指标定义 |
| `evaluation/baselines.py` | 基线（若有扩展评估时参考） |

## 配置文件读哪里

- **单日 screening 示例**：`configs/screening_2026-10-14.yaml` — `task`（cutoff、target_session、top_k）、`data` 路径、`features` / `screening` / `ranking` / `retrieval` / `agents` / `revision` / `output.runs_path`。
- **多日评估**：`configs/evaluation_2026-10-01_2026-10-14.yaml`。
- **行业敞口启发式**：`configs/sector_exposures.json`（macro 暴露相关）。

改权重或阈值时：同步检查 `ranking/ranker.py` 是否假设了相同字段名，并跑相关 tests。

## 一次 `screen` run 里有什么（`runs/<run_id>/`）

调试时通常需要对照磁盘产物：

| 产物 | 含义 |
|------|------|
| `config.yaml` / `task.json` | 冻结的配置与任务 |
| `snapshot/*` | 时点快照：universe、parquet bars、evidence jsonl、metadata/manifest |
| `features.parquet` | 确定性特征 |
| `screening.json` | 候选、粗分、被拒 symbol |
| `macro_context.json` | 宏观 agent 输出 |
| `initial_research_cards.json` / `research_cards.json` | 研究卡片（含校验结果） |
| `watchlist.json` | 最终排序列表 |
| `model_attempts.jsonl` / `llm_responses/*.json` | LLM 调用审计 |
| `retrieval_traces.jsonl` | BM25 检索轨迹 |

失败 run 也会发布到 `runs/`（`service.py` 在异常时仍尽量保留已物化输入）。

## 任务导向：我要改 X → 先读这些

| 目标 | 优先阅读 | 建议测试 |
|------|----------|----------|
| 特征定义或缺失 session 行为 | `features/engine.py`, `config` 的 `features.*` | `tests/test_gateway_and_features.py` |
| 粗筛 / 候选数量 | `ranking/screen.py`, `screening` 配置块 | `tests/test_retrieval_ranking_metrics.py` |
| 最终分数与 watchlist | `ranking/ranker.py`, `schemas/models.py`（Prediction/Card） | 同上 + `test_end_to_end.py` |
| 时点 / 证据可见性 | `data/gateway.py`, `data/providers.py` | `tests/test_gateway_and_features.py`, `test_task_and_calendar.py` |
| 任务合法性（cutoff、lookback） | `task/validator.py` | `test_task_and_calendar.py` |
| Agent 提示词 / 角色输出 schema | `agents/roles.py`, `llm/client.py` | `test_llm_controls.py`, `test_end_to_end.py`（scripted） |
| LangGraph 拓扑或并行 | `workflow/graph.py`, `workflow/symbol_graph.py` | `test_end_to_end.py` |
| 确定性校验规则 | `verification/deterministic.py` | 相关 unit tests + e2e |
| 评估与指标 | `evaluation/runner.py`, `evaluation/metrics.py` | `test_end_to_end.py` 及 evaluation 配置 |
| 配置 schema 变更 | `config.py` + 所有 YAML 示例 | `tests/test_config.py` |
| 审计 / 复现 | `audit/writer.py`, `audit/replay.py` | `tests/test_audit_and_cache.py`, `test_end_to_end.py` |

## 编码约定（给 agent）

1. **契约单一来源**：跨模块数据结构以 `schemas/models.py` 为准；改字段要同步 agent 输出模型（`agents/roles.py` 内 `*Output`）与校验器。
2. **检索保持确定性**：证据选择走 `retrieval/bm25.py` + gateway 已过滤的 records，避免让 LLM 直接「读 data 目录」的设计。
3. **配置严格**：`config.py` 与 schema 使用 `extra=forbid`；新配置项要进 Pydantic 模型和示例 YAML。
4. **最小 diff**：研究代码；不要为 agent 方便而改未请求的业务逻辑。
5. **验证**：改代码后默认 `uv run pytest`；动 ranking/features/gateway 时跑对应专项 test，不只跑一个文件。

## 与 README 的分工

- **README.md**：安装、快照来源、环境变量、免责声明。
- **AGENTS.md（本文件）**：架构、文件地图、改代码路径、run 产物说明。

大重构（拆分 `service.py`、改 graph 节点名、移动数据路径）后请 **更新本文件对应章节**，避免地图过期。
