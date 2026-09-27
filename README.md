# 企业经营风险智能分析与决策平台

一套面向企业经营风险的**智能分析与决策支持系统**：用 ReAct Agent 编排「内部数据 + 知识库 + 联网公开资料」三方证据，
产出**来源分层、引用可核对**的分析结论，并支持风险规则预警、审批流与审计追溯。

> 本文档面向**部署与运维**。功能操作见 [`使用手册.md`](./使用手册.md)。

---

## 1. 系统构成

```
┌──────────────────────────────────────────────────────────────┐
│  浏览器                                                       │
│     ↓ HTTP                                                   │
│  Spring Boot 应用（8080）  ← 单 jar，内含前端页面              │
│     ├─ 静态页面  /            （前端产物，hash 路由资源管理）    │
│     ├─ 业务 API  /api/**      （需 JWT）                       │
│     └─ AI 转发   /api/ai/**   → Python Agent（8081）           │
│                                                               │
│  Python Agent（8081）                                         │
│     ├─ ReAct Agent / 工具编排 / 护栏 / 留痕                     │
│     ├─ RAG 混合检索（SQLite FTS5 + 本地零成本向量）             │
│     └─ 联网检索（火山方舟 Responses API + web_search）          │
│                                                               │
│  MySQL 3306（risk_platform 库）· Redis 6379 · RabbitMQ 5672    │
└──────────────────────────────────────────────────────────────┘
```

**为什么 AI 部分在 Python 侧？** 原 Java agent 栈已整体删除，`/api/ai/*` 由 Python 承接；
Spring 只负责按白名单转发，因此**前端零改动**。回退途径见 §8。

| 服务 | 端口 | 必需 | 说明 |
|---|---|---|---|
| Spring Boot（业务 + 页面） | **8080** | ✅ | 打成一个 fat jar，启动后**直接就是完整系统** |
| Python Agent（AI 引擎） | **8081** | ✅ | 不启动则 AI 分析不可用，其余功能正常 |
| MySQL | 3306 | ✅ | 库名 `risk_platform` |
| Redis | 6379 | ➖ | 未起时设 `APP_REDIS_ENABLED=false` |
| RabbitMQ | 5672 | ➖ | 未起时设 `APP_MQ_ENABLED=false` |

---

## 2. 快速开始

### 2.1 前置条件

| 组件 | 版本 | 备注 |
|---|---|---|
| JDK | **17** | 本机 `D:\Java17` |
| Maven | 3.6.3+ | 本机用的是隔离安装版 3.9.9 |
| Python | 3.10+ | 需要能建 venv |
| MySQL | 8.x | 库 `risk_platform` |

### 2.2 三步跑起来

```powershell
# ① 启动 Python Agent（AI 引擎）
cd pyagent
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
AI_MODEL=deepseek-v4-flash-ga-260731 APP_RAG_EMBEDDING_ENABLED=true APP_RAG_EMBEDDING_MODE=local `
  .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8081

# ② 启动后端（已打包好的 fat jar）
.\start-backend.ps1          # 或直接 java -jar backend\target\enterprise-risk-platform-1.0.0.jar --server.port=8080

# ③ 打开浏览器
start http://localhost:8080      # 账号 admin / Admin@123
```

> **开发改前端时**才需要 `cd frontend && npm run dev`（5173 端口，`/api` 代理到 8080）。
> 交付与部署**不需要跑 Node**——页面已经打进 jar 里了。

### 2.3 一键启动（Windows）

仓库根目录提供了三个脚本：

| 脚本 | 作用 |
|---|---|
| `start-backend.ps1` | 启动后端 jar（自动找最新 jar，会等待健康检查） |
| `start-python-agent.ps1` | 启动 Python Agent（自动挑解释器、装缺失依赖） |
| `start-all.bat` | 依次拉起上述两者 |
| `stop-backend.ps1` | **按命令行精确匹配**停掉本项目 jar（不会误杀 IDE 的其它 java 进程） |
| `stop-python-agent.ps1` | 只停止本项目 8081 端口的 Uvicorn Agent |
| `stop-all.bat` | 一键关闭 Spring Boot 与 Python Agent |

---

## 3. 打包：生成可独立运行的 fat jar

```powershell
cd backend
mvn clean package -DskipTests
# 产物：backend\target\enterprise-risk-platform-1.0.0.jar（约 116 MB）
```

### 3.1 RAG 入库链路升级

知识文档不再在上传请求中同步解析，也不再截断到 6 万字符。当前链路为：

```text
上传与安全校验 -> 原文件持久化 -> rag_ingest_job 后台解析
-> Office 结构化提取/可选 PDF OCR -> 质量评分 -> rag_outbox_event
-> Python 单文档切片与增量索引 -> Java 确认切片数并完成任务
```

首次升级已有数据库时执行一次：

```powershell
Get-Content -Raw -Encoding UTF8 mysql\init\07_rag_ingestion_pipeline.sql |
  mysql --default-character-set=utf8mb4 -uroot -p123456 risk_platform
```

新建数据库时，Docker/MySQL 初始化目录会按文件名顺序自动执行 `01` 到 `07`。原文件默认保存在
`./data/knowledge-files`，生产环境应通过 `APP_KNOWLEDGE_STORAGE_DIR` 指向持久化磁盘。

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `APP_KNOWLEDGE_STORAGE_DIR` | `./data/knowledge-files` | 原文件目录 |
| `APP_KNOWLEDGE_OCR_ENABLED` | `false` | 是否对低文本量 PDF 启用 OCR |
| `APP_KNOWLEDGE_OCR_COMMAND` | `tesseract` | Tesseract 命令路径 |
| `APP_KNOWLEDGE_OCR_MAX_PAGES` | `10` | 单文档 OCR 页数上限 |

支持 `txt/md/csv/pdf/doc/docx/xls/xlsx/ppt/pptx/rtf`。宏文件、可执行文件、路径穿越和异常膨胀的
Office 压缩容器会被拒绝；同一企业内按 SHA-256 去重。服务异常退出后，`RUNNING/SENDING`
状态会在重启时自动恢复，Python 暂时不可用时 Outbox 会指数退避重试。

**这个 jar 自带前端页面**，启动后访问 `http://localhost:8080` 即是完整系统。

### 3.2 前端是怎么进 jar 的

前端产物经 `frontend/dist` → 复制到 `backend/src/main/resources/static/`，由 Spring Boot
从 `classpath:/static` 提供。因此：

```powershell
cd frontend
npm run build                                   # 产出 dist/
cp -r dist/. ..\backend\src\main\resources\static\
cd ..\backend
mvn clean package -DskipTests                   # 重新打包
```

> 前端用的是 **`createWebHistory()`**（history 模式），所以有两处配套实现缺一不可：
> 1. `SecurityConfig` 放行 `/`、`/index.html`、`/assets/**` 及各子路由；
> 2. `SpaFallbackController` 把子路由 forward 回 `index.html`。
>
> 少了第 2 步，**刷新任意子页面就是 404**；两者依赖 `permitAll` 清单**保持一致**。

### 3.3 打包的两个硬性约束

| 坑 | 症状 | 解法 |
|---|---|---|
| **jar 正在运行时打包** | `Unable to rename ...jar to ...jar.original`（文件被占用） | 先 `stop-backend.ps1` 停服务再打包 |
| **不 clean 就 package** | 几百个「找不到符号」——**Lombok 增量编译失效**的典型表现（`@Data` 的 setter、`@Slf4j` 的 `log` 全部消失） | 用 `mvn clean package`，不要只 `package` |

> 第 2 条极具误导性：报错全是业务类名，很容易去翻代码找哪个字段没写。
> 判据很明确——**错误数量几百个、且全是"找不到符号"**，就是 Lombok，不是你的代码。

---

## 4. 配置

配置集中在 `backend/src/main/resources/application.yml`，全部支持「环境变量覆盖」。

### 4.1 数据与环境

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` | `127.0.0.1` / `3306` / `risk_platform` / `root` / — | MySQL 连接 |
| `REDIS_HOST` / `REDIS_PORT` | `127.0.0.1` / `6379` | Redis 缓存 |
| `RABBIT_HOST` / `RABBIT_PORT` / `RABBIT_USER` / `RABBIT_PASS` | `127.0.0.1` / `5672` / `guest` | 消息队列 |
| `APP_REDIS_ENABLED` | `true` | 没起 Redis 时设 `false` |
| `APP_MQ_ENABLED` | `true` | 没起 RabbitMQ 时设 `false` |

### 4.2 AI（火山方舟）

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `AI_BASE_URL` | `https://ark.cn-beijing.volces.com/api/v3` | 底座地址 |
| `AI_MODEL` | — | 主模型；**建议留空**由平台自选 + 自动降级 |
| `AI_MODEL_CANDIDATES` | — | 候选模型链（逗号分隔），主模型不可用时依次降级 |
| `AI_API_KEY` | — | **唯一的推理 Key。优先级最高** |
| `OPENAI_API_KEY` | — | 兜底变量。**仅在 `AI_API_KEY` 为空时**才被采用 |
| `APP_AI_AUTO_CONSUME` | `false` | **成本红线**：后台自动任务默认不消耗额度 |

> **⚠ 换 Key 的两个高频坑（都实测过）**
>
> 1. **放错变量**：`AI_API_KEY` 与 `OPENAI_API_KEY` 是**先到先得**，不是择优。
>    只要 `AI_API_KEY` 存在就用它，**哪怕它已作废也不会回落到另一把**——等于没换 Key。
>    ✅ 正确做法：把有效 Key 写进 `AI_API_KEY`。
> 2. **没重启**：已运行的进程持有**启动那一刻**的环境变量副本，
>    改注册表对正在跑的服务无效。✅ 必须重启 8080 与 8081。
>
> 不想重启时，可在 **AI 分析页 → 「应用并自检」** 热更新（只存内存，重启即回落环境变量）。

### 4.3 Agent 与检索

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `APP_AGENT_PYTHON_ENABLED` / `APP_AGENT_PYTHON_BASE_URL` | `true` / `http://127.0.0.1:8081` | Java → Python 转发开关与地址 |
| `APP_AGENT_ORCHESTRATOR` | `langgraph` | **编排实现**。自研 legacy 已删除，LangGraph 是唯一实现；配成别的值会告警后回退（§4.4）|
| `APP_AGENT_LANGGRAPH_CHECKPOINT` | `./data/langgraph.sqlite` | LangGraph 分支的状态快照**路径** |
| `APP_AGENT_LANGGRAPH_CHECKPOINT_ENABLED` | `false` | 快照开关。**默认关**：thread_id 用的是每次唯一的 trace_id，快照只写不读 |
| **`APP_AGENT_DEADLINE_ENABLED`** | `true` | **时间预算与超时降级**总开关（见 §4.5）。关掉＝回到旧行为 |
| `APP_AGENT_DEADLINE_SECONDS` | `0` | 显式指定预算（秒）。`0` = 跟随深度档位（快答 75 / 标准 240 / 深度 600）|
| `APP_AGENT_DEADLINE_SOFT_RATIO` | `0.7` | 用掉预算的这个比例就进入「收敛」：不再给工具 + 只再跑一轮 |
| `APP_AGENT_DEADLINE_MIN_CALL_SECONDS` | `15` | 单次模型调用超时的下限，再赶也不低于它 |
| `APP_AGENT_DEADLINE_CONVERGE_TOKENS` | `0` | 收敛轮的 `max_tokens`。`0` = 不收紧（默认）|
| `APP_AGENT_PARALLEL_TOOLS` | `true` | 同一轮里的多个工具**并行执行** |
| `APP_AGENT_PARALLEL_TOOLS_MAX` | `4` | 并行工具上限。**它是保护下游的闸门，不是性能参数** |
| `APP_RAG_EMBEDDING_ENABLED` | `true` | 语义向量开关 |
| `APP_RAG_EMBEDDING_MODE` | — | `local` = 本地零成本向量（不联网、不需要 Key） |
| `APP_WEB_SEARCH_ENABLED` | `true` | 联网检索总开关 |
| `APP_WEB_SEARCH_MCP_ENABLED` | `true` | MCP 通道优先 |
| `APP_EMBEDDING_API_KEY` | — | 仅在显式使用第三方 embedding 时才需要 |

### 4.4 编排层：LangGraph（唯一实现）

AI 引擎的**编排层**由 `pyagent/app/agent/langgraph_impl.py` 承担：
`StateGraph` 的 `agent ⇄ tools` + 条件边 `should_continue` + 可选 SQLite 快照。

自研 ReAct 循环（`AgentService._tool_loop`）已于 **2026-09-23 删除**，此前它与 LangGraph
**并存可切换**（`APP_AGENT_ORCHESTRATOR=legacy|langgraph`）。删除的判据不是"新旧之争"，
而是三条：

1. **功能等价** —— 把全量 pytest 强制切到 LangGraph 编排，**566 条全绿**。这套用例原本
   就是为 legacy 写的全链路用例（路由 / 工具 / 引用核对 / 护栏 / 缓存 / 降级 / 审计），
   也就是说新编排已经不需要 legacy 兜底。
2. **性能归因闭环** —— 此前「LangGraph 慢 3 倍」的根因是**漏发 `thinking` 参数**
   （服务端因此默认开启思考推理），已修复并被端到端测试钉住（见下面 ⚠️）。
3. **没有收益的复杂度** —— 继续并存意味着每次改编排要维护两条路径，任何"只在默认
   编排下成立"的行为都会变成隐藏分歧。

> 备份：`D:\backup\pyagent-legacy-orchestrator-20260923\`（含删除前的 `agent_service.py` /
> `langgraph_impl.py` / `config.py`）。

LangGraph 依赖**现在是必需的**（不再是可选），已合入主 `requirements.txt`：

```powershell
python -m pip install -r requirements.txt
.\start-python-agent.ps1
```

> ⚠️ **不要随意升级 langchain-openai**。`thinking` 这类扩展字段的传参入口在版本间变过：
> 早期代码用 `inspect` 探测 `extra_body` 是否可用，**那个探测恒为 False**（pydantic v2 的
> `__init__` 签名是动态生成的），于是落到 `model_kwargs`，而 langchain-openai 1.6.x 会把
> `model_kwargs` 原样展开喂给 `Completions.create()` ——
> 结果是 `TypeError: ... unexpected keyword argument 'thinking'`，**请求一个字节都没发出去**。
> 升级后务必跑 `pytest tests/test_stage12_langgraph_perf.py -k wire` 复核。

> 三条约定（改动前必读）：
> ① **工具只能走 `RiskAgentTools.execute_with_meta`** —— 数据权限闸门在那里，绕过即越权；
> ② **工具 Schema 单一数据源** = `specs()`，LangGraph 侧不再维护第二份；
> ③ **模型轮换降级不丢** —— `ChatOpenAI` 原生没有"429 换下一个候选"，由 `ModelChain` 补回来
> （401/403 不换，404/429/5xx 才换）。

> **SSE 逐 token 流式**：被逼成稿的那一轮边生成边推，中间轮的草稿攒到本轮结束再说——
> 没要工具就是提前成稿、补推，要了工具就丢掉（那是它的自言自语）。

> **空正文不再空转**：图语义是"没有工具调用即收敛"，所以空正文不会像旧循环那样
> 连跑 6 轮；自愈责任在上层 `_retry_final_answer`（不带工具、给足预算再问一次）。

跑单编排评测（产出报告；`--mode fast` 是 142 条确定性用例，**零 LLM 推理**）：

```powershell
python qa\eval_compare.py                      # fast：142 条确定性用例，零 LLM 推理
python qa\eval_compare.py --mode live --yes    # ★ 只跑 8 条端到端，**会调用大模型**
python qa\eval_compare.py --mode full --yes    # 142 + 8 条，额度消耗约为 live 的十几倍
```

> ★★ 想验证"编排跑得好不好"用 `--mode live`，别用 `full`
> 150 例里只有 8 条端到端真正穿过编排层，`full` 里另外 142 条是确定性的、两侧共用实现，
> 跑它们纯属烧额度。实测踩过：一轮 full 就把一个模型的免费额度跑爆，第二轮后半程全线 429。

报告落在 `qa/eval-compare/single-<时间戳>.md`。

> 对照能力已随 legacy 删除：`eval_compare.py` 现在只跑单侧。历史 A/B 报告仍在
> `qa/eval-compare/compare-*.json`，可用 `qa/_perf_attrib.py --stamp <时间戳>` 回看。

脚本自带**可用性探测**：起飞前、1/3、2/3、落地后各探一次（1 token）。起飞前不过就直接退出
（退出码 4，不白烧额度）；航程中任一次失败，结论标为 **无法判定（环境不可用）**，
而不是给你一个看起来很像结论的数字 —— 中途被限流的那半程全是环境假失败。

> ⚠️ **端到端样本只有 8 条**：一条翻转就是 12.5 分，单次 live 结论不可靠。
> 要真验收就先把 e2e 用例扩到 20~30 条，让结果有统计意义。

### 4.5 时间预算：到点得不到结果就降级

一次分析最多等多久，由**深度档位自带的挂钟预算**决定（`AgentBudget` 里一直就有）：

| 档位 | 挂钟预算 | 收敛起点（70%） | 收敛后单次调用上限 |
|---|---|---|---|
| 快答 quick | 75 s | 52.5 s | ≤ 22.5 s |
| 标准 standard | 240 s | 168 s | ≤ 72 s |
| 深度 deep | 600 s | 420 s | ≤ 180 s |

到点之后走**三态降级**，不是报错、也不是干等：

```
RUNNING ──(用掉 70%)──▶ CONVERGING ──(预算用尽)──▶ EXPIRED
                            │                         │
               不再给工具                    不再发起任何模型调用
               + 把「只剩 X 秒，请立即成稿」        + 工具也不再执行
                 原样交给模型                     + 直落零 token 本地确定性报告
               + 只再跑这一轮
```

**它省 token，不多花**：砍掉的正是"注定来不及、回来也用不上"的调用。
实测对照组（空正文时）旧行为会跑满 6 轮，收敛后只跑 1 轮。

几个刻意的设计：

- **理由必须送到模型面前**。旧代码算出了「预算已用尽，请立即成稿」却只用它决定不带工具，
  **文案被丢掉** —— 模型根本不知道该收敛，于是接着空转。现在这段提示语会进 messages。
- **单次调用超时按剩余时间收紧**（`llm_call_timeout`，ContextVar 实现，按执行上下文隔离）。
  常见故障是"一次卡住的请求吃掉整份预算"，收紧后它自己会在到点前断开。
  客户端是进程内单例，所以**不能**把超时写到实例上（A 的预算会掐死并发的 B）。
- **降级不抛异常**：用户拿到本地确定性报告 + `PARTIAL` + 明确理由，接口照旧 200。
- **唯一编排同样要接**：自研循环删除后，LangGraph 侧补齐了 `timeout_fn`
  （**每轮重新求值**剩余时间，用 `contextvars` 隔离），否则"到点降级"会缺掉这一层。

诊断看每次分析返回的 `diagnostics.deadline`：`state` / `budgetSeconds` / `elapsedSeconds` /
`remainingSeconds`。排障时第一眼就能分清"这次 PARTIAL 是环境慢，还是我们主动掐的"。

```powershell
# 三个档位统一压到 90 秒
$env:APP_AGENT_DEADLINE_SECONDS = "90"
# 或整层关掉（回到旧行为：到点只是不再给工具）
$env:APP_AGENT_DEADLINE_ENABLED = "false"
.\start-python-agent.ps1
```

> 回归：`pyagent/tests/test_stage11_deadline.py`（27 条，**全程离线、零 token**）：
> `pytest tests/test_stage11_deadline.py`

---

## 4.6 LangGraph 分支为什么慢：一次归因复盘

> 结论先行：**慢的不是框架，是"两侧给服务端发的请求不一样"。**
> 排查过程与工具都留在 `qa/`，可以直接复现。

### 现象

2026-09-22 的 live 对照（8 条 e2e，`qa/eval-compare/compare-20260922-213640.json`）：

| | legacy | langgraph |
|---|---:|---:|
| 单条均值 | 64.8 s | **196.2 s**（3.03×）|
| 最慢一条 | 146.6 s | **432.0 s** |

第一反应是「LangGraph 的图开销大」，于是先做了微基准 `python qa/bench_orchestrators.py`：

| 项目 | 触发频率 | 中位数 |
|---|---|---:|
| StateGraph 构造 + compile | 每次分析一次 | 2.38 ms |
| 折算每轮图开销 | 每轮一次 | 0.84 ms |
| ChatOpenAI 构造 | 每轮 × 每候选 | 0.39 ms |
| bind_tools | 每轮一次 | 0.02 ms |

**一次 6 轮分析的全部本地开销 ≈ 4.8 ms。** 而端到端差了几百秒——差四个数量级。
所以"框架慢"是被证伪的第一个假设。

### 第二个假设：多跑了轮次 —— 也被证伪

逐条看 `python qa/_perf_attrib.py --stamp 20260922-213640`：

| 用例 | legacy | 轮数 | langgraph | 轮数 | 倍数 |
|---|---:|---:|---:|---:|---:|
| EVAL-904 | 39.8 s | 2 | **259.8 s** | **1** | **6.53×** |
| EVAL-906 | 44.4 s | 2 | 160.1 s | 1 | 3.61× |
| EVAL-907 | 35.9 s | 1 | 81.4 s | 1 | 2.27× |

**轮数更少，却慢好几倍。** 所以慢发生在"单次调用之内"，不在"多跑了几轮"。

### 真因：少发了一个字段

`python qa/_payload_diff.py`（纯构造、**零 token**）把两条编排真正会下发的 body 摊开：

```
thinking: legacy={'type': 'disabled'}    langgraph='<缺>'      ⚠ 会影响服务端行为
```

自研 client 每次请求都显式带 `thinking: {"type": "disabled"}`（`llm_client._finish_body`），
而 `ChatOpenAI` 原生不知道这个约定 —— **不下发时服务端按模型默认走**，
对支持推理的模型（seed / v4 系列）等于默认打开了思考：每条请求多出成倍的 reasoning token。

它之所以难查，是因为：

- **不报错**。服务端没有任何异常，只是换了一条生成路径；
- **不像参数问题**。现象是"langgraph 更慢"，没人会怀疑到一个没发的字段上；
- **被 confounder 掩护**。当时同时还在排查 429 额度问题，两条混在一起。

### 修了什么

| # | 修复 | 位置 | 实测 |
|---|---|---|---|
| 1 | **`thinking` 跟随自研配置下发** | `langgraph_impl.ModelChain._new_client` | ★ 主因，预计端到端 2~6× 收益 |

> ⚠️ 上表 #1 **当初差点没修成**：第一版用 `inspect` 探测 `extra_body` 是否可用，
> 而 pydantic v2 的 `__init__` 签名是动态生成的，**那个探测恒为 False**，于是落到
> `model_kwargs`，请求直接 `TypeError`、一个字节都没发出去。更糟的是当时验它是用
> `_payload_diff.py` —— 那只走到 body *构造* 层，看着完全正确。
> 现在是**构造后回读**（`_thinking_kwargs`），验收则是本机 HTTP 端到端
> （`test_stage12_langgraph_perf.py -k wire`）。

| 2 | 流式 chunk 合并 **O(n²) → O(n)** | `ModelChain._chunks_to_ai` | 800 片 14.80ms → **5.25ms**（2.8×）|
| 3 | 模型客户端**按 model 缓存** | `ModelChain._build` | 省掉每轮重建 HTTP 连接池（含 TLS 握手）|
| 4 | 关掉 `stream_options.include_usage` | 同上 | 少一次服务端 usage 计算 |
| 5 | **同轮工具并行执行** | `agent_service._run_tool_calls` | 4×50ms 串行 201ms → 并行 53ms（3.8×）|

> **#5 两条编排共用**（legacy 删除前它也受益）。
> 注意顺序：**预算判定仍在主线程串行完成**（`budget.try_consume` 不是线程安全的，
> 让并发的两个工具各自通过意味着闸门被穿透），只有"跑"这一步并行；
> 后处理按**原始调用顺序**写回，保证留痕确定性。

### 没做的优化（以及为什么）

- **缓存 `StateGraph` 编译结果**：收益上限 2.38 ms/次分析，占端到端 0.001%；
  代价是图的复用正确性风险。**不值得。**
- **`bind_tools` 结果缓存**：0.02 ms/轮。同上，不值得。

这两种「看起来很该优化」的东西，做了只会增加复杂度。**优化要先有量化数据。**

### 怎么复现

```powershell
python qa/bench_orchestrators.py            # 微基准：本地开销只有几毫秒
python qa/_payload_diff.py                  # 请求体对照：找出真正的行为差异（零 token）
python qa/_perf_attrib.py --stamp <时间戳>   # 从历史对照报告里挖逐条耗时与轮数
```

回归：`pyagent/tests/test_stage12_langgraph_perf.py`（8 条，离线零 token）。

---

## 5. 排障入口

**判断问题的顺序很重要**：先看「服务活着吗」，再看「配置对不对」，最后才看代码。

| 端点 | 成本 | 用途 |
|---|---|---|
| `GET /api/ai/health` | **零** | 就绪自检与问题列表。**只校验配置存不存在** |
| `GET /api/ai/models` | 零 token | 当前 Key 可见的模型数、`unavailable` 清单 |
| `GET /api/ai/diagnose` | 极小 | **Key 失效 / 欠费 / 无权限**的真实诊断，会回显 Key 指纹与来源变量 |
| `GET /api/ai/web/probe` | 零（`live=1` 才真发） | 联网通道体检；`tried` 列出**每条通道为什么没被采纳** |
| `GET /api/ai/rag/status` | 零 | 语料库索引状态 |
| `POST /api/ai/evaluate/async` | 中（含真实联网） | 回归评估；返回 `taskId` 轮询，不要同步等待 |

### 5.1 为什么 health 显示 UP，分析却失败？

`/api/ai/health` 按设计**不发网络请求**（零成本），所以只能回答「Key 填没填」，
**回答不了「Key 还能不能用」**。要看这把 Key 是不是活的，用 `GET /api/ai/diagnose`。

快速自查脚本（零 token，逐个探测环境变量里各把 Key 的有效性）：

```powershell
python qa\_key_check.py      # 输出脱敏指纹 + 每把 Key 的 200/401 结论
```

### 5.2 「改了什么却不生效」的固定排查顺序

1. **跑的是不是旧进程**？Python 侧 uvicorn **不热加载**，改完 `pyagent/**` 必须重启 8081。
   判据：脚本跑对了、接口跑不对 ⇒ 一定是没重启。
2. **跑的是不是旧 jar**？看进程命令行里的 `-jar <路径>` 与启动时间。
3. 环境变量是否真的进了这个进程？（见 §4.2 的坑 2）

---

## 6. 目录结构

```
├── backend/                 Spring Boot 3（Java 17）
│   ├── src/main/java/…/      业务 + 转发过滤器 + 安全配置
│   ├── src/main/resources/   application.yml、static/（前端产物）
│   └── target/*.jar          打包产物
├── pyagent/                 Python Agent（FastAPI）
│   ├── app/agent/            ReAct 编排、评估、护栏、长期记忆
│   ├── app/web/              联网三通道（MCP → 方舟 API → 免 Key 直连）
│   ├── app/rag/              混合检索、相关度回检、索引
│   ├── data/rag-index/        Lucene/SQLite 索引（运行时生成）
│   └── requirements.txt
├── frontend/                Vue3 + Pinia + Element Plus（仅开发时需要）
├── mysql/                   初始化 SQL
├── qa/                      验证脚本与诊断工具（*_check.py 等）
├── start-backend.ps1 / start-python-agent.ps1 / start-all.bat / stop-backend.ps1
└── README.md、使用手册.md
```

---

## 7. 测试

```powershell
cd pyagent
pytest tests -q            # Python 侧回归（约 500 条）
```

```powershell
cd backend
mvn test -Dtest=PythonAgentForwardFilterTest -DfailIfNoTests=false   # 单个类，避免拉起整个上下文
```

> 后端**不建议直接跑 `mvn test` 全量**：骨架测试会启动完整上下文（连 MySQL/Redis/MQ），
> 本机这些中间件常常没起，失败信息与你改的代码无关。

---

## 8. 备份与回滚

项目**没有 git**，备份是唯一的后悔药：

| 目录 | 内容 |
|---|---|
| `D:\backup\risk-java-*` | 删除前的 Java agent 栈源码（如需回退 Python 承接方案） |
| `D:\backup\cleanup-2026*` | 历次清理移除的文件 |

Python Agent 若需回退到 Java 实现：把 `risk-java-src-*` 复制回 `backend/src` 并重新打包。

---

## 9. 安全注意事项

- **API Key 不要落盘**。历史上 `qa/` 下曾出现含明文 Key 的临时文件，已清除。
  如 Key 曾随仓库外发，请立即到控制台**重新生成**——删文件不能撤回已经发生的泄露。
- 静态资源（页面 shell）无需登录即可访问，但**所有业务数据在 `/api/**` 之后、必须带 JWT**。
- 默认账号请在首次部署后立即修改。
