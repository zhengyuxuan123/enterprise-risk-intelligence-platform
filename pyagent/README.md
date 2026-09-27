# pyagent —— Agent 栈的 Python 实现

企业经营风险智能分析平台的 **Agent 栈**用 Python 重写，与现有 Java 版**并存**，
由 Spring 按开关把 `/api/ai/*` 转发过来。前端**零改动**。

```
┌──────────┐      /api/ai/*       ┌──────────────────┐    转发（可开关）    ┌─────────────────┐
│  Vue SPA │ ───────────────────► │ Spring Boot 8080 │ ──────────────────► │ FastAPI 8081    │
└──────────┘                      │ 鉴权/CRUD/导入   │ ◄──── 回退到 Java ── │ agent/rag/web   │
                                  └──────────────────┘    （不可用时）      └─────────────────┘
                                                                                    │
                                                                            MySQL / 向量索引 / MCP
```

**为什么这么切**：Agent 栈（编排 + 检索 + 联网 + 模型调用）≈ 1.45 万行，
是整个项目里最需要 Python 生态的部分；而鉴权、CRUD、导入、MQ、报表是 Spring 的强项，
且已经稳定，没必要重写。两边端口错开，可以**同时跑、逐条对账**。

---

## 快速开始

```bash
# 依赖（本项目使用受管虚拟环境）
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt

# 跑对账测试（拿 Java 版实测结果当基准，不需要起任何服务）
python -m pytest tests -q

# 起服务
cp .env.example .env          # 按需填写 AI_API_KEY 等
python -m uvicorn app.main:app --host 127.0.0.1 --port 8081
```

自检入口：

| 端点 | 用途 |
|---|---|
| `GET /healthz` | 进程级探活（不含依赖） |
| `GET /api/ai/health` | 依赖体检（会跑**带负样本的自检**，不是只报"模块已装配"） |
| `GET /api/ai/agents?q=...` | 分工表 + 该提问的路由 dry-run |
| `GET /api/ai/models` | 账号可用模型 + 推荐排序（只读列举，零 token） |
| `GET /api/ai/diagnose` | **模型自检**（会真实发 1~3 个极小请求，只该由用户主动点击触发） |
| `GET /api/ai/migration` | 迁移进度自查（Python 侧独有：现在能跑哪一半、还差哪一半） |

> **端口绑定**：只监听 `127.0.0.1`。Python 侧不重复实现鉴权 —— 鉴权由 Spring 统一负责，
> 内部服务不对公网暴露。这是"Spring 转发"这个选择的题中之义。

---

## 迁移方法论：契约先行、切片对账

1.45 万行不可能一口气译完再调。所以按**能独立验证的层**切，每迁一层就用现有
**49 条评估用例**做双跑对账，只有逐字段相等才算过。

| 阶段 | 内容 | 状态 | 对账方式 |
|---|---|---|---|
| 1 | 确定性层（零 token）：`registry` / `router` / `budget` / `guardrail` | **完成** | 对比 EVAL-701~705 的路由结果与工具白名单 |
| 2a | 模型目录（零 token）：`isChatModel` / `modelScore` / `rankChatModels` / `isModelUnavailable` / `describe` / Key 解析 / 只读列举与缓存 | **完成** | 拿账号真实的 **133 个模型**逐字节对比 `available` 与 `recommended` |
| 2b | 模型调用：`complete` / `stream` / `chatWithTools`、工具循环、冷却状态机 | **完成** | 冷却状态机与错误翻译：jshell 跑 Java 真身取基准，76/76 逐字节对账；**真实调用已做线上双跑对账**（2026-09-21，见下） |
| 3a | 检索确定性层：`RagText` / `TextSplitter` / `QueryRewrite` / `LocalEmbedding` / `WebQueryCleaner` / `SourceRelevanceFilter` | **完成** | jshell 跑 Java 真身取基准，69/69 逐字节对账 |
| 3b | 检索索引层：`RagIndexStore` → **SQLite FTS5 + numpy** | **完成** | jshell 加载**真实 Lucene 9.11** 跑基准，对比 27 条查询的候选排名，54/55 一致 |
| 3c | 检索 IO 层：DB 访问层、结构化语料源（8 个 `CorpusSource`）、写入即索引、四阶段检索流水线 | **完成** | 端到端：回填索引 → 检索；并发与串行结果一致（曾修掉"兜底路径用 sync 把 217 条索引删到 22 条"的破坏性 bug） |
| 4 | 联网层：MCP / 直连 / 方舟 API 三通道 | **完成** | 解析层用固定 HTML 对账 40 passed；真实联网需联网 Key，本机未配 |
| 5 | 编排主链路：`AgentService`、留痕、记忆、引用核对、复核回环、工具层 | **完成** | 43 passed；含空正文自愈、企业不存在闸门、护栏拦截不调模型等分支 |
| 6 | 评估与治理：评估回归、历史存档、健康度 | **完成** | 同一份 `ai-eval-cases.json`（**150 条**，Java 与 Python 共用）；fast 模式 142/142 全通过（另 8 条 e2e 仅 live 模式执行） |
| 7 | 追溯 / 预警 / 通知 / 导出：`TraceQueryService` / `ProactiveRiskService` / `NotificationService` / `ReportExportService` | **完成** | 47 passed；导出已端到端产出真实 PDF 与 DOCX |

> 已移植约 **15,649 / 17,870** 行（按行数估）。**2026-09-21 起 Java 侧 agent 栈已删除**
> （18,946 → 2,047 行），`/api/ai/*` 全部由本服务承接，没有可回退的第二份实现 ——
> 回退只能从 `D:\backup\risk-java-*` 恢复源码。

### 阶段 1 对账结果（已通过）

```
registrySize  java=5  py=5
分工表字段比对完成（5 个 Agent × 6 个字段：id / name / duty / tools / bundles / core）

用例         Java 专员                            Python 专员                          工具
EVAL-701   ['风控总监', '外部情报员']                  ['风控总监', '外部情报员']                  OK
EVAL-702   ['风控总监', '内部数据专员']                 ['风控总监', '内部数据专员']                 OK
EVAL-703   ['风控总监', '处置审批专员', '内部数据专员']       ['风控总监', '处置审批专员', '内部数据专员']       OK
EVAL-704   ['风控总监', '资料库检索专员', '内部数据专员']      ['风控总监', '资料库检索专员', '内部数据专员']      OK
EVAL-705   ['风控总监', '内部数据专员']                 ['风控总监', '内部数据专员']                 OK
X-01..X-06（6 条压边界提问，含兜底分支 / 无工具分组 / 候补补位）                              全部 OK

结论：全部一致
```

复现方式：两个服务同时跑（Java 8080 + Python 8081），执行
`python ../qa/_parity_agents.py`。

### 阶段 2a 对账结果（已通过）

这一层用的是**真实账号数据**而不是构造样本：`GET /v3/models` 返回的 133 个模型，
喂给两边的同一套筛选/打分/排序逻辑，逐字节比对。

```
available     java=133                      py=133                        OK（含顺序）
recommended   java=20 项                    py=20 项                       OK（逐项相等）
current / candidates / autoFallback / discovery / autoConsume / hint      全部 OK
被判定「不适合对话」而剔除：47 个（向量化 8 / 图像视频与 3D 13 / 角色扮演与基座 12 / …）
```

`isModelUnavailable` 的 16 条状态码-正文组合、`describe` 的 5 个文案分支、
`parseModelIds` 的 4 种结构、Key 解析与脱敏、发现器的 TTL 与负缓存，均已逐条单测覆盖。

复现方式：两个服务同时跑，执行 `python ../qa/_parity_models.py`。
基准快照冻结在 `tests/data/ark_models_baseline.json`，刷新用 `python ../qa/_baseline_models.py`。

> **这些测试不联网、不消耗任何 token。** 单元测试读的是冻结快照，
> 只有 `_parity_models.py` 与 `_baseline_models.py` 会真的打一次只读列举接口。

### 阶段 2b 对账结果（冷却状态机 + 错误翻译，已通过）

这一刀先迁**不烧 token 也能验证**的那一半：模型降级的状态机和上游错误翻译。

```
describe                  26 组 (status, body)   与 Java 逐字节相等
isModelUnavailable        26 组                  全部相等
trim                      10 组（含 astral 字符） 全部相等
parseCandidates           11 组                  全部相等
                          ─────────────────────
                          76 / 76
```

**基准不是手抄的期望值。** `qa/_java_ground.jsh` 用 jshell 把 `LlmClient` 真加载起来，
跑它那些 `static` 方法（`describe` 是 public，`isModelUnavailable` / `trim` /
`parseCandidates` 是 private，用反射 `setAccessible` 取），输出 base64 写进
`tests/data/java_ground_fallback.txt`。Python 侧解码后逐字节比对。

这一步值在哪：见下面**第 12 条坑** —— 我原先那份实现（在 `model_catalog.py` 里）
已经和 Java 漂移了，而读代码一点也看不出来。

复现方式：

```bash
python qa/_javacp_prepare.py      # 从 fat jar 拆出 classes + BOOT-INF/lib（只做一次）
jshell -q -R-Dground.out="<abs>/pyagent/tests/data/java_ground_fallback.txt" \
       --class-path "<classes>;<lib>/*" qa/_java_ground.jsh
python qa/_parity_fallback.py     # 76/76
python -m pytest tests -q         # 120 passed（含这份基准的离线回归）
```

冷却状态机本身依赖 `System.currentTimeMillis()`，没法在两个进程间对拍，
改用**注入假时钟**的单元测试守住：冷却排序、10 分钟窗口边界、promote / 不 promote、
「401 不换模型」、「第二轮只拉一次可用列表」等 12 条。

---

### 阶段 3a 对账结果（检索确定性层，已通过）

新增 `app/rag/`：`textnorm.py` / `text.py` / `query_rewrite.py` / `embedding.py` /
`web_cleaner.py` / `relevance.py`（约 970 行）。

对账 69 条用例，**逐字节一致**：分片 9、分词 6、二元组 5、词频 4、查询改写 13、
词表顺序 1、检索词清洗 12、本地向量 6（按 IEEE-754 **位模式**比，不是比十进制）、
来源回检 3。

用例与基准都是**单一来源**（`tests/data/rag_cases.txt` + `tests/data/java_ground_rag.txt`），
Java 侧由 jshell 读、Python 侧由 `tests/rag_parity.py` 读。

#### 这一刀顺手抓出的 Java 侧问题

`QueryRewrite.LEXICON` 用 `Map.ofEntries(...)` 构建，而 **`Map.of` / `Map.ofEntries`
的迭代顺序是未规定的**（探测表顺序），实测**不等于声明顺序**：

```
声明顺序：续费, 留存, 流失, 投诉, sla, 故障, 合规, 风险, 现金流, ...
实际迭代：sla, 故障, 现金流, 投诉, 库存, 数据, 大促, 利润, ...
```

后果：`expanded()` 在**多个词条同时命中**时（如"现金流与债务风险"同时命中 现金流 与 风险），
拼出来的扩展词顺序取决于这张探测表，而这直接决定多路召回的第二条路径。
改词表时无法预测输出 —— 属于"能跑但不可推理"的那类设计。

Python 侧按 dump 出来的真实顺序重排，并在 `query_rewrite.LEXICON` 上方写了
`.. DANGER::` 警告；对账用例 `LEXICON_ORDER` 专门盯着它，换 JDK 导致顺序变化会立刻报红。

### 阶段 3b 对账结果（索引层，已通过）

Java 的 `RagIndexStore` 建在 **Lucene 9.11** 上，Python 没有对应物。选定的替代方案是
**SQLite FTS5（倒排 + BM25）+ numpy（向量余弦）**。

**基准不是手抄的**：`qa/_java_ground_index.jsh` 把 fat jar 里的 lucene jar 抽出来，
用 jshell 加载**真实的 Lucene**，按 `RagIndexStore.buildDoc` / `aclFilter` 同样的构造
建索引、跑查询，把排名 dump 出来（base64 防 GBK 乱码）。

```
upsert total=14 added=14 updated=0 skipped=0 ok=True
  Q0   OK   资金链断裂风险怎么处置      Q14  OK   管理
  Q1   OK   毛利率为什么下滑           Q15  OK   客户投诉如何处理
  Q2   DIFF 客户集中度风险  （并列区换位，见下）
  ...  Q3~Q13 / Q16~Q26             全部 OK
对账：54 项一致，1 项不一致
```

比的是三件事：**入库 token 串**（14 条）、**内容 hash**（14 条）、**候选排名**（27 条查询）。
分数量纲**故意不比** —— Lucene 的 BM25 与 SQLite 的 BM25 是两套值，
而上层 `RagService` 只用排名（RRF 是 `1/(60+rank)`），排名对了检索行为就对了。

**唯一一处差异（已知并接受）**：`客户集中度风险` 这一条，Java 给 k:1 / k:5 的分数是
`0.693125` / `0.687222` —— **相差 0.9%**，处在并列区；FTS5 把这两位换了位置。
根因是 idf 公式不同：SQLite 用 `log((N-df+0.5)/(df+0.5))`，Lucene 用 `log(1+x)`。

为什么**不去**做逐字节的 BM25 对齐：FTS5 的 `fts5vocab` 只提供 df、**不提供逐文档 tf**
（实测：'row' 变体的 `doc` 列是"含该词的文档数"，不是文档号）。要算 Lucene 那份精确 BM25
就得自建一张倒排表 —— 等于给同一份索引维护**两个真相**，写入放大一倍。
相比之下，并列区偶尔换位只影响 RRF 里的 `1/64` 与 `1/65`，最终顺序还由精排决定。
测试里对这条**只断言集合相等**，其余 26 条断言顺序完全相同（`_RANK_ORDER_EXEMPT`）。

另外两处刻意保留的差异，写在 `index_store.py` 模块头：

* **向量通道改成精确余弦**。Java 走 HNSW 近似最近邻，这里用 numpy 暴力点积 ——
  万级语料下暴力更快，而且没有召回抖动；入库向量已归一化时两者排序一致。
* **没有 Lucene 的"同索引必须同一维度"约束**。SQLite 不限制 BLOB 长度，
  所以 Java 那段「维度冲突 → 全量重建后重试」的分支不存在，
  改由 `vector()` 只比同维度的行（等价于重建后只剩一种维度）。

## 移植时踩到的坑（都是"不报错但行为悄悄变"的那类）

### 1. 正则必须加 `re.ASCII`

Java 的 `\b` / `\d` / `\w` **只认 ASCII**，Python 默认是 Unicode 感知。
`\b\d{17}[\dXx]\b` 去匹配「身份证110101199003078888」：

* Java：「证」不是 `\w` → 与 `1` 之间**是**词边界 → 命中；
* Python 默认：「证」是 `\w` → 不是词边界 → **不命中**。

忘了加标志，身份证/手机号检测在中文正文里会**静默失效**。
`tests/test_stage1_parity.py::test_pii_detection_survives_chinese_context` 专门守这条。

### 2. 档位解析不是对称的：`quick` 是前缀，`fast` 是全等

Java 原文是 `d.startsWith("quick") || d.equals("fast")`。
若"顺手统一成前缀匹配"，`fastest` 就会落到 QUICK 而 Java 落到 STANDARD ——
两边同一份配置跑出不同档位，双跑对照直接失真。

### 3. 工具配额是「三笔账」，不是一笔

预取证 / 模型只读 / 写动作 `propose_*` 各有各的额度。
**这是两次线上事故换来的**：混在一起算时，快答档的总配额（5 次）会被预取证一次用满，
紧接着"写证据"的循环第一步就撞上「配额已用尽」→ **刚取到的证据整批被丢弃** →
回退多轮 → PARTIAL。表现为"取证取够了反而失败、问题越多面越容易触发"。
回归测试：`test_budget_three_ledgers_do_not_collide`。

### 4. 三个上限是「按工具」而非「按类别总量」

`web_search_limit` / `write_tool_limit` / `per_tool_limit` 的判定都是
`used >= per_limit`，而 `used` 是**该工具**的调用次数，不是该类别的总数。
对 `web_search` 无害（全网只有一个联网工具），但 `propose_*` 有 **3 个**工具，
于是 `writeToolLimit=2` 实际是"每个提案工具最多 2 次"，而拒绝文案写的是
「本次分析允许的动作提案（2 个）已提满」—— **三处口径不一致**。
迁移期原样保留（改了就和 Java 对不上），详见 `app/agent/budget.py` 里的 NOTE。

### 5. `calls_so_far_for_this_tool` 由调用方维护，且被拒绝也照样 +1

Java 的 `consumeBudget()`：

```java
int used = toolUsage.getOrDefault(toolName, 0);
String denied = budget.tryConsume(toolName, used, prefetch);
toolUsage.put(toolName, used + 1);   // 被拒也 +1，且预取证与模型共用这张表
```

第 2 条意味着：快答档（`per_tool_limit=1`）下，某个工具被预取证用过一次之后，
模型再请求同一个工具就可能被拒。**这会不会又变成一次"取证取够了反而失败"，
取决于 `runTools` 里 `callCache` 与配额检查的先后顺序 —— 阶段 5 必须实测确认**，
不靠读代码下结论。

### 6. 排序必须**稳定**：`List.sort` ↔ `sorted` 要一一对应

Java 的 `rankChatModels` 用 `List.sort((a,b) -> Integer.compare(score(b), score(a)))`（TimSort，稳定）；
Python 必须写 `out.sort(key=model_score, reverse=True)`。

**`reverse=True` 不会破坏稳定性**：同分元素仍保持原始列表顺序。
而方舟的原始列表是按上架时间排的 —— 所以"同分保持原序"本身就带着语义。
若哪天有人改成 `heapq.nlargest` 之类的不稳定实现，表现是"每次点刷新，下拉框里同分模型的顺序都在变"，
很难归因到排序上。回归测试：`test_available_order_is_preserved_for_equal_scores`
（构造法：把输入逆序再排一次，同分组的相对顺序必须跟着反过来）。

### 7. 版本日期取**最后一组** 6 位数字，不是第一组

Java 是 `while (dm.find()) date = Integer.parseInt(dm.group(1));` —— 循环覆盖，
留下的是**最后**一个匹配。直接译成 `findall()[0]` 会取到第一组。

对 `deepseek-v3-1-250821` 这种只有一组的看不出差别；只有 `xxx-2024-250821` 这类
"ID 里本来就带 6 位数字"的模型才会错，且错得毫无征兆（排序微调，没人会去查）。
回归测试：`test_score_uses_last_date_group_not_first`。

### 8. `String.valueOf(true)` 是 `"true"`，`str(True)` 是 `"True"`

`parseModelIds` 里对非 Map 的元素走 `String.valueOf(o)`。模型 ID 本来是字符串，
但既然要对账，就不留一个"理论上会不一致"的口子 —— `_java_string_value()` 专门处理这一处。

### 9. 同名同值的字段，语义可能不同：`\b` 的教训在这里换了个形态

`describe()` 里的截断条件是 `b.length() > 200`。Java 的 `length()` 按 **UTF-16 码元**，
Python 按**码点**。错误正文里出现中文时截断位置会差一点。

这类"看着一样、量纲不同"的差异比第 1 条更隐蔽：不会让逻辑走错分支，只会让输出差几个字符。
处理方式是**明确选择**并按选择加注释（这里选码点，因为这段文案是给人读的），
而不是"没注意到所以默认"。

> **【2026-09-23 现状】** 这一条后来被彻底统一了：Java 侧下线后，
> `fallback` / `corpus` / `text` 全部改成 Python 原生（按码点），
> 原来的 `jcompat`（UTF-16 码元 + `<= U+0020` 的 trim + HALF_UP）已删除，
> 详见 `app/rag/textnorm.py`。上面这段保留作为"当时是怎么踩到的"记录。

### 10. **对账通过 ≠ 可以转发** —— 这是迁移期最容易犯的错

阶段 2a 完成后，`/api/ai/models` 的 `available` 与 `recommended` 已经和 Java **逐项相等**
（133 个真实模型）。很自然会想把这条路径加进转发白名单，让模型下拉框跑在 Python 上。

**不能加。** 该端点还有两个字段 Python 现在只能说假话：

* `unavailable` —— 模型冷却期表，来自尚未移植的 `withModelFallback` 状态机；
* `lastSwitch`  —— 最近一次自动降级的说明。

（阶段 2b 把冷却状态机搬完之后这两条已成立，端点才准入。留下这段是为了记住**准入条件是怎么来的**。）

前端拿这两个字段渲染「✕ 已判定不可用（10 分钟内不再优先尝试）」和「⚑ 已自动切换到 xxx」。
转发过去，这两行会**静默消失** —— 正是本项目一路在防的那个失败模式。

于是把准入条件从"实现对账通过"收紧成三条：

1. Python 侧该端点不再是 501；
2. **该端点返回的每一个字段**，Python 都能给出真实值（不是空占位）；
3. 前端真的会读到的每一个字段，都在第 2 条里。

反例就写在 `application.yml` 的 `forward-paths` 注释里，免得后人再踩一次。

**2026-09-22 补记**：这三条把 `/api/ai/models` 挡住了，却没人给 `/api/ai/diagnose` 做同样的检查 ——
它一直停在"未移植 501"，而 Java 侧代码已被删除，于是界面上那个「模型自检」按钮**两边都没实现**，
点下去只看到一行红字。补它的时候又抓出三个静默失败：`capabilities()` 键名对不上前端、
`/models/adopt` 不回模型列表（点一下下拉就空）、`LlmClient.list_available_models` 构造
`ModelDiscovery` 时参数名写错导致**自动降级的第二轮从来没跑起来**。
`tests/test_diagnose_contract.py` 把这三个都钉住了。
教训是：**"未移植"清单本身需要一条定期复核的规则**，否则它会随着 Java 侧被删而悄悄变成"两边都没有"。

### 11. 体检接口不许自己发网络请求

Java 的 `health()` 在模型这一段特意注明"只查配置，不发起任何推理请求"。
Python 侧照抄这条纪律时有个陷阱：`ModelDiscovery.list_available_models()` 在缓存为空时
**是真的会发请求的**。所以体检里只能读 `cached_models` 属性。

如果图省事直接调 `list_available_models()`，体检就变成了"刷一次页面打一次外部接口"的接口 ——
在排障时人会反复刷新，正好把这个坑踩实。

### 12. 基准要靠 JVM 真跑，不能靠读代码转写

阶段 2b 之前，我对账的期望值是**照着 Java 源码手抄**的。这只能证明
"我读到的和我写的一致"，证不出"和 Java 一致"。后来改成用 jshell 把 `LlmClient`
真加载起来跑（`describe` 是 public static；`isModelUnavailable` / `trim` /
`parseCandidates` 是 private static，用反射 `setAccessible(true)` 取），
结果立刻抓出一个已经发生的漂移：

```python
# model_catalog.py 里那份（我写的）
return f"模型服务返回 HTTP {status}：{b[:200]}"      # 按码点切

# Java 的
return "模型服务返回 HTTP " + status + "：" + (b.length() > 200 ? b.substring(0, 200) : b);
#                                                ↑ length() 是 UTF-16 码元
```

BMP 内字符两者相等 —— 中文、ASCII 全测过都是绿的。只有 **emoji 这类 astral 字符**
（1 码点 = 2 码元）能把差异暴露出来：240 个码元的串，Java 截出 100 个 emoji，
Python 截出 120 个。

同一份逻辑后来在 `fallback.py` 里又出现了一份，于是**两份实现各自漂移**。
现在的规矩：同一份逻辑只允许一处实现，其余模块一律再导出。

### 13. 取基准的脚本必须保持纯 ASCII

jshell 按**平台编码**（Windows 上是 GBK）读 `.jsh` 脚本文件。脚本里一旦出现
非 ASCII 字面量（比如 `"风险"`），它会被读成乱码，然后这个乱码被写进基准 ——
从此基准不可信，而且看起来一切正常。

所以 `qa/_java_ground.jsh` 里的中文测试串一律写 `\uXXXX` 转义。
同理，让 Java 自己 `Files.write(..., UTF_8)` 写文件，不要走 stdout
（stdout 在 Windows 控制台上也是 GBK）。

### 14. 日志占位符是 `%s`，不是 SLF4J 的 `{}`

照抄 Java 的 `log.info("[AI] 模型 {} 不可用：{}", m, e)` 不会在导入时报错，
只会在**真正走到那行**时抛 `TypeError: not all arguments converted during string
formatting` —— 而那行恰恰在异常处理路径上，于是把原始错误彻底盖住。

### 15. 对账要比**最终端点**的键集合，不能比它内部复用的子结构

`/api/ai/models` 返回的是 `modelInfo(refresh)` **再额外 put 一个 `baseUrl`**。
我拿 `/api/ai/diagnose` 的 `models` 子对象当对照做字段对账 —— 那份不含 `baseUrl`
—— 于是"全部一致"，直到真的转发之后才发现 Java 12 个键、Python 11 个。

现在 `qa/_parity_models.py` 里有一条显式的**字段契约**检查。

### 16. `Map.of` / `Map.ofEntries` 的迭代顺序不是声明顺序

见上面「阶段 3a」那一节。`expanded()` 的拼词顺序依赖它，
而它取决于 JDK 的探测表实现 —— 换 JDK 可能变。

### 17. Python 内置 `re` 不认识 `\p{...}`

Java 的 `RagText` 用 `\p{IsHan}` 切中文（Unicode 脚本类，含扩展 B 区）。
Python `re` 只认 `\w` `\d` `\s`，没有脚本类；装 `regex` 模块用 `\p{Han}` 才对齐。
**退而求其次用码点范围（如 `[\u4e00-\u9fff]`）会漏掉扩展 B 区的生僻字**，
而且漏得毫无征兆。

### 18. 字符串下标 / 长度的量纲必须一次定死

`TextSplitter` 全程 `substring` / `length()`，Java 是 UTF-16 码元。
上一轮在 `describe` 的截断上已经栽过一次（判断用码元、切片用码点），
所以当时把所有码元操作收进 `jcompat` 的 `to_units` / `from_units` /
`utf16_len` / `utf16_head`，并把「判断 + 切片」合成一个 `utf16_head` ——
让调用方**没有机会**再写出量纲不一致的代码。

> **【2026-09-23 现状】** `jcompat` 已删除，量纲统一为**码点**（Python 原生），
> 对应模块是 `app/rag/textnorm.py`（`length` / `head` 仍然成对提供，
> "判断 + 切片必须同量纲"这条约束保留了下来）。
> ⚠️ 分片规则变了，**已入库的 chunk 需要重建索引**。

### 19. `String.trim()` 不删全角空格，`isBlank()` 不算 NBSP

| 输入 | Java `trim()` | Python `strip()` |
|---|---|---|
| `　全角`（U+3000） | 保留 | **删掉** |
| NBSP（U+00A0） | `isBlank()` = false | `isspace()` = **true** |

全角空格在知识库正文里是常客：一段以它开头的语料，Java 侧留着、Python 侧删掉，
分片起点就差一个字符，后面所有分片跟着错位。

> **【2026-09-23 现状】** 现在**采用 Python 侧的行为**（`str.strip()` 删全角空格、
> NBSP 算空白）。理由：Java 侧已下线，而"留着全角空格"这条规则只会在中文语料里
> 制造看不见的偏移。对应 `app/rag/textnorm.py` 的 `trim` / `is_blank`。

> **【2026-09-23 现状】** 现在**采用 Python 侧的行为**（`str.strip()` 删全角空格、
> NBSP 算空白）。理由：Java 侧已下线，而"留着全角空格"这条规则只会在中文语料里
> 制造看不见的偏移。对应 `app/rag/textnorm.py` 的 `trim` / `is_blank`。

### 20. 别写「Python 的默认行为不同」这种对照断言

写单测时顺手加了 `assert round(0.0005, 3) == 0.0` 想证明银行家舍入的差异，
结果它是 `0.001` —— **十进制字面量在二进制里几乎不可能正好落在半值上**
（`0.0005` 的实际值略大于 0.0005）。这类"证明差异"的断言看似有理、实则脆弱，
应改为直接断言期望值。

### 21. jshell 脚本的每条语句必须写成**一行**

续行写法（下一行以 `.append(...)` 开头）在 Java 源文件里合法，
在 jshell 里会被当成新表达式 → `非法的表达式开始`。
另外 `--class-path` 的长选项形式在本机不生效（类找不到），
必须写短选项 `-c`，并且**不能用通配符** `lib/*`（要逐个 jar 列出来）。

### 22. `fts5vocab` 的 `doc` 列不是文档号

`CREATE VIRTUAL TABLE v USING fts5vocab(t, 'row')` 的列是 `(term, doc, cnt)`，
其中 **`doc` = 含该词的文档数、`cnt` = 总出现次数**，不是 (term, docid) 的明细。
想拿逐文档 tf 只能自建倒排表 —— 这是阶段 3b 决定"不做精确 BM25 对齐"的直接依据。

### 23. `LocalEmbedding` 会把维度夹到 `[64, 4096]`

写 `LocalEmbedding(16)` 或 `LocalEmbedding(32)` 得到的都是 **64 维**。
测试里想造"两个不同维度"必须写 64 / 128 —— 否则两个向量其实同维，
"维度冲突 → 重建" 这条用例会**假通过**。

### 24. FTS5 的 `bm25()` 是**负值且越小越相关**

与 Lucene（正值越大越相关）相反。对外统一翻成正数（`-bm25(...)`）再排序，
否则 `ORDER BY` 会把最不相关的排在最前面，而且不会报错。

### 25. 兜底检索路径不能用 `sync`（会**删光索引**）

`RagIndexStore.sync()` 是**对账式**的：本次没提到的语料会被删掉。
结构化切片受 `structured_max` 截断，一次兜底触发 sync 就把 217 条索引删到 22 条，
而且**不报错** —— 表现是"第一次检索正常，之后越检索越差"。
兜底路径只该用非破坏的 `upsert()`。

### 26. SQLite 连接不能跨线程共享

同一个 `sqlite3.Connection` 给多线程用，并发下同一条查询会返回 0/1/3 条（串行是 5）。
排查时最容易误判成"检索有随机性"。修法是加一层串行化代理：
**所有语句在锁内执行并立即取回结果**，不能让游标在锁外被别的线程打乱。

### 27. 评估用例文件是 camelCase，数据类是 snake_case

`ai-eval-cases.json` 里写的是 `expectWeb` / `minWebResults` / `companyId`，
而 `EvalCase` 的字段是 `expect_web` / `min_web_results`。
解析时按"未知字段忽略"处理 → 所有断言**根本没执行**，报告显示 44/44 全过。
**假绿灯比红绿灯危险得多**：迁移期最需要的是"确实跑过"的证据。

### 28. heredoc 里写 Python 会把 `\n` 变成真换行

用 `python - <<'PYEOF'` 写文件时，`"\n"` 会被 shell 展开成真实换行，
产出 `SyntaxError: unterminated string literal`。
涉及转义的改动一律用 Edit 工具直接改文件，改完 grep 确认。

### 29. 插入点拿 `execute()` 的返回值当自增 id

`execute()` 返回的是**影响行数**，自增表插入恒为 1。
拿它当 id 会让"审批 #1"永远指向同一条记录，且**完全不报错**。
需要 id 的插入点必须走 `insert_id()`。

### 30. `GET /v3/models` 列的是**平台全量**，不是本账号已开通

2026-09-22 用 `qa/_probe_models.mjs` 把本账号 134 个模型逐个真调了一遍，结果：

| 结果 | 个数 | 方舟返回 |
|---|---|---|
| 真能调通 | **14** | HTTP 200 |
| 不存在于本账号 | 91 | 404 `InvalidEndpointOrModel.NotFound` |
| 已开通但没开通这个模型 | 16 | 404 `ModelNotOpen` |
| 开通了但超自设限额被暂停 | 1 | 429 `SetLimitExceeded` |

也就是说**"列出 134 个"和"能用 134 个"是两件事**，前者只说明这个平台有这些模型。
教训有三层：

1. **文案层**：`hint` 里绝不能写"本账号共 N 个可用模型"——数字是真的，含义是错的。
2. **行为层**：自动降级的第二轮是"从列举结果里补试"，若照**上游返回顺序**取前 4 个，
   必然挑到 2024 年的老模型（列表按上架时间**从旧到新**排），全 404 ——
   对外表现成"自动降级从来不生效"。现改为先 `rank_chat_models()` 排序再取。
3. **运维层**：换 Key、或新开通了模型之后，跑一次
   `node ../qa/_probe_models.mjs`（失败请求不计费，只有 200 那次计几个 token），
   把真能调通的填进 `AI_MODEL` / `AI_MODEL_CANDIDATES`。
   这比让平台自己去猜要可靠得多。

### 31. `getModel()` 是**生效模型**，不是配置值

Java 的 `noteModelSuccess` 里是直接 `model = m` —— 自动降级会把 `getModel()` 的返回值
改写成**切换后**的模型。Python 侧 `get_model()` 一度返回构造时读到的配置值，
于是降级真的生效了、界面顶部横幅却还显示那个已经失效的旧模型名，
而同一屏右侧「模型服务」卡片显示的是切换后的模型：**同一屏两处自相矛盾**。

配置值与生效值要分开记：

* `self.model` —— 配置里写的（环境变量 / 运行时入口设的）；
* `self._cooldown.primary` —— 真正在用的，降级时被 promote 改写。

连带一个静默失败：`apply_runtime_config` 里判断"模型名有没有变"若拿 `self.model` 比，
则降级到 X 之后用户再提交恰好等于配置值 Y 的名字，会被判成"没变化"而直接跳过 ——
界面提示"应用成功"，实际用的还是 X。

---

## 线上双跑对账（2026-09-21）

前面各阶段的对账都是**离线**的：基准来自 jshell 跑 Java 真身，或固定样本。
第一次**两侧服务真起、真打模型、真联网**互相对照，才发现离线对账抓不到的那一类差异：

> 接口照样 200、正文照样有字，但**前端某块会静默变空**。

一共抓出 8 处（详见根目录 `QA-线上对账-Java与Python双跑-2026-09-21.md`）：
`reindex` 直接 500（索引只有 48 条 vs Java 1065）、`flush`/`rag/status`/`health` 键集合不重叠、
索引里 48 条无主切片、`answer_json` 少 13 个键、`provider` 少前缀、
模型循环取的数不进证据池、企业不存在返回 200（Java 是 400）。

**教训（这是本次最值钱的一条）**：离线对账能保证"函数算得对"，
保证不了"端点吐出来的结构对得上"。凡是**前端直接渲染**的结构
（health / answer_json / evidence_json / provider），必须拿两个真服务互打一遍，
只比对**契约字段**，不比正文——模型输出天然不逐字节相等，拿它做逐字节比对是假对账。

复跑方式：

```bash
python ../qa/_online_recon.py reindex   # 两侧全量回填（先做，否则索引快照新旧不可比）
python ../qa/_online_recon.py a         # 零 token 层：health / models / agents / rag
python ../qa/_online_recon.py b         # 烧 token 层：analyze 契约字段
python ../qa/_model_probe.py            # 挑一个"列得出来且真调得动"的模型
```

---

## 有意为之的差异

| 项 | Java | Python | 原因 |
|---|---|---|---|
| 向量索引 | Lucene 9.11（倒排 + HNSW） | SQLite FTS5 + numpy | Python 无对应物；上层只用排名，排序语义一致即可 |
| 端口 | 8080 | 8081 | 必须能同时跑，否则无法对账 |
| 鉴权 | Spring Security | 无（只绑 127.0.0.1） | 鉴权由 Spring 统一负责，内部服务不重复实现 |
| 未移植端点 | 正常实现 | 业务码 **501** + 一句可操作说明 | 迁移期最危险的不是"缺功能"，而是"接口在、行为不对"——那会让对账得出完全错误的结论。**2026-09-22 起已无未移植端点**，这条通道保留以备将来新增 |
| Key 热更新入口 | 只要 Bean 在就能收 Key | 同样收，且**客户端实例不会被丢掉** | 首次进页面时本来就没有 Key，`if not is_configured(): llm = None` 会让"粘贴 Key"只写进一个随即被丢弃的一次性实例里 |
| `always-offer` | 启动期 `@Value` 注入 | **每次读配置** | 迁移期要能"改一个环境变量就换策略"，不必重启来对比两种行为 |

---

## 目录结构

```
pyagent/
├── app/
│   ├── config.py            # 环境变量契约，键名与默认值逐一对齐 application.yml
│   ├── main.py              # FastAPI 入口 + 异常处理器（对齐 GlobalExceptionHandler）
│   ├── core/                # errors / logging
│   ├── api/                 # ai.py（路由）、schemas.py（ApiResponse）
│   ├── core/                # errors / logging / security（请求级用户上下文）
│   ├── db/                  # tables（表定义）/ engine（SQLAlchemy Core 只读）
│   ├── agent/               # registry / router / budget / guardrail（阶段 1）
│   │                        # agent_service / trace / memory / grounding（阶段 5）
│   │                        # review_loop / resilience / approvals
│   │                        # evaluation / eval_history / health（阶段 6）
│   │                        # trace_query / proactive / notification / report_export（阶段 7）
│   ├── ai/                  # keys / model_catalog / discovery / fallback（阶段 2）
│   │                        # llm_client / tools / tool_context（阶段 5）
│   ├── rag/                 # textnorm / text / query_rewrite / embedding / web_cleaner
│   │                        # / relevance（3a）+ index_store（3b）
│   │                        # + corpus / sources / structured / indexing / rag_service（3c）
│   └── web/                 # mcp_client / direct_client / search_service（阶段 4）
└── tests/                   # 对账测试，基准来自 Java 版实测输出
    ├── ground_cases.py      # 阶段 2b 的用例清单（qa 脚本与回归测试共用一份）
    └── data/
        ├── ark_models_baseline.json       # 133 个真实模型的 Java 实测快照
        ├── java_ground_fallback.txt       # jshell 跑 Java 真身取回的纯函数基准
        ├── java_ground_rag.txt            # 阶段 3a：检索确定性层基准
        └── java_ground_index.txt          # 阶段 3b：真实 Lucene 的排名基准
```

---

## 验证入口

| 命令 | 覆盖 |
|---|---|
| `python -m pytest tests -q` | **395 passed**，全离线：含 jshell 基准的逐字节回归 |
| `node ../qa/_probe_models.mjs` | **实探账号哪些模型真能调通**（404/429 不计费；换 Key 后跑一次） |
| `node ../qa/_diag_check.mjs` | `/diagnose` 与 `/models` 的键集契约（零 token，走 mock 底座） |
| `node ../qa/_diag_ui_check.mjs` | 真机（Chrome/CDP）点「模型自检」，12 项断言 |
| `python ../qa/_parity_agents.py` | 路由对账（阶段 1） |
| `python ../qa/_parity_models.py` | 模型层全字段对账 + 字段契约（阶段 2a + 2b） |
| `python ../qa/_parity_fallback.py` | 纯函数对账，读 jshell 基准（阶段 2b） |
| `python ../qa/_parity_index.py` | 索引层对账：FTS5 vs 真实 Lucene（阶段 3b） |
| `python ../qa/_verify_pyfwd.py [--expect-fallback]` | 转发 5 项 / 自动回退 |
| `python ../qa/_baseline_models.py` | 刷新模型目录基准快照 |
| `python ../qa/_javacp_prepare.py` | 从 fat jar 拆 classpath（jshell 取真前跑一次） |

---

## 回退

> **2026-09-21 起，这条已不成立**：Java 侧的 Agent 栈（`service/ai`、`service/web`、
> `service/rag`、`service/agent`、`AgentService`、`AiController`，约 1.7 万行）
> 已删除，归档在 `D:\backup\risk-java-deleted-20260921\`（删前全量在
> `D:\backup\risk-java-src-20260921-src`）。

所以 `APP_AGENT_PYTHON_ENABLED` **不再是"切回 Java"的开关**：

* `true`（默认）：`/api/ai/*` 转发给 Python 服务（8081）—— **Python 必须处于运行状态**；
* 关掉它：这些路径在 Java 侧已无实现，会**直接 404**，不会自动降级回 Java。

要真正回退，得先从备份恢复源码再重新打包。启动与依赖顺序见根目录
`RUNBOOK-系统运行手册.md`，接管过程见 `QA-Java-agent栈删除与接管-2026-09-21.md`。
