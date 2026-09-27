<template><div class="page">
  <div class="page-head">
    <div>
      <h2>AI Agent 智能分析（ReAct · 混合检索 · 联网溯源 · 来源分层）</h2>
      <p>Agent 按需调用取数与联网检索工具，结论按「内部知识库与经营数据」「外部公开资料」两份来源分开给出，每条建议都标注依据，引用可点开核对。</p>
    </div>
    <div class="runbar" v-if="ai.running">
      <el-tag type="warning" effect="dark">流式分析进行中</el-tag>
      <span class="tip">{{(ai.elapsedMs/1000).toFixed(1)}}s · {{ai.tokenCount}} 片段 · 期间可自由切换模块，分析会在后台继续</span>
      <el-button size="small" type="danger" plain @click="ai.cancel()">取消分析</el-button>
    </div>
  </div>

  <el-alert v-if="ai.status==='ERROR'" :title="'流式分析失败：'+ai.error" type="error" show-icon :closable="false" style="margin-bottom:12px"/>

  <el-alert v-if="diag" :type="diag.ok?'success':'warning'" show-icon style="margin-bottom:12px"
            :title="diag.ok?'模型服务连接正常':'模型服务不可用'"
            :description="(diag.ok?'':'原因：'+diag.issue+'　')
              +'base-url '+(diag.baseUrl||'-')+' · 模型 '+(diag.model||'-')
              +' · Key '+(diag.keyHint||'-')+'（来源 '+(diag.keySource||'-')+'）'
              +(diag.processStartedAt?('　· 本进程启动于 '+diag.processStartedAt):'')"/>
  <div v-if="diag && diag.envHint" class="hint envhint">
    ⓘ 改了系统环境变量却不生效？{{diag.envHint}}
    <span v-if="diag.configSource">当前配置来源：{{diag.configSource}}。</span>
    <span v-if="diag.webSearch && !diag.webSearch.enabled">联网检索：{{diag.webSearch.reason}}</span>
    <span v-if="channelLine" class="chanline">　ⓘ {{channelLine}}</span>
  </div>

  <el-card v-if="diag && diag.capabilities" style="margin-bottom:12px" class="caps">
    <template #header><div class="cardhead"><span>服务商能力体检</span><span class="tip">换 LLM 厂商后先看这张表，缺哪项就少了哪项功能</span></div></template>
    <el-descriptions :column="1" border size="small">
      <el-descriptions-item v-for="(v,k) in capsMeta" :key="k" :label="v.label">
        <el-tag size="small" :type="capsTag(k)">{{capsText(k)}}</el-tag>
        <span class="capmsg">
          <template v-if="diag.capabilities[k]!=='ok'">　{{diag.capabilities[k]}}</template>
          <template v-else>　{{v.ok}}</template>
        </span>
      </el-descriptions-item>
    </el-descriptions>
    <div class="hint">影响：JSON 模式缺失 → 结构化答案抽取失败；工具调用缺失 → Agent 取不到内部数据。向量化默认走本地零成本向量（不联网、无需 Key），只有显式配置第三方 embedding 后才会远程调用。</div>
  </el-card>

  <el-card v-if="ragStatus" style="margin-bottom:12px" class="ragcard">
    <template #header><div class="cardhead"><span>本地资料库（写入即索引）</span><span class="tip">业务数据一落库就增量进本地向量库，不再等提问时才同步</span></div></template>
    <div class="row" style="gap:10px;flex-wrap:wrap;align-items:center">
      <el-tag size="small" :type="ragStatus.enabled?'success':'info'">写入即索引 {{ragStatus.enabled?'已开启':'已关闭'}}</el-tag>
      <span class="tip">库内 {{ragStatus.index?.docs||0}} 条语料 · 向量 {{ragStatus.index?.vectorKey||'—'}} · 合并窗口 {{ragStatus.mergeWindowMs}}ms</span>
      <el-button size="small" plain :loading="ragBusy" @click="reindexNow">重新回填</el-button>
    </div>
    <div class="hint" v-if="ragTypes.length">
      来源分布：<span v-for="t in ragTypes" :key="t.k" class="ragtype">　{{t.label}} {{t.n}}</span>
      <span v-if="ragStatus.pending">　· 待写入 {{ragStatus.pending}} 条</span>
      <span v-if="ragStatus.lastFlushAt">　· 最近落库 {{String(ragStatus.lastFlushAt).replace('T',' ')}}</span>
    </div>
    <div class="hint">点「重新回填」可按当前数据库重建全部切片（幂等：内容没变的连向量都不重算）。资料库落后于数据库时，语义检索会漏召回。</div>
  </el-card>

  <!-- 长期记忆（跨会话）：与「本地资料库」并列，但语义完全不同 ——
       资料库是证据（要占 [n] / 来源ID=x 编号），记忆只是历史快照（不占编号、数字必须重新核实）。 -->
  <el-card v-if="memStats" style="margin-bottom:12px" class="memcard">
    <template #header><div class="cardhead">
      <span>长期记忆（跨会话）</span>
      <span class="tip">零 token：抽取与召回全走本地规则 + 本地哈希向量，不消耗模型额度</span>
    </div></template>
    <div class="row" style="gap:10px;flex-wrap:wrap;align-items:center">
      <el-tag size="small" :type="memStats.enabled?'success':'info'">长期记忆 {{memStats.enabled?'已开启':'已关闭'}}</el-tag>
      <el-tag size="small" effect="plain" :type="memStats.writeEnabled?'success':'info'">写入 {{memStats.writeEnabled?'已开启':'已关闭'}}</el-tag>
      <span class="tip">生效 {{memStats.active}} 条<span v-if="memStats.superseded">（另有 {{memStats.superseded}} 条已被新口径取代）</span>
        · 每次最多召回 {{memStats.topK}} 条 / 注入 {{memStats.budget}} 字</span>
      <el-button size="small" plain :loading="memBusy" @click="loadMemory()">刷新</el-button>
      <el-button size="small" plain :loading="memRebuildBusy" @click="rebuildMemory">从历史分析回填</el-button>
      <el-button size="small" plain :loading="memProbeBusy" @click="probeMemory">召回预演</el-button>
    </div>
    <div class="hint" v-if="memKindLine">构成：{{memKindLine}}</div>
    <div class="hint" style="margin-top:6px">
      记忆<b>不是证据</b>：它只用来判断「这次的说法与以前是否一致」，不占 [n] 与「来源ID=x」编号；
      其中的数字都是历史值，必须重新核实后才可写进结论。同主题出现新口径时不覆盖旧条目，而是标记为「已被取代」——口径漂移本身就是要被看见的信息。
    </div>

    <div v-if="memProbe" class="memprobe">
      <div class="tip">召回预演「{{memProbe.q}}」→ 命中 {{memProbe.hits.length}} 条、注入 {{memProbe.contextChars}} 字（阈值 {{memProbe.minScore}}）</div>
      <div v-if="!memProbe.hits.length" class="hint">本次没有命中。长期记忆是「按需想起」，不是把全部历史堆给模型。</div>
      <div v-else>
        <div class="memrow" v-for="h in memProbe.hits" :key="h.id">
          <span class="mtag">{{kindLabel(h.kind)}}</span>
          <span class="mtext">{{h.content}}</span>
          <span class="tip">记忆#{{h.id}} · 分 {{h.score}} · {{h.ageDays}} 天前</span>
        </div>
      </div>
    </div>

    <el-collapse style="margin-top:6px">
      <el-collapse-item :title="'已沉淀的记忆（'+memItems.length+' 条）'" name="mem">
        <div class="memtools">
          <el-checkbox v-model="memIncludeSuperseded" size="small" @change="loadMemory()">显示已被取代的历史口径</el-checkbox>
          <span class="tip">「忘掉」是物理删除（合规要求），点了就真的没了。</span>
          <el-button size="small" type="danger" plain :loading="memBusy" @click="clearMemory">清空本企业记忆</el-button>
        </div>
        <el-table :data="memItems" size="small" v-if="memItems.length">
          <el-table-column label="类型" width="96">
            <template #default="{row}"><el-tag size="small" effect="plain" :type="kindTag(row.kind)">{{kindLabel(row.kind)}}</el-tag></template>
          </el-table-column>
          <el-table-column prop="content" label="记忆内容" min-width="300" show-overflow-tooltip/>
          <el-table-column label="状态" width="118">
            <template #default="{row}">
              <span v-if="row.status==='SUPERSEDED'" class="tip">已被 #{{row.supersededBy}} 取代</span>
              <span v-else class="ok">生效中</span>
            </template>
          </el-table-column>
          <el-table-column label="命中" width="56"><template #default="{row}">{{row.hits}}</template></el-table-column>
          <el-table-column prop="updatedAt" label="更新时间" width="152"/>
          <el-table-column label="操作" width="72" fixed="right">
            <template #default="{row}">
              <el-button link type="danger" size="small" @click="removeMemory(row)">忘掉</el-button>
            </template>
          </el-table-column>
        </el-table>
        <div v-else class="hint">还没有沉淀任何记忆。做一次分析后会自动抽取；也可以点「从历史分析回填」把以前的记录补进来。</div>
      </el-collapse-item>
    </el-collapse>
  </el-card>

  <el-card v-if="diag" style="margin-bottom:12px" class="keyfix">
    <template #header><div class="cardhead"><span>模型服务（热更新，无需重启后端）</span><span class="tip">仅管理员可用 · 只存内存，重启后回到环境变量</span></div></template>

    <!-- 账号可用模型：把「到底有哪些模型能用」摆出来，并从下拉里直接选一个，不必知道模型 ID 怎么拼 -->
    <div class="row modelsbar">
      <span class="tip">账号可用模型：</span>
      <el-select v-model="pickedModel" filterable allow-create default-first-option clearable
                 placeholder="先点右侧「刷新列表」拉取，或直接手填模型名" style="flex:1"
                 @change="onPickModel">
        <el-option v-for="m in availableModels" :key="m" :label="m+(m===currentModel?'（当前）':'')" :value="m"/>
      </el-select>
      <el-button size="small" plain :loading="modelsLoading" @click="refreshModels(true)">刷新列表</el-button>
      <el-button size="small" type="primary" plain :loading="adoptLoading" @click="adoptModel">自动选一个能用的</el-button>
    </div>
    <div class="hint">
      列表来自 <code>GET /v3/models</code>（只读列举，<b>不消耗 token</b>）
      <!-- 措辞不能再写「本账号共 N 个可用模型」：该接口返回的是**平台全量**清单，
           跟"本账号开通了哪些"是两码事。2026-09-22 实测本账号 134 个里只有 14 个
           真能调通，91 个调用直接 404 NotFound。数字是真的，但含义是错的 ——
           会让人以为闭着眼睛挑一个都行。（校验脚本：qa/_probe_models.mjs） -->
      <span v-if="modelsMeta.availableCount">，共 {{modelsMeta.availableCount}} 个候选 —— 这是<b>平台全量</b>清单，<b>不等于本账号已开通</b>（未开通的调用会返回 404）。下拉里是其中适合对话的（已按「新且强」排序）。</span>
      <span v-else>。</span>
      <span v-if="currentModel">当前生效模型：<b>{{currentModel}}</b>。</span>
      <span v-else>当前未指定主模型，运行时按可用列表自选。</span>
      <span v-if="modelsMeta.autoFallback">已开启自动降级：主模型报「未开通 / 不存在 / 限流」时会自动换下一个候选。</span>
      <span v-if="modelsMeta.lastSwitch" class="switchline">　⚑ {{modelsMeta.lastSwitch}}</span>
      <span v-if="unavailableList.length" class="switchline">　✕ 已判定不可用（10 分钟内不再优先尝试）：{{unavailableList.join('；')}}</span>
    </div>

    <!-- 后台自动任务是否允许消耗模型额度：默认禁止，只有使用者主动触发才花 token -->
    <div class="row autobar">
      <el-switch v-model="autoConsume" :loading="autoConsumeLoading" @change="applyAutoConsume"/>
      <span class="tip">允许后台自动研判调用模型（MQ 事件驱动的主动预警）</span>
      <el-tag size="small" :type="autoConsume?'warning':'success'">{{autoConsume?'后台会消耗额度':'仅你主动发起才消耗'}}</el-tag>
    </div>
    <div class="hint">
      默认<b>关闭</b>：未经你主动触发，系统不会调用你的模型、不消耗额度。页面上的「同步分析 / 流式分析」不受此开关影响，随时可用。
      （此开关只存内存，长期生效请设环境变量 <code>APP_AI_AUTO_CONSUME=true</code>。）
    </div>

    <div class="row presets">
      <span class="tip">一键填充：</span>
      <el-button size="small" plain @click="fillPreset('dashscope')">阿里云百炼</el-button>
      <el-button size="small" plain @click="fillPreset('ark')">火山方舟</el-button>
      <el-button size="small" plain @click="fillPreset('openai')">OpenAI</el-button>
    </div>
    <div class="row">
      <el-input v-model="cfgForm.baseUrl" placeholder="base-url，例如 https://ark.cn-beijing.volces.com/api/v3" style="flex:1"/>
    </div>
    <div class="row">
      <el-input v-model="cfgForm.model" placeholder="模型名：填方舟模型 ID（如 doubao-seed-1-6-250615）或接入点 ep-…" style="flex:1"/>
    </div>
    <div class="row">
      <el-input v-model="cfgForm.key" type="password" show-password :placeholder="'粘贴 API Key（'+providerName+'）'" style="flex:1" @keyup.enter="applyKey"/>
      <el-button type="primary" :loading="keyLoading" @click="applyKey">应用并自检</el-button>
    </div>
    <div class="hint">{{providerHint}} 三项可只填要改的：换厂商时 base-url、模型名、Key 必须一起换，只改一个必然 401。</div>
  </el-card>

  <el-alert v-if="ai.status==='CANCELLED'" type="info" show-icon :closable="false" style="margin-bottom:12px"
            title="上一次流式分析已被取消，已接收的部分内容仍保留在下方。"/>

  <!-- 服务端主动收工：这不是失败，分析可能仍在后台跑，必须告诉用户去哪里取结果 -->
  <el-alert v-if="ai.notice" type="warning" show-icon :closable="true" style="margin-bottom:12px"
            :title="ai.notice" @close="ai.notice=''"/>

  <!-- 流式进行中：把「现在在做什么 / 已经等了多久」实时说清楚。
       等待之所以难熬，多半不是因为慢，而是屏幕上没有任何证据表明系统还活着。 -->
  <div class="progbar" v-if="ai.running">
    <div class="prow">
      <span class="pstage">{{ai.stage || '正在准备本次分析…'}}</span>
      <span class="ptip">已等待 {{elapsedSec}} 秒</span>
    </div>
    <!-- 自绘往返条：Element Plus 的 indeterminate 在部分版本不存在，不必赌组件行为 -->
    <div class="ptrack"><div class="pfill"/></div>
    <div class="pfoot">
      <span>{{depthHint}}</span>
      <span v-if="elapsedSec>=60" class="pwarn">耗时偏长时建议改用「异步提交」，不必把浏览器一直挂在这里。</span>
    </div>
  </div>

  <el-row :gutter="16">
    <el-col :span="9">
      <el-card><template #header>发起分析</template>
        <el-form label-width="86px">
          <el-form-item label="企业ID"><el-input-number v-model="form.companyId" :min="1"/></el-form-item>
          <el-form-item label="问题"><el-input type="textarea" :rows="6" v-model="form.question" placeholder="例如：为什么最近客户流失风险升高？应该采取哪些措施？"/></el-form-item>
          <el-form-item label="RAG条数"><el-input-number v-model="form.topK" :min="1" :max="10"/></el-form-item>
          <el-form-item label="会话续聊">
            <div class="row">
              <el-select v-model="form.sessionId" placeholder="新建会话" clearable style="flex:1">
                <el-option v-for="c in conversations" :key="c.id" :label="(c.title||'会话')+' #'+c.id" :value="c.id"/>
              </el-select>
              <el-button v-if="form.sessionId" type="danger" plain size="small" @click="removeConversation(form.sessionId)">删除</el-button>
            </div>
          </el-form-item>
          <el-form-item label="多智能体"><el-switch v-model="form.useMultiAgent"/><span class="tip">检索员→分析师→复核员</span></el-form-item>
          <!-- 联网不是"有开关就等于会联网"：路由只负责把 web_search 放进工具清单，
               模型完全可以不调它然后写一句「本次未联网核实」交差。这个开关会把它变成硬要求。 -->
          <el-form-item label="联网检索">
            <el-switch v-model="form.forceWeb"/>
            <span class="tip">强制联网核实并写入外部结论；提问里写「请联网核实」等措辞同样生效</span>
          </el-form-item>
          <!-- 分析深度：耗时的大头是模型写多少字，把选择权交给用的人 -->
          <el-form-item label="分析深度">
            <el-radio-group v-model="form.depth" size="small">
              <el-radio-button label="quick">快答</el-radio-button>
              <el-radio-button label="standard">标准</el-radio-button>
              <el-radio-button label="deep">深度</el-radio-button>
            </el-radio-group>
            <div class="tip depth-tip">{{depthHint}}</div>
          </el-form-item>
          <!-- 评估口径必须让用户选得明白：完整回归会真实调用大模型（每条端到端一次完整推理），
               快速档零 token。原先固定 full 且界面上没有入口，用户点一下就在消耗自己的模型额度。 -->
          <el-form-item label="评估模式">
            <el-radio-group v-model="evalMode" size="small">
              <el-radio-button label="fast">快速校验（零 token）</el-radio-button>
              <el-radio-button label="full">完整回归（调用大模型）</el-radio-button>
            </el-radio-group>
            <div class="tip depth-tip">
              <template v-if="evalMode==='fast'">只跑确定性用例（检索/护栏/引用/路由），不调用大模型，整轮约 1 分钟。</template>
              <template v-else>额外跑真实端到端用例：真实取数 + 真实推理成稿，会消耗模型额度，整轮数分钟。</template>
            </div>
          </el-form-item>
          <div class="btns">
            <el-button type="primary" :loading="loading" :disabled="ai.running" @click="analyze">同步分析</el-button>
            <el-button type="success" :disabled="ai.running" @click="startStream">流式(SSE)</el-button>
            <el-button v-if="ai.running" type="danger" @click="ai.cancel()">取消</el-button>
            <el-button :disabled="ai.running" @click="analyzeAsync">异步提交</el-button>
            <el-button type="warning" plain :loading="evalRunning" :disabled="ai.running||evalRunning" @click="evaluate">评估回归</el-button>
            <el-button plain :loading="diagLoading" @click="diagnose">模型自检</el-button>
          </div>
        </el-form>
      </el-card>

      <el-card style="margin-top:16px" v-if="evalReport||evalRunning||evalError"><template #header>
        <div class="cardhead">
          <span>评估与回归集</span>
          <span class="tip" v-if="evalReport">{{evalReport.total}} 条用例 · 并行执行 · 最慢单例 {{(evalReport.wallClockHintMs/1000).toFixed(1)}}s（串行合计 {{(evalReport.totalCaseMs/1000).toFixed(1)}}s）</span>
        </div>
      </template>
        <div v-if="evalRunning" class="evalprog">
          <el-progress :percentage="evalPercent" :stroke-width="16" :text-inside="true" status="warning"/>
          <div class="hint">
            已完成 {{evalStatus.done||0}} / {{evalStatus.total||0}} 条用例。
            <template v-if="evalMode!=='fast'">含真实大模型端到端分析（每条一次完整推理），整轮可达数分钟；</template>
            <template v-else>确定性校验，零 token，整轮约 1 分钟；</template>
            期间可切到其他模块，回来仍是这个进度。
          </div>
        </div>
        <el-alert v-if="evalError" :title="'回归执行失败：'+evalError" type="error" show-icon :closable="false"/>
        <template v-if="evalReport">
          <el-alert :title="'通过 '+evalReport.passed+' / '+evalReport.total" :type="evalReport.passed===evalReport.total?'success':'warning'" :closable="false"/>
          <div class="catline">
            <el-tag v-for="(v,k) in evalReport.byCategory" :key="k" size="small" :type="v.split('/')[0]===v.split('/')[1]?'success':'danger'" effect="plain">{{k}} {{v}}</el-tag>
            <el-tag size="small" :type="evalReport.webSearchEnabled?'info':'danger'" effect="plain">联网检索 {{evalReport.webSearchEnabled?'已启用':'未启用'}}</el-tag>
          </div>
          <!-- 回归基线对比：这才是「回归」，而不是单次体检 -->
          <div v-if="evalReport.comparison" class="baseline">
            <div class="brow">
              <el-tag size="small" effect="dark" :type="cmpType(evalReport.comparison.verdict)">{{cmpLabel(evalReport.comparison.verdict)}}</el-tag>
              <span>通过率 {{evalReport.passRate}}% <span class="tip">基线 {{evalReport.comparison.baselinePassRate}}%（{{fmtDelta(evalReport.comparison.passRateDelta)}} pp）</span></span>
              <!-- 耗时只在同一口径下才比：基线可能是 fast-only（不调模型），本轮是 full/live（真调模型），
                   两者直接相减会得出「变慢 30 倍」这种把人吓一跳、又没有意义的结论。 -->
              <span v-if="evalReport.comparison.sameMode!==false">耗时 {{(evalReport.wallClockMs/1000).toFixed(1)}}s <span class="tip">基线 {{(evalReport.comparison.baselineWallClockMs/1000).toFixed(1)}}s（{{fmtDelta(evalReport.comparison.wallClockDeltaPct)}}%）</span></span>
              <span v-else>耗时 {{(evalReport.wallClockMs/1000).toFixed(1)}}s <span class="tip warn-text">基线口径不同（{{evalReport.comparison.baselineMode}} vs {{evalReport.comparison.currentMode}}），不做耗时对比</span></span>
              <span class="tip">对比基线 #{{evalReport.comparison.baselineId}}</span>
            </div>
            <div class="bsec" v-if="evalReport.comparison.regressions&&evalReport.comparison.regressions.length">
              <div class="blabel danger">新增失败 {{evalReport.comparison.regressions.length}} 条（上轮通过、本轮失败）</div>
              <div class="bitem" v-for="r in evalReport.comparison.regressions" :key="r.id">
                <b>{{r.id}}</b> {{r.question}}
                <div class="treason" v-if="r.reasons&&r.reasons.length">{{r.reasons.join('；')}}</div>
              </div>
            </div>
            <div class="bsec" v-if="evalReport.comparison.recoveries&&evalReport.comparison.recoveries.length">
              <div class="blabel ok">已修复 {{evalReport.comparison.recoveries.length}} 条（上轮失败、本轮通过）</div>
              <div class="bitem" v-for="r in evalReport.comparison.recoveries" :key="r.id"><b>{{r.id}}</b> {{r.question}}</div>
            </div>
            <div class="bsec" v-if="evalReport.comparison.slowed&&evalReport.comparison.slowed.length">
              <div class="blabel warn">耗时显著增长 {{evalReport.comparison.slowed.length}} 条</div>
              <div class="bitem" v-for="r in evalReport.comparison.slowed" :key="r.id">
                <b>{{r.id}}</b> {{r.question}} <span class="tip">{{r.prevDurationMs}}ms → {{r.durationMs}}ms</span>
              </div>
            </div>
          </div>
          <div v-else class="hint">本次为首次留档，尚无历史基线可对比；再跑一次即可看到通过率与耗时趋势。</div>

          <!-- 这套回归是「确定性校验」：绝大多数用例不调大模型，真实耗时就是亚毫秒。
               所以耗时列显示 0 不是"没跑"，而是"本用例不需要推理"——这里如实标注，别再让人误会。 -->
          <div class="tip" style="margin-top:8px">
            <template v-if="evalReport.liveCases">
              本轮包含 <b>{{evalReport.liveCases}} 条真实端到端用例</b>（{{evalReport.mode==='full'?'确定性 + 真实大模型':'仅真实大模型'}}）：
              由 Agent 真实取数、真实调用大模型成稿，模型调用次数、成稿字数、档位与耗时全部来自真实执行。
              端到端合计 {{(evalReport.liveTotalMs/1000).toFixed(1)}}s，单条均值 {{(evalReport.liveAvgMs/1000).toFixed(1)}}s。
              其余 {{evalReport.total - evalReport.liveCases}} 条确定性用例不调用大模型，「模型调用」列显示「未调用」，
              合计约 {{((evalReport.wallClockMs - evalReport.liveTotalMs)/1000).toFixed(1)}}s，耗时按各自真实执行时间显示。
            </template>
            <template v-else>
              本轮为纯确定性回归：全部用例都不调用大模型，因此「模型调用」列显示「未调用」、耗时即真实执行时间。
            </template>
          </div>
          <el-table :data="evalReport.results" size="small" style="margin-top:8px" max-height="420">
            <el-table-column prop="id" label="用例" width="84"/>
            <el-table-column prop="category" label="类别" width="74"/>
            <el-table-column prop="companyId" label="企业" width="52"/>
            <el-table-column prop="question" label="问题" min-width="150" show-overflow-tooltip/>
            <el-table-column label="证据" width="76">
              <template #default="{row}">
                <span v-if="row.webHits!=null">{{row.webHits}} 网页</span>
                <span v-else-if="row.ragHits!=null">{{row.ragHits}} 文档</span>
                <span v-else-if="row.probeTool" class="ok">✓探针</span>
                <span v-else-if="row.citationWebCited!==null&&row.citationWebCited!==undefined">引用 {{row.citationWebCited}}</span>
                <span v-else>-</span>
              </template>
            </el-table-column>
            <el-table-column label="模型调用" width="96">
              <template #default="{row}">
                <!-- 「—」会被读成"数据缺失/功能坏了"，而这里其实是"本用例不调模型"。
                     写「未调用」把话说清楚，别再让人以为漏填了。 -->
                <span v-if="row.llmCalls===null||row.llmCalls===undefined" class="tip"
                      title="确定性校验：只验检索/护栏/引用等逻辑，不调用大模型，因此没有模型调用次数">未调用</span>
                <span v-else :class="row.singleRound?'ok':''">{{row.llmCalls}} 次<template v-if="row.singleRound"> ·单轮</template></span>
              </template>
            </el-table-column>
            <el-table-column label="成稿" width="82">
              <template #default="{row}">
                <span v-if="row.answerChars===null||row.answerChars===undefined" class="tip"
                      title="确定性校验不产出成稿；只有端到端用例才由大模型写出正文">不适用</span>
                <span v-else>{{row.answerChars}} 字</span>
              </template>
            </el-table-column>
            <el-table-column label="Top分" width="104">
              <template #default="{row}">
                <span v-if="row.ragTopScore===null||row.ragTopScore===undefined" class="tip">—</span>
                <span v-else :title="topScoreTip(row)">{{row.ragTopScore}}<span v-if="row.ragTopRawScore!==null&&row.ragTopRawScore!==undefined" class="tip"> / 原{{row.ragTopRawScore}}</span></span>
              </template>
            </el-table-column>
            <el-table-column label="耗时" width="92">
              <template #default="{row}">
                <!-- 这里曾经按 live 分叉：确定性用例一律显示「—」。
                     理由写的是"避免一片 0s 看着像故障"，实际效果是用户看到
                     「好多指标都没有」—— 隐藏真实数据比展示 0 更糟。
                     现在如实显示每条的耗时，口径差异交给表头 tooltip 讲。 -->
                <el-tooltip placement="top" :content="row.live
                  ? '端到端真实耗时（含取数 + 大模型成稿），档位 '+(row.depth||'默认')
                  : '确定性校验的真实执行耗时（不调用大模型，通常亚毫秒级）'">
                  <span :class="row.live?'':'tip'">{{fmtMs(row.durationMs)}}</span>
                </el-tooltip>
              </template>
            </el-table-column>
            <el-table-column label="档位" width="72">
              <template #default="{row}">
                <span v-if="row.live" :class="row.depth==='quick'?'ok':'tip'">{{depthLabel(row.depth)}}</span>
                <span v-else class="tip" title="确定性校验不受分析深度影响">—</span>
              </template>
            </el-table-column>
            <el-table-column label="结果" width="62">
              <template #default="{row}"><el-tag :type="row.passed?'success':'danger'" size="small">{{row.passed?'通过':'失败'}}</el-tag></template>
            </el-table-column>
            <el-table-column label="原因" width="58">
              <template #default="{row}"><el-button v-if="row.reasons&&row.reasons.length" link type="warning" size="small" @click="showReasons(row)">查看</el-button></template>
            </el-table-column>
          </el-table>

          <template v-if="evalHistory.length">
            <el-divider content-position="left">历史趋势（最近 {{evalHistory.length}} 次）</el-divider>
            <el-table :data="evalHistory" size="small" max-height="200">
              <el-table-column prop="id" label="报告" width="138"/>
              <el-table-column prop="generatedAt" label="时间" min-width="150"/>
              <el-table-column prop="model" label="底座" min-width="120" show-overflow-tooltip/>
              <el-table-column label="通过" width="110">
                <template #default="{row}">{{row.passed}} / {{row.total}}（{{row.passRate}}%）</template>
              </el-table-column>
              <el-table-column label="耗时" width="76">
                <template #default="{row}">{{(row.wallClockMs/1000).toFixed(1)}}s</template>
              </el-table-column>
            </el-table>
          </template>
        </template>
      </el-card>

      <el-card style="margin-top:16px" v-if="asyncTaskId"><template #header>异步任务</template>
        <div>任务ID：{{asyncTaskId}} 状态：<el-tag size="small">{{asyncStatus.status}}</el-tag></div>
        <div v-if="asyncStatus.error" class="err">{{asyncStatus.error}}</div>
      </el-card>
    </el-col>

    <el-col :span="15">
      <el-card><template #header>
        <div class="cardhead">
          <span>分析结果</span>
          <span class="tip" v-if="view">
            #{{view.analysisId||'-'}} · {{view.provider}}
            <template v-if="view.source==='stream'"> · 流式耗时 {{(ai.elapsedMs/1000).toFixed(1)}}s</template>
          </span>
        </div>
      </template>
        <el-empty v-if="!view && !ai.running" description="提交问题后显示分析结果"/>
        <template v-else>
          <!-- 先行事实卡：工具取数一结束就出，不等正文写完 -->
          <el-card v-if="metaInfo" shadow="never" class="metacard">
            <template #header>
              <div class="cardhead">
                <span>先行事实</span>
                <span class="tip">工具取数已完成即刻判定 · 早于正文生成</span>
              </div>
            </template>
            <div class="metarow">
              <span class="mlabel">风险等级</span>
              <el-tag size="small" effect="dark" :type="metaInfo.riskLevel==='PENDING'?'info':riskType(metaInfo.riskLevel)">
                {{metaInfo.riskLevel==='PENDING'?'等级待定（证据不足）':metaInfo.riskLevel}}
              </el-tag>
              <span class="mlabel">证据依据</span>
              <span class="mmixes">
                <el-tag size="small" effect="plain" v-if="riskStats.riskHigh!==undefined">高风险事件 {{riskStats.riskHigh}}</el-tag>
                <el-tag size="small" effect="plain" v-if="riskStats.riskMedium!==undefined">中风险 {{riskStats.riskMedium}}</el-tag>
                <el-tag size="small" effect="plain" v-if="riskStats.complaintSla!==undefined">超SLA投诉 {{riskStats.complaintSla}}</el-tag>
                <el-tag size="small" effect="plain" v-if="riskStats.complaintChurnHigh!==undefined">高流失投诉 {{riskStats.complaintChurnHigh}}</el-tag>
              </span>
            </div>
            <div class="metarow">
              <span class="mlabel">来源构成</span>
              <span class="mmixes">
                <el-tag size="small" type="info" effect="plain">内部取数 {{metaSplit.internalCalls||0}} 项</el-tag>
                <el-tag size="small" type="success" effect="plain">知识库 {{metaSplit.kbSources||0}} 篇</el-tag>
                <el-tag size="small" type="warning" effect="plain">网页 {{metaSplit.webSources||0}} 条</el-tag>
              </span>
              <span class="tip" v-if="metaSplit.internalTools&&metaSplit.internalTools.length">
                {{metaSplit.internalTools.join('、')}}
              </span>
            </div>
            <div class="tip" v-if="metaInfo.rag">检索流水线：{{metaInfo.rag.mode||'-'}} · 查询变体 {{metaInfo.rag.queryVariants||1}} · 候选 {{metaInfo.rag.candidates||0}}</div>
          </el-card>

          <el-descriptions :column="3" border size="small">
            <el-descriptions-item label="Provider">{{view?.provider||'-'}}</el-descriptions-item>
            <el-descriptions-item label="置信度">{{view?.confidence||'-'}}</el-descriptions-item>
            <el-descriptions-item label="风险等级">
              <el-tag :type="riskType(structured?.risk_level)" size="small">{{structured?.risk_level||'UNKNOWN'}}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="需人工复核">
              <el-tag :type="structured?.needs_human_review?'danger':'info'" size="small">{{structured?.needs_human_review?'是':'否'}}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="来源构成" :span="2">
              <span class="mixes">
                <el-tag size="small" type="info" effect="plain">内部取数 {{split.internal_tool_calls||0}} 项</el-tag>
                <el-tag size="small" type="success" effect="plain">知识库 {{kbSources.length}} 篇</el-tag>
                <el-tag size="small" type="warning" effect="plain">网页 {{webSources.length}} 条</el-tag>
              </span>
            </el-descriptions-item>
            <!-- 追溯信息：结论被质疑时，凭追溯号能调出当时的完整决策过程 -->
            <el-descriptions-item label="追溯号">
              <span class="tip">{{view?.traceId||'-'}}</span>
            </el-descriptions-item>
            <el-descriptions-item label="执行耗时">{{view?.durationMs?view.durationMs+' ms':'-'}}</el-descriptions-item>
            <el-descriptions-item label="模型/工具调用">
              {{view?.llmCalls!=null?view.llmCalls:'-'}} 次 / {{view?.toolCalls!=null?view.toolCalls:'-'}} 次
            </el-descriptions-item>
            <el-descriptions-item label="降级状态" :span="3">
              <el-tag size="small" :type="degradeTagType(view?.degradeLevel)" effect="plain">
                {{degradeText(view?.degradeLevel)}}
              </el-tag>
              <span class="tip" v-if="view?.degradeLevel && view.degradeLevel!=='NONE' && view?.degradeReason">
                &nbsp;{{view.degradeReason}}
              </span>
            </el-descriptions-item>
          </el-descriptions>

          <!-- 命中结果缓存时必须说清楚：用户问了同一个问题却看不到"正在输出"的过程，
               如果不点破，就会被理解成"这次没结果"。 -->
          <el-alert v-if="structured?.cached" type="info" show-icon :closable="false" style="margin-top:10px"
                    :title="'本次直接复用了 #'+(structured.cachedFrom||'?')+' 的结论（未重新调用模型）'"
                    description="这个问题你刚才问过：为避免重复消耗额度，直接返回了上次那条答案；本次已单独留下分析记录与追溯号，工具调用与证据仍是那次的结果。想重新跑一次，换个说法或稍后（10 分钟后）再问即可。"/>

          <el-alert v-if="structured?.degraded" type="warning" show-icon :closable="false" style="margin-top:10px"
                    title="降级报告：模型服务不可用"
                    :description="'原因：'+(structured.degraded_reason||'未知')+'。本报告由系统规则基于本地真实数据生成，未经大模型推理、未联网核实，结论仅供内部初筛参考；模型服务恢复后建议对同一问题重新分析。'"/>

          <div v-if="groundingCheck" class="groundbar">
            <el-tag size="small" :type="groundingCheck.ok?'success':'danger'" effect="dark">
              引用核对：{{structured.grounding_summary}}
            </el-tag>
            <span class="tip" v-if="groundingCheck.webUnused&&groundingCheck.webUnused.length">
              未引用的来源序号：{{groundingCheck.webUnused.join('、')}}
            </span>
          </div>

          <!-- 本次对照过的历史记忆：刻意与「参考来源」分开显示 —— 记忆不是证据、没有编号，
               否则用户会把它当成这次结论的依据。 -->
          <div v-if="longTerm && longTerm.hits" class="membar">
            <el-tag size="small" type="info" effect="plain">
              历史记忆对照：命中 {{longTerm.hits}} 条 · 注入 {{longTerm.injectedChars}} 字
            </el-tag>
            <span class="tip">记忆只用于判断「这次的说法与以前是否一致」，其中的数字须重新核实，不作为本次证据。</span>
            <el-collapse class="memhits">
              <el-collapse-item title="查看本次对照的历史记忆" name="lt">
                <div class="memrow" v-for="m in longTerm.items" :key="m.id">
                  <span class="mtag">{{kindLabel(m.kind)}}</span>
                  <span class="mtext">{{m.content}}</span>
                  <span class="tip">记忆#{{m.id}} · 分 {{m.score}} · {{m.ageDays}} 天前<span v-if="m.sourceAnalysisId"> · 出自分析 #{{m.sourceAnalysisId}}</span></span>
                </div>
              </el-collapse-item>
            </el-collapse>
          </div>

          <!-- 正文区域在运行前后保持同一个 DOM，不在 done 到达时被结构化卡片替换。 -->
          <section class="report-shell">
            <div class="ghead"><span class="badge ib">报告</span>综合风险报告</div>
            <div class="answer live">{{displayAnswer||'正在汇总证据并生成最终报告…'}}<span v-if="ai.running" class="cursor">▌</span></div>
          </section>

          <!-- 最终结构化结论作为补充视图追加在正文后，不再替换流式正文。 -->
          <template v-if="isSplit">
            <section class="group internal">
              <div class="ghead"><span class="badge ib">内部</span>一、结论（基于内部知识库与经营数据）</div>
              <div class="answer">{{structured.conclusion_internal||'—'}}</div>
              <div class="gsub">建议动作 · 基于内部证据</div>
              <ol class="recs" v-if="recsInternal.length">
                <li v-for="(r,i) in recsInternal" :key="i">
                  <span>{{r.action}}</span>
                  <span class="basis" v-if="r.basis"> —— 依据：{{r.basis}}</span>
                  <a v-for="(rf,j) in r.refs" :key="j" class="ref" @click="focusRef(rf)">{{refLabel(rf)}}</a>
                  <span class="noref" v-if="!r.refs.length">（未标注内部依据，建议人工核对）</span>
                </li>
              </ol>
              <div v-else class="hint">本次未给出基于内部证据的动作。</div>
            </section>

            <section class="group web">
              <div class="ghead"><span class="badge wb">外部</span>二、结论（基于外部公开资料）</div>
              <div class="answer">{{structured.conclusion_web||'—'}}</div>
              <div class="gsub">建议动作 · 基于外部资料（据公开资料）</div>
              <ol class="recs" v-if="recsWeb.length">
                <li v-for="(r,i) in recsWeb" :key="i">
                  <span>{{r.action}}</span>
                  <span class="basis" v-if="r.basis"> —— 依据：{{r.basis}}</span>
                  <a v-for="(rf,j) in r.refs" :key="j" class="ref" @click="focusRef(rf)">{{refLabel(rf)}}</a>
                  <span class="noref" v-if="!r.refs.length">（未标注外部依据，建议人工核对）</span>
                </li>
              </ol>
              <div v-else class="hint">本次未给出基于外部资料的动作。</div>
            </section>

            <h4>三、不确定性 / 需人工确认</h4><div class="answer">{{structured.uncertainties||'—'}}</div>
          </template>

          <!-- 旧版结构（未做来源分层）保持兼容 -->
          <template v-else-if="structured">
            <el-alert type="info" :closable="false" show-icon style="margin-top:10px"
                      title="该记录由旧版结构生成，未做来源分层。重新分析即可获得「内部 / 外部」分开的结论与建议。"/>
            <h4>结论</h4><div class="answer">{{structured.conclusion}}</div>
            <h4>建议动作</h4>
            <ol class="recs"><li v-for="(r,i) in toArr(structured.recommendations)" :key="i">{{r}}</li></ol>
            <h4>不确定性 / 需人工确认</h4><div class="answer">{{structured.uncertainties}}</div>
          </template>

          <template v-if="agentTeam.length">
            <h4>本次参与的智能体<span class="tip">由提问意图自动路由 · 未选中的专员本轮不占用上下文</span></h4>
            <div class="agentteam">
              <el-tooltip v-for="a in agentTeam" :key="a.id" placement="top" :content="a.duty">
                <el-tag size="small" :type="a.core?'info':'success'" effect="plain">
                  {{a.name}}<span v-if="a.core" class="tip"> · 常驻</span>
                </el-tag>
              </el-tooltip>
            </div>
          </template>

          <template v-if="webSources.length">
            <h4>参考来源（网页）<span class="tip">共 {{webSources.length}} 条，均为正文实际引用过的来源</span></h4>
            <div class="srclist">
              <div class="src" v-for="s in webSources" :key="s.url" :id="'web-src-'+(s.index||0)"
                   :class="{hot:hot.kind==='web'&&String(hot.key)===String(s.index)}">
                <span class="idx">[{{s.index}}]</span>
                <a :href="s.url" target="_blank" rel="noopener noreferrer">{{s.title||s.url}}</a>
                <div class="url">{{s.url}}<span v-if="s.siteName" class="site"> · {{s.siteName}}</span></div>
              </div>
            </div>
          </template>

          <template v-if="kbSources.length">
            <h4>参考来源（知识库）<span class="tip">共 {{kbSources.length}} 篇，引用写作「来源ID=x」</span></h4>
            <div class="srclist">
              <div class="src" v-for="s in kbSources" :key="s.documentId" :id="'kb-src-'+s.documentId"
                   :class="{hot:hot.kind==='kb'&&String(hot.key)===String(s.documentId)}">
                <span class="idx">ID</span>
                <span class="kbtitle">{{s.title||('文档 #'+s.documentId)}}</span>
                <div class="url">来源ID={{s.documentId}}<span v-if="s.score!==undefined" class="site"> · 相关度 {{s.score}}</span></div>
              </div>
            </div>
          </template>

          <template v-if="webTrails.length || webUnusedSources.length">
            <el-collapse style="margin-top:6px">
              <el-collapse-item name="wtrail">
                <template #title>
                  <span>联网检索经过（{{webTrails.length}} 次调用<el-text type="info" size="small"> · 另有 {{webUnusedSources.length}} 条未被引用的结果</el-text>）</span>
                </template>
                <div v-if="webTrails.length" class="wtrail">
                  <div class="wt" v-for="(t,i) in webTrails" :key="i">
                    <div><span class="tip">第 {{i+1}} 次</span> 原始词：<b>{{t.rawQuery}}</b></div>
                    <div v-if="t.usedQuery && t.usedQuery!==t.rawQuery">清洗后用于检索：<b>{{t.usedQuery}}</b></div>
                    <div v-if="!t.usedQuery"><b>已驳回，未发起检索</b></div>
                    <div class="tip">保留 {{t.kept||0}} 条 / 丢弃 {{t.dropped||0}} 条<span v-if="t.note"> · {{t.note}}</span></div>
                  </div>
                </div>
                <div v-if="webUnusedSources.length" style="margin-top:8px">
                  <div class="tip">以下结果被检索到但正文未引用（不进"参考来源"，避免把"搜过"当成"依据"）：</div>
                  <div class="srclist">
                    <div class="src" v-for="s in webUnusedSources" :key="'u'+s.index">
                      <span class="idx">[{{s.index}}]</span>
                      <a :href="s.url" target="_blank" rel="noopener noreferrer">{{s.title||s.url}}</a>
                      <div class="url">{{s.url}}</div>
                    </div>
                  </div>
                </div>
              </el-collapse-item>
            </el-collapse>
          </template>

          <div class="hint">结论与建议在上方；来源、证据溯源、工具轨迹与原始 JSON 默认折叠，需要核对时展开。</div>
          <el-collapse>
            <el-collapse-item title="证据溯源（工具 / 来源类型 / 耗时 / 重试）" name="1">
              <el-table :data="evidenceList" size="small" v-if="evidenceList.length">
                <el-table-column prop="tool" label="工具" width="150"/>
                <el-table-column prop="sourceType" label="来源" width="96">
                  <template #default="{row}">
                    <el-tag size="small" :type="row.sourceType==='web'?'warning':(row.sourceType==='knowledge'?'success':'info')" effect="plain">
                      {{row.sourceType==='web'?'联网':(row.sourceType==='knowledge'?'知识库':'内部数据')}}
                    </el-tag>
                  </template>
                </el-table-column>
                <el-table-column label="条数" width="60">
                  <template #default="{row}">{{row.sources?row.sources.length:'-'}}</template>
                </el-table-column>
                <el-table-column label="字符" width="70">
                  <template #default="{row}">{{row.chars!==undefined?row.chars:'-'}}</template>
                </el-table-column>
                <el-table-column label="尝试" width="60">
                  <template #default="{row}">
                    <span v-if="row.cached" class="ok">复用</span>
                    <span v-else>{{row.attempts||1}}</span>
                  </template>
                </el-table-column>
                <el-table-column prop="preview" label="摘要"/>
              </el-table>
              <div v-else class="hint">无</div>
            </el-collapse-item>
            <el-collapse-item title="工具调用轨迹" name="2"><pre>{{traceList.join(' → ')||'（无）'}}</pre></el-collapse-item>
            <el-collapse-item title="原始答案（正文纯文本）" name="3"><pre>{{displayAnswer||'（无）'}}</pre></el-collapse-item>
            <el-collapse-item title="原始答案 JSON（answer_json）" name="4">
              <div class="copybar"><el-button size="small" plain @click="copy(rawAnswerJson)">复制</el-button></div>
              <pre v-if="rawAnswerJson">{{rawAnswerJson}}</pre>
              <div v-else class="hint">这条记录没有 answerJson。同步/流式分析会写入结构化 JSON；若为本地规则兜底模式（未启用外部大模型）则不产生。</div>
            </el-collapse-item>
            <el-collapse-item title="原始证据 JSON（evidence_json）" name="5">
              <div class="copybar"><el-button size="small" plain @click="copy(rawEvidenceJson)">复制</el-button></div>
              <pre v-if="rawEvidenceJson">{{rawEvidenceJson}}</pre>
              <div v-else class="hint">这条记录没有证据 JSON。</div>
            </el-collapse-item>
          </el-collapse>
        </template>
      </el-card>
    </el-col>
  </el-row>

  <el-card style="margin-top:16px"><template #header>
    <div class="cardhead">
      <span>历史分析</span>
      <span>
        <el-button size="small" @click="history">刷新</el-button>
        <el-button size="small" type="danger" plain :disabled="!historyRows.length" @click="clearHistory">清空历史</el-button>
      </span>
    </div>
  </template>
    <el-table :data="historyRows">
      <el-table-column prop="createdAt" label="时间" width="170"/>
      <el-table-column prop="id" label="ID" width="70"/>
      <el-table-column prop="companyId" label="企业ID" width="76"/>
      <el-table-column prop="question" label="问题" min-width="220" show-overflow-tooltip/>
      <el-table-column prop="provider" label="Provider" width="150"/>
      <el-table-column prop="confidence" label="置信度" width="72"/>
      <el-table-column prop="answer" label="结果摘要" min-width="220" show-overflow-tooltip/>
      <el-table-column label="操作" width="196" fixed="right">
        <template #default="{row}">
          <el-button link type="primary" size="small" @click="viewDetail(row)">查看</el-button>
          <el-button link type="info" size="small" @click="gotoAudit(row)">追溯与导出</el-button>
          <el-button link type="danger" size="small" @click="removeHistory(row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>
  </el-card>
</div></template>

<script setup>
import {ref, computed, onMounted, onUnmounted, watch} from 'vue'
import {useRouter} from 'vue-router'
import {ElMessage, ElMessageBox} from 'element-plus'
import {api} from '../api'
import {useAiTaskStore} from '../stores/aiTask'

const ai = useAiTaskStore()
const router = useRouter()

/** 跳到审计页并把这条分析直接选中：结论是怎么得出的、依据了什么，都在那边。 */
function gotoAudit(row){
  router.push({path:'/audit', query:{analysisId: row.id, companyId: row.companyId || form.value.companyId}})
}

const form = ref({companyId:1, question:'为什么最近客户流失风险升高？请结合经营指标、投诉、竞品和知识库给出证据与建议，并联网对照行业做法。', topK:8, sessionId:null, useMultiAgent:false, depth:'standard', forceWeb:false})
/**
 * 三档深度的取舍说清楚：等待时间几乎全部由「模型写了多少字」决定，
 * 而且历史上比这些更长的等待会被流式连接直接掐掉、界面显示成"没有结果"，
 * 所以默认给 standard 而不是 deep。
 */
const DEPTH_HINTS = {
  quick: '结论优先：约 700 字、工具最少，通常一分钟内出结论。适合日常快速核验。',
  standard: '默认：约 1500 字，覆盖大多数风控问询。',
  deep: '正式报告：允许更多轮取证与联网，字数与时延上限最高，请配合「异步提交」使用。'
}
const depthHint = computed(() => DEPTH_HINTS[form.value.depth] || DEPTH_HINTS.standard)

/** 已耗时（秒），用于判断是否该提示用户换异步通道。 */
const elapsedSec = computed(() => Math.round((ai.elapsedMs || 0) / 1000))
const loading = ref(false)
const record = ref(null)            // 同步/异步/历史详情的 AiAnalysis 实体
const historyRows = ref([]), conversations = ref([])
const evalReport = ref(null)
const asyncTaskId = ref(''), asyncStatus = ref({status:''}), pollTimer = ref(null)

// 评估回归：异步任务 + 进度（整轮含真实联网检索，远超浏览器 30s 超时）
// 评估模式：full=确定性+真实大模型 / live=仅真实大模型 / fast=仅确定性（零 token）
// 默认 full：用户要看的是「真正的性能」，只有真实调用的那一类才测得出来。
const evalMode = ref('full')
const evalTaskId = ref(''), evalStatus = ref({status:'', done:0, total:0, percent:0}), evalTimer = ref(null), evalError = ref('')
const evalHistory = ref([])         // 历史报告摘要（通过率 / 耗时趋势）

/** SSE meta 事件：正仍未写出时先到达的「风险等级 / 来源构成」。 */
const metaInfo = computed(()=> ai.meta)
const metaSplit = computed(()=> (metaInfo.value && metaInfo.value.sourceSplit) || {})
const riskStats = computed(()=> (metaInfo.value && metaInfo.value.riskEvidence) || {})

function loadEvalHistory(){
  api.aiEvalReports().then(r=>{ evalHistory.value = (r.data||r||[]).slice(0,10) }).catch(()=>{})
}

function fmtDelta(v){
  if(v===null||v===undefined) return '-'
  return (v>0?'+':'') + v
}
/** 耗时格式化：确定性用例常是 0~5ms，端到端是十几秒，同一列要都能读。 */
function fmtMs(ms){
  if(ms===null||ms===undefined) return '—'
  return ms<1000 ? ms+'ms' : (ms/1000).toFixed(1)+'s'
}
/** 深度档位的中文名：报告里显示英文 key 会让人以为是内部字段。 */
function depthLabel(d){
  return {quick:'快答',standard:'标准',deep:'深度'}[d] || d || '默认'
}
function cmpType(verdict){
  return verdict==='REGRESSED' ? 'danger' : verdict==='IMPROVED' ? 'success' : 'info'
}
function cmpLabel(verdict){
  return verdict==='REGRESSED' ? '相比基线有回归' : verdict==='IMPROVED' ? '相比基线有改善' : '与基线持平'
}
const evalRunning = computed(()=> evalStatus.value.status==='RUNNING')
const evalPercent = computed(()=> evalStatus.value.percent || 0)
const hot = ref({kind:'', key:''})   // 点击引用后高亮的来源
const diag = ref(null), diagLoading = ref(false)
const ragStatus = ref(null), ragBusy = ref(false)
const RAG_LABELS = {knowledge:'知识库文档',metric:'经营指标',event:'风险事件',complaint:'客户投诉',competitor:'竞品',company:'企业档案',rule:'风控规则',aggregate:'聚合统计'}
const ragTypes = computed(()=> Object.entries(ragStatus.value?.byType||{}).map(([k,v])=>({k,label:RAG_LABELS[k]||k,n:(v&&v.chunks)||0})))

/** 模型服务自检：把「Key 无效 / 账户欠费 / 模型无权限」直接说清楚。 */
async function diagnose(){
  diagLoading.value = true
  try{
    const r = await api.aiDiagnose()
    diag.value = r.data || r
    if(diag.value){
      cfgForm.value.baseUrl = diag.value.baseUrl || cfgForm.value.baseUrl
      cfgForm.value.model = diag.value.model || cfgForm.value.model
      // 自检响应里已经带了模型可用列表 / 自动降级记录 / 后台消耗授权，直接同步，省一次请求
      applyModelsMeta(diag.value.models)
    }
    if(diag.value && diag.value.ok) ElMessage.success('模型服务连接正常')
    else ElMessage.warning('模型服务不可用，详见上方诊断')
  }catch(e){
    diag.value = {ok:false, issue:'自检请求失败：'+(e?.message||e)}
  }finally{ diagLoading.value = false }
  loadRagStatus()
}

/** 本地资料库状态：写了多少片、各类来源分别多少、有没有欠着没入的。 */
async function loadRagStatus(){
  try{ const r = await api.aiRagStatus(); ragStatus.value = r.data || r }
  catch(e){ /* 索引状态拿不到不影响主流程，静默即可 */ }
}
async function reindexNow(){
  ragBusy.value = true
  try{
    const r = await api.aiRagReindex(null)
    const d = r.data || r
    ElMessage.success('回填完成：'+(d.chunks||0)+' 条语料，耗时 '+(d.costMs||0)+'ms')
    await loadRagStatus()
  }catch(e){ ElMessage.error('回填失败：'+(e?.message||e)) }
  finally{ ragBusy.value = false }
}

// ---------------- 长期记忆（跨会话）：清单 / 回填 / 召回预演 ----------------
// 与「本地资料库」的区别（改这块前先读）：资料库是证据，占 [n] / 来源ID=x 编号；
// 记忆只是历史快照，只做一致性对照，**不占编号**。后端已把这条规则写进提示词，
// 前端也必须把两者分开呈现，否则用户会把"记忆里的旧数字"当成"本次结论的依据"。
const memStats = ref(null), memItems = ref([]), memBusy = ref(false)
const memRebuildBusy = ref(false), memProbeBusy = ref(false), memProbe = ref(null)
const memIncludeSuperseded = ref(false)
const MEM_KIND = {
  FACT:       { label:'事实', tag:'warning' },   // 带数字的风险断言
  EPISODE:    { label:'经过', tag:'info' },      // 高风险 / 降级处置经过
  PREFERENCE: { label:'偏好', tag:'success' }    // 用户长期偏好
}
function kindLabel(k){ return (MEM_KIND[k]||{}).label || k || '-' }
function kindTag(k){ return (MEM_KIND[k]||{}).tag || 'info' }
const memKindLine = computed(()=> Object.entries(memStats.value?.byKind||{})
  .map(([k,v])=> kindLabel(k)+' '+v).join('　·　'))

/** 清单 + 统计。取不到不挡主流程（记忆是增强项，不是主链路的必经环节）。 */
async function loadMemory(){
  memBusy.value = true
  try{
    const r = await api.memoryList({companyId: form.value.companyId, includeSuperseded: memIncludeSuperseded.value})
    const d = r.data || r
    memStats.value = d.stats || null
    memItems.value = d.items || []
  }catch(e){ /* 静默：记忆面板拿不到不影响分析与评估 */ }
  finally{ memBusy.value = false }
}
/** 从历史分析回填：功能上线前积累的分析记录也能变成记忆，零 token。 */
async function rebuildMemory(){
  memRebuildBusy.value = true
  try{
    const r = await api.memoryRebuild({companyId: form.value.companyId, limit: 50})
    const d = r.data || r
    ElMessage.success('回填完成：扫描 '+(d.scanned||0)+' 条历史分析，新增 '+(d.written||0)
      +' 条、强化 '+(d.reinforced||0)+' 条、取代 '+(d.superseded||0)+' 条')
    await loadMemory()
  }catch(e){ ElMessage.error('回填失败：'+(e?.message||e)) }
  finally{ memRebuildBusy.value = false }
}
/** 召回预演：用当前问题试一次「这次会想起什么」，回答"为什么这次没想起那件事"。 */
async function probeMemory(){
  memProbeBusy.value = true
  try{
    const q = (form.value.question||'').trim()
    const r = await api.memoryProbe({q, companyId: form.value.companyId})
    const d = r.data || r
    memProbe.value = { q, hits: d.hits||[], contextChars: d.contextChars||0,
      minScore: (d.diagnostics && d.diagnostics.minScore!==undefined) ? d.diagnostics.minScore : '-' }
  }catch(e){ ElMessage.error('召回预演失败：'+(e?.message||e)) }
  finally{ memProbeBusy.value = false }
}
/** 「忘掉」= 物理删除（合规诉求：用户说忘就必须真的忘，置失效会把内容留在库里）。 */
async function removeMemory(row){
  try{
    await ElMessageBox.confirm('将从库中物理删除「'+(row.content||'').slice(0, 40)+'…」，删除后不可恢复。确定吗？',
      '删除长期记忆', {type:'warning'})
  }catch{ return }
  try{ await api.deleteMemory(row.id); ElMessage.success('已删除 记忆#'+row.id); await loadMemory() }
  catch(e){ ElMessage.error('删除失败：'+(e?.message||e)) }
}
async function clearMemory(){
  try{
    await ElMessageBox.confirm('将清空企业 #'+form.value.companyId+' 的全部长期记忆（物理删除，不可恢复）。确定吗？',
      '清空长期记忆', {type:'warning'})
  }catch{ return }
  try{
    const r = await api.clearMemory({companyId: form.value.companyId})
    ElMessage.success('已清空 '+(r.data||r||0)+' 条')
    await loadMemory(); memProbe.value = null
  }catch(e){ ElMessage.error('清空失败：'+(e?.message||e)) }
}

const cfgForm = ref({ baseUrl:'', model:'', key:'' }), keyLoading = ref(false)

// ---------------- 多模型：可用列表 / 自动降级 / 后台消耗授权 ----------------
// 目的：让「用哪个模型」不再是配置里写死的一个字符串。可用列表来自只读的 GET /v3/models，
// 不消耗 token；自动降级由后端完成，前端只负责把「当前是谁、换过没有、哪些已失效」说清楚。
const modelsMeta = ref({}), modelsLoading = ref(false), adoptLoading = ref(false)
const pickedModel = ref(''), autoConsume = ref(false), autoConsumeLoading = ref(false)
// 下拉里用「推荐」那一份（后端已剔掉向量化/视频等不能当对话用的，并按「新且强」排序）；
// 拉不到时才退回全量列表，仍然可选。
const availableModels = computed(()=> {
  const d = modelsMeta.value || {}
  const rec = d.recommended || []
  if (rec.length) return rec
  return d.available || []
})
const currentModel = computed(()=> modelsMeta.value.current || diag.value?.model || '')
const unavailableList = computed(()=> Object.entries(modelsMeta.value.unavailable || {})
  .map(([k,v])=> k + '（' + (v||'不可用') + '）'))

function applyModelsMeta(d){
  if(!d) return
  modelsMeta.value = d
  if(typeof d.autoConsume === 'boolean') autoConsume.value = d.autoConsume
  if(d.current) pickedModel.value = d.current
}

/** 拉取账号可用模型列表（force=true 时强制重新拉一次；只读列举，不花 token）。 */
async function refreshModels(force){
  modelsLoading.value = true
  try{
    const r = await api.aiModels(force === true)
    applyModelsMeta(r.data || r)
  }catch(e){
    ElMessage.warning('拉取可用模型失败：'+(e?.response?.data?.message||e?.message||e))
  }finally{ modelsLoading.value = false }
}

/** 从下拉里选了一个模型 → 直接热更新并自检。 */
function onPickModel(v){
  if(!v) return
  cfgForm.value.model = v
  applyKey()
}

/** 让平台自己挑一个「现在真能调通」的模型（会真实发一次 1 token 探测）。 */
async function adoptModel(){
  adoptLoading.value = true
  try{
    const r = await api.aiAdoptModel()
    const d = r.data || r
    applyModelsMeta(d)
    diag.value = Object.assign({}, diag.value, { ok:d.ok, issue:d.issue, model:d.current, models:d })
    if(d.ok){ cfgForm.value.model = d.current || ''; ElMessage.success('已切到可用模型：'+d.current) }
    else ElMessage.warning('没找到可用模型：'+(d.issue||'未知原因'))
  }catch(e){ ElMessage.error('操作失败：'+(e?.response?.data?.message||e?.message||e)) }
  finally{ adoptLoading.value = false }
}

/** 后台自动研判开关：默认关闭 = 未经你主动触发就不消耗额度。 */
async function applyAutoConsume(v){
  autoConsumeLoading.value = true
  try{
    const r = await api.aiAutoConsume(v)
    const d = r.data || r
    autoConsume.value = !!d.autoConsume
    ElMessage.success(d.note || '已更新')
  }catch(e){
    autoConsume.value = !v
    ElMessage.error('更新失败：'+(e?.response?.data?.message||e?.message||e))
  }finally{ autoConsumeLoading.value = false }
}

/** 常见服务商预设：换厂商时三项一起填，避免"只换了 Key 没换 base-url"这类必 401 的组合。 */
const PRESETS = {
  dashscope: { baseUrl:'https://dashscope.aliyuncs.com/compatible-mode/v1', model:'qwen-plus' },
  // 方舟：模型名故意留空——留空即由平台从「账号可用模型」里自选并自动降级，
  // 比写死一个（可能没开通/已下架的）模型名可靠得多。
  ark:       { baseUrl:'https://ark.cn-beijing.volces.com/api/v3',           model:'' },
  openai:    { baseUrl:'https://api.openai.com/v1',                          model:'gpt-5-mini' }
}
function fillPreset(k){
  const p = PRESETS[k]; if(!p) return
  cfgForm.value.baseUrl = p.baseUrl
  cfgForm.value.model = p.model
  ElMessage.info('已填入 '+providerNameOf(p.baseUrl)+' 的 base-url'+(p.model?('与模型名 '+p.model):'（模型名留空 = 平台自动从可用列表里选一个）')+'，请再粘贴对应的 API Key')
}
function providerNameOf(b){
  b = String(b||'').toLowerCase()
  if (b.includes('dashscope')) return '阿里云百炼'
  if (b.includes('volces')) return '火山方舟'
  if (b.includes('openai.com')) return 'OpenAI'
  return '当前服务商'
}

/** 按当前 base-url 判断底座厂商，把「去哪儿拿 Key」这句话说对。 */
const providerName = computed(() => providerNameOf(diag.value?.baseUrl))
const providerHint = computed(() => {
  const b = String(diag.value?.baseUrl || '').toLowerCase()
  if (b.includes('volces')) {
    return '火山方舟 → API Key：https://ark.volcengine.com/region:cn-beijing/apikey ；base-url 用 https://ark.cn-beijing.volces.com/api/v3 ；'
      + '模型可以填模型 ID（如 doubao-seed-1-6-250615）或接入点 ep-…，也可以留空——留空时平台会自动从「账号可用模型」里选一个能用的，并在某个模型报「未开通 / 限流」时自动降级到下一个。'
  }
  if (b.includes('dashscope')) {
    return '百炼控制台 → API Key 管理 → 复制当前有效的 Key。注意地域：账号在北京用 dashscope.aliyuncs.com，在新加坡用 dashscope-intl.aliyuncs.com（需同步改 app.ai.base-url）。'
  }
  return '到你的模型服务控制台复制 API Key；注意 base-url 要与该服务商的 OpenAI 兼容端点一致。'
})
const capsMeta = {
  chat:       { label: '对话补全', ok: '主链路可用' },
  jsonMode:   { label: 'JSON 模式（response_format）', ok: '结构化答案抽取可用' },
  tools:      { label: '工具 / 函数调用', ok: 'Agent 可调用内部取数工具' },
  embedding:  { label: '向量化（语义检索）', ok: '语义检索可用' }
}

/**
 * 一句话说清「语义检索」和「联网检索」各走哪条通道。
 * 默认两条都是免 Key 的（本地哈希向量 / 直连抓取），密钥只在你显式配置第三方时才下发。
 */
const channelLine = computed(() => {
  const d = diag.value
  if (!d) return ''
  const e = d.embedding || {}
  const w = d.webSearch || {}
  const emb = e.mode === 'remote' ? '第三方 embedding'
    : e.mode === 'auto' ? '第三方 embedding（失败自动回落本地）'
      : '本地哈希向量（零成本 · 不联网 · 无需 Key）'
  const ws = !w.enabled ? ('联网检索不可用：' + (w.reason || '未知原因'))
    : w.mode === 'api' ? '方舟 Responses API（需 Key，且须已开通联网插件）'
      : ('免 Key 直连抓取' + (w.lastChannel ? '（本次实际：' + w.lastChannel + '）' : ''))
  return '语义检索：' + emb + '　·　联网检索：' + ws
})
/** 能力体检三项状态：ok / 明确关闭（中性）/ 失败。 */
function capsVal(k){ return (diag.value?.capabilities||{})[k] }
/** 本地零成本向量是「可用」，只是不走服务商——不能因为文案里没有 ok 就标成红色不可用。 */
function capsLocalOk(v){ return String(v || '').startsWith('本地向量') }
function capsTag(k){
  const v = capsVal(k)
  if (v === 'ok' || capsLocalOk(v)) return 'success'
  if (String(v||'').startsWith('已关闭') || String(v||'').startsWith('未配置')) return 'info'
  return 'danger'
}
function capsText(k){
  const v = capsVal(k)
  if (v === 'ok') return '支持'
  if (capsLocalOk(v)) return '支持（本地向量）'
  if (String(v||'').startsWith('已关闭') || String(v||'').startsWith('未配置')) return '未启用'
  return '不可用'
}

/** 提交 base-url / 模型名 / Key（只填要改的）→ 后端热更新 → 立刻回显能不能用。 */
async function applyKey(){
  const payload = {
    baseUrl: (cfgForm.value.baseUrl||'').trim(),
    model:   (cfgForm.value.model||'').trim(),
    key:     (cfgForm.value.key||'').trim()
  }
  if(!payload.baseUrl && !payload.model && !payload.key){
    ElMessage.warning('请至少填写 base-url、模型名、API Key 之一')
    return
  }
  keyLoading.value = true
  try{
    const r = await api.aiUpdateKey(payload)
    diag.value = r.data || r
    applyModelsMeta(diag.value?.models)
    if(diag.value && diag.value.ok){
      cfgForm.value.key = ''
      ElMessage.success('新配置已生效，模型服务连接正常')
    }else{
      ElMessage.warning('新配置仍不可用：'+(diag.value?.issue||'未知原因'))
    }
  }catch(e){
    ElMessage.error('更新失败：'+(e?.response?.data?.message || e?.message || e))
  }finally{ keyLoading.value = false }
}

function parseJsonStr(v){ if(!v) return null; try{ return typeof v==='string'?JSON.parse(v):v }catch{ return null } }
function parseArr(v){ const j = parseJsonStr(v); return Array.isArray(j)?j:[] }
function toArr(v){ if(Array.isArray(v)) return v; const j = parseJsonStr(v); return Array.isArray(j)?j:[] }
function pretty(v){ if(v===null||v===undefined||v==='') return ''; try{ return JSON.stringify(typeof v==='string'?JSON.parse(v):v, null, 2) }catch{ return String(v) } }
function riskType(l){ return l==='HIGH'?'danger':l==='MEDIUM'?'warning':'info' }

/** 降级状态：NONE=正常，PARTIAL=带病交付（工具失败或模型降级），FULL=完全走本地规则。 */
function degradeText(l){
  if(l==='FULL') return '完全降级（本地规则）'
  if(l==='PARTIAL') return '部分降级（带病交付）'
  return '正常'
}
function degradeTagType(l){
  if(l==='FULL') return 'danger'
  if(l==='PARTIAL') return 'warning'
  return 'success'
}

/** 建议动作归一化：兼容字符串数组与 {action,basis,refs} 对象数组。 */
function toRecs(v){
  const arr = toArr(v)
  return arr.map(x=>{
    if(x && typeof x==='object') return {action:x.action||'', basis:x.basis||'', refs:Array.isArray(x.refs)?x.refs:(x.refs?[x.refs]:[])}
    return {action:String(x||''), basis:'', refs:[]}
  }).filter(r=>r.action)
}

/** 统一的展示视图：流式结果优先，其次同步/异步/历史详情。 */
const view = computed(()=>{
  if(ai.outcome){
    return {
      source:'stream',
      analysisId: ai.outcome.analysisId,
      provider: ai.outcome.provider,
      confidence: ai.outcome.confidence,
      answer: ai.outcome.answer || ai.answer,
      answerJson: ai.outcome.answerJson,
      evidenceJson: ai.outcome.evidenceJson,
      toolTraceJson: ai.outcome.toolTraceJson,
      traceId: ai.outcome.traceId,
      degradeLevel: ai.outcome.degradeLevel,
      degradeReason: ai.outcome.degradeReason,
      durationMs: ai.outcome.durationMs,
      llmCalls: ai.outcome.llmCalls,
      toolCalls: ai.outcome.toolCalls
    }
  }
  const r = record.value
  if(!r) return null
  return {
    source:'record',
    analysisId: r.id,
    provider: r.provider,
    confidence: r.confidence,
    answer: r.answer,
    answerJson: r.answerJson,
    evidenceJson: r.evidenceJson,
    toolTraceJson: r.toolTraceJson,
    traceId: r.traceId,
    degradeLevel: r.degradeLevel,
    degradeReason: r.degradeReason,
    durationMs: r.durationMs,
    llmCalls: r.llmCalls,
    toolCalls: r.toolCalls
  }
})

const structured = computed(()=>parseJsonStr(view.value?.answerJson))
const split = computed(()=> structured.value?.source_split || {})
const groundingCheck = computed(()=> structured.value?.grounding_check || null)
/** 本次命中的历史记忆（后端写进 answer_json.memory.longTerm）。
 *  与「参考来源」严格分开显示：记忆不是证据、没有编号。 */
const longTerm = computed(()=> structured.value?.memory?.longTerm || null)
const recsInternal = computed(()=> toRecs(structured.value?.recommendations_internal))
const recsWeb = computed(()=> toRecs(structured.value?.recommendations_web))
/** 是否新版「来源分层」结构；旧记录回落到合并视图。 */
const isSplit = computed(()=>{
  const s = structured.value
  if(!s) return false
  return !!(s.conclusion_internal || s.conclusion_web || recsInternal.value.length || recsWeb.value.length)
})

/** 正文：运行中显示流式累积（打字机效果）；完成后以服务端持久化的完整正文为准。 */
const displayAnswer = computed(()=>{
  if(ai.running) return ai.answer || view.value?.answer || ''
  return view.value?.answer || ai.answer || ''
})

const evidenceList = computed(()=>{
  if(ai.outcome) return ai.outcome.evidence || []
  const r = record.value
  if(!r) return []
  const direct = parseArr(r.evidenceJson)
  if(direct.length) return direct
  const s = parseJsonStr(r.answerJson)
  return (s && Array.isArray(s.evidence)) ? s.evidence : []
})

const traceList = computed(()=>{
  if(ai.outcome) return ai.outcome.trace || []
  return parseArr(record.value?.toolTraceJson)
})

/** 网页来源：优先后端持久化的 web_sources（序号已全局唯一），否则从证据里汇总。 */
const webSources = computed(()=>{
  const persisted = structured.value?.web_sources
  if(Array.isArray(persisted) && persisted.length) return persisted
  const out=[]; const seen=new Set()
  for(const e of evidenceList.value){
    if(e && e.sourceType==='web' && Array.isArray(e.sources)){
      for(const s of e.sources){
        const k = s && (s.index!==undefined ? 'i'+s.index : s.url)
        if(s && s.url && !seen.has(k)){ seen.add(k); out.push(s) }
      }
    }
  }
  return out.sort((a,b)=>(a.index||0)-(b.index||0))
})

/** 未被引用的网页来源：搜出来但没进正文依据的，单独折叠，不混进「参考来源」。 */
const webUnusedSources = computed(()=>{
  const persisted = structured.value?.web_sources_unused
  return Array.isArray(persisted) ? persisted : []
})

/** 联网检索经过：每次调用的原始词、清洗后的词、保留/丢弃条数。 */
const webTrails = computed(()=>{
  const t = structured.value?.web_trails
  return Array.isArray(t) ? t : []
})

/** 本次参与的智能体（后端按意图路由的结果）。 */
const agentTeam = computed(()=>{
  const a = structured.value?.agents
  return Array.isArray(a) ? a : []
})

/** 知识库来源：内部依据，与「来源ID=x」引用一一对应。 */
const kbSources = computed(()=>{
  const persisted = structured.value?.kb_sources
  if(Array.isArray(persisted) && persisted.length) return persisted
  const out=[]; const seen=new Set()
  for(const e of evidenceList.value){
    if(e && e.sourceType==='knowledge' && Array.isArray(e.sources)){
      for(const s of e.sources){
        if(s && s.documentId!==undefined && !seen.has(String(s.documentId))){ seen.add(String(s.documentId)); out.push(s) }
      }
    }
  }
  return out
})

const rawAnswerJson = computed(()=>pretty(view.value?.answerJson))
const rawEvidenceJson = computed(()=>pretty(view.value?.evidenceJson))

/** 把引用标注解析成「网页 [n]」或「知识库 来源ID=x」。 */
function refInfo(ref){
  const s = String(ref==null?'':ref).trim()
  const m = s.match(/(\d+)/)
  if(!m) return null
  const key = m[1]
  if(/来源|知识库|文档|ID/i.test(s) && !/^\[/.test(s)) return {kind:'kb', key, label:'来源ID='+key}
  return {kind:'web', key, label:'['+key+']'}
}
function refLabel(ref){ const i = refInfo(ref); return i?i.label:String(ref||'') }
/** 点击引用 → 滚动并高亮对应来源，让「可核对」变成可操作。 */
function focusRef(ref){
  const i = refInfo(ref)
  if(!i) return
  hot.value = i
  const el = document.getElementById((i.kind==='web'?'web-src-':'kb-src-')+i.key)
  if(el && el.scrollIntoView) el.scrollIntoView({behavior:'smooth', block:'center'})
  setTimeout(()=>{ hot.value = {kind:'', key:''} }, 2500)
}

async function copy(text){
  if(!text) return
  try{ await navigator.clipboard.writeText(text); ElMessage.success('已复制到剪贴板') }
  catch{ ElMessage.warning('浏览器拒绝了剪贴板访问，请手动选中复制') }
}

function showReasons(row){
  ElMessageBox.alert((row.reasons||[]).join('\n') || '无', '用例 '+row.id+' 未通过原因', {customStyle:{maxWidth:'560px'}})
}

// Top 分的两种口径：左边是 query 内归一化分（Top1 恒为 100，只能看出"排第一"），
// 右边「原」是归一化之前的原始余弦分（×100）——它才是跨 query 比得出的质量刻度。
function topScoreTip(row){
  const base = '左：本次查询内部归一化分（Top1 恒为 100，只能说明排序，不能横向比较）'
  return row.ragTopRawScore===null||row.ragTopRawScore===undefined
    ? base + '；本条无向量通道，未产出原始分'
    : base + '。右：原始相关度 ' + row.ragTopRawScore + '（跨查询可比，适合设绝对阈值）'
}

async function analyze(){
  if(!form.value.question) return ElMessage.warning('请输入问题')
  loading.value = true
  record.value = null
  try{
    const r = await api.analyze(form.value)
    record.value = r
    ElMessage.success('分析完成'); history(); loadConversations()
  }catch(e){} finally{ loading.value = false }
}

function startStream(){
  if(!form.value.question) return ElMessage.warning('请输入问题')
  // 交给全局 store：切走模块也不会中断，回来仍在/已完成
  ai.start(form.value)
}

function analyzeAsync(){
  if(!form.value.question) return ElMessage.warning('请输入问题')
  api.aiAsyncSubmit(form.value).then(r=>{
    asyncTaskId.value = r.taskId; asyncStatus.value = {status:'RUNNING'}
    if(pollTimer.value) clearInterval(pollTimer.value)
    pollTimer.value = setInterval(async ()=>{
      const st = await api.aiAsyncStatus(r.taskId)
      asyncStatus.value = st
      if(st.status==='DONE'){ clearInterval(pollTimer.value); if(st.analysis) record.value = st.analysis; history(); loadConversations() }
      else if(st.status==='FAILED'){ clearInterval(pollTimer.value); ElMessage.error('异步分析失败: '+st.error) }
    }, 1500)
  })
}

/** 评估回归：提交异步任务 → 轮询进度 → 取报告（避免同步等待被 30s 超时掐断）。 */
async function evaluate(){
  if(evalRunning.value) return
  evalReport.value = null; evalError.value = ''
  try{
    const r = await api.aiEvaluateAsync({companyId: form.value.companyId, mode: evalMode.value})
    evalTaskId.value = r.taskId
    evalStatus.value = {status:'RUNNING', done:0, total:r.total||0, percent:0}
    if(evalTimer.value) clearInterval(evalTimer.value)
    evalTimer.value = setInterval(pollEval, 1000)
  }catch(e){}
}

async function pollEval(){
  try{
    const st = await api.aiEvaluateStatus(evalTaskId.value)
    evalStatus.value = st
    if(st.status==='DONE'){
      clearEvalTimer()
      evalReport.value = st.report
      loadEvalHistory()
      if(!st.report) return
      const cmp = st.report.comparison
      if(cmp && cmp.regressions && cmp.regressions.length){
        ElMessage.error('发现 '+cmp.regressions.length+' 条回归（上轮通过、本轮失败），见「评估与回归集」对比区')
      }else if(st.report.passed!==st.report.total) ElMessage.warning('部分用例未通过，可点「查看」看原因')
      else ElMessage.success('回归全部通过（'+st.report.total+' 条）')
    }else if(st.status==='FAILED'){
      clearEvalTimer(); evalError.value = st.error || '未知错误'; ElMessage.error('回归执行失败')
    }else if(st.status==='NOT_FOUND'){
      clearEvalTimer(); evalError.value = '任务已失效，请重新发起'
    }
  }catch(e){}
}
function clearEvalTimer(){ if(evalTimer.value){ clearInterval(evalTimer.value); evalTimer.value = null } }

async function viewDetail(row){
  try{ record.value = await api.aiHistoryDetail(row.id); ai.reset(); ElMessage.success('已加载分析 #'+row.id) }catch(e){}
}

async function removeHistory(row){
  try{
    await ElMessageBox.confirm('确认删除分析记录 #'+row.id+'？该操作不可恢复。','删除确认',
      {type:'warning', confirmButtonText:'删除', cancelButtonText:'取消'})
  }catch{ return }
  try{
    await api.deleteAiHistory(row.id)
    if(record.value && record.value.id===row.id) record.value = null
    if(ai.outcome && ai.outcome.analysisId===row.id) ai.reset()
    ElMessage.success('已删除')
    history()
  }catch(e){}
}

async function clearHistory(){
  const scope = form.value.companyId ? '企业 '+form.value.companyId+' 的' : '全部'
  try{
    await ElMessageBox.confirm('确认清空'+scope+'分析历史？该操作不可恢复。','清空确认',
      {type:'warning', confirmButtonText:'清空', cancelButtonText:'取消'})
  }catch{ return }
  try{
    const n = await api.clearAiHistory(form.value.companyId?{companyId:form.value.companyId}:{})
    record.value = null; ai.reset()
    ElMessage.success('已清空 '+(n||0)+' 条')
    history()
  }catch(e){}
}

async function removeConversation(id){
  try{
    await ElMessageBox.confirm('确认删除会话 #'+id+' 及其全部消息？该操作不可恢复。','删除确认',
      {type:'warning', confirmButtonText:'删除', cancelButtonText:'取消'})
  }catch{ return }
  try{
    await api.deleteAiConversation(id)
    if(form.value.sessionId===id) form.value.sessionId = null
    ElMessage.success('会话已删除')
    loadConversations()
  }catch(e){}
}

async function history(){ historyRows.value = (await api.aiHistory(form.value.companyId?{companyId:form.value.companyId}:{})) || [] }
async function loadConversations(){ conversations.value = (await api.aiConversations(form.value.companyId?{companyId:form.value.companyId}:{})) || [] }

// 流式在后台跑完时（哪怕用户当时在别的模块）回来后自动刷新历史。
// 记忆也要一起刷：这次分析可能刚沉淀了新记忆、或取代了旧口径。
watch(()=>ai.status, s=>{ if(s==='DONE'){ history(); loadConversations(); loadMemory() } })
// 切换企业必须重取记忆 —— 记忆按企业隔离，不重取会把上一个企业的记忆显示成这个企业的。
watch(()=>form.value.companyId, ()=>{ loadMemory(); memProbe.value = null })

onMounted(()=>{
  history(); loadConversations(); loadEvalHistory(); loadMemory()
  if(ai.status==='DONE' && ai.outcome) { history(); loadConversations() }
})
onUnmounted(()=>{ clearEvalTimer(); if(pollTimer.value) clearInterval(pollTimer.value) })
</script>

<style scoped>
.page-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
.runbar{display:flex;align-items:center;gap:10px;background:#fff7e6;border:1px solid #ffd591;border-radius:8px;padding:8px 12px}
.answer{white-space:pre-wrap;line-height:1.8;padding:10px 4px;color:#1e293b}
.live{background:#f8fafc;border-radius:6px}
.report-shell{margin:12px 0 16px;border:1px solid #cbd5e1;border-left:4px solid #2563eb;border-radius:8px;padding:8px 12px 12px;background:#fff}
.report-shell .answer{min-height:84px;background:#f8fafc;padding:12px;transition:min-height .2s ease}
.report-shell .badge.ib{background:#2563eb}
.cursor{color:#409eff;animation:blink 1s steps(2,start) infinite}
@keyframes blink{to{visibility:hidden}}
.hint{margin:6px 4px 10px;font-size:12px;color:#94a3b8}
.envhint{margin:-4px 4px 12px;font-size:12px;color:#64748b;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:8px 10px;line-height:1.8}
.envhint span{display:block}
.chanline{color:#0f766e;font-weight:500}
.presets{align-items:center;margin-bottom:6px}
/* 模型服务卡：可用模型下拉 + 后台消耗授权 */
.modelsbar{align-items:center;margin-bottom:6px}
.autobar{align-items:center;margin:8px 0 2px}
.autobar .tip{margin-left:0}
.switchline{color:#b45309;font-weight:500}
code{background:#f1f5f9;border-radius:4px;padding:0 4px;color:#0f766e}
.capmsg{font-size:12px;color:#64748b}
.caps .el-descriptions__label{width:240px}
.tip{font-size:12px;color:#94a3b8;margin-left:8px}
.warn-text{color:#d97706}
.btns{display:flex;flex-wrap:wrap;gap:8px}
h4{margin:14px 0 6px;color:#0f172a}
.recs{padding-left:20px;line-height:1.9;color:#1e293b}
.recs li{margin-bottom:4px}
.basis{color:#475569}
.ref{display:inline-block;margin-left:6px;padding:0 6px;border-radius:10px;background:#eef2ff;color:#2563eb;font-size:12px;cursor:pointer;border:1px solid #c7d2fe}
.ref:hover{background:#e0e7ff;text-decoration:underline}
.noref{color:#b45309;font-size:12px;margin-left:6px}
.err{color:#dc2626;font-size:13px;margin-top:6px}
.row{display:flex;gap:8px;width:100%}
.cardhead{display:flex;justify-content:space-between;align-items:center;gap:10px}
.catline{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
/* 先行事实卡 */
.metacard{margin-bottom:12px;border-color:#bfdbfe;background:linear-gradient(180deg,#f8fbff,#fff)}
.metacard :deep(.el-card__header){padding:8px 12px;border-bottom:1px solid #e2e8f0}
.metacard :deep(.el-card__body){padding:10px 12px}
.metarow{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:2px 0}
.mlabel{font-size:12px;color:#64748b;min-width:56px}
.mmixes{display:flex;gap:6px;flex-wrap:wrap}
.metacard .tip{margin-left:6px}
/* 回归基线对比 */
.baseline{margin-top:10px;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;background:#fff}
.brow{display:flex;align-items:center;gap:14px;flex-wrap:wrap;font-size:13px;color:#334155}
.bsec{margin-top:8px}
.blabel{font-size:12px;margin-bottom:4px}
.blabel.danger{color:#dc2626}
.blabel.ok{color:#16a34a}
.blabel.warn{color:#d97706}
.bitem{font-size:12px;color:#475569;line-height:1.7;padding-left:8px;border-left:2px solid #e2e8f0;margin-bottom:4px}
.treason{color:#94a3b8;font-size:12px}
.ok{color:#67c23a;font-weight:700}
.evalprog{margin-bottom:10px}
.mixes{display:inline-flex;gap:6px;flex-wrap:wrap}
.groundbar{display:flex;align-items:center;gap:10px;margin:10px 0 4px}
/* 来源分层：内部 / 外部两块视觉上明确分开 */
.group{border:1px solid #e2e8f0;border-radius:10px;padding:6px 12px 12px;margin:12px 0}
.group.internal{border-left:4px solid #16a34a;background:#f8fdfa}
.group.web{border-left:4px solid #ea9a1a;background:#fffdf7}
.ghead{font-weight:700;color:#0f172a;margin:8px 0 2px;display:flex;align-items:center;gap:8px}
.badge{font-size:12px;padding:1px 8px;border-radius:10px;color:#fff}
.badge.ib{background:#16a34a}
.badge.wb{background:#ea9a1a}
.gsub{margin-top:10px;font-weight:600;color:#334155;font-size:13px}
.combine{margin-top:10px}
.srclist{margin:6px 0 10px}
.agentteam{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px}
.wtrail{margin:6px 0}
.wt{border-left:3px solid #e2e8f0;padding:4px 0 6px 8px;margin-bottom:6px;font-size:12px;line-height:1.7}
.src{padding:6px 8px;border-bottom:1px dashed #e2e8f0;font-size:13px;border-radius:6px;transition:background .3s}
.src:last-child{border-bottom:none}
.src.hot{background:#fff7e6;box-shadow:0 0 0 2px #ffd591 inset}
.src a{color:#2563eb;text-decoration:none;word-break:break-all}
.src a:hover{text-decoration:underline}
.src .idx{display:inline-block;min-width:28px;color:#94a3b8}
.src .kbtitle{color:#0f172a}
.src .url{font-size:12px;color:#94a3b8;word-break:break-all;margin-left:28px}
.src .site{color:#64748b}
.copybar{display:flex;justify-content:flex-end;margin-bottom:6px}
pre{white-space:pre-wrap;word-break:break-all;background:#f8fafc;padding:12px;max-height:420px;overflow:auto}
/* 流式进度条：告诉用户「系统在动」比让它转一圈更能降低焦虑 */
.progbar{border:1px solid #fde68a;background:#fffbeb;border-radius:8px;padding:10px 12px;margin-bottom:12px}
.progbar .prow{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}
.progbar .pstage{font-size:13px;color:#92400e;font-weight:600}
.progbar .ptip{font-size:12px;color:#b45309;font-variant-numeric:tabular-nums}
.progbar .ptrack{height:6px;border-radius:3px;background:#fde68a;overflow:hidden}
.progbar .pfill{height:100%;width:40%;border-radius:3px;background:#f59e0b;animation:pslide 1.4s ease-in-out infinite}
@keyframes pslide{0%{margin-left:-40%}100%{margin-left:100%}}
.progbar .pfoot{display:flex;justify-content:space-between;gap:12px;margin-top:6px;font-size:12px;color:#b45309}
.progbar .pwarn{font-weight:600}
.depth-tip{margin-top:4px}
/* 长期记忆：刻意与「参考来源」用不同底色的区块，避免被误读成本次证据 */
.memcard{border-left:4px solid #6366f1}
.memprobe{margin:8px 0 4px;padding:8px 10px;border:1px solid #e0e7ff;background:#f5f7ff;border-radius:8px}
.memrow{display:flex;align-items:baseline;gap:8px;font-size:13px;line-height:1.7;padding:5px 0;border-bottom:1px dashed #e2e8f0}
.memrow:last-child{border-bottom:none}
.memrow .mtag{flex:0 0 auto;font-size:12px;color:#4338ca;background:#eef2ff;border-radius:8px;padding:0 6px}
.memrow .mtext{flex:1;color:#0f172a;word-break:break-all}
.memtools{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px}
.memtools .el-button{margin-left:auto}
.membar{margin:8px 0 4px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.memhits{flex:1 1 100%}
</style>
