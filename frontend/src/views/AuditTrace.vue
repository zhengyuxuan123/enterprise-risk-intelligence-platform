<template>
<div class="page">
  <div class="page-head">
    <div>
      <h2>审计与追溯</h2>
      <p>每一次 AI 结论都能回答三个问题：<b>它依据了什么</b>（数据血缘）、<b>它是怎么得出的</b>（决策留痕）、<b>谁批了后续动作</b>（审批队列）。可导出 PDF/Word 归档。</p>
    </div>
  </div>

  <!-- 检索区：既可按追溯号反查，也可从分析历史里挑一条 -->
  <el-card shadow="never" style="margin-bottom:14px">
    <el-form :inline="true" @submit.prevent>
      <el-form-item label="追溯号">
        <el-input v-model="traceNo" placeholder="X-Trace-Id，跨系统排查的唯一锚点" clearable style="width:290px" @keyup.enter="loadByTraceNo"/>
      </el-form-item>
      <el-form-item>
        <el-button :loading="loading" @click="loadByTraceNo">按追溯号查</el-button>
      </el-form-item>
      <el-form-item label="或选一条分析">
        <el-select v-model="analysisId" filterable clearable placeholder="选择分析记录" style="width:330px" @change="loadAll">
          <el-option v-for="h in histories" :key="h.id" :value="h.id"
                     :label="'#'+h.id+' · '+h.question+(h.traceId?' · '+h.traceId.slice(0,8):'')"/>
        </el-select>
      </el-form-item>
      <el-form-item label="企业ID">
        <el-input-number v-model="companyId" :min="1" controls-position="right" style="width:120px" @change="loadHistories"/>
      </el-form-item>
      <el-form-item>
        <el-button :loading="loading" @click="loadHistories">刷新列表</el-button>
      </el-form-item>
      <el-form-item>
        <el-button-group>
          <el-button :disabled="!analysisId" :loading="exporting==='pdf'" @click="doExport('pdf')">导出 PDF</el-button>
          <el-button :disabled="!analysisId" :loading="exporting==='docx'" @click="doExport('docx')">导出 Word</el-button>
        </el-button-group>
      </el-form-item>
    </el-form>
  </el-card>

  <el-alert v-if="traceRaw" type="info" show-icon :closable="false" style="margin-bottom:14px">
    <template #title>追溯号命中：分析 #{{traceRaw.analysis_id}}　模型 {{traceRaw.model||'—'}}　降级 {{traceRaw.degrade_level||'NONE'}}　工具调用 {{traceRaw.tool_calls}}　耗时 {{traceRaw.duration_ms}}ms</template>
    <el-button link type="primary" @click="useAnalysis(traceRaw.analysis_id)">查看该分析的完整留痕与血缘</el-button>
  </el-alert>

  <el-tabs v-model="tab">
    <!-- ============ 决策留痕 ============ -->
    <el-tab-pane label="决策留痕" name="trace">
      <el-empty v-if="!trace" description="选择一条分析，或输入追溯号"/>
      <template v-else>
        <el-descriptions :column="4" border size="small" style="margin-bottom:14px">
          <el-descriptions-item label="追溯号">
            <span style="font-family:monospace">{{trace.traceId}}</span>
            <el-button link type="primary" @click="copy(trace.traceId)">复制</el-button>
          </el-descriptions-item>
          <el-descriptions-item label="分析ID">{{trace.analysisId}}</el-descriptions-item>
          <el-descriptions-item label="问题" :span="2">{{trace.question}}</el-descriptions-item>
          <el-descriptions-item label="底座">{{trace.provider||'—'}}</el-descriptions-item>
          <el-descriptions-item label="生效模型">{{trace.model||'—'}}</el-descriptions-item>
          <el-descriptions-item label="置信度">{{trace.confidence||'—'}}</el-descriptions-item>
          <el-descriptions-item label="有据可依">
            <el-tag :type="trace.grounded?'success':'warning'" size="small">{{trace.grounded?'是':'否'}}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="降级等级">
            <el-tag :type="degradeType(trace.degradeLevel)" size="small">{{trace.degradeLevel||'NONE'}}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="降级原因" :span="3">{{trace.degradeReason||'—'}}</el-descriptions-item>
          <el-descriptions-item label="耗时(ms)">{{trace.durationMs}}</el-descriptions-item>
          <el-descriptions-item label="模型调用">{{trace.llmCalls}}</el-descriptions-item>
          <el-descriptions-item label="工具调用">{{trace.toolCalls}}</el-descriptions-item>
          <el-descriptions-item label="时间">{{trace.createdAt}}</el-descriptions-item>
        </el-descriptions>

        <el-alert v-if="trace.detailNote" type="warning" show-icon :closable="false" style="margin-bottom:14px" :title="trace.detailNote"/>
        <el-alert v-if="trace.detailError" type="error" show-icon :closable="false" style="margin-bottom:14px" :title="trace.detailError"/>

        <template v-if="trace.detail">
          <el-row :gutter="14">
            <el-col :span="12">
              <el-card shadow="never">
                <template #header>路由与开放的工具</template>
                <div v-if="!route.openedTools || !Object.keys(route.openedTools).length">—</div>
                <div v-for="(tools,agent) in route.openedTools" :key="agent" style="margin-bottom:8px">
                  <b>{{agentName(agent)}}</b>
                  <el-tag v-for="t in tools" :key="t" size="small" type="info" style="margin-left:6px">{{t}}</el-tag>
                </div>
                <el-divider style="margin:10px 0"/>
                <el-tag size="small" :type="route.multiAgent?'success':'info'">{{route.multiAgent?'多智能体协作':'单智能体'}}</el-tag>
                <el-tag size="small" :type="route.needsWeb?'warning':'info'" style="margin-left:6px">{{route.needsWeb?'需要联网':'未联网'}}</el-tag>
              </el-card>
            </el-col>
            <el-col :span="12">
              <el-card shadow="never">
                <template #header>引用核对（确定性核对，不消耗额度）</template>
                <el-descriptions :column="2" size="small">
                  <el-descriptions-item label="内部来源">{{g.kbSources}}</el-descriptions-item>
                  <el-descriptions-item label="外部来源">{{g.webSources}}</el-descriptions-item>
                  <el-descriptions-item label="引用覆盖率">{{pct(g.citationCoverage)}}</el-descriptions-item>
                  <el-descriptions-item label="结论">
                    <el-tag :type="g.ok?'success':'danger'" size="small">{{g.ok?'引用自洽':'存在引用问题'}}</el-tag>
                  </el-descriptions-item>
                </el-descriptions>
                <div v-if="citedRefs.length" style="margin-top:6px">
                  <b>正文引用过的来源：</b>
                  <el-tag v-for="x in citedRefs" :key="x" size="small" type="success" style="margin:2px">{{x}}</el-tag>
                </div>
                <div v-if="dangling.length" style="margin-top:6px">
                  <b>悬空引用（正文标了但证据里没有）：</b>
                  <el-tag v-for="x in dangling" :key="x" size="small" type="danger" style="margin:2px">{{x}}</el-tag>
                </div>
                <div v-if="unused.length" style="margin-top:6px">
                  <b>未引用来源（召回了但正文没用）：</b>
                  <el-tag v-for="x in unused" :key="x" size="small" type="warning" style="margin:2px">{{x}}</el-tag>
                </div>
                <div v-if="g.issues && g.issues.length" style="margin-top:6px">
                  <b>问题：</b>
                  <div v-for="x in g.issues" :key="x" style="color:#c45656">{{x}}</div>
                </div>
              </el-card>
            </el-col>
          </el-row>

          <el-card shadow="never" style="margin-top:14px">
            <template #header>工具调用明细（{{toolDetail.length}}）</template>
            <el-table :data="toolDetail" size="small" stripe>
              <el-table-column prop="name" label="工具" width="180"/>
              <el-table-column label="耗时" width="100">
                <template #default="s">{{s.row.ms!=null?s.row.ms+'ms':'—'}}</template>
              </el-table-column>
              <el-table-column label="结果" width="86">
                <template #default="s">
                  <el-tag size="small" :type="s.row.ok===false?'danger':'success'">{{s.row.ok===false?'失败':'成功'}}</el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="sourceType" label="来源类型" width="110"/>
              <el-table-column label="方式" width="100">
                <template #default="s">
                  <el-tag v-if="s.row.prefetch" size="small" type="info">预取</el-tag>
                  <span v-else>按需</span>
                </template>
              </el-table-column>
              <el-table-column label="返回" width="110">
                <template #default="s">{{s.row.chars!=null?s.row.chars+' 字符':'—'}}</template>
              </el-table-column>
              <el-table-column label="失败原因">
                <template #default="s"><span style="color:#c45656">{{s.row.error||'—'}}</span></template>
              </el-table-column>
              <el-empty v-if="!toolDetail.length" description="本次分析未调用工具"/>
            </el-table>
            <div class="tip" style="margin-top:6px;color:#909399;font-size:12px">
              「预取」= 在模型开口要之前就主动跑掉的工具（用来解释为什么工具耗时几乎没有进入推理等待）。
            </div>
          </el-card>

          <el-card shadow="never" style="margin-top:14px">
            <template #header>原始留痕 JSON</template>
            <el-collapse>
              <el-collapse-item title="展开（排障 / 对账用）">
                <pre style="max-height:320px;overflow:auto;background:#f7f8fa;padding:10px;font-size:12px">{{pretty(trace.detail)}}</pre>
              </el-collapse-item>
            </el-collapse>
          </el-card>
        </template>
      </template>
    </el-tab-pane>

    <!-- ============ 数据血缘 ============ -->
    <el-tab-pane label="数据血缘" name="lineage">
      <el-empty v-if="!lineage" description="选择一条分析"/>
      <template v-else>
        <el-row :gutter="14" style="margin-bottom:14px">
          <el-col :span="6"><el-card shadow="never"><el-statistic title="证据总数" :value="lineage.total"/></el-card></el-col>
          <el-col :span="6"><el-card shadow="never"><el-statistic title="正文已引用" :value="lineage.cited"/></el-card></el-col>
          <el-col :span="6"><el-card shadow="never"><el-statistic title="未引用（潜在漏引）" :value="lineage.uncited"/></el-card></el-col>
          <el-col :span="6"><el-card shadow="never"><el-statistic title="正文字数" :value="lineage.answerChars"/></el-card></el-col>
        </el-row>

        <el-table :data="lineage.items" size="small" stripe>
          <el-table-column prop="ref" label="引用标识" width="150"><template #default="s"><span style="font-family:monospace">{{s.row.ref}}</span></template></el-table-column>
          <el-table-column prop="type" label="类型" width="90">
            <template #default="s"><el-tag size="small">{{typeName(s.row.type)}}</el-tag></template>
          </el-table-column>
          <el-table-column prop="title" label="材料" min-width="200" show-overflow-tooltip/>
          <el-table-column label="正文引用" width="100" align="center">
            <template #default="s">
              <el-tag size="small" :type="s.row.citedInAnswer?'success':'warning'">{{s.row.citedInAnswer?'已引用':'未引用'}}</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="已入库" width="90" align="center">
            <template #default="s">
              <el-tag size="small" :type="s.row.indexed?'success':'info'">{{s.row.indexed?'是':'—'}}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="sourceType" label="源类型" width="100"/>
          <el-table-column prop="sourceId" label="源记录ID" width="100"/>
          <el-table-column prop="chunkId" label="语料块" width="140"><template #default="s"><span style="font-family:monospace">{{s.row.chunkId||'—'}}</span></template></el-table-column>
          <el-table-column prop="chunkStatus" label="块状态" width="90"/>
          <el-table-column label="说明" min-width="160">
            <template #default="s">
              <span v-if="s.row.note" style="color:#909399">{{s.row.note}}</span>
              <span v-else-if="!s.row.citedInAnswer" style="color:#e6a23c">证据召回了但正文没提——可能是漏引用</span>
              <span v-else>—</span>
            </template>
          </el-table-column>
          <el-empty v-if="!lineage.items.length" description="本次分析没有可追溯的证据引用"/>
        </el-table>

        <el-card shadow="never" style="margin-top:14px">
          <template #header>引用核对</template>
          <div v-if="lineage.grounding && lineage.grounding.issues && lineage.grounding.issues.length">
            <div v-for="x in lineage.grounding.issues" :key="x" style="color:#c45656">{{x}}</div>
          </div>
          <el-tag v-else type="success" size="small">引用自洽，未发现悬空引用</el-tag>
        </el-card>
      </template>
    </el-tab-pane>

    <!-- ============ 审批队列 ============ -->
    <el-tab-pane :label="'处置审批'+(actions.length?'（'+actions.length+'）':'')" name="actions">
      <div style="margin-bottom:12px">
        <el-radio-group v-model="actionScope" @change="loadActions">
          <el-radio-button label="all">全部企业</el-radio-button>
          <el-radio-button label="one">按企业ID</el-radio-button>
        </el-radio-group>
        <el-input-number v-if="actionScope==='one'" v-model="companyId" :min="1" controls-position="right" style="width:120px;margin-left:8px" @change="loadActions"/>
        <el-button style="margin-left:8px" :loading="actionLoading" @click="loadActions">刷新</el-button>
        <span style="margin-left:12px;color:#909399;font-size:12px">AI 的写动作只入审批队列，不直接改业务数据；批准后才由执行方落地。</span>
      </div>
      <el-table :data="actions" v-loading="actionLoading" size="small" stripe>
        <el-table-column prop="id" label="ID" width="60"/>
        <el-table-column prop="company_id" label="企业" width="70"/>
        <el-table-column prop="action_type" label="动作" width="150"/>
        <el-table-column label="风险等级" width="90">
          <template #default="s"><el-tag size="small" :type="riskType(s.row)">{{riskLevel(s.row)}}</el-tag></template>
        </el-table-column>
        <el-table-column label="处置内容" min-width="260" show-overflow-tooltip>
          <template #default="s">{{actionTitle(s.row)}}</template>
        </el-table-column>
        <el-table-column label="依据" min-width="240" show-overflow-tooltip>
          <template #default="s">{{s.row.reason||'—'}}</template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="90"/>
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="s">
            <el-button link type="success" :disabled="!auth.has('ai:approve')" @click="decide(s.row,true)">批准</el-button>
            <el-button link type="danger" :disabled="!auth.has('ai:approve')" @click="decide(s.row,false)">驳回</el-button>
            <el-button link type="primary" v-if="s.row.analysis_id" @click="useAnalysis(s.row.analysis_id)">留痕</el-button>
          </template>
        </el-table-column>
        <el-empty v-if="!actions.length" description="当前没有待审批的动作"/>
      </el-table>
    </el-tab-pane>
  </el-tabs>
</div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api } from '../api'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()
const rt = useRoute()   // 变量名不与下面「路由方案」的 computed 撞车
const tab = ref('trace')
const companyId = ref(1)
const analysisId = ref(null)
const traceNo = ref('')
const loading = ref(false)
const exporting = ref('')
const histories = ref([])
const trace = ref(null)
const traceRaw = ref(null)
const lineage = ref(null)
const actions = ref([])
const actionLoading = ref(false)
const actionScope = ref('all')

const route = computed(() => (trace.value && trace.value.detail && trace.value.detail.route) || {})
const g = computed(() => {
  const d = trace.value && trace.value.detail
  return (d && d.grounding) || {}
})
// 留痕里 toolDetail 的键是 name/ok/ms/sourceType/prefetch/chars/error（**不是** tool/success/durationMs），
// 照名字硬取会只留空单元格——这类「键名不符」在本项目里已经反复出现过，改前先 curl 一次看真身。
const toolDetail = computed(() => {
  const d = trace.value && trace.value.detail
  const list = (d && d.toolDetail) || []
  return list.map(x => (typeof x === 'string' ? { name: x } : x))
})
const dangling = computed(() => [].concat(g.value.webDangling || [], g.value.kbDangling || []).map(String))
// 内部来源的被引用情况在 kbCited（ref 列表），外部在 webUnused（未被引用的 URL）
const citedRefs = computed(() => (g.value.kbCited || []).map(String))
const unused = computed(() => (g.value.webUnused || []).map(String))

const AGENT_NAMES = {
  'risk-director': '风控总监', 'internal-analyst': '内部数据专员',
  'knowledge-agent': '知识库专员', 'external-scout': '外部情报员',
  'disposition-agent': '处置审批专员',
}
const agentName = a => AGENT_NAMES[a] || a
const TYPE_NAMES = { knowledge: '知识库', metric: '经营指标', event: '风险事件', complaint: '投诉', competitor: '竞品', company: '企业档案', web: '外部网页', aggregate: '聚合切片' }
const typeName = t => TYPE_NAMES[t] || t || '—'
const degradeType = v => (v === 'FULL' ? 'danger' : v === 'PARTIAL' ? 'warning' : 'success')
const pct = v => (v == null ? '—' : Math.round(v * 100) + '%')
const pretty = o => { try { return JSON.stringify(o, null, 2) } catch (e) { return String(o) } }

function copy (t) {
  navigator.clipboard?.writeText(t)
  ElMessage.success('追溯号已复制')
}

function parseArgs (row) {
  if (!row) return {}
  if (typeof row.action_args === 'object' && row.action_args) return row.action_args
  try { return JSON.parse(row.action_args || '{}') } catch (e) { return {} }
}
const actionTitle = r => parseArgs(r).title || r.action_type
const riskLevel = r => parseArgs(r).riskLevel || '—'
const riskType = r => ({ HIGH: 'danger', MEDIUM: 'warning', LOW: 'info' }[riskLevel(r)] || 'info')

async function loadHistories () {
  loading.value = true
  try {
    histories.value = await api.aiHistory({ companyId: companyId.value, limit: 50 }) || []
  } catch (e) { /* http 拦截器已提示 */ } finally { loading.value = false }
}

async function loadAll () {
  if (!analysisId.value) { trace.value = null; lineage.value = null; return }
  loading.value = true
  try {
    const [t, l] = await Promise.all([
      api.aiTrace(analysisId.value).catch(() => null),
      api.aiLineage(analysisId.value).catch(() => null),
    ])
    if (t && t.found === false) ElMessage.warning('分析记录不存在：' + analysisId.value)
    trace.value = (t && t.found) ? t : (t || null)
    lineage.value = (l && l.found) ? l : null
  } finally { loading.value = false }
}

async function loadByTraceNo () {
  const t = (traceNo.value || '').trim()
  if (!t) { ElMessage.warning('请输入追溯号'); return }
  loading.value = true
  try {
    const r = await api.aiTraceByNo(t)
    traceRaw.value = r
    if (r && r.analysis_id) await useAnalysis(r.analysis_id)
  } catch (e) { traceRaw.value = null } finally { loading.value = false }
}

async function useAnalysis (id) {
  analysisId.value = id
  tab.value = 'trace'
  await loadAll()
}

async function doExport (format) {
  if (!analysisId.value) return
  exporting.value = format
  try {
    const name = await api.aiExport(analysisId.value, format)
    ElMessage.success('已导出：' + name)
  } catch (e) { /* 已提示 */ } finally { exporting.value = '' }
}

async function loadActions () {
  actionLoading.value = true
  try {
    actions.value = actionScope.value === 'one'
      ? await api.aiActions(companyId.value) || []
      : await api.aiActionsAll() || []
  } catch (e) { actions.value = [] } finally { actionLoading.value = false }
}

async function decide (row, approve) {
  if (!auth.has('ai:approve')) { ElMessage.warning('当前角色没有处置审批权限'); return }
  try {
    const { value } = await ElMessageBox.prompt(
      (approve ? '批准' : '驳回') + '「' + actionTitle(row) + '」，请填写意见（会随审批记录一并留档）',
      approve ? '批准处置动作' : '驳回处置动作',
      { inputPlaceholder: '审批意见', inputType: 'textarea' })
    await api.aiDecide(row.id, {
      approve,
      comment: value || '',
      approverUserId: auth.user?.id || null,
    })
    ElMessage.success(approve ? '已批准' : '已驳回')
    await loadActions()
  } catch (e) {
    if (e !== 'cancel') { /* 已提示 */ }
  }
}

onMounted(async () => {
  // 从 AI 分析页「追溯与导出」跳过来时带上分析号，直接落到那一条上
  const q = rt.query || {}
  if (q.companyId) companyId.value = Number(q.companyId) || 1
  await loadHistories()
  await loadActions()
  if (q.analysisId) await useAnalysis(Number(q.analysisId))
})
</script>
