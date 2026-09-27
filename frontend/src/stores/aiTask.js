import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { ElMessage } from 'element-plus'

/**
 * AI 流式分析的全局任务。
 *
 * 为什么必须放在 store 而不是页面组件里：
 *  1) 请求与组件生命周期解耦 —— 切到别的模块时流式仍在后台继续，回来还能看到完整结果；
 *  2) token 渲染节流 —— 逐 token 直接写响应式变量会产生上千次重渲染，
 *     这里先攒进缓冲区、按 ~80ms 批量 flush，页面不再被渲染压力拖住导致点不动菜单。
 */
export const useAiTaskStore = defineStore('aiTask', () => {
  const status = ref('IDLE')        // IDLE | RUNNING | DONE | ERROR | CANCELLED
  const answer = ref('')            // 已渲染的正文（节流后）
  const outcome = ref(null)         // done 事件原始数据（含 answerJson / evidenceJson / toolTraceJson）
  const error = ref('')
  const question = ref('')
  const startedAt = ref(null)
  const elapsedMs = ref(0)
  const tokenCount = ref(0)
  /** SSE meta 事件：工具取数结束即刻送达的「风险等级 / 来源构成」，先于正文。 */
  const meta = ref(null)
  /** 当前进度播报（"正在调用 3 个工具取证…"）。等待难熬多半是因为屏幕上没有任何动静。 */
  const stage = ref('')
  /** 服务端主动收工的原因（例如流式连接到达等待上限），与失败不同：分析可能仍在后台进行。 */
  const notice = ref('')

  const running = computed(() => status.value === 'RUNNING')
  const evidence = computed(() => (outcome.value && outcome.value.evidence) || [])
  const webSources = computed(() => {
    const out = []
    const seen = new Set()
    for (const e of evidence.value) {
      if (e && e.sourceType === 'web' && Array.isArray(e.sources)) {
        for (const s of e.sources) {
          if (s && s.url && !seen.has(s.url)) { seen.add(s.url); out.push(s) }
        }
      }
    }
    return out
  })

  let controller = null
  let buffer = ''          // 待渲染的 token 缓冲
  let flushTimer = null
  let tickTimer = null
  let pendingDone = null
  // Async callbacks from an older run must not mutate a new login session.
  let generation = 0

  function finalize (d) {
    stopTimers()
    outcome.value = d || null
    // The token stream is the live model draft. The done event is authoritative:
    // citation repair, review, or output guardrails may have adjusted it afterwards.
    if (outcome.value && typeof outcome.value.answer === 'string') {
      answer.value = outcome.value.answer
    } else if (outcome.value) {
      outcome.value.answer = answer.value
    }
    status.value = 'DONE'
    elapsedMs.value = startedAt.value ? (Date.now() - startedAt.value) : elapsedMs.value
  }

  function flush (force = false) {
    if (flushTimer) { clearTimeout(flushTimer); flushTimer = null }
    if (buffer) {
      // Providers sometimes buffer hundreds of SSE chunks and release them in
      // under 200ms. Drain that burst over roughly two seconds so users can see
      // progress, while long answers do not turn into a slow typewriter demo.
      const size = force ? buffer.length : Math.max(1, Math.min(32, Math.ceil(buffer.length / 80)))
      answer.value += buffer.slice(0, size)
      buffer = buffer.slice(size)
    }
    if (buffer && !force) {
      scheduleFlush()
    } else if (!buffer && pendingDone) {
      const done = pendingDone
      pendingDone = null
      finalize(done)
    }
  }

  function scheduleFlush () {
    if (flushTimer) return
    flushTimer = setTimeout(() => flush(false), 24)
  }

  function stopTimers () {
    if (flushTimer) { clearTimeout(flushTimer); flushTimer = null }
    if (tickTimer) { clearInterval(tickTimer); tickTimer = null }
  }

  function reset () {
    pendingDone = null
    stopTimers()
    status.value = 'IDLE'
    answer.value = ''
    outcome.value = null
    error.value = ''
    question.value = ''
    startedAt.value = null
    elapsedMs.value = 0
    tokenCount.value = 0
    meta.value = null
    stage.value = ''
    notice.value = ''
    buffer = ''
    pendingDone = null
  }

  /**
   * End a task at an authentication boundary and remove all partial data.
   * Unlike cancel(), this intentionally retains nothing for the next user.
   */
  function discardSession () {
    generation++
    const activeController = controller
    controller = null
    if (activeController) activeController.abort()
    reset()
  }

  /** 取消正在进行的流式请求（连接会被真正中止，后端也会停止推送）。 */
  function cancel () {
    generation++
    const activeController = controller
    controller = null
    if (activeController) activeController.abort()
    pendingDone = null
    flush(true)
    stopTimers()
    status.value = 'CANCELLED'
    ElMessage.info('已取消本次流式分析')
  }

  function parseEvent (chunk) {
    let name = 'message'
    const data = []
    for (const line of chunk.split('\n')) {
      if (line.startsWith('event:')) name = line.slice(6).trim()
      // 只切掉「data:」这 5 个字符，不能 trim 也不能再吃掉一个空格：
      // 服务端（Spring SseEmitter）写的是 `data:` + 原文，没有补空格，
      // 所以行首缩进本身就是正文内容，削掉就会让流式正文与最终答案对不上。
      else if (line.startsWith('data:')) data.push(line.slice(5))
    }
    return { name, data: data.join('\n') }
  }

  function parseJsonStr (v) {
    if (!v) return null
    try { return typeof v === 'string' ? JSON.parse(v) : v } catch { return null }
  }

  function finish (d) {
    pendingDone = d || {}
    if (buffer) scheduleFlush()
    else {
      const done = pendingDone
      pendingDone = null
      finalize(done)
    }
  }

  /**
   * 发起一次流式分析。payload 与 /api/ai/analyze/stream 的请求体一致。
   * 组件卸载不会中断它。
   */
  async function start (payload) {
    if (running.value) { ElMessage.warning('已有流式分析在进行中'); return }
    reset()
    const runGeneration = ++generation
    status.value = 'RUNNING'
    question.value = payload.question
    startedAt.value = Date.now()
    tickTimer = setInterval(() => {
      if (startedAt.value) elapsedMs.value = Date.now() - startedAt.value
    }, 500)

    controller = new AbortController()
    try {
      const token = localStorage.getItem('token')
      const resp = await fetch('/api/ai/analyze/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token },
        body: JSON.stringify(payload),
        signal: controller.signal
      })
      if (runGeneration !== generation) return
      if (!resp.ok) {
        status.value = 'ERROR'
        error.value = '流式请求失败: HTTP ' + resp.status
        stopTimers()
        ElMessage.error(error.value)
        return
      }
      const reader = resp.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buf = ''
      while (true) {
        const { done, value } = await reader.read()
        if (runGeneration !== generation) return
        if (done) break
        buf += decoder.decode(value, { stream: true })
        let idx
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const chunk = buf.slice(0, idx)
          buf = buf.slice(idx + 2)
          handleEvent(parseEvent(chunk))
        }
      }
      if (buf.trim()) handleEvent(parseEvent(buf))
      if (status.value === 'RUNNING' && !pendingDone) {
        // 读完流却没等到 done，而且服务端也没明说过原因 —— 不能把这种情况显示成"分析结束但没有结果"。
        // 历史上这类情况正是界面出现「没有结果」的主要来源：后端也许还在写第 3000 个字。
        if (!outcome.value && !notice.value) {
          notice.value = '流式连接已结束，但未收到完成事件。本次分析可能仍在后台进行，请稍后在历史记录中查看。'
        }
        finish(outcome.value || {})
      }
    } catch (e) {
      if (runGeneration !== generation) return
      pendingDone = null
      flush(true)
      stopTimers()
      if (e && e.name === 'AbortError') {
        status.value = 'CANCELLED'
      } else {
        status.value = 'ERROR'
        error.value = '流式异常: ' + (e && e.message ? e.message : e)
        ElMessage.error(error.value)
      }
    } finally {
      if (runGeneration === generation) controller = null
    }
  }

  function handleEvent (ev) {
    if (ev.name === 'token') {
      // Python SSE JSON-encodes token strings, while the legacy Java emitter
      // wrote raw text. Accept both forms so quotes and escaped newlines never
      // leak into the report body.
      const decoded = parseJsonStr(ev.data)
      buffer += typeof decoded === 'string' ? decoded : ev.data
      tokenCount.value++
      scheduleFlush()
    } else if (ev.name === 'meta') {
      // 先行事实 + 进度播报走同一个事件名：合并而不是替换，
      // 否则每一句「正在调用…」都会把刚才那轮取到的「风险等级 / 来源构成」抹掉。
      const d = parseJsonStr(ev.data) || {}
      meta.value = { ...(meta.value || {}), ...d }
      if (d.stage) stage.value = d.stage
    } else if (ev.name === 'done') {
      finish(parseJsonStr(ev.data))
    } else if (ev.name === 'error') {
      pendingDone = null
      flush(true)
      status.value = 'ERROR'
      error.value = ev.data
      stopTimers()
      ElMessage.error('流式错误: ' + ev.data)
    } else if (ev.name === 'timeout') {
      // 不是失败：分析可能还在后台跑。如实告诉用户去哪里取，
      // 而不是让他以为这次分析"没有任何结果"。
      notice.value = ev.data
      ElMessage.warning(ev.data)
    }
    // comment（`:ping ...`）被 parseEvent 解析为无名行，走到这里 name 不匹配任何分支，安全忽略。
  }

  return {
    status, answer, outcome, error, question, meta, stage, notice,
    startedAt, elapsedMs, tokenCount,
    running, evidence, webSources,
    start, cancel, reset, discardSession
  }
})
