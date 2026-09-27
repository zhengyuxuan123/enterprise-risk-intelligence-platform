<template>
  <div class="page">
    <div class="page-head"><div><h2>知识库 / RAG</h2><p>原文件持久化保存，后台解析、切片并增量写入检索索引。</p></div></div>
    <div class="summary">
      <div><strong>{{ rows.length }}</strong><span>文档</span></div>
      <div><strong>{{ stats.indexed }}</strong><span>已索引</span></div>
      <div><strong>{{ stats.running }}</strong><span>处理中</span></div>
      <div><strong :class="{danger:stats.failed}">{{ stats.failed }}</strong><span>失败</span></div>
      <div><strong>{{ stats.chunks }}</strong><span>切片</span></div>
      <div><strong>{{ stats.quality }}</strong><span>平均质量</span></div>
    </div>
    <el-row :gutter="16">
      <el-col :xs="24" :lg="10"><el-card><template #header>上传文档</template>
        <el-form label-width="85px">
          <el-form-item label="企业ID"><el-input-number v-model="upload.companyId" :min="1"/></el-form-item>
          <el-form-item label="部门ID"><el-input-number v-model="upload.deptId" :min="1"/></el-form-item>
          <el-form-item label="文档类型"><el-input v-model="upload.docType" placeholder="产品资料/政策/历史报告"/></el-form-item>
          <el-form-item label="密级"><el-input-number v-model="upload.securityLevel" :min="1" :max="5"/></el-form-item>
          <el-form-item label="标题"><el-input v-model="upload.title"/></el-form-item>
          <el-form-item label="文件"><input type="file" accept=".txt,.md,.csv,.pdf,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.rtf" @change="pick"/></el-form-item>
          <el-button type="primary" :loading="uploading" v-if="auth.has('knowledge:write')" @click="doUpload">创建入库任务</el-button>
        </el-form>
      </el-card></el-col>
      <el-col :xs="24" :lg="14"><el-card><template #header>RAG 检索测试</template>
        <el-form inline><el-form-item label="企业ID"><el-input-number v-model="search.companyId" :min="1"/></el-form-item>
          <el-form-item label="问题"><el-input v-model="search.q" class="query" placeholder="例如：续费价格政策是什么？"/></el-form-item>
          <el-button type="primary" :loading="searching" @click="doSearch">检索</el-button></el-form>
        <el-empty v-if="!hits.length" description="未检索到相关文档，可换一种问法"/>
        <div v-for="(h,i) in hits" :key="i" class="hit"><div class="hit-head"><b>{{i+1}}. {{h.title}}</b><el-tag size="small" effect="plain" :type="scoreType(h.score)">相关度 {{Number(h.score||0).toFixed(1)}}</el-tag></div><p>{{h.snippet}}</p></div>
      </el-card></el-col>
    </el-row>
    <el-card class="docs"><template #header><div class="card-head"><span>知识文档</span><el-button @click="load">刷新</el-button></div></template>
      <el-table :data="rows" stripe>
        <el-table-column prop="id" label="ID" width="65"/><el-table-column prop="companyId" label="企业" width="70"/>
        <el-table-column prop="title" label="标题" min-width="170" show-overflow-tooltip/>
        <el-table-column label="版本" width="70"><template #default="s">v{{s.row.documentVersion||1}}</template></el-table-column>
        <el-table-column label="入库状态" width="120"><template #default="s"><el-tag size="small" :type="statusType(s.row.ingestStatus)">{{statusText(s.row.ingestStatus)}}</el-tag></template></el-table-column>
        <el-table-column label="进度" width="130"><template #default="s"><el-progress :percentage="jobOf(s.row).progress||statusProgress(s.row.ingestStatus)" :status="s.row.ingestStatus==='FAILED'?'exception':undefined"/></template></el-table-column>
        <el-table-column label="质量" width="90"><template #default="s">{{quality(s.row.parseQualityScore)}}</template></el-table-column>
        <el-table-column prop="chunkCount" label="切片" width="70"/><el-table-column prop="originalFilename" label="原文件" min-width="150" show-overflow-tooltip/>
        <el-table-column label="异常/提示" min-width="190" show-overflow-tooltip><template #default="s">{{s.row.ingestError||s.row.parseWarning||'-'}}</template></el-table-column>
        <el-table-column label="操作" width="185" fixed="right" v-if="auth.has('knowledge:write')"><template #default="s">
          <el-button link type="primary" @click="chooseReplacement(s.row)">换版</el-button>
          <el-button v-if="canRetry(s.row)" link type="warning" @click="retry(s.row.id)">重试</el-button>
          <el-button link type="danger" @click="remove(s.row.id)">删除</el-button>
        </template></el-table-column>
      </el-table>
      <input ref="replaceInput" type="file" class="hidden" @change="replaceSelected"/>
    </el-card>
  </div>
</template>

<script setup>
import{computed,onBeforeUnmount,onMounted,ref}from'vue';
import{ElMessage,ElMessageBox}from'element-plus';
import{api}from'../api';import{useAuthStore}from'../stores/auth';
const auth=useAuthStore(),rows=ref([]),jobs=ref([]),hits=ref([]),file=ref(null),uploading=ref(false),searching=ref(false);
const replaceInput=ref(null),replaceRow=ref(null);
const upload=ref({companyId:1,deptId:2,docType:'业务资料',securityLevel:1,title:''});
const search=ref({companyId:1,q:'客户流失风险应该如何分析？'});let timer;
const stats=computed(()=>{const active=new Set(['PENDING','STORING','PARSING','INDEX_PENDING','INDEX_RETRY']);const q=rows.value.map(x=>Number(x.parseQualityScore)).filter(Number.isFinite);return{indexed:rows.value.filter(x=>x.ingestStatus==='INDEXED').length,running:rows.value.filter(x=>active.has(x.ingestStatus)).length,failed:rows.value.filter(x=>x.ingestStatus==='FAILED').length,chunks:rows.value.reduce((n,x)=>n+Number(x.chunkCount||0),0),quality:q.length?(q.reduce((a,b)=>a+b,0)/q.length).toFixed(1):'-'}});
function pick(e){file.value=e.target.files?.[0]||null}
async function load(){const [docs,taskRows]=await Promise.all([api.knowledge(),api.knowledgeJobs()]);rows.value=docs||[];jobs.value=taskRows||[]}
function jobOf(row){return jobs.value.find(j=>j.documentId===row.id&&j.documentVersion===row.documentVersion)||{}}
async function doUpload(){if(!file.value)return ElMessage.warning('请选择文件');const fd=new FormData();Object.entries(upload.value).forEach(([k,v])=>{if(v!==null&&v!=='')fd.append(k,v)});fd.append('file',file.value);uploading.value=true;try{await api.uploadKnowledge(fd);ElMessage.success('入库任务已创建，可在列表查看进度');await load()}finally{uploading.value=false}}
async function doSearch(){if(!search.value.q)return;searching.value=true;try{hits.value=(await api.searchKnowledge({...search.value,topK:5}))||[]}finally{searching.value=false}}
async function retry(id){await api.retryKnowledge(id);ElMessage.success('已重新加入任务队列');await load()}
function chooseReplacement(row){replaceRow.value=row;replaceInput.value?.click()}
async function replaceSelected(e){const f=e.target.files?.[0];if(!f||!replaceRow.value)return;const fd=new FormData();fd.append('file',f);try{await api.replaceKnowledge(replaceRow.value.id,fd);ElMessage.success('新版本已提交，旧索引将在新版成功后替换');await load()}finally{e.target.value='';replaceRow.value=null}}
async function remove(id){await ElMessageBox.confirm('删除后会同步清理检索切片，原文件保留用于审计。','删除文档');await api.deleteKnowledge(id);ElMessage.success('已提交删除事件');await load()}
function canRetry(r){return['FAILED','INDEX_RETRY'].includes(r.ingestStatus)}
function quality(v){return v==null?'-':`${Number(v).toFixed(1)}分`}
function scoreType(s){const v=Number(s||0);return v>=60?'success':v>=30?'warning':'info'}
function statusType(s){if(s==='INDEXED')return'success';if(s==='FAILED')return'danger';if(s==='INDEX_RETRY')return'warning';return'info'}
function statusText(s){return({STORING:'保存文件',PENDING:'等待解析',PARSING:'解析中',INDEX_PENDING:'等待索引',INDEX_RETRY:'索引重试',INDEXED:'已索引',FAILED:'失败'}[s]||s||'未知')}
function statusProgress(s){return({STORING:10,PENDING:15,PARSING:30,INDEX_PENDING:75,INDEX_RETRY:80,INDEXED:100,FAILED:30}[s]||0)}
onMounted(()=>{load();timer=setInterval(()=>{if(stats.value.running)load()},2500)});onBeforeUnmount(()=>clearInterval(timer));
</script>

<style scoped>
.summary{display:grid;grid-template-columns:repeat(6,minmax(90px,1fr));gap:1px;background:#e2e8f0;border:1px solid #e2e8f0;margin-bottom:16px}.summary div{background:#fff;padding:14px 16px;display:flex;align-items:baseline;gap:8px}.summary strong{font-size:22px;color:#0f172a}.summary span{font-size:13px;color:#64748b}.summary .danger{color:#dc2626}.query{width:320px}.docs{margin-top:16px}.card-head,.hit-head{display:flex;align-items:center;justify-content:space-between;gap:10px}.hit{padding:12px;border-bottom:1px solid #eee}.hit p{white-space:pre-wrap;color:#475569;line-height:1.6}.hidden{display:none}@media(max-width:900px){.summary{grid-template-columns:repeat(3,1fr)}.query{width:220px}}
</style>
