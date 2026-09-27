import http from'./http';import axios from'axios';export const api={login:d=>http.post('/auth/login',d),dashboard:()=>http.get('/dashboard/summary'),companies:(p={})=>http.get('/companies',{params:p}),createCompany:d=>http.post('/companies',d),updateCompany:(id,d)=>http.put(`/companies/${id}`,d),deleteCompany:id=>http.delete(`/companies/${id}`),metrics:(p={})=>http.get('/metrics',{params:p}),createMetric:d=>http.post('/metrics',d),rules:()=>http.get('/risk-rules'),createRule:d=>http.post('/risk-rules',d),updateRule:(id,d)=>http.put(`/risk-rules/${id}`,d),deleteRule:id=>http.delete(`/risk-rules/${id}`),risks:(p={})=>http.get('/risks',{params:p}),assignRisk:(id,d)=>http.post(`/risks/${id}/assign`,d),handleRisk:(id,d)=>http.post(`/risks/${id}/handle`,d),reviewRisk:(id,d)=>http.post(`/risks/${id}/review`,d),closeRisk:id=>http.post(`/risks/${id}/close`),complaints:(p={})=>http.get('/complaints',{params:p}),createComplaint:d=>http.post('/complaints',d),competitors:(p={})=>http.get('/competitors',{params:p}),createCompetitor:d=>http.post('/competitors',d),knowledge:(p={})=>http.get('/knowledge',{params:p}),uploadKnowledge:fd=>http.post('/knowledge/upload',fd,{headers:{'Content-Type':'multipart/form-data'}}),searchKnowledge:p=>http.get('/knowledge/search',{params:p}),deleteKnowledge:id=>http.delete(`/knowledge/${id}`),analyze:d=>http.post('/ai/analyze',d,{timeout:660000}),aiHistory:p=>http.get('/ai/history',{params:p}),aiConversations:p=>http.get('/ai/conversations',{params:p}),aiEvaluate:p=>http.get('/ai/evaluate',{params:p,timeout:600000}),aiAsyncSubmit:d=>http.post('/ai/analyze/async',d),aiAsyncStatus:t=>http.get('/ai/async/'+t),users:()=>http.get('/system/users'),roles:()=>http.get('/system/roles'),departments:()=>http.get('/system/departments'),createUser:d=>http.post('/system/users',d),importExcel:(fd)=>http.post('/import/excel',fd,{headers:{'Content-Type':'multipart/form-data'}}),importHistory:()=>http.get('/import/history'),aiHistoryDetail:id=>http.get('/ai/history/'+id),deleteAiHistory:id=>http.delete('/ai/history/'+id),clearAiHistory:p=>http.delete('/ai/history',{params:p}),deleteAiConversation:id=>http.delete('/ai/conversations/'+id),aiEvaluateAsync:p=>http.post('/ai/evaluate/async',null,{params:p}),aiEvaluateStatus:t=>http.get('/ai/evaluate/async/'+t),aiDiagnose:()=>http.get('/ai/diagnose',{timeout:660000}),aiUpdateKey:p=>http.post('/ai/key',typeof p==='string'?{key:p}:p,{timeout:120000}),aiEvalReports:()=>http.get('/ai/evaluate/reports'),aiEvalReport:id=>http.get('/ai/evaluate/reports/'+id),aiModels:(refresh)=>http.get('/ai/models',{params:{refresh:!!refresh}}),aiAdoptModel:()=>http.post('/ai/models/adopt',{},{timeout:180000}),aiAutoConsume:enabled=>http.post('/ai/auto-consume',{enabled:!!enabled}),aiRagStatus:()=>http.get('/ai/rag/status'),aiRagFlush:()=>http.post('/ai/rag/flush'),aiRagReindex:cid=>http.post('/ai/rag/reindex',null,{params:{companyId:cid}}),aiRagProbe:p=>http.get('/ai/rag/probe',{params:p}),
// --- 审计与追溯：留痕 / 血缘 / 导出 / 审批队列 ---
aiTrace:id=>http.get('/ai/analysis/'+id+'/trace'),
aiLineage:id=>http.get('/ai/analysis/'+id+'/lineage'),
aiTraceByNo:tid=>http.get('/ai/trace/'+encodeURIComponent(tid)),
aiActions:cid=>http.get('/ai/actions',{params:cid?{companyId:cid}:{}}),
aiActionsAll:()=>http.get('/ai/actions/all'),
aiDecide:(id,d)=>http.post('/ai/actions/'+id+'/decide',d),
aiExport:async(id,format)=>{
  // 导出返回的是裸文件（不是 ApiResponse），http 拦截器会对它没有 code 字段而误判失败，
  // 所以这里单独发一次带令牌的请求并自己拼下载。
  const r=await axios.get((import.meta.env.VITE_API_BASE_URL||'/api')+'/ai/analysis/'+id+'/export',
    {params:{format:format||'pdf'},responseType:'blob',headers:{Authorization:'Bearer '+(localStorage.getItem('token')||'')}});
  const ct=(r.headers&&r.headers['content-type'])||'application/pdf';
  const blob=r.data instanceof Blob?r.data:new Blob([r.data],{type:ct});
  let name='analysis-'+id+(format==='docx'?'.docx':'.pdf');
  const cd=(r.headers&&r.headers['content-disposition'])||'';
  const m=/filename\*=UTF-8''(.+)/i.exec(cd);if(m){try{name=decodeURIComponent(m[1])}catch(e){}}
  const url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download=name;document.body.appendChild(a);a.click();
  document.body.removeChild(a);URL.revokeObjectURL(url);return name}}
// --- 长期记忆（跨会话）：清单 / 召回预演 / 删除 / 清空 / 回填 ---
// 全部零 token，不消耗模型额度；DELETE 能过转发白名单（FORWARDABLE_METHODS 已含 DELETE）。
Object.assign(api,{
  knowledgeJobs:p=>http.get('/knowledge/jobs',{params:p||{}}),
  retryKnowledge:id=>http.post(`/knowledge/${id}/retry`),
  replaceKnowledge:(id,fd)=>http.post(`/knowledge/${id}/replace`,fd,{headers:{'Content-Type':'multipart/form-data'}}),
  memoryList:p=>http.get('/ai/memory',{params:p}),
  memoryProbe:p=>http.get('/ai/memory/probe',{params:p}),
  deleteMemory:id=>http.delete('/ai/memory/'+id),
  clearMemory:p=>http.post('/ai/memory/clear',null,{params:p}),
  memoryRebuild:p=>http.post('/ai/memory/rebuild',null,{params:p})});
