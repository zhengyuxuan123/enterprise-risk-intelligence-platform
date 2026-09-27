<template>
  <div class="page">
    <div class="page-head"><div><h2>经营风险看板</h2><p>汇总当前权限范围内的企业、风险与关键状态。</p></div><el-button @click="load">刷新</el-button></div>
    <el-row :gutter="16" v-loading="loading">
      <el-col :span="6" v-for="c in cards" :key="c.label"><el-card><div class="stat-label">{{c.label}}</div><div class="stat-value">{{c.value}}</div></el-card></el-col>
    </el-row>
    <el-row :gutter="16" style="margin-top:16px">
      <el-col :span="12"><el-card><template #header>风险等级分布</template><div ref="riskChart" class="chart"></div></el-card></el-col>
      <el-col :span="12"><el-card><template #header>风险状态分布</template><div ref="statusChart" class="chart"></div></el-card></el-col>
    </el-row>
    <el-card style="margin-top:16px"><template #header>使用建议</template><el-alert type="info" :closable="false" show-icon title="先从风险事件定位异常，再进入 AI Agent 分析，系统会自动汇总指标、投诉、竞品和知识库证据。"/></el-card>
  </div>
</template>
<script setup>
import{ref,onMounted,nextTick,onBeforeUnmount}from'vue';import*as echarts from'echarts';import{api}from'../api';
const loading=ref(false),summary=ref({}),riskChart=ref(),statusChart=ref();let c1,c2;
const cards=ref([{label:'可见企业',value:0},{label:'风险事件',value:0},{label:'高风险',value:0},{label:'待处理',value:0}]);
function pairs(v){if(!v)return[];return Object.entries(v).map(([name,value])=>({name,value}))}
async function load(){loading.value=true;try{const r=await api.dashboard();summary.value=r||{};const s=summary.value;cards.value=[{label:'可见企业',value:s.companyCount||0},{label:'风险事件',value:s.riskCount||0},{label:'高风险',value:s.highRiskCount||0},{label:'待处理',value:s.pendingRiskCount||0}];await nextTick();c1?.dispose();c2?.dispose();c1=echarts.init(riskChart.value);c2=echarts.init(statusChart.value);c1.setOption({tooltip:{},series:[{type:'pie',radius:['40%','68%'],data:pairs(s.riskLevelDistribution)}]});c2.setOption({tooltip:{},xAxis:{type:'category',data:Object.keys(s.riskStatusDistribution||{})},yAxis:{type:'value'},series:[{type:'bar',data:Object.values(s.riskStatusDistribution||{})}]});}finally{loading.value=false}}
onMounted(load);onBeforeUnmount(()=>{c1?.dispose();c2?.dispose()});
</script>
