<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h2>数据接入中心</h2>
        <p>批量导入经营指标、投诉或竞品 Excel，自动校验并记录导入结果；指标导入成功后会继续触发风险检测。</p>
      </div>
    </div>

    <el-card>
      <el-form inline>
        <el-form-item label="导入类型">
          <el-select v-model="type" style="width:190px">
            <el-option label="经营指标 METRIC" value="METRIC" />
            <el-option label="投诉 COMPLAINT" value="COMPLAINT" />
            <el-option label="竞品 COMPETITOR" value="COMPETITOR" />
          </el-select>
        </el-form-item>
        <el-form-item label="Excel文件">
          <input type="file" accept=".xlsx,.xls" @change="e => file = e.target.files?.[0]" />
        </el-form-item>
        <el-button type="primary" :loading="loading" @click="upload">开始导入</el-button>
      </el-form>
      <el-alert
        :closable="false"
        type="info"
        show-icon
        title="字段可使用英文或中文表头；至少需要企业ID（或企业编码）以及对应业务必填字段。同一份文件重复上传会被自动识别并跳过；文件中与库内已有记录内容完全相同的行也不会重复写入。" />
    </el-card>

    <el-card style="margin-top:16px">
      <template #header>
        <div style="display:flex;justify-content:space-between">
          <span>我的导入历史</span>
          <el-button @click="load">刷新</el-button>
        </div>
      </template>
      <el-table :data="rows" stripe>
        <el-table-column prop="taskNo" label="任务号" width="190" />
        <el-table-column prop="importType" label="类型" width="120" />
        <el-table-column prop="originalFilename" label="文件" min-width="180" />
        <el-table-column prop="totalRows" label="总行数" width="80" />
        <el-table-column prop="successRows" label="成功" width="70" />
        <el-table-column prop="skippedRows" label="跳过" width="70" />
        <el-table-column prop="failedRows" label="失败" width="70" />
        <el-table-column prop="status" label="状态" width="110">
          <template #default="s">
            <el-tag :type="tagType(s.row.status)">{{ s.row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="说明" min-width="260" show-overflow-tooltip>
          <template #default="s">{{ s.row.message || s.row.errorMessage || '—' }}</template>
        </el-table-column>
        <el-table-column prop="createdAt" label="创建时间" width="180" />
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { ElMessage, ElNotification } from 'element-plus';
import { api } from '../api';

const type = ref('METRIC'), file = ref(null), loading = ref(false), rows = ref([]);

function tagType(status) {
  if (status === 'SUCCESS') return 'success';
  if (status === 'FAILED') return 'danger';
  if (status === 'SKIPPED') return 'info';
  return 'warning';
}

async function upload() {
  if (!file.value) return ElMessage.warning('请选择Excel文件');
  const fd = new FormData();
  fd.append('importType', type.value);
  fd.append('file', file.value);
  loading.value = true;
  try {
    const r = await api.importExcel(fd);
    if (r.status === 'SKIPPED') {
      ElNotification({ title: '已跳过重复导入', message: r.message || '该文件内容与此前导入一致，未重复写入。', type: 'warning', duration: 6000 });
    } else {
      const parts = [`成功 ${r.successRows || 0} 行`];
      if (r.skippedRows) parts.push(`跳过重复 ${r.skippedRows} 行`);
      if (r.failedRows) parts.push(`失败 ${r.failedRows} 行`);
      ElNotification({ title: '导入完成', message: parts.join('，') + (r.message ? `。${r.message}` : ''), type: r.failedRows ? 'warning' : 'success', duration: 6000 });
    }
    file.value = null;
    load();
  } finally {
    loading.value = false;
  }
}

async function load() {
  rows.value = (await api.importHistory()) || [];
}

onMounted(load);
</script>
