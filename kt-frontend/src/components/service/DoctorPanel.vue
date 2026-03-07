<template>
  <div class="doctor-panel">
    <div class="header">
      <h3>System Diagnostics</h3>
      <el-button type="primary" @click="runDiagnostics" :loading="running">
        Run Diagnostics
      </el-button>
    </div>

    <el-alert v-if="errorMsg" :title="errorMsg" type="error" show-icon style="margin-top: 12px" />

    <el-table v-if="checkItems.length > 0" :data="checkItems" style="margin-top: 20px">
      <el-table-column label="Check" prop="name" width="180" />
      <el-table-column label="Status" width="90">
        <template #default="{ row }">
          <el-icon v-if="row.status === 'ok'" color="#67C23A" :size="20"><CircleCheck /></el-icon>
          <el-icon v-else-if="row.status === 'warning'" color="#E6A23C" :size="20"><Warning /></el-icon>
          <el-icon v-else color="#F56C6C" :size="20"><CircleClose /></el-icon>
        </template>
      </el-table-column>
      <el-table-column label="Value" prop="value" />
      <el-table-column label="Notes" prop="hint" show-overflow-tooltip>
        <template #default="{ row }">
          <span v-if="row.hint" class="hint">{{ row.hint }}</span>
        </template>
      </el-table-column>
    </el-table>

    <el-empty v-else-if="!running" description="Click 'Run Diagnostics' to check your environment" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { CircleCheck, Warning, CircleClose } from '@element-plus/icons-vue'

const running = ref(false)
const checkItems = ref<any[]>([])
const errorMsg = ref('')

async function runDiagnostics() {
  running.value = true
  errorMsg.value = ''
  checkItems.value = []
  try {
    const result = await window.electronAPI.doctor.run()
    if (result.success === false && result.error) {
      errorMsg.value = result.error
    }
    // doctor_json.py returns { checks: [...], success: bool }
    // Each check: { name, status ('ok'/'warning'/'error'), value, hint }
    checkItems.value = (result.checks || []).map((c: any) => ({
      name: c.name,
      status: c.status === 'ok' ? 'ok' : c.status === 'warning' ? 'warning' : 'error',
      value: c.value || '',
      hint: c.hint || ''
    }))
  } catch (e: any) {
    errorMsg.value = e.message || 'Diagnostics failed'
  } finally {
    running.value = false
  }
}
</script>

<style scoped>
.doctor-panel { padding: 20px; }
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.hint { color: #909399; font-size: 12px; }
</style>
