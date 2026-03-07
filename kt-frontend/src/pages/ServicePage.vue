<template>
  <div class="service-page">
    <h2>{{ t('service.title') }}</h2>
    <ServerControl />

    <el-tabs v-model="activeTab" style="margin-top: 20px">
      <el-tab-pane label="Logs" name="logs">
        <el-card style="height: 500px">
          <LogViewer :logs="logs" @clear="handleClearLogs" />
        </el-card>
      </el-tab-pane>

      <el-tab-pane label="Diagnostics" name="diagnostics">
        <DoctorPanel />
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useServerStore } from '@/stores/server'
import { storeToRefs } from 'pinia'
import ServerControl from '@/components/service/ServerControl.vue'
import LogViewer from '@/components/service/LogViewer.vue'
import DoctorPanel from '@/components/service/DoctorPanel.vue'

const { t } = useI18n()
const serverStore = useServerStore()
const { logs } = storeToRefs(serverStore)
const activeTab = ref('logs')

onMounted(() => {
  window.electronAPI.server.onLog((log) => serverStore.addLog(log))
})

function handleClearLogs() {
  serverStore.clearLogs()
}
</script>

<style scoped>
.service-page {
  padding: 20px;
}
</style>
