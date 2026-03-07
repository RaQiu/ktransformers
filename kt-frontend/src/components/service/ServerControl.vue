<template>
  <div class="server-control">
    <el-card>
      <div class="control-panel">
        <div class="status-section">
          <el-tag :type="status.running ? 'success' : 'info'" size="large">
            {{ status.running ? 'Running' : 'Stopped' }}
          </el-tag>
          <div v-if="status.running" class="status-details">
            <span>PID: {{ status.pid }}</span>
            <span>Uptime: {{ uptime }}</span>
          </div>
        </div>
        <div class="action-buttons">
          <el-button v-if="!status.running" type="primary" size="large" @click="showConfig = true">
            Configure & Start
          </el-button>
          <el-button v-else type="danger" size="large" @click="handleStop">
            Stop Server
          </el-button>
        </div>
      </div>
    </el-card>

    <el-drawer v-model="showConfig" title="Server Configuration" size="50%">
      <el-form label-width="150px">
        <el-form-item label="Model">
          <el-select v-model="config.model" placeholder="Select model">
            <el-option v-for="m in models" :key="m.name" :label="m.name" :value="m.name" />
          </el-select>
        </el-form-item>
        <el-form-item label="Host">
          <el-input v-model="config.host" />
        </el-form-item>
        <el-form-item label="Port">
          <el-input-number v-model="config.port" :min="1024" :max="65535" />
        </el-form-item>
        <el-form-item label="GPU Experts">
          <el-input-number v-model="config.gpuExperts" :min="0" />
        </el-form-item>
        <el-form-item label="CPU Threads">
          <el-input-number v-model="config.cpuThreads" :min="1" />
        </el-form-item>
        <el-form-item label="NUMA Nodes">
          <el-input-number v-model="config.numaNodes" :min="1" />
        </el-form-item>
        <el-form-item label="Tensor Parallel">
          <el-input-number v-model="config.tensorParallel" :min="1" />
        </el-form-item>
        <el-form-item label="Max Total Tokens">
          <el-input-number v-model="config.maxTokens" :min="1024" :step="1024" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showConfig = false">Cancel</el-button>
        <el-button type="primary" @click="handleStart">Start Server</el-button>
      </template>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useServerStore } from '@/stores/server'
import { useModelsStore } from '@/stores/models'
import { storeToRefs } from 'pinia'
import { ElMessage } from 'element-plus'

const serverStore = useServerStore()
const modelsStore = useModelsStore()
const { running, pid } = storeToRefs(serverStore)
const { models } = storeToRefs(modelsStore)

const showConfig = ref(false)
const startTime = ref<number>(0)
const uptime = ref('0s')

const status = computed(() => ({
  running: running.value,
  pid: pid.value
}))

const config = ref({
  model: '',
  host: '0.0.0.0',
  port: 30000,
  gpuExperts: 8,
  cpuThreads: 16,
  numaNodes: 2,
  tensorParallel: 1,
  maxTokens: 32768
})

onMounted(async () => {
  await serverStore.checkStatus()
  await modelsStore.load()
  if (models.value.length > 0) {
    config.value.model = models.value[0].name
  }
  setInterval(updateUptime, 1000)
})

function updateUptime() {
  if (running.value && startTime.value) {
    const seconds = Math.floor((Date.now() - startTime.value) / 1000)
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = seconds % 60
    uptime.value = `${h}h ${m}m ${s}s`
  }
}

async function handleStart() {
  const result = await serverStore.start(config.value)
  if (result.success) {
    startTime.value = Date.now()
    showConfig.value = false
    ElMessage.success('Server started')
  } else {
    ElMessage.error(result.error || 'Failed to start server')
  }
}

async function handleStop() {
  const result = await serverStore.stop()
  if (result.success) {
    startTime.value = 0
    ElMessage.success('Server stopped')
  }
}
</script>

<style scoped>
.control-panel {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.status-section {
  display: flex;
  align-items: center;
  gap: 20px;
}
.status-details {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 14px;
  color: #606266;
}
</style>
