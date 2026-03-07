<template>
  <div class="dashboard">
    <!-- Connection Status Bar -->
    <el-card class="status-bar" :body-style="{ padding: '16px 24px' }">
      <div class="status-row">
        <div class="status-left">
          <span class="status-dot" :class="serverAlive ? 'online' : 'offline'" />
          <span class="status-text">{{ serverAlive ? 'Connected' : 'Disconnected' }}</span>
          <el-tag v-if="serverAlive && serverInfo" size="small" type="success">{{ serverInfo.status }}</el-tag>
          <el-tag v-if="serverInfo?.version" size="small">SGLang v{{ serverInfo.version }}</el-tag>
        </div>
        <div class="status-right">
          <el-input
            v-model="currentUrl"
            size="small"
            style="width: 320px"
            placeholder="Server URL"
            @change="reconnect"
          >
            <template #prepend>URL</template>
            <template #append>
              <el-button @click="reconnect" :loading="checking">Connect</el-button>
            </template>
          </el-input>
        </div>
      </div>
    </el-card>

    <!-- Server Overview Cards -->
    <el-row :gutter="16" class="section">
      <el-col :span="6">
        <el-card shadow="hover" class="metric-card">
          <div class="metric-icon" style="background: #e8f5e9"><span style="color: #4caf50">M</span></div>
          <div class="metric-body">
            <div class="metric-label">Model</div>
            <div class="metric-value">{{ modelName || '—' }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover" class="metric-card">
          <div class="metric-icon" style="background: #e3f2fd"><span style="color: #2196f3">T</span></div>
          <div class="metric-body">
            <div class="metric-label">Throughput</div>
            <div class="metric-value">{{ throughputDisplay }} tok/s</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover" class="metric-card">
          <div class="metric-icon" style="background: #fff3e0"><span style="color: #ff9800">V</span></div>
          <div class="metric-body">
            <div class="metric-label">VRAM Weight</div>
            <div class="metric-value">{{ serverInfo?.memoryUsage ? serverInfo.memoryUsage.weight.toFixed(1) + ' GB' : '—' }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover" class="metric-card">
          <div class="metric-icon" style="background: #fce4ec"><span style="color: #e91e63">K</span></div>
          <div class="metric-body">
            <div class="metric-label">KV Cache</div>
            <div class="metric-value">{{ serverInfo?.memoryUsage ? serverInfo.memoryUsage.kvcache.toFixed(2) + ' GB' : '—' }}</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Detailed Panels -->
    <el-row :gutter="16" class="section">
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>Model Information</span>
              <el-tag v-if="modelArch?.modelType" size="small">{{ modelArch.modelType }}</el-tag>
            </div>
          </template>
          <div v-if="serverAlive" class="info-list">
            <div class="info-item"><span>Name</span><span>{{ modelName || '—' }}</span></div>
            <div class="info-item"><span>Architecture</span><span>{{ modelArch?.architectures?.join(', ') || '—' }}</span></div>
            <div class="info-item"><span>Max Context</span><span>{{ maxContext ? maxContext.toLocaleString() + ' tokens' : '—' }}</span></div>
            <div class="info-item"><span>Token Capacity</span><span>{{ serverInfo?.memoryUsage?.tokenCapacity?.toLocaleString() || '—' }}</span></div>
            <div class="info-item"><span>Model Path</span><span class="path-text">{{ serverInfo?.modelPath || '—' }}</span></div>
          </div>
          <el-empty v-else description="Not connected" :image-size="60" />
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>Server Configuration</span>
              <el-tag v-if="serverInfo?.device" size="small">{{ serverInfo.device.toUpperCase() }}</el-tag>
            </div>
          </template>
          <div v-if="serverAlive && serverInfo" class="info-list">
            <div class="info-item"><span>TP Size</span><span>{{ serverInfo.tpSize }}</span></div>
            <div class="info-item"><span>DP Size</span><span>{{ serverInfo.dpSize }}</span></div>
            <div class="info-item"><span>Max Tokens</span><span>{{ serverInfo.maxTotalTokens.toLocaleString() }}</span></div>
            <div class="info-item" v-if="serverInfo.ktMethod">
              <span>KT Method</span>
              <span>{{ serverInfo.ktMethod }}</span>
            </div>
            <div class="info-item" v-if="serverInfo.ktCpuInfer">
              <span>CPU Infer Threads</span>
              <span>{{ serverInfo.ktCpuInfer }}</span>
            </div>
            <div class="info-item" v-if="serverInfo.ktNumGpuExperts">
              <span>GPU Experts</span>
              <span>{{ serverInfo.ktNumGpuExperts }}</span>
            </div>
            <div class="info-item" v-if="serverInfo.ktWeightPath">
              <span>GGUF Weights</span>
              <span class="path-text">{{ serverInfo.ktWeightPath }}</span>
            </div>
          </div>
          <el-empty v-else description="Not connected" :image-size="60" />
        </el-card>
      </el-col>
    </el-row>

    <!-- Memory Usage Visual -->
    <el-row :gutter="16" class="section" v-if="serverAlive && serverInfo?.memoryUsage">
      <el-col :span="24">
        <el-card>
          <template #header>VRAM Usage</template>
          <div class="vram-bar-container">
            <div class="vram-bar">
              <div class="vram-segment weight" :style="{ width: vramWeightPct + '%' }" />
              <div class="vram-segment kvcache" :style="{ width: vramKvPct + '%' }" />
              <div class="vram-segment graph" :style="{ width: vramGraphPct + '%' }" />
            </div>
            <div class="vram-legend">
              <span><i class="legend-dot" style="background:#409eff"/> Weight: {{ serverInfo.memoryUsage.weight.toFixed(1) }} GB</span>
              <span><i class="legend-dot" style="background:#67c23a"/> KV Cache: {{ serverInfo.memoryUsage.kvcache.toFixed(2) }} GB</span>
              <span><i class="legend-dot" style="background:#e6a23c"/> Graph: {{ serverInfo.memoryUsage.graph.toFixed(2) }} GB</span>
              <span class="vram-total">Total: {{ vramTotal.toFixed(1) }} GB</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Throughput Chart -->
    <el-row :gutter="16" class="section">
      <el-col :span="24">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>Throughput</span>
              <el-tag v-if="serverAlive" size="small" type="success">Live</el-tag>
            </div>
          </template>
          <ThroughputChart :server-url="resolvedUrl" :server-alive="serverAlive" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useConfigStore } from '@/stores/config'
import ThroughputChart from '@/components/dashboard/ThroughputChart.vue'
import { fetchServerInfo, fetchServerModels, fetchModelArchInfo } from '@/api/server-client'
import type { ServerInfo, ModelInfo } from '@/api/server-client'

const configStore = useConfigStore()

const checking = ref(false)
const serverAlive = ref(false)
const serverInfo = ref<ServerInfo | null>(null)
const modelArch = ref<{ modelType: string; architectures: string[]; modelPath: string } | null>(null)
const remoteModels = ref<ModelInfo[]>([])
const currentUrl = ref('')

const resolvedUrl = computed(() => {
  const url = currentUrl.value.trim().replace(/\/+$/, '')
  return url || 'http://localhost:30000'
})

const modelName = computed(() => {
  if (serverInfo.value?.servedModelName) return serverInfo.value.servedModelName
  if (remoteModels.value.length) return remoteModels.value[0].id
  return ''
})

const maxContext = computed(() => {
  if (remoteModels.value.length && remoteModels.value[0].maxModelLen) return remoteModels.value[0].maxModelLen
  return serverInfo.value?.maxTotalTokens || 0
})

const throughputDisplay = computed(() => {
  if (!serverInfo.value) return '—'
  return serverInfo.value.lastGenThroughput.toFixed(1)
})

const vramTotal = computed(() => {
  const m = serverInfo.value?.memoryUsage
  if (!m) return 0
  return m.weight + m.kvcache + m.graph
})

const vramWeightPct = computed(() => {
  if (!vramTotal.value) return 0
  return (serverInfo.value!.memoryUsage!.weight / vramTotal.value) * 100
})
const vramKvPct = computed(() => {
  if (!vramTotal.value) return 0
  return (serverInfo.value!.memoryUsage!.kvcache / vramTotal.value) * 100
})
const vramGraphPct = computed(() => {
  if (!vramTotal.value) return 0
  return (serverInfo.value!.memoryUsage!.graph / vramTotal.value) * 100
})

let pollInterval: any

async function fetchAll() {
  checking.value = true
  try {
    const [models, info, arch] = await Promise.all([
      fetchServerModels(resolvedUrl.value),
      fetchServerInfo(resolvedUrl.value),
      fetchModelArchInfo(resolvedUrl.value),
    ])
    remoteModels.value = models
    serverInfo.value = info
    modelArch.value = arch
    serverAlive.value = !!info || models.length > 0
  } catch {
    serverAlive.value = false
  } finally {
    checking.value = false
  }
}

async function reconnect() {
  await fetchAll()
}

function startPolling() {
  pollInterval = setInterval(async () => {
    if (!serverAlive.value) return
    // Light poll: just server info for throughput + memory updates
    const info = await fetchServerInfo(resolvedUrl.value)
    if (info) {
      serverInfo.value = info
    } else {
      serverAlive.value = false
    }
  }, 5000)
}

onMounted(async () => {
  await configStore.load()
  // Init URL from config
  const cfg = configStore.config?.server
  if (cfg?.mode === 'remote' && cfg.remoteUrl) {
    currentUrl.value = cfg.remoteUrl.replace(/\/+$/, '')
  } else {
    const host = (cfg?.host === '0.0.0.0' ? 'localhost' : cfg?.host) || 'localhost'
    const port = cfg?.port || 30000
    currentUrl.value = `http://${host}:${port}`
  }
  await fetchAll()
  startPolling()
})

onUnmounted(() => {
  if (pollInterval) clearInterval(pollInterval)
})
</script>

<style scoped>
.dashboard {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}
.section { margin-top: 16px; }

/* Status bar */
.status-bar { margin-bottom: 4px; }
.status-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}
.status-left {
  display: flex;
  align-items: center;
  gap: 10px;
}
.status-right { display: flex; align-items: center; gap: 8px; }
.status-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  display: inline-block;
}
.status-dot.online { background: #67c23a; box-shadow: 0 0 6px #67c23a; }
.status-dot.offline { background: #909399; }
.status-text { font-weight: 500; font-size: 14px; }

/* Metric cards */
.metric-card :deep(.el-card__body) {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 18px;
}
.metric-icon {
  width: 48px; height: 48px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  font-weight: 700;
  flex-shrink: 0;
}
.metric-body { flex: 1; min-width: 0; }
.metric-label { font-size: 12px; color: #909399; }
.metric-value {
  font-size: 16px;
  font-weight: 600;
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Card header */
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

/* Info list */
.info-list {
  display: flex;
  flex-direction: column;
  gap: 0;
}
.info-item {
  display: flex;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px solid #f5f5f5;
  font-size: 13px;
}
.info-item:last-child { border-bottom: none; }
.info-item span:first-child { color: #909399; flex-shrink: 0; margin-right: 12px; }
.path-text {
  font-family: monospace;
  font-size: 12px;
  word-break: break-all;
  text-align: right;
  max-width: 350px;
}

/* VRAM bar */
.vram-bar-container { padding: 8px 0; }
.vram-bar {
  height: 28px;
  background: #f0f2f5;
  border-radius: 6px;
  display: flex;
  overflow: hidden;
}
.vram-segment {
  height: 100%;
  transition: width 0.5s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  color: white;
  font-weight: 500;
}
.vram-segment.weight { background: #409eff; }
.vram-segment.kvcache { background: #67c23a; }
.vram-segment.graph { background: #e6a23c; }
.vram-legend {
  display: flex;
  gap: 20px;
  margin-top: 10px;
  font-size: 13px;
  color: #606266;
}
.legend-dot {
  display: inline-block;
  width: 10px; height: 10px;
  border-radius: 2px;
  margin-right: 4px;
  vertical-align: middle;
}
.vram-total { margin-left: auto; font-weight: 600; }
</style>
