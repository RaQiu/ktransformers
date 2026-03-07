<template>
  <div class="throughput-chart" style="height: 300px">
    <div v-if="!serverAlive" class="offline-hint">
      <el-empty description="Server offline — connect to see live metrics" :image-size="80" />
    </div>
    <v-chart v-else :option="option" autoresize />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { fetchServerMetrics, fetchThroughputFromServerInfo } from '@/api/server-client'

use([LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer])

const props = defineProps<{
  serverUrl: string
  serverAlive: boolean
}>()

const timestamps = ref<string[]>([])
const prefillData = ref<number[]>([])
const decodeData = ref<number[]>([])

const option = computed(() => ({
  tooltip: { trigger: 'axis' },
  legend: { data: ['Prefill', 'Decode'] },
  grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
  xAxis: { type: 'category', boundaryGap: false, data: timestamps.value },
  yAxis: { type: 'value', name: 'tokens/s' },
  series: [
    { name: 'Prefill', type: 'line', data: prefillData.value, smooth: true, areaStyle: { opacity: 0.1 }, itemStyle: { color: '#409eff' } },
    { name: 'Decode', type: 'line', data: decodeData.value, smooth: true, areaStyle: { opacity: 0.1 }, itemStyle: { color: '#67c23a' } }
  ]
}))

let interval: ReturnType<typeof setInterval> | null = null

async function poll() {
  if (!props.serverAlive) return
  // Try /metrics first, fallback to /get_server_info
  let snapshot = await fetchServerMetrics(props.serverUrl)
  if (!snapshot) {
    snapshot = await fetchThroughputFromServerInfo(props.serverUrl)
  }
  if (!snapshot) return
  timestamps.value.push(snapshot.timestamp)
  prefillData.value.push(snapshot.prefill)
  decodeData.value.push(snapshot.decode)
  if (timestamps.value.length > 30) {
    timestamps.value.shift()
    prefillData.value.shift()
    decodeData.value.shift()
  }
}

function startPolling() {
  stopPolling()
  if (props.serverAlive) {
    poll()
    interval = setInterval(poll, 3000)
  }
}

function stopPolling() {
  if (interval) { clearInterval(interval); interval = null }
}

function clearData() {
  timestamps.value = []
  prefillData.value = []
  decodeData.value = []
}

watch(() => props.serverAlive, (alive) => {
  if (alive) {
    startPolling()
  } else {
    stopPolling()
    clearData()
  }
})

watch(() => props.serverUrl, () => {
  clearData()
  if (props.serverAlive) startPolling()
})

onMounted(() => {
  if (props.serverAlive) startPolling()
})

onUnmounted(() => stopPolling())
</script>

<style scoped>
.throughput-chart { position: relative; }
.offline-hint { display: flex; align-items: center; justify-content: center; height: 100%; }
</style>
