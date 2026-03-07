<template>
  <div class="bench-page">
    <h2>Benchmark</h2>

    <el-row :gutter="20">
      <el-col :span="8">
        <el-card header="Run Benchmark">
          <el-form label-width="110px">
            <el-form-item label="Type">
              <el-select v-model="benchType" style="width: 100%">
                <el-option label="All" value="all" />
                <el-option label="Inference" value="inference" />
                <el-option label="MLA" value="mla" />
                <el-option label="MoE" value="moe" />
                <el-option label="Linear" value="linear" />
                <el-option label="Attention" value="attention" />
              </el-select>
            </el-form-item>
            <el-form-item label="Model">
              <el-select v-model="selectedModel" clearable placeholder="Default model" style="width: 100%">
                <el-option v-for="m in models" :key="m.name" :label="m.name" :value="m.name" />
              </el-select>
            </el-form-item>
            <el-form-item label="Iterations">
              <el-input-number v-model="iterations" :min="1" :max="1000" style="width: 100%" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="runBench" :loading="running" style="width: 100%">
                {{ running ? 'Running...' : 'Run Benchmark' }}
              </el-button>
            </el-form-item>
          </el-form>

          <el-alert v-if="error" :title="error" type="error" show-icon style="margin-top: 10px" />

          <div v-if="lastResult" class="result-summary">
            <div class="result-item" v-for="(val, key) in flatResults" :key="key">
              <span class="label">{{ key }}</span>
              <span class="value">{{ val }}</span>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="16">
        <el-card header="Results">
          <div v-if="chartData.length > 0" style="height: 400px">
            <v-chart :option="chartOption" autoresize />
          </div>
          <el-empty v-else description="Run a benchmark to see results" />

          <div v-if="history.length > 1" style="margin-top: 20px">
            <el-divider>History</el-divider>
            <el-table :data="history" size="small" max-height="200">
              <el-table-column prop="time" label="Time" width="150" />
              <el-table-column prop="type" label="Type" width="100" />
              <el-table-column prop="summary" label="Summary" />
            </el-table>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { useModelsStore } from '@/stores/models'
import { storeToRefs } from 'pinia'
import { ElMessage } from 'element-plus'

use([BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer])

const modelsStore = useModelsStore()
const { models } = storeToRefs(modelsStore)

const benchType = ref('inference')
const selectedModel = ref('')
const iterations = ref(10)
const running = ref(false)
const error = ref('')
const lastResult = ref<any>(null)
const history = ref<any[]>([])

interface ChartEntry { name: string; value: number; unit: string }
const chartData = ref<ChartEntry[]>([])

const flatResults = computed(() => {
  if (!lastResult.value) return {}
  const flat: Record<string, string> = {}
  function walk(obj: any, prefix = '') {
    for (const [k, v] of Object.entries(obj)) {
      const key = prefix ? `${prefix}.${k}` : k
      if (typeof v === 'object' && v !== null) walk(v, key)
      else flat[key] = String(v)
    }
  }
  walk(lastResult.value)
  return flat
})

const chartOption = computed(() => ({
  tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
  grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
  xAxis: { type: 'category', data: chartData.value.map(d => d.name) },
  yAxis: { type: 'value', name: chartData.value[0]?.unit || '' },
  series: [{
    type: 'bar',
    data: chartData.value.map(d => d.value),
    itemStyle: { color: '#409eff' }
  }]
}))

onMounted(() => modelsStore.load())

async function runBench() {
  running.value = true
  error.value = ''
  try {
    const result = await window.electronAPI.bench.run({
      type: benchType.value,
      model: selectedModel.value || undefined,
      iterations: iterations.value
    })

    if (result.success && result.results) {
      lastResult.value = result.results
      buildChartData(result.results)
      history.value.unshift({
        time: new Date().toLocaleTimeString(),
        type: benchType.value,
        summary: summarize(result.results)
      })
    } else {
      error.value = result.error || 'Benchmark failed'
      ElMessage.error(error.value)
    }
  } finally {
    running.value = false
  }
}

function buildChartData(results: any) {
  chartData.value = []
  // Look for throughput, latency, or any numeric metric
  function extract(obj: any, prefix = '') {
    for (const [k, v] of Object.entries(obj)) {
      if (typeof v === 'number') {
        const unit = k.includes('latency') ? 'ms' : k.includes('throughput') ? 'tokens/s' : ''
        chartData.value.push({ name: prefix ? `${prefix}.${k}` : k, value: v, unit })
      } else if (typeof v === 'object' && v !== null) {
        extract(v, k)
      }
    }
  }
  extract(results)
}

function summarize(results: any): string {
  const nums = Object.values(results).filter(v => typeof v === 'number') as number[]
  if (nums.length === 0) return 'Done'
  return `${nums.length} metrics`
}
</script>

<style scoped>
.bench-page { padding: 20px; }
.result-summary {
  margin-top: 16px;
  padding: 12px;
  background: #f5f7fa;
  border-radius: 4px;
  font-family: monospace;
  font-size: 12px;
  max-height: 200px;
  overflow-y: auto;
}
.result-item {
  display: flex;
  justify-content: space-between;
  padding: 2px 0;
  border-bottom: 1px solid #e8e8e8;
}
.label { color: #606266; }
.value { font-weight: 500; }
</style>
