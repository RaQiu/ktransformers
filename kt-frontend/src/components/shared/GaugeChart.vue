<template>
  <div class="gauge-chart" :style="{ width: size + 'px', height: size + 'px' }">
    <v-chart :option="option" autoresize />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { GaugeChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'

use([GaugeChart, CanvasRenderer])

const props = defineProps<{
  value: number
  max?: number
  size?: number
  unit?: string
}>()

const option = computed(() => ({
  series: [{
    type: 'gauge',
    startAngle: 180,
    endAngle: 0,
    min: 0,
    max: props.max || 100,
    splitNumber: 4,
    axisLine: {
      lineStyle: {
        width: 6,
        color: [[0.7, '#67C23A'], [0.9, '#E6A23C'], [1, '#F56C6C']]
      }
    },
    pointer: { itemStyle: { color: 'auto' } },
    axisTick: { show: false },
    splitLine: { length: 8, lineStyle: { width: 2, color: '#999' } },
    axisLabel: { distance: 15, color: '#999', fontSize: 10 },
    detail: {
      valueAnimation: true,
      formatter: `{value}${props.unit || '%'}`,
      color: 'auto',
      fontSize: 16
    },
    data: [{ value: props.value }]
  }]
}))
</script>

<style scoped>
.gauge-chart {
  display: inline-block;
}
</style>
