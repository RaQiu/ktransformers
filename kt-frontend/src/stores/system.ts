import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSystemStore = defineStore('system', () => {
  const info = ref<any>({})
  const metrics = ref<any>({})

  async function loadInfo() {
    info.value = await window.electronAPI.system.info()
  }

  async function loadMetrics() {
    metrics.value = await window.electronAPI.system.monitor()
  }

  return { info, metrics, loadInfo, loadMetrics }
})
