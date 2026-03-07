import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

const hasElectron = typeof window !== 'undefined' && !!window.electronAPI?.server

export const useServerStore = defineStore('server', () => {
  const running = ref(false)
  const pid = ref<number | null>(null)
  const logs = ref<string[]>([])
  const startTime = ref<Date | null>(null)

  const uptime = computed(() => {
    if (!running.value || !startTime.value) return null
    const secs = Math.floor((Date.now() - startTime.value.getTime()) / 1000)
    const h = Math.floor(secs / 3600)
    const m = Math.floor((secs % 3600) / 60)
    const s = secs % 60
    return [h, m, s].map(n => String(n).padStart(2, '0')).join(':')
  })

  async function start(config: any) {
    if (!hasElectron) return { success: false, error: 'Not in Electron' }
    const result = await window.electronAPI.server.start(config)
    if (result.success) {
      running.value = true
      pid.value = result.pid ?? null
      startTime.value = new Date()
      window.electronAPI.tray.setServerState(true)
    }
    return result
  }

  async function stop() {
    if (!hasElectron) return { success: false, error: 'Not in Electron' }
    const result = await window.electronAPI.server.stop()
    if (result.success) {
      running.value = false
      pid.value = null
      startTime.value = null
      window.electronAPI.tray.setServerState(false)
    }
    return result
  }

  async function checkStatus() {
    if (!hasElectron) return
    const status = await window.electronAPI.server.status()
    running.value = status.running
    pid.value = status.pid
    if (status.running && !startTime.value) startTime.value = new Date()
    window.electronAPI.tray.setServerState(status.running)
  }

  function addLog(log: string) {
    logs.value.push(log)
    if (logs.value.length > 5000) logs.value.shift()
  }

  function clearLogs() {
    logs.value = []
  }

  return { running, pid, logs, startTime, uptime, start, stop, checkStatus, addLog, clearLogs }
})
