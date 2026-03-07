import { defineStore } from 'pinia'
import { ref, toRaw } from 'vue'

const STORAGE_KEY = 'kt-config'
const hasElectron = typeof window !== 'undefined' && !!window.electronAPI?.config

function defaultConfig() {
  return {
    general: { language: 'auto', color: true, verbose: false },
    paths: { models: [], cache: '', weights: '' },
    server: { mode: 'local', host: '0.0.0.0', port: 30000, remoteUrl: '' },
    inference: { env: {} },
    download: { mirror: '', resume: true, verify: true },
    advanced: { env: {}, sglang_args: [], llamafactory_args: [] }
  }
}

export const useConfigStore = defineStore('config', () => {
  const config = ref<any>(defaultConfig())
  const dirty = ref(false)

  async function load() {
    if (hasElectron) {
      const data = await window.electronAPI.config.getAll()
      config.value = { ...defaultConfig(), ...data }
      // Ensure server section has all fields
      config.value.server = { ...defaultConfig().server, ...config.value.server }
    } else {
      // Browser fallback: use localStorage
      try {
        const stored = localStorage.getItem(STORAGE_KEY)
        if (stored) {
          const parsed = JSON.parse(stored)
          config.value = { ...defaultConfig(), ...parsed }
          config.value.server = { ...defaultConfig().server, ...config.value.server }
        }
      } catch {}
    }
    dirty.value = false
  }

  async function save() {
    // Unwrap Vue reactive proxies — Electron IPC structured clone cannot handle Proxy objects
    const raw = JSON.parse(JSON.stringify(toRaw(config.value)))
    if (hasElectron) {
      for (const key in raw) {
        await window.electronAPI.config.set(key, raw[key])
      }
    } else {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(raw))
    }
    dirty.value = false
  }

  async function reset() {
    if (hasElectron) {
      await window.electronAPI.config.reset()
    } else {
      localStorage.removeItem(STORAGE_KEY)
    }
    config.value = defaultConfig()
    dirty.value = false
  }

  function update(key: string, value: any) {
    const parts = key.split('.')
    let obj = config.value
    for (let i = 0; i < parts.length - 1; i++) {
      if (!obj[parts[i]]) obj[parts[i]] = {}
      obj = obj[parts[i]]
    }
    obj[parts[parts.length - 1]] = value
    dirty.value = true
  }

  return { config, dirty, load, save, reset, update }
})
