import { contextBridge, ipcRenderer } from 'electron'
import { IPC_CHANNELS } from './ipc/channels'

const api = {
  config: {
    getAll: () => ipcRenderer.invoke(IPC_CHANNELS.CONFIG_GET_ALL),
    get: (key: string) => ipcRenderer.invoke(IPC_CHANNELS.CONFIG_GET, key),
    set: (key: string, value: any) => ipcRenderer.invoke(IPC_CHANNELS.CONFIG_SET, key, value),
    reset: () => ipcRenderer.invoke(IPC_CHANNELS.CONFIG_RESET)
  },
  models: {
    list: () => ipcRenderer.invoke(IPC_CHANNELS.MODELS_LIST),
    add: (model: any) => ipcRenderer.invoke(IPC_CHANNELS.MODELS_ADD, model),
    remove: (name: string) => ipcRenderer.invoke(IPC_CHANNELS.MODELS_REMOVE, name),
    update: (name: string, updates: any) => ipcRenderer.invoke(IPC_CHANNELS.MODELS_UPDATE, name, updates),
    scan: (path: string) => ipcRenderer.invoke(IPC_CHANNELS.MODELS_SCAN, path),
    verify: (name: string) => ipcRenderer.invoke(IPC_CHANNELS.MODELS_VERIFY, name)
  },
  server: {
    start: (config: any) => ipcRenderer.invoke(IPC_CHANNELS.SERVER_START, config),
    stop: () => ipcRenderer.invoke(IPC_CHANNELS.SERVER_STOP),
    status: () => ipcRenderer.invoke(IPC_CHANNELS.SERVER_STATUS),
    onLog: (callback: (log: string) => void) => {
      ipcRenderer.on(IPC_CHANNELS.SERVER_LOG, (_, log) => callback(log))
    }
  },
  system: {
    info: () => ipcRenderer.invoke(IPC_CHANNELS.SYSTEM_INFO),
    monitor: () => ipcRenderer.invoke(IPC_CHANNELS.SYSTEM_MONITOR)
  },
  doctor: {
    run: () => ipcRenderer.invoke(IPC_CHANNELS.DOCTOR_RUN)
  },
  bench: {
    run: (params: { type: string; model?: string; iterations: number }) =>
      ipcRenderer.invoke(IPC_CHANNELS.BENCH_RUN, params)
  },
  kvCache: {
    calc: (params: { modelPath: string; maxTokens: number; tp?: number; dtype?: string }) =>
      ipcRenderer.invoke(IPC_CHANNELS.KV_CACHE_CALC, params)
  },
  download: {
    start: (params: { repoId: string; outputDir?: string; token?: string }) =>
      ipcRenderer.invoke(IPC_CHANNELS.DOWNLOAD_START, params),
    cancel: (id: string) => ipcRenderer.invoke(IPC_CHANNELS.DOWNLOAD_CANCEL, id),
    onProgress: (callback: (data: { id: string; text: string; percent: number }) => void) => {
      ipcRenderer.on(IPC_CHANNELS.DOWNLOAD_PROGRESS, (_, data) => callback(data))
    },
    onComplete: (callback: (data: { id: string }) => void) => {
      ipcRenderer.on(IPC_CHANNELS.DOWNLOAD_COMPLETE, (_, data) => callback(data))
    },
    onError: (callback: (data: { id: string; error: string }) => void) => {
      ipcRenderer.on(IPC_CHANNELS.DOWNLOAD_ERROR, (_, data) => callback(data))
    }
  },
  version: {
    get: () => ipcRenderer.invoke(IPC_CHANNELS.VERSION_GET)
  },
  nav: {
    onGoto: (callback: (route: string) => void) => {
      ipcRenderer.on(IPC_CHANNELS.NAV_GOTO, (_, route) => callback(route))
    }
  },
  tray: {
    setServerState: (running: boolean) => ipcRenderer.send('tray:server-state', running)
  },
  fs: {
    selectDir: () => ipcRenderer.invoke(IPC_CHANNELS.FS_SELECT_DIR),
    selectFile: () => ipcRenderer.invoke(IPC_CHANNELS.FS_SELECT_FILE)
  }
}

contextBridge.exposeInMainWorld('electronAPI', api)
