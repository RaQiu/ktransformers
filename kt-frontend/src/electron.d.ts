export interface ElectronAPI {
  config: {
    getAll: () => Promise<any>
    get: (key: string) => Promise<any>
    set: (key: string, value: any) => Promise<void>
    reset: () => Promise<void>
  }
  models: {
    list: () => Promise<any[]>
    add: (model: any) => Promise<void>
    remove: (name: string) => Promise<void>
    update: (name: string, updates: any) => Promise<void>
    scan: (path: string) => Promise<{ success: boolean; found?: number; error?: string }>
    verify: (name: string) => Promise<{ success: boolean; status?: string; output?: string; error?: string }>
  }
  server: {
    start: (config: any) => Promise<{ success: boolean; pid?: number; error?: string }>
    stop: () => Promise<{ success: boolean; error?: string }>
    status: () => Promise<{ running: boolean; pid: number | null }>
    onLog: (callback: (log: string) => void) => void
  }
  system: {
    info: () => Promise<any>
    monitor: () => Promise<any>
  }
  doctor: {
    run: () => Promise<{ checks: any[]; success: boolean; error?: string }>
  }
  bench: {
    run: (params: { type: string; model?: string; iterations: number }) => Promise<{ success: boolean; results?: any; error?: string }>
  }
  kvCache: {
    calc: (params: { modelPath: string; maxTokens: number; tp?: number; dtype?: string }) => Promise<{ success: boolean; kv_cache_gb?: number; details?: any; error?: string }>
  }
  download: {
    start: (params: { repoId: string; outputDir?: string; token?: string }) => Promise<{ success: boolean; id?: string; error?: string }>
    cancel: (id: string) => Promise<{ success: boolean }>
    onProgress: (callback: (data: { id: string; text: string; percent: number }) => void) => void
    onComplete: (callback: (data: { id: string }) => void) => void
    onError: (callback: (data: { id: string; error: string }) => void) => void
  }
  version: {
    get: () => Promise<{ success: boolean; ktKernelVersion?: string; cpuVariant?: string; pythonVersion?: string; cudaVersion?: string; rawOutput?: string }>
  }
  nav: {
    onGoto: (callback: (route: string) => void) => void
  }
  tray: {
    setServerState: (running: boolean) => void
  }
  fs: {
    selectDir: () => Promise<string | undefined>
    selectFile: () => Promise<string | undefined>
  }
}

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}
