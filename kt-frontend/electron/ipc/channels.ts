// IPC channel constants
export const IPC_CHANNELS = {
  // Config
  CONFIG_GET_ALL: 'config:get-all',
  CONFIG_GET: 'config:get',
  CONFIG_SET: 'config:set',
  CONFIG_RESET: 'config:reset',

  // Models
  MODELS_LIST: 'models:list',
  MODELS_ADD: 'models:add',
  MODELS_REMOVE: 'models:remove',
  MODELS_UPDATE: 'models:update',
  MODELS_SCAN: 'models:scan',

  // Server Process
  SERVER_START: 'server:start',
  SERVER_STOP: 'server:stop',
  SERVER_STATUS: 'server:status',
  SERVER_LOG: 'server:log',

  // System Monitor
  SYSTEM_INFO: 'system:info',
  SYSTEM_MONITOR: 'system:monitor',

  // Doctor
  DOCTOR_RUN: 'doctor:run',

  // Benchmark
  BENCH_RUN: 'bench:run',

  // KV Cache Calculator
  KV_CACHE_CALC: 'kv-cache:calc',

  // Model verification
  MODELS_VERIFY: 'models:verify',

  // Download
  DOWNLOAD_START: 'download:start',
  DOWNLOAD_PROGRESS: 'download:progress',
  DOWNLOAD_COMPLETE: 'download:complete',
  DOWNLOAD_ERROR: 'download:error',
  DOWNLOAD_CANCEL: 'download:cancel',

  // Version info
  VERSION_GET: 'version:get',

  // Navigation (keyboard shortcuts → renderer)
  NAV_GOTO: 'nav:goto',

  // File System
  FS_SELECT_DIR: 'fs:select-dir',
  FS_SELECT_FILE: 'fs:select-file'
} as const
