import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import { IPC_CHANNELS } from './channels'
import { ConfigService } from '../services/config-service'
import { ModelService } from '../services/model-service'
import { ProcessService } from '../services/process-service'
import { SystemMonitorService } from '../services/system-monitor-service'
import { DownloadService } from '../services/download-service'
import { join } from 'path'
import { execFile, spawn } from 'child_process'

export function registerIpcHandlers(win: BrowserWindow) {
  const configService = new ConfigService()
  const modelService = new ModelService()
  const processService = new ProcessService(win)
  const systemMonitor = new SystemMonitorService()
  const downloadService = new DownloadService(win)

  // Config handlers
  ipcMain.handle(IPC_CHANNELS.CONFIG_GET_ALL, () => configService.getAll())
  ipcMain.handle(IPC_CHANNELS.CONFIG_GET, (_, key: string) => configService.get(key))
  ipcMain.handle(IPC_CHANNELS.CONFIG_SET, (_, key: string, value: any) => configService.set(key, value))
  ipcMain.handle(IPC_CHANNELS.CONFIG_RESET, () => configService.reset())

  // Model handlers
  ipcMain.handle(IPC_CHANNELS.MODELS_LIST, () => modelService.list())
  ipcMain.handle(IPC_CHANNELS.MODELS_ADD, (_, model: any) => modelService.add(model))
  ipcMain.handle(IPC_CHANNELS.MODELS_REMOVE, (_, name: string) => modelService.remove(name))
  ipcMain.handle(IPC_CHANNELS.MODELS_UPDATE, (_, name: string, updates: any) => modelService.update(name, updates))
  ipcMain.handle(IPC_CHANNELS.MODELS_SCAN, (_, path: string) => modelService.scan(path))
  ipcMain.handle(IPC_CHANNELS.MODELS_VERIFY, (_, name: string) => modelService.verify(name))

  // Server handlers
  ipcMain.handle(IPC_CHANNELS.SERVER_START, (_, config: any) => processService.start(config))
  ipcMain.handle(IPC_CHANNELS.SERVER_STOP, () => processService.stop())
  ipcMain.handle(IPC_CHANNELS.SERVER_STATUS, () => processService.getStatus())

  // System monitor
  ipcMain.handle(IPC_CHANNELS.SYSTEM_INFO, () => systemMonitor.getSystemInfo())
  ipcMain.handle(IPC_CHANNELS.SYSTEM_MONITOR, () => systemMonitor.getCurrentMetrics())

  // Doctor — calls scripts/doctor_json.py
  ipcMain.handle(IPC_CHANNELS.DOCTOR_RUN, async () => {
    const scriptPath = join(app.getAppPath(), '..', 'scripts', 'doctor_json.py')
    return new Promise((resolve) => {
      let output = ''
      let errOutput = ''
      const proc = spawn('python3', [scriptPath], { shell: false })
      proc.stdout.on('data', (d) => { output += d.toString() })
      proc.stderr.on('data', (d) => { errOutput += d.toString() })
      proc.on('close', () => {
        try {
          resolve(JSON.parse(output))
        } catch {
          resolve({ checks: [], success: false, error: errOutput || 'Failed to parse output' })
        }
      })
      proc.on('error', (e) => resolve({ checks: [], success: false, error: e.message }))
    })
  })

  // Benchmark — calls kt bench with JSON output file
  ipcMain.handle(IPC_CHANNELS.BENCH_RUN, async (_, { type, model, iterations }) => {
    const os = await import('os')
    const fs = await import('fs')
    const tmpFile = join(os.default.tmpdir(), `kt-bench-${Date.now()}.json`)
    const args = ['bench', '--type', type, '--iterations', String(iterations)]
    if (model) args.push('--model', model)
    args.push('--output', tmpFile)
    return new Promise((resolve) => {
      const proc = spawn('kt', args, { shell: true })
      let stderr = ''
      proc.stderr.on('data', (d) => { stderr += d.toString() })
      proc.on('close', (code) => {
        try {
          if (fs.default.existsSync(tmpFile)) {
            const data = JSON.parse(fs.default.readFileSync(tmpFile, 'utf8'))
            fs.default.unlinkSync(tmpFile)
            resolve({ success: true, results: data })
          } else {
            resolve({ success: false, error: stderr || `Exit code ${code}` })
          }
        } catch (e: any) {
          resolve({ success: false, error: e.message })
        }
      })
      proc.on('error', (e) => resolve({ success: false, error: e.message }))
    })
  })

  // KV Cache Calculator — calls scripts/kv_cache_json.py
  ipcMain.handle(IPC_CHANNELS.KV_CACHE_CALC, async (_, { modelPath, maxTokens, tp, dtype }) => {
    const scriptPath = join(app.getAppPath(), '..', 'scripts', 'kv_cache_json.py')
    const args = [scriptPath, modelPath, String(maxTokens), String(tp || 1), dtype || 'auto']
    return new Promise((resolve) => {
      let output = ''
      let errOutput = ''
      const proc = spawn('python3', args, { shell: false })
      proc.stdout.on('data', (d) => { output += d.toString() })
      proc.stderr.on('data', (d) => { errOutput += d.toString() })
      proc.on('close', () => {
        try {
          resolve(JSON.parse(output))
        } catch {
          resolve({ success: false, error: errOutput || 'Failed to calculate KV cache' })
        }
      })
      proc.on('error', (e) => resolve({ success: false, error: e.message }))
    })
  })

  // Download handlers
  ipcMain.handle(IPC_CHANNELS.DOWNLOAD_START, (_, params: any) => downloadService.start(params))
  ipcMain.handle(IPC_CHANNELS.DOWNLOAD_CANCEL, (_, id: string) => downloadService.cancel(id))

  // Version info — calls kt version
  ipcMain.handle(IPC_CHANNELS.VERSION_GET, async () => {
    return new Promise((resolve) => {
      let output = ''
      let errOutput = ''
      const proc = spawn('kt', ['version'], { shell: true })
      proc.stdout.on('data', (d) => { output += d.toString() })
      proc.stderr.on('data', (d) => { errOutput += d.toString() })
      proc.on('close', () => {
        // Parse kt version output lines like "kt-kernel 0.5.1 (amx)"
        const lines = (output + errOutput).split('\n').filter(Boolean)
        const info: Record<string, string> = {}
        for (const line of lines) {
          const m = line.match(/kt[-\s]?kernel[\s:]+([^\s(]+)(?:\s+\(([^)]+)\))?/i)
          if (m) { info.ktKernelVersion = m[1]; info.cpuVariant = m[2] || '' }
          const py = line.match(/python[\s:]+([^\s]+)/i)
          if (py) info.pythonVersion = py[1]
          const cuda = line.match(/cuda[\s:]+([^\s]+)/i)
          if (cuda) info.cudaVersion = cuda[1]
        }
        info.rawOutput = output.trim()
        resolve({ success: true, ...info })
      })
      proc.on('error', (e) => resolve({ success: false, error: e.message }))
    })
  })

  // File system
  ipcMain.handle(IPC_CHANNELS.FS_SELECT_DIR, async () => {
    const result = await dialog.showOpenDialog(win, { properties: ['openDirectory'] })
    return result.filePaths[0]
  })
  ipcMain.handle(IPC_CHANNELS.FS_SELECT_FILE, async () => {
    const result = await dialog.showOpenDialog(win, { properties: ['openFile'] })
    return result.filePaths[0]
  })
}
