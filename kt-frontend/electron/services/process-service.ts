import { BrowserWindow } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import { IPC_CHANNELS } from '../ipc/channels'

export class ProcessService {
  private process: ChildProcess | null = null
  private win: BrowserWindow

  constructor(win: BrowserWindow) {
    this.win = win
  }

  start(config: any) {
    if (this.process) return { success: false, error: 'Server already running' }

    const args = ['run', config.model]
    if (config.host) args.push('--host', config.host)
    if (config.port) args.push('--port', config.port.toString())
    if (config.gpuExperts) args.push('--gpu-experts', config.gpuExperts.toString())
    if (config.cpuThreads) args.push('--cpu-threads', config.cpuThreads.toString())
    if (config.useMmap !== undefined) {
      args.push('--weight-strategy', config.useMmap ? 'tiered' : 'legacy')
    }

    this.process = spawn('kt', args, { shell: true })

    this.process.stdout?.on('data', (data) => {
      this.win.webContents.send(IPC_CHANNELS.SERVER_LOG, data.toString())
    })

    this.process.stderr?.on('data', (data) => {
      this.win.webContents.send(IPC_CHANNELS.SERVER_LOG, data.toString())
    })

    this.process.on('close', () => {
      this.process = null
    })

    return { success: true, pid: this.process.pid }
  }

  stop() {
    if (this.process) {
      this.process.kill()
      this.process = null
      return { success: true }
    }
    return { success: false, error: 'No server running' }
  }

  getStatus() {
    return {
      running: this.process !== null,
      pid: this.process?.pid || null
    }
  }
}
