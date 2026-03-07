import { BrowserWindow } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import { join } from 'path'
import { mkdirSync } from 'fs'
import { homedir } from 'os'
import { IPC_CHANNELS } from '../ipc/channels'

interface DownloadTask {
  id: string
  process: ChildProcess
}

export class DownloadService {
  private win: BrowserWindow
  private tasks: Map<string, DownloadTask> = new Map()

  constructor(win: BrowserWindow) {
    this.win = win
  }

  start(params: { repoId: string; outputDir?: string; token?: string }): { success: boolean; id?: string; error?: string } {
    const id = `dl-${Date.now()}`
    const outputDir = params.outputDir || join(homedir(), '.ktransformers', 'models')

    try {
      mkdirSync(outputDir, { recursive: true })
    } catch {}

    // Use kt download command or huggingface-cli
    const args = ['model', 'download', params.repoId]
    const proc = spawn('kt', args, {
      shell: true,
      env: { ...process.env, HF_TOKEN: params.token || process.env.HF_TOKEN || '' }
    })

    const task: DownloadTask = { id, process: proc }
    this.tasks.set(id, task)

    proc.stdout?.on('data', (data: Buffer) => {
      const text = data.toString()
      // Parse progress lines like "Downloading: 45%|████ | 2.3G/5.1G"
      const pctMatch = text.match(/(\d+)%/)
      const percent = pctMatch ? parseInt(pctMatch[1]) : -1
      this.win.webContents.send(IPC_CHANNELS.DOWNLOAD_PROGRESS, { id, text, percent })
    })

    proc.stderr?.on('data', (data: Buffer) => {
      const text = data.toString()
      const pctMatch = text.match(/(\d+)%/)
      const percent = pctMatch ? parseInt(pctMatch[1]) : -1
      this.win.webContents.send(IPC_CHANNELS.DOWNLOAD_PROGRESS, { id, text, percent })
    })

    proc.on('close', (code) => {
      this.tasks.delete(id)
      if (code === 0) {
        this.win.webContents.send(IPC_CHANNELS.DOWNLOAD_COMPLETE, { id })
      } else {
        this.win.webContents.send(IPC_CHANNELS.DOWNLOAD_ERROR, { id, error: `Process exited with code ${code}` })
      }
    })

    proc.on('error', (err) => {
      this.tasks.delete(id)
      this.win.webContents.send(IPC_CHANNELS.DOWNLOAD_ERROR, { id, error: err.message })
    })

    return { success: true, id }
  }

  cancel(id: string): { success: boolean } {
    const task = this.tasks.get(id)
    if (task) {
      task.process.kill()
      this.tasks.delete(id)
      return { success: true }
    }
    return { success: false }
  }
}
