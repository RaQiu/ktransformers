import { homedir } from 'os'
import { join } from 'path'
import { readFileSync, writeFileSync, existsSync } from 'fs'
import { spawn } from 'child_process'
import * as yaml from 'js-yaml'

const MODELS_FILE = join(homedir(), '.ktransformers', 'user_models.yaml')

export class ModelService {
  private models: any[] = []

  constructor() {
    this.load()
  }

  private load() {
    if (existsSync(MODELS_FILE)) {
      const data = yaml.load(readFileSync(MODELS_FILE, 'utf8')) as any
      this.models = data?.models || []
    }
  }

  private save() {
    writeFileSync(MODELS_FILE, yaml.dump({ version: '1.0', models: this.models }))
  }

  list() {
    this.load() // Always read fresh from disk
    return [...this.models]
  }

  add(model: any) {
    this.load()
    this.models.push(model)
    this.save()
  }

  remove(name: string) {
    this.load()
    this.models = this.models.filter(m => m.name !== name)
    this.save()
  }

  update(name: string, updates: any) {
    this.load()
    const model = this.models.find(m => m.name === name)
    if (model) {
      Object.assign(model, updates)
      this.save()
    }
  }

  /** Run `kt model add <path>` which scans and registers models, then reload registry */
  scan(path: string): Promise<{ success: boolean; found?: number; error?: string }> {
    return new Promise((resolve) => {
      const proc = spawn('kt', ['model', 'add', path], { shell: true })
      let stderr = ''
      proc.stderr?.on('data', (d) => { stderr += d.toString() })
      proc.on('close', (code) => {
        this.load() // Reload from disk after kt model add
        if (code === 0) {
          resolve({ success: true, found: this.models.length })
        } else {
          resolve({ success: false, error: stderr || `Exit code ${code}` })
        }
      })
      proc.on('error', (e) => resolve({ success: false, error: e.message }))
    })
  }

  /** Run `kt model verify <name>` to check integrity */
  verify(name: string): Promise<{ success: boolean; status?: string; error?: string }> {
    return new Promise((resolve) => {
      const proc = spawn('kt', ['model', 'verify', name], { shell: true })
      let stdout = ''
      let stderr = ''
      proc.stdout?.on('data', (d) => { stdout += d.toString() })
      proc.stderr?.on('data', (d) => { stderr += d.toString() })
      proc.on('close', (code) => {
        const output = stdout + stderr
        // kt model verify exits 0 on pass, non-zero on fail
        const status = code === 0 ? 'passed' : 'failed'
        // Update model's sha256_status in registry
        this.load()
        const model = this.models.find(m => m.name === name)
        if (model) {
          model.sha256_status = status
          this.save()
        }
        resolve({ success: true, status, output: output.trim() })
      })
      proc.on('error', (e) => resolve({ success: false, error: e.message }))
    })
  }
}
