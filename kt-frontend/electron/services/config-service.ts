import { homedir } from 'os'
import { join } from 'path'
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs'
import * as yaml from 'js-yaml'

const CONFIG_DIR = join(homedir(), '.ktransformers')
const CONFIG_FILE = join(CONFIG_DIR, 'config.yaml')

const DEFAULT_CONFIG = {
  general: { language: 'auto', color: true, verbose: false },
  paths: { models: join(CONFIG_DIR, 'models'), cache: join(CONFIG_DIR, 'cache'), weights: '' },
  server: { mode: 'local', host: '0.0.0.0', port: 30000, remoteUrl: '' },
  inference: { env: { PYTORCH_ALLOC_CONF: 'expandable_segments:True', SGLANG_ENABLE_JIT_DEEPGEMM: '0' } },
  download: { mirror: '', resume: true, verify: true },
  advanced: { env: {}, sglang_args: [], llamafactory_args: [] }
}

export class ConfigService {
  private config: any

  constructor() {
    this.load()
  }

  private load() {
    if (!existsSync(CONFIG_DIR)) mkdirSync(CONFIG_DIR, { recursive: true })
    if (existsSync(CONFIG_FILE)) {
      const data = yaml.load(readFileSync(CONFIG_FILE, 'utf8')) as any
      this.config = this.deepMerge(DEFAULT_CONFIG, data || {})
    } else {
      this.config = JSON.parse(JSON.stringify(DEFAULT_CONFIG))
      this.save()
    }
  }

  private save() {
    writeFileSync(CONFIG_FILE, yaml.dump(this.config))
  }

  private deepMerge(base: any, override: any): any {
    const result = { ...base }
    for (const key in override) {
      if (typeof override[key] === 'object' && !Array.isArray(override[key]) && override[key] !== null) {
        result[key] = this.deepMerge(base[key] || {}, override[key])
      } else {
        result[key] = override[key]
      }
    }
    return result
  }

  getAll() {
    return JSON.parse(JSON.stringify(this.config))
  }

  get(key: string) {
    const parts = key.split('.')
    let value = this.config
    for (const part of parts) {
      if (value && typeof value === 'object') value = value[part]
      else return undefined
    }
    return value
  }

  set(key: string, value: any) {
    const parts = key.split('.')
    let obj = this.config
    for (let i = 0; i < parts.length - 1; i++) {
      if (!obj[parts[i]]) obj[parts[i]] = {}
      obj = obj[parts[i]]
    }
    obj[parts[parts.length - 1]] = value
    this.save()
  }

  reset() {
    this.config = JSON.parse(JSON.stringify(DEFAULT_CONFIG))
    this.save()
  }
}
