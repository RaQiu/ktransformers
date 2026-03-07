/**
 * HTTP client for SGLang / KTransformers server.
 * Fetches metrics, models, and server info from the running inference server.
 */

export interface ThroughputSnapshot {
  prefill: number  // tokens/s
  decode: number   // tokens/s
  timestamp: string
}

export interface ServerMemoryUsage {
  weight: number      // GB
  kvcache: number     // GB
  tokenCapacity: number
  graph: number       // GB
}

export interface ServerInfo {
  status: string
  version: string
  modelPath: string
  servedModelName: string
  device: string
  tpSize: number
  dpSize: number
  maxTotalTokens: number
  lastGenThroughput: number
  memoryUsage: ServerMemoryUsage | null
  // KTransformers specific
  ktWeightPath: string
  ktMethod: string
  ktCpuInfer: number
  ktNumGpuExperts: number
}

export interface ModelInfo {
  id: string
  maxModelLen: number
  modelType: string
  architectures: string[]
  modelPath: string
  isGeneration: boolean
}

/** Check if server is reachable */
export async function checkServerHealth(baseUrl: string): Promise<boolean> {
  try {
    const resp = await fetch(`${baseUrl}/v1/models`, { signal: AbortSignal.timeout(3000) })
    return resp.ok
  } catch {
    return false
  }
}

/** Fetch model list with details */
export async function fetchServerModels(baseUrl: string): Promise<ModelInfo[]> {
  try {
    const response = await fetch(`${baseUrl}/v1/models`, { signal: AbortSignal.timeout(3000) })
    if (!response.ok) return []
    const data = await response.json()
    return (data.data || []).map((m: any) => ({
      id: m.id,
      maxModelLen: m.max_model_len || 0,
      modelType: '',
      architectures: [],
      modelPath: '',
      isGeneration: true,
    }))
  } catch {
    return []
  }
}

/** Fetch detailed server info from /get_server_info */
export async function fetchServerInfo(baseUrl: string): Promise<ServerInfo | null> {
  try {
    const response = await fetch(`${baseUrl}/get_server_info`, { signal: AbortSignal.timeout(5000) })
    if (!response.ok) return null
    const data = await response.json()

    // Parse internal_states for runtime metrics
    const state = data.internal_states?.[0] || data

    const memUsage = state.memory_usage
    let memoryUsage: ServerMemoryUsage | null = null
    if (memUsage) {
      memoryUsage = {
        weight: memUsage.weight || 0,
        kvcache: memUsage.kvcache || 0,
        tokenCapacity: memUsage.token_capacity || 0,
        graph: memUsage.graph || 0,
      }
    }

    return {
      status: data.status || state.status || 'unknown',
      version: data.version || '',
      modelPath: data.model_path || '',
      servedModelName: data.served_model_name || '',
      device: data.device || 'unknown',
      tpSize: data.tp_size || 1,
      dpSize: data.dp_size || 1,
      maxTotalTokens: data.max_total_num_tokens || state.max_total_num_tokens || 0,
      lastGenThroughput: state.last_gen_throughput || 0,
      memoryUsage,
      ktWeightPath: data.kt_weight_path || '',
      ktMethod: data.kt_method || '',
      ktCpuInfer: data.kt_cpuinfer || 0,
      ktNumGpuExperts: data.kt_num_gpu_experts || 0,
    }
  } catch {
    return null
  }
}

/** Fetch model architecture info from /get_model_info */
export async function fetchModelArchInfo(baseUrl: string): Promise<{
  modelType: string
  architectures: string[]
  modelPath: string
  isGeneration: boolean
} | null> {
  try {
    const response = await fetch(`${baseUrl}/get_model_info`, { signal: AbortSignal.timeout(3000) })
    if (!response.ok) return null
    const data = await response.json()
    return {
      modelType: data.model_type || '',
      architectures: data.architectures || [],
      modelPath: data.model_path || '',
      isGeneration: data.is_generation ?? true,
    }
  } catch {
    return null
  }
}

/** Fetch throughput from /metrics (Prometheus format) */
export async function fetchServerMetrics(baseUrl: string): Promise<ThroughputSnapshot | null> {
  try {
    const response = await fetch(`${baseUrl}/metrics`, { signal: AbortSignal.timeout(2000) })
    if (!response.ok) return null
    const text = await response.text()
    return parsePrometheusMetrics(text)
  } catch {
    return null
  }
}

/** Fetch throughput from /get_server_info as fallback when /metrics is disabled */
export async function fetchThroughputFromServerInfo(baseUrl: string): Promise<ThroughputSnapshot | null> {
  try {
    const response = await fetch(`${baseUrl}/get_server_info`, { signal: AbortSignal.timeout(3000) })
    if (!response.ok) return null
    const data = await response.json()
    const state = data.internal_states?.[0] || data
    return {
      prefill: 0,
      decode: state.last_gen_throughput || 0,
      timestamp: new Date().toLocaleTimeString(),
    }
  } catch {
    return null
  }
}

function parsePrometheusMetrics(text: string): ThroughputSnapshot {
  const lines = text.split('\n')
  let prefill = 0
  let decode = 0

  for (const line of lines) {
    if (line.startsWith('#')) continue
    const prefillMatch = line.match(/sglang[_:]prefill_throughput[^\s]*\s+([\d.]+)/)
    const decodeMatch = line.match(/sglang[_:]decode_throughput[^\s]*\s+([\d.]+)/)

    if (prefillMatch) prefill = parseFloat(prefillMatch[1])
    if (decodeMatch) decode = parseFloat(decodeMatch[1])
  }

  return { prefill, decode, timestamp: new Date().toLocaleTimeString() }
}
