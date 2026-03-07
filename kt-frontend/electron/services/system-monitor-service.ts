import { cpus, freemem, totalmem } from 'os'
import { execSync } from 'child_process'

export class SystemMonitorService {
  getSystemInfo() {
    const cpu = cpus()[0]
    return {
      cpu: { model: cpu.model, cores: cpus().length },
      memory: { total: totalmem(), free: freemem() },
      gpu: this.getGPUInfo()
    }
  }

  getCurrentMetrics() {
    return {
      cpu: { usage: this.getCPUUsage() },
      memory: { used: totalmem() - freemem(), total: totalmem() },
      gpu: this.getGPUMetrics()
    }
  }

  private getCPUUsage() {
    const cpuList = cpus()
    let totalIdle = 0, totalTick = 0
    cpuList.forEach(cpu => {
      for (const type in cpu.times) totalTick += cpu.times[type as keyof typeof cpu.times]
      totalIdle += cpu.times.idle
    })
    return 100 - (100 * totalIdle / totalTick)
  }

  private getGPUInfo() {
    try {
      const output = execSync('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits', { encoding: 'utf8' })
      const lines = output.trim().split('\n')
      return lines.map(line => {
        const [name, memory] = line.split(',')
        return { name: name.trim(), memory: parseInt(memory) }
      })
    } catch {
      return []
    }
  }

  private getGPUMetrics() {
    try {
      const output = execSync('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits', { encoding: 'utf8' })
      const lines = output.trim().split('\n')
      return lines.map(line => {
        const [util, used, total, temp] = line.split(',').map(v => parseInt(v.trim()))
        return { utilization: util, memoryUsed: used, memoryTotal: total, temperature: temp }
      })
    } catch {
      return []
    }
  }
}
