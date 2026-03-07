<template>
  <div class="models-page">
    <div class="header">
      <h2>{{ t('models.title') }}</h2>
      <el-space>
        <el-button type="primary" @click="showAddDialog = true">{{ t('models.add') }}</el-button>
        <el-button @click="handleScan">{{ t('models.scan') }}</el-button>
      </el-space>
    </div>

    <div class="content">
      <div class="table-section">
        <el-table :data="models" @row-click="handleRowClick" highlight-current-row>
          <el-table-column prop="name" :label="t('models.name')" width="200" />
          <el-table-column prop="format" :label="t('models.format')" width="120">
            <template #default="{ row }">
              <el-tag :type="getFormatType(row.format)">{{ row.format }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="path" :label="t('models.path')" show-overflow-tooltip />
          <el-table-column label="MoE" width="80">
            <template #default="{ row }">
              <el-icon v-if="row.is_moe" color="#67C23A"><Check /></el-icon>
            </template>
          </el-table-column>
          <el-table-column label="Status" width="100">
            <template #default="{ row }">
              <el-tag v-if="row.sha256_status === 'passed'" type="success" size="small">Valid</el-tag>
              <el-tag v-else-if="row.sha256_status === 'failed'" type="danger" size="small">Invalid</el-tag>
              <el-tag v-else type="info" size="small">Unknown</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="Actions" width="200">
            <template #default="{ row }">
              <el-button size="small" @click.stop="handleUseModel(row)">Use</el-button>
              <el-button size="small" type="warning" @click.stop="handleVerify(row.name)" :loading="verifyingModel === row.name">Verify</el-button>
              <el-button size="small" type="danger" @click.stop="handleRemove(row.name)">Remove</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <div class="detail-section">
        <el-card v-if="selectedModel">
          <template #header>Model Details</template>
          <div class="detail-content">
            <div class="detail-item">
              <span class="label">Name:</span>
              <span>{{ selectedModel.name }}</span>
            </div>
            <div class="detail-item">
              <span class="label">Path:</span>
              <span class="path">{{ selectedModel.path }}</span>
            </div>
            <div class="detail-item">
              <span class="label">Format:</span>
              <span>{{ selectedModel.format }}</span>
            </div>
            <div class="detail-item">
              <span class="label">Repository:</span>
              <span>{{ selectedModel.repo_id || 'N/A' }}</span>
            </div>
            <div v-if="selectedModel.is_moe" class="detail-item">
              <span class="label">MoE Experts:</span>
              <span>{{ selectedModel.moe_num_experts }} total, {{ selectedModel.moe_num_experts_per_tok }} active</span>
            </div>
            <div v-if="selectedModel.amx_source_model" class="detail-item">
              <span class="label">AMX Source:</span>
              <span>{{ selectedModel.amx_source_model }}</span>
            </div>
            <div v-if="selectedModel.amx_quant_method" class="detail-item">
              <span class="label">Quantization:</span>
              <span>{{ selectedModel.amx_quant_method }}</span>
            </div>
            <div class="detail-item">
              <span class="label">Created:</span>
              <span>{{ formatDate(selectedModel.created_at) }}</span>
            </div>
            <el-button type="primary" style="margin-top: 20px; width: 100%" @click="handleUseModel(selectedModel)">
              Start Server with This Model
            </el-button>
          </div>
          <KvCachePanel :model-path="selectedModel.path" />
        </el-card>
        <el-empty v-else description="Select a model to view details" />
      </div>
    </div>

    <!-- Download Panel -->
    <el-card style="margin-top: 20px">
      <template #header>
        <span>Download Model from HuggingFace</span>
      </template>
      <el-form :inline="true" style="display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap">
        <el-form-item label="Repo ID" style="flex: 1; min-width: 250px">
          <el-input v-model="downloadRepoId" placeholder="e.g. deepseek-ai/DeepSeek-V3" style="width: 100%" />
        </el-form-item>
        <el-form-item label="Save to">
          <el-input v-model="downloadOutputDir" placeholder="~/.ktransformers/models" style="width: 240px">
            <template #append><el-button @click="selectDownloadDir">Browse</el-button></template>
          </el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="startDownload" :disabled="!downloadRepoId || !!activeDownloadId">Download</el-button>
          <el-button v-if="activeDownloadId" type="warning" @click="cancelDownload">Cancel</el-button>
        </el-form-item>
      </el-form>
      <div v-if="downloadLogs.length > 0" class="download-log">
        <el-progress v-if="downloadPercent >= 0" :percentage="downloadPercent" style="margin-bottom: 8px" />
        <div class="log-text">{{ downloadLogs[downloadLogs.length - 1] }}</div>
      </div>
    </el-card>

    <!-- Add Model Dialog -->
    <el-dialog v-model="showAddDialog" title="Add Model" width="600px">
      <el-form :model="form" label-width="120px">
        <el-form-item label="Name">
          <el-input v-model="form.name" placeholder="Model name" />
        </el-form-item>
        <el-form-item label="Path">
          <el-input v-model="form.path">
            <template #append>
              <el-button @click="selectPath">Browse</el-button>
            </template>
          </el-input>
        </el-form-item>
        <el-form-item label="Format">
          <el-select v-model="form.format">
            <el-option label="Safetensors" value="safetensors" />
            <el-option label="GGUF" value="gguf" />
            <el-option label="AMX" value="amx" />
          </el-select>
        </el-form-item>
        <el-form-item label="Repository ID">
          <el-input v-model="form.repo_id" placeholder="e.g., deepseek-ai/DeepSeek-V3" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAddDialog = false">Cancel</el-button>
        <el-button type="primary" @click="handleAdd">Add</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { useModelsStore } from '@/stores/models'
import { storeToRefs } from 'pinia'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Check } from '@element-plus/icons-vue'
import KvCachePanel from '@/components/models/KvCachePanel.vue'

const { t } = useI18n()
const router = useRouter()
const modelsStore = useModelsStore()
const { models, selectedModel } = storeToRefs(modelsStore)

const showAddDialog = ref(false)
const verifyingModel = ref('')
const downloadRepoId = ref('')
const downloadOutputDir = ref('')
const activeDownloadId = ref('')
const downloadPercent = ref(-1)
const downloadLogs = ref<string[]>([])
const form = ref({
  name: '',
  path: '',
  format: 'safetensors',
  repo_id: ''
})

onMounted(() => {
  modelsStore.load()
  // Listen to download events
  window.electronAPI.download.onProgress(({ id, text, percent }) => {
    if (id !== activeDownloadId.value) return
    downloadLogs.value.push(text.trim())
    if (downloadLogs.value.length > 100) downloadLogs.value.shift()
    if (percent >= 0) downloadPercent.value = percent
  })
  window.electronAPI.download.onComplete(({ id }) => {
    if (id !== activeDownloadId.value) return
    activeDownloadId.value = ''
    downloadPercent.value = 100
    ElMessage.success('Download complete!')
    modelsStore.load()
  })
  window.electronAPI.download.onError(({ id, error }) => {
    if (id !== activeDownloadId.value) return
    activeDownloadId.value = ''
    ElMessage.error(`Download failed: ${error}`)
  })
})

function handleRowClick(row: any) {
  modelsStore.select(row)
}

async function selectPath() {
  const path = await window.electronAPI.fs.selectDir()
  if (path) {
    form.value.path = path
    if (!form.value.name) {
      form.value.name = path.split('/').pop() || ''
    }
  }
}

async function handleAdd() {
  if (!form.value.name || !form.value.path) {
    ElMessage.warning('Please fill in required fields')
    return
  }
  await modelsStore.add({
    ...form.value,
    id: crypto.randomUUID(),
    created_at: new Date().toISOString()
  })
  showAddDialog.value = false
  form.value = { name: '', path: '', format: 'safetensors', repo_id: '' }
  ElMessage.success('Model added')
}

async function handleRemove(name: string) {
  await ElMessageBox.confirm(`Remove model "${name}"?`, 'Confirm', { type: 'warning' })
  await modelsStore.remove(name)
  ElMessage.success('Model removed')
}

async function handleScan() {
  const path = await window.electronAPI.fs.selectDir()
  if (path) {
    const loading = ElMessage({ message: 'Scanning directory for models...', type: 'info', duration: 0 })
    const result = await window.electronAPI.models.scan(path)
    loading.close()
    if (result.success) {
      await modelsStore.load()
      ElMessage.success(`Scan complete. Registry now has ${result.found ?? 0} models.`)
    } else {
      ElMessage.error(result.error || 'Scan failed')
    }
  }
}

async function handleVerify(name: string) {
  verifyingModel.value = name
  try {
    const result = await window.electronAPI.models.verify(name)
    if (result.success) {
      await modelsStore.load()
      const type = result.status === 'passed' ? 'success' : 'error'
      ElMessage({ type, message: `Integrity ${result.status}: ${name}` })
    } else {
      ElMessage.error(result.error || 'Verification failed')
    }
  } finally {
    verifyingModel.value = ''
  }
}

async function startDownload() {
  if (!downloadRepoId.value) return
  downloadLogs.value = []
  downloadPercent.value = 0
  const result = await window.electronAPI.download.start({
    repoId: downloadRepoId.value,
    outputDir: downloadOutputDir.value || undefined
  })
  if (result.success && result.id) {
    activeDownloadId.value = result.id
    ElMessage.info(`Download started: ${downloadRepoId.value}`)
  } else {
    ElMessage.error(result.error || 'Failed to start download')
  }
}

async function cancelDownload() {
  if (activeDownloadId.value) {
    await window.electronAPI.download.cancel(activeDownloadId.value)
    activeDownloadId.value = ''
    ElMessage.warning('Download cancelled')
  }
}

async function selectDownloadDir() {
  const path = await window.electronAPI.fs.selectDir()
  if (path) downloadOutputDir.value = path
}

function handleUseModel(model: any) {
  router.push('/service')
}

function getFormatType(format: string) {
  const types: Record<string, string> = {
    safetensors: 'success',
    gguf: 'warning',
    amx: 'primary'
  }
  return types[format] || 'info'
}

function formatDate(dateStr: string) {
  if (!dateStr) return 'N/A'
  return new Date(dateStr).toLocaleString()
}
</script>

<style scoped>
.models-page {
  padding: 20px;
}
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
.content {
  display: grid;
  grid-template-columns: 60% 40%;
  gap: 20px;
}
.table-section {
  min-height: 400px;
}
.detail-section {
  min-height: 400px;
}
.detail-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.detail-item {
  display: flex;
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
}
.detail-item .label {
  width: 140px;
  color: #909399;
  flex-shrink: 0;
}
.detail-item .path {
  word-break: break-all;
  font-family: monospace;
  font-size: 12px;
}
.download-log {
  margin-top: 12px;
  padding: 10px;
  background: #1e1e1e;
  border-radius: 4px;
}
.log-text {
  font-family: monospace;
  font-size: 12px;
  color: #a8ff78;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
