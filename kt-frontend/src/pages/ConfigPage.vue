<template>
  <div class="config-page">
    <h2>{{ t('config.title') }}</h2>
    <el-tabs v-model="activeTab">
      <el-tab-pane label="General" name="general">
        <el-form label-width="150px">
          <el-form-item label="Language">
            <el-select v-model="config.general.language" @change="markDirty">
              <el-option label="Auto Detect" value="auto" />
              <el-option label="English" value="en" />
              <el-option label="中文" value="zh" />
            </el-select>
          </el-form-item>
          <el-form-item label="Color Output">
            <el-switch v-model="config.general.color" @change="markDirty" />
          </el-form-item>
          <el-form-item label="Verbose Output">
            <el-switch v-model="config.general.verbose" @change="markDirty" />
          </el-form-item>
        </el-form>
      </el-tab-pane>

      <el-tab-pane label="Paths" name="paths">
        <el-form label-width="150px">
          <el-form-item label="Model Paths">
            <div v-for="(path, idx) in modelPaths" :key="idx" style="margin-bottom: 8px">
              <el-input v-model="modelPaths[idx]" @change="updateModelPaths">
                <template #append>
                  <el-button @click="selectModelPath(idx)">Browse</el-button>
                  <el-button v-if="modelPaths.length > 1" @click="removeModelPath(idx)" type="danger">Remove</el-button>
                </template>
              </el-input>
            </div>
            <el-button @click="addModelPath" size="small">Add Path</el-button>
          </el-form-item>
          <el-form-item label="Cache Path">
            <el-input v-model="config.paths.cache" @change="markDirty">
              <template #append>
                <el-button @click="selectCachePath">Browse</el-button>
              </template>
            </el-input>
          </el-form-item>
          <el-form-item label="Weights Path">
            <el-input v-model="config.paths.weights" @change="markDirty" placeholder="Optional custom weights">
              <template #append>
                <el-button @click="selectWeightsPath">Browse</el-button>
              </template>
            </el-input>
          </el-form-item>
        </el-form>
      </el-tab-pane>

      <el-tab-pane label="Server" name="server">
        <el-form label-width="180px">
          <el-form-item label="Connection Mode">
            <el-radio-group v-model="config.server.mode" @change="markDirty">
              <el-radio value="local">Local</el-radio>
              <el-radio value="remote">Remote</el-radio>
            </el-radio-group>
          </el-form-item>

          <template v-if="config.server.mode !== 'remote'">
            <el-form-item label="Host">
              <el-input v-model="config.server.host" @change="markDirty" />
            </el-form-item>
            <el-form-item label="Port">
              <el-input-number v-model="config.server.port" @change="markDirty" :min="1024" :max="65535" />
            </el-form-item>
          </template>

          <template v-else>
            <el-form-item label="Remote Server URL">
              <el-input v-model="config.server.remoteUrl" @change="markDirty" placeholder="https://your-server:8443">
                <template #append>
                  <el-button @click="testRemoteConnection" :loading="testingConnection">Test</el-button>
                </template>
              </el-input>
            </el-form-item>
            <el-alert v-if="connectionTestResult" :title="connectionTestResult.message" :type="connectionTestResult.type" show-icon closable style="margin-bottom: 12px" />
          </template>
        </el-form>
      </el-tab-pane>

      <el-tab-pane label="Inference" name="inference">
        <el-form label-width="200px">
          <el-form-item label="Environment Variables">
            <div v-for="(value, key) in config.inference.env" :key="key" style="margin-bottom: 8px">
              <el-input :model-value="key" disabled style="width: 40%; margin-right: 8px" />
              <el-input v-model="config.inference.env[key]" @change="markDirty" style="width: 50%" />
              <el-button @click="removeInferenceEnv(key)" type="danger" size="small" style="margin-left: 8px">Remove</el-button>
            </div>
            <el-button @click="showAddEnvDialog = true" size="small">Add Variable</el-button>
          </el-form-item>
        </el-form>
      </el-tab-pane>

      <el-tab-pane label="Download" name="download">
        <el-form label-width="150px">
          <el-form-item label="Mirror URL">
            <el-input v-model="config.download.mirror" @change="markDirty" placeholder="HuggingFace mirror" />
          </el-form-item>
          <el-form-item label="Resume Downloads">
            <el-switch v-model="config.download.resume" @change="markDirty" />
          </el-form-item>
          <el-form-item label="Verify Downloads">
            <el-switch v-model="config.download.verify" @change="markDirty" />
          </el-form-item>
        </el-form>
      </el-tab-pane>

      <el-tab-pane label="Advanced" name="advanced">
        <el-form label-width="200px">
          <el-form-item label="Extra Environment Variables">
            <div v-for="(value, key) in config.advanced.env" :key="key" style="margin-bottom: 8px">
              <el-input :model-value="key" disabled style="width: 40%; margin-right: 8px" />
              <el-input v-model="config.advanced.env[key]" @change="markDirty" style="width: 50%" />
              <el-button @click="removeAdvancedEnv(key)" type="danger" size="small" style="margin-left: 8px">Remove</el-button>
            </div>
            <el-button @click="showAddAdvEnvDialog = true" size="small">Add Variable</el-button>
          </el-form-item>
          <el-form-item label="SGLang Arguments">
            <el-input v-model="sglangArgsStr" @change="updateSglangArgs" type="textarea" :rows="3" placeholder="--arg1 value1 --arg2 value2" />
          </el-form-item>
          <el-form-item label="LlamaFactory Arguments">
            <el-input v-model="llamafactoryArgsStr" @change="updateLlamafactoryArgs" type="textarea" :rows="3" placeholder="--arg1 value1 --arg2 value2" />
          </el-form-item>
        </el-form>
      </el-tab-pane>
    </el-tabs>

    <div class="footer-bar">
      <el-tag v-if="dirty" type="warning">Unsaved changes</el-tag>
      <div class="spacer"></div>
      <el-button @click="handleRevert" :disabled="!dirty">Revert</el-button>
      <el-button @click="handleReset">Reset to Defaults</el-button>
      <el-button type="primary" @click="handleSave" :disabled="!dirty">Save</el-button>
    </div>

    <el-dialog v-model="showAddEnvDialog" title="Add Environment Variable" width="500px">
      <el-form>
        <el-form-item label="Key">
          <el-input v-model="newEnvKey" />
        </el-form-item>
        <el-form-item label="Value">
          <el-input v-model="newEnvValue" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAddEnvDialog = false">Cancel</el-button>
        <el-button type="primary" @click="addInferenceEnv">Add</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="showAddAdvEnvDialog" title="Add Advanced Environment Variable" width="500px">
      <el-form>
        <el-form-item label="Key">
          <el-input v-model="newAdvEnvKey" />
        </el-form-item>
        <el-form-item label="Value">
          <el-input v-model="newAdvEnvValue" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAddAdvEnvDialog = false">Cancel</el-button>
        <el-button type="primary" @click="addAdvancedEnv">Add</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useConfigStore } from '@/stores/config'
import { storeToRefs } from 'pinia'
import { ElMessage, ElMessageBox } from 'element-plus'

const { t } = useI18n()
const configStore = useConfigStore()
const { config, dirty } = storeToRefs(configStore)

const activeTab = ref('general')
const showAddEnvDialog = ref(false)
const showAddAdvEnvDialog = ref(false)
const newEnvKey = ref('')
const newEnvValue = ref('')
const newAdvEnvKey = ref('')
const newAdvEnvValue = ref('')

const modelPaths = ref<string[]>([])
const sglangArgsStr = ref('')
const llamafactoryArgsStr = ref('')
const testingConnection = ref(false)
const connectionTestResult = ref<{ type: 'success' | 'error'; message: string } | null>(null)

async function testRemoteConnection() {
  const url = config.value.server?.remoteUrl
  if (!url) {
    connectionTestResult.value = { type: 'error', message: 'Please enter a URL' }
    return
  }
  testingConnection.value = true
  connectionTestResult.value = null
  try {
    const resp = await fetch(`${url.replace(/\/+$/, '')}/v1/models`, { signal: AbortSignal.timeout(5000) })
    if (resp.ok) {
      const data = await resp.json()
      const models = (data.data || []).map((m: any) => m.id).join(', ')
      connectionTestResult.value = { type: 'success', message: `Connected! Models: ${models || 'none'}` }
    } else {
      connectionTestResult.value = { type: 'error', message: `HTTP ${resp.status}` }
    }
  } catch (e: any) {
    connectionTestResult.value = { type: 'error', message: e.message || 'Connection failed' }
  } finally {
    testingConnection.value = false
  }
}

onMounted(async () => {
  await configStore.load()
  syncModelPaths()
  syncArgs()
})

watch(() => config.value.paths?.models, syncModelPaths)
watch(() => config.value.advanced?.sglang_args, syncArgs)
watch(() => config.value.advanced?.llamafactory_args, syncArgs)

function syncModelPaths() {
  const models = config.value.paths?.models
  if (Array.isArray(models)) {
    modelPaths.value = [...models]
  } else if (typeof models === 'string') {
    modelPaths.value = [models]
  } else {
    modelPaths.value = []
  }
}

function syncArgs() {
  sglangArgsStr.value = (config.value.advanced?.sglang_args || []).join(' ')
  llamafactoryArgsStr.value = (config.value.advanced?.llamafactory_args || []).join(' ')
}

function markDirty() {
  configStore.dirty = true
}

function updateModelPaths() {
  if (!config.value.paths) config.value.paths = {}
  config.value.paths.models = modelPaths.value.length === 1 ? modelPaths.value[0] : modelPaths.value
  markDirty()
}

async function selectModelPath(idx: number) {
  const path = await window.electronAPI.fs.selectDir()
  if (path) {
    modelPaths.value[idx] = path
    updateModelPaths()
  }
}

function addModelPath() {
  modelPaths.value.push('')
  updateModelPaths()
}

function removeModelPath(idx: number) {
  modelPaths.value.splice(idx, 1)
  updateModelPaths()
}

async function selectCachePath() {
  const path = await window.electronAPI.fs.selectDir()
  if (path) {
    config.value.paths.cache = path
    markDirty()
  }
}

async function selectWeightsPath() {
  const path = await window.electronAPI.fs.selectDir()
  if (path) {
    config.value.paths.weights = path
    markDirty()
  }
}

function addInferenceEnv() {
  if (newEnvKey.value && newEnvValue.value) {
    if (!config.value.inference) config.value.inference = { env: {} }
    if (!config.value.inference.env) config.value.inference.env = {}
    config.value.inference.env[newEnvKey.value] = newEnvValue.value
    newEnvKey.value = ''
    newEnvValue.value = ''
    showAddEnvDialog.value = false
    markDirty()
  }
}

function removeInferenceEnv(key: string) {
  delete config.value.inference.env[key]
  markDirty()
}

function addAdvancedEnv() {
  if (newAdvEnvKey.value && newAdvEnvValue.value) {
    if (!config.value.advanced) config.value.advanced = { env: {} }
    if (!config.value.advanced.env) config.value.advanced.env = {}
    config.value.advanced.env[newAdvEnvKey.value] = newAdvEnvValue.value
    newAdvEnvKey.value = ''
    newAdvEnvValue.value = ''
    showAddAdvEnvDialog.value = false
    markDirty()
  }
}

function removeAdvancedEnv(key: string) {
  delete config.value.advanced.env[key]
  markDirty()
}

function updateSglangArgs() {
  if (!config.value.advanced) config.value.advanced = {}
  config.value.advanced.sglang_args = sglangArgsStr.value.split(/\s+/).filter(Boolean)
  markDirty()
}

function updateLlamafactoryArgs() {
  if (!config.value.advanced) config.value.advanced = {}
  config.value.advanced.llamafactory_args = llamafactoryArgsStr.value.split(/\s+/).filter(Boolean)
  markDirty()
}

async function handleSave() {
  try {
    await configStore.save()
    ElMessage.success('Configuration saved')
  } catch (e: any) {
    ElMessage.error(`Save failed: ${e.message}`)
  }
}

async function handleRevert() {
  await configStore.load()
  syncModelPaths()
  syncArgs()
  ElMessage.info('Changes reverted')
}

async function handleReset() {
  await ElMessageBox.confirm('Reset all configuration to defaults?', 'Warning', { type: 'warning' })
  await configStore.reset()
  syncModelPaths()
  syncArgs()
  ElMessage.success('Configuration reset to defaults')
}
</script>

<style scoped>
.config-page {
  padding: 20px;
}
.footer-bar {
  display: flex;
  align-items: center;
  margin-top: 30px;
  padding-top: 20px;
  border-top: 1px solid #e8e8e8;
}
.spacer {
  flex: 1;
}
</style>
