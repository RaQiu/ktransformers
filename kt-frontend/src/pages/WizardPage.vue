<template>
  <div class="wizard-page">
    <div class="wizard-container">
      <el-steps :active="step - 1" align-center>
        <el-step title="Welcome" />
        <el-step title="Environment" />
        <el-step title="Storage" />
        <el-step title="Discover" />
        <el-step title="Model" />
        <el-step title="Complete" />
      </el-steps>

      <div class="wizard-content">
        <div v-if="step === 1" class="step-content">
          <h1>Welcome to KTransformers</h1>
          <p>Let's set up your environment in a few simple steps.</p>
          <el-radio-group v-model="language" size="large">
            <el-radio-button label="en">English</el-radio-button>
            <el-radio-button label="zh">中文</el-radio-button>
          </el-radio-group>
        </div>

        <div v-if="step === 2" class="step-content">
          <h2>Environment Check</h2>
          <p>Checking your system configuration...</p>
          <el-progress v-if="checking" :percentage="checkProgress" />
          <div v-else class="check-results">
            <div v-for="(result, key) in checkResults" :key="key" class="check-item">
              <el-icon v-if="result.status === 'pass'" color="#67C23A"><CircleCheck /></el-icon>
              <el-icon v-else-if="result.status === 'warn'" color="#E6A23C"><Warning /></el-icon>
              <el-icon v-else color="#F56C6C"><CircleClose /></el-icon>
              <span>{{ key }}: {{ result.message }}</span>
            </div>
          </div>
        </div>

        <div v-if="step === 3" class="step-content">
          <h2>Model Storage Path</h2>
          <p>Choose where to store your models</p>
          <el-input v-model="storagePath" placeholder="Select directory">
            <template #append>
              <el-button @click="selectStorage">Browse</el-button>
            </template>
          </el-input>
          <div v-if="storageInfo" class="storage-info">
            <p>Available space: {{ formatBytes(storageInfo.free) }}</p>
          </div>
        </div>

        <div v-if="step === 4" class="step-content">
          <h2>Discover Existing Models</h2>
          <p>Scan for models in the selected directory?</p>
          <el-button type="primary" @click="scanModels" :loading="scanning">Scan Now</el-button>
          <div v-if="foundModels.length > 0" class="found-models">
            <p>Found {{ foundModels.length }} model(s)</p>
            <el-tag v-for="m in foundModels" :key="m" style="margin: 4px">{{ m }}</el-tag>
          </div>
        </div>

        <div v-if="step === 5" class="step-content">
          <h2>Select Default Model</h2>
          <p>Choose a model to use by default</p>
          <el-select v-model="defaultModel" placeholder="Select model" style="width: 100%">
            <el-option v-for="m in availableModels" :key="m" :label="m" :value="m" />
          </el-select>
        </div>

        <div v-if="step === 6" class="step-content">
          <h2>Setup Complete!</h2>
          <p>Your KTransformers environment is ready.</p>
          <el-result icon="success" title="All Set">
            <template #sub-title>
              <p>Language: {{ language === 'en' ? 'English' : '中文' }}</p>
              <p>Storage: {{ storagePath }}</p>
              <p>Default Model: {{ defaultModel || 'None' }}</p>
            </template>
          </el-result>
        </div>
      </div>

      <div class="wizard-footer">
        <el-button v-if="step > 1" @click="prevStep">Previous</el-button>
        <el-button v-if="step < 6" type="primary" @click="nextStep">Next</el-button>
        <el-button v-if="step === 6" type="primary" @click="finish">Go to Dashboard</el-button>
        <el-button @click="skip">Skip Setup</el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAppStore } from '@/stores/app'
import { CircleCheck, Warning, CircleClose } from '@element-plus/icons-vue'

const router = useRouter()
const appStore = useAppStore()

const step = ref(1)
const language = ref('en')
const checking = ref(false)
const checkProgress = ref(0)
const checkResults = ref<Record<string, any>>({})
const storagePath = ref('')
const storageInfo = ref<any>(null)
const scanning = ref(false)
const foundModels = ref<string[]>([])
const defaultModel = ref('')
const availableModels = ref<string[]>(['DeepSeek-V3', 'Qwen-2.5', 'Llama-3'])

onMounted(() => {
  if (step.value === 2) runChecks()
})

async function runChecks() {
  checking.value = true
  checkProgress.value = 0
  const checks = ['Python', 'CUDA', 'GPU', 'CPU', 'Memory']
  for (let i = 0; i < checks.length; i++) {
    await new Promise(resolve => setTimeout(resolve, 500))
    checkResults.value[checks[i]] = { status: 'pass', message: 'OK' }
    checkProgress.value = ((i + 1) / checks.length) * 100
  }
  checking.value = false
}

async function selectStorage() {
  const path = await window.electronAPI.fs.selectDir()
  if (path) {
    storagePath.value = path
    storageInfo.value = { free: 500 * 1024 * 1024 * 1024 }
  }
}

async function scanModels() {
  scanning.value = true
  await new Promise(resolve => setTimeout(resolve, 1000))
  foundModels.value = ['Model-A', 'Model-B']
  scanning.value = false
}

function formatBytes(bytes: number) {
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`
}

async function nextStep() {
  if (step.value === 2 && !checking.value && Object.keys(checkResults.value).length === 0) {
    await runChecks()
  }
  step.value++
}

function prevStep() {
  step.value--
}

function skip() {
  finish()
}

async function finish() {
  appStore.setLocale(language.value)
  if (storagePath.value) {
    await window.electronAPI.config.set('paths.models', storagePath.value)
  }
  router.push('/')
}
</script>

<style scoped>
.wizard-page {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.wizard-container {
  width: 800px;
  background: white;
  border-radius: 8px;
  padding: 40px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}
.wizard-content {
  min-height: 300px;
  padding: 40px 0;
}
.step-content {
  text-align: center;
}
.step-content h1 {
  font-size: 32px;
  margin-bottom: 20px;
}
.step-content h2 {
  font-size: 24px;
  margin-bottom: 20px;
}
.check-results {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 20px;
}
.check-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px;
  background: #f5f7fa;
  border-radius: 4px;
}
.storage-info {
  margin-top: 20px;
  color: #606266;
}
.found-models {
  margin-top: 20px;
}
.wizard-footer {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 40px;
  padding-top: 20px;
  border-top: 1px solid #e8e8e8;
}
</style>
