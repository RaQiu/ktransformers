<template>
  <el-card class="kv-cache-panel">
    <template #header>
      <span>KV Cache Calculator</span>
    </template>
    <el-form label-width="130px" size="small">
      <el-form-item label="Max Tokens">
        <el-input-number v-model="maxTokens" :min="1024" :max="2000000" :step="4096" style="width: 100%" />
      </el-form-item>
      <el-form-item label="Tensor Parallel">
        <el-input-number v-model="tp" :min="1" :max="8" style="width: 100%" />
      </el-form-item>
      <el-form-item label="Data Type">
        <el-select v-model="dtype" style="width: 100%">
          <el-option label="Auto (bfloat16)" value="auto" />
          <el-option label="float16" value="float16" />
          <el-option label="bfloat16" value="bfloat16" />
          <el-option label="float8_e4m3fn" value="float8_e4m3fn" />
        </el-select>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="calculate" :loading="loading" :disabled="!modelPath" style="width: 100%">
          Calculate
        </el-button>
      </el-form-item>
    </el-form>

    <div v-if="result" class="result">
      <div class="kv-gb">
        <span class="big-number">{{ result.kv_cache_gb }}</span>
        <span class="unit">GB</span>
      </div>
      <div class="sub">KV Cache VRAM required</div>
      <el-divider />
      <div class="details">
        <div class="detail-row" v-for="(val, key) in result.details" :key="key">
          <span>{{ key }}</span><span>{{ val }}</span>
        </div>
      </div>
    </div>

    <el-alert v-if="error" :title="error" type="error" show-icon style="margin-top: 8px" />
    <el-empty v-if="!modelPath" description="Select a model to use the calculator" :image-size="60" />
  </el-card>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'

const props = defineProps<{ modelPath?: string }>()

const maxTokens = ref(131072)
const tp = ref(1)
const dtype = ref('auto')
const loading = ref(false)
const result = ref<any>(null)
const error = ref('')

watch(() => props.modelPath, () => { result.value = null; error.value = '' })

async function calculate() {
  if (!props.modelPath) return
  loading.value = true
  error.value = ''
  result.value = null
  try {
    const res = await window.electronAPI.kvCache.calc({
      modelPath: props.modelPath,
      maxTokens: maxTokens.value,
      tp: tp.value,
      dtype: dtype.value
    })
    if (res.success) {
      result.value = res
    } else {
      error.value = res.error || 'Calculation failed'
    }
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.kv-cache-panel { margin-top: 16px; }
.result { text-align: center; }
.kv-gb { display: flex; align-items: baseline; justify-content: center; gap: 4px; margin: 12px 0 4px; }
.big-number { font-size: 48px; font-weight: 700; color: #409eff; }
.unit { font-size: 18px; color: #606266; }
.sub { color: #909399; font-size: 13px; margin-bottom: 8px; }
.details { font-size: 12px; }
.detail-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #f0f0f0; }
.detail-row span:first-child { color: #909399; }
</style>
