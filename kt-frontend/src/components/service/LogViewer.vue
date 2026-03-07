<template>
  <div class="log-viewer">
    <div class="toolbar">
      <el-input v-model="searchText" placeholder="Search logs..." style="width: 300px" clearable>
        <template #prefix><el-icon><Search /></el-icon></template>
      </el-input>
      <el-button @click="paused = !paused" :type="paused ? 'warning' : 'default'">
        {{ paused ? 'Resume' : 'Pause' }}
      </el-button>
      <el-button @click="handleClear">Clear</el-button>
      <el-button @click="handleCopy">Copy All</el-button>
    </div>
    <div ref="logContainer" class="log-container" @scroll="handleScroll">
      <div v-for="(log, i) in filteredLogs" :key="i" class="log-line">{{ log }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'
import { Search } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const props = defineProps<{
  logs: string[]
}>()

const emit = defineEmits<{
  clear: []
}>()

const searchText = ref('')
const paused = ref(false)
const autoScroll = ref(true)
const logContainer = ref<HTMLElement>()

const filteredLogs = computed(() => {
  if (!searchText.value) return props.logs
  return props.logs.filter(log => log.toLowerCase().includes(searchText.value.toLowerCase()))
})

watch(() => props.logs.length, async () => {
  if (!paused.value && autoScroll.value) {
    await nextTick()
    scrollToBottom()
  }
})

function handleScroll() {
  if (!logContainer.value) return
  const { scrollTop, scrollHeight, clientHeight } = logContainer.value
  autoScroll.value = scrollTop + clientHeight >= scrollHeight - 50
}

function scrollToBottom() {
  if (logContainer.value) {
    logContainer.value.scrollTop = logContainer.value.scrollHeight
  }
}

function handleClear() {
  emit('clear')
}

function handleCopy() {
  navigator.clipboard.writeText(props.logs.join('\n'))
  ElMessage.success('Logs copied to clipboard')
}
</script>

<style scoped>
.log-viewer {
  display: flex;
  flex-direction: column;
  height: 100%;
}
.toolbar {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
}
.log-container {
  flex: 1;
  overflow-y: auto;
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 10px;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 4px;
}
.log-line {
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
