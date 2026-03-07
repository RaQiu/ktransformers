<template>
  <div class="chat-page">
    <!-- Sidebar: Conversation History -->
    <div class="chat-sidebar" :class="{ collapsed: sidebarCollapsed }">
      <div class="sidebar-header">
        <el-button type="primary" size="small" @click="newChat" style="width: 100%">
          + New Chat
        </el-button>
      </div>
      <div class="conversation-list">
        <div
          v-for="conv in chatStore.sortedConversations"
          :key="conv.id"
          class="conv-item"
          :class="{ active: conv.id === chatStore.currentId }"
          @click="chatStore.switchTo(conv.id)"
        >
          <div class="conv-title">{{ conv.title }}</div>
          <div class="conv-meta">
            <span>{{ conv.model || 'unknown' }}</span>
            <el-button
              link
              size="small"
              type="danger"
              @click.stop="deleteConv(conv.id)"
              class="conv-delete"
            >Del</el-button>
          </div>
        </div>
        <div v-if="!chatStore.conversations.length" class="empty-sidebar">
          No conversations yet
        </div>
      </div>
      <div class="sidebar-toggle" @click="sidebarCollapsed = !sidebarCollapsed">
        {{ sidebarCollapsed ? '>' : '<' }}
      </div>
    </div>

    <!-- Main Chat Area -->
    <div class="chat-main">
      <!-- Connection Bar -->
      <div class="connection-bar">
        <div class="conn-left">
          <el-select
            v-model="activeUrl"
            filterable
            allow-create
            default-first-option
            placeholder="Server URL"
            size="small"
            style="width: 300px"
            @change="onUrlChange"
          >
            <el-option
              v-for="conn in savedConnections"
              :key="conn"
              :label="conn"
              :value="conn"
            />
          </el-select>
          <span class="conn-status-dot" :class="serverAlive ? 'online' : 'offline'" />
        </div>
        <div class="conn-right">
          <el-select
            v-model="selectedModel"
            placeholder="Model"
            size="small"
            style="width: 220px"
            :loading="loadingModels"
          >
            <el-option v-for="m in detectedModels" :key="m" :label="m" :value="m" />
          </el-select>
          <el-button size="small" @click="saveConnection" :disabled="!activeUrl">Save URL</el-button>
        </div>
      </div>

      <!-- Warning -->
      <el-alert
        v-if="!serverAlive && !checking"
        title="Server is not reachable. Check your connection or enter a different URL."
        type="warning"
        show-icon
        :closable="false"
        style="margin: 0 16px"
      />

      <!-- Messages -->
      <div class="messages" ref="messagesEl">
        <div v-if="!currentMessages.length" class="empty-hint">
          Send a message to start chatting
        </div>
        <div v-for="(msg, i) in currentMessages" :key="i" :class="['message', msg.role]">
          <div class="role">{{ msg.role === 'user' ? 'You' : 'Assistant' }}</div>
          <div class="content" v-html="renderContent(msg.content)" />
        </div>
        <div v-if="streaming" class="message assistant">
          <div class="role">Assistant</div>
          <div class="content" v-html="renderContent(streamBuffer)" />
          <span class="cursor">▋</span>
        </div>
      </div>

      <!-- Input -->
      <div class="input-area">
        <el-input
          v-model="inputText"
          type="textarea"
          :rows="3"
          placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
          @keydown.enter.exact.prevent="sendMessage"
          :disabled="streaming || !serverAlive"
        />
        <div class="input-actions">
          <el-popover trigger="click" width="400">
            <template #reference>
              <el-button size="small">System Prompt</el-button>
            </template>
            <el-input v-model="systemPrompt" type="textarea" :rows="4" placeholder="Optional system prompt" />
          </el-popover>
          <el-button
            type="primary"
            @click="sendMessage"
            :disabled="!inputText.trim() || streaming || !serverAlive"
            :loading="streaming"
          >
            Send
          </el-button>
          <el-button v-if="streaming" type="warning" @click="stopStream">Stop</el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, onMounted, onUnmounted, watch } from 'vue'
import { useConfigStore } from '@/stores/config'
import { useChatStore } from '@/stores/chat'
import { checkServerHealth, fetchServerModels } from '@/api/server-client'
import { ElMessage, ElMessageBox } from 'element-plus'

const configStore = useConfigStore()
const chatStore = useChatStore()

const SAVED_CONNECTIONS_KEY = 'kt-saved-connections'

const sidebarCollapsed = ref(false)
const inputText = ref('')
const systemPrompt = ref('')
const streaming = ref(false)
const streamBuffer = ref('')
const messagesEl = ref<HTMLElement>()
const checking = ref(false)
const serverAlive = ref(false)
const loadingModels = ref(false)
const activeUrl = ref('')
const selectedModel = ref('')
const detectedModels = ref<string[]>([])
const savedConnections = ref<string[]>([])

let abortController: AbortController | null = null
let healthInterval: any

const currentMessages = computed(() => chatStore.current?.messages || [])

// --- Connection management ---

function loadSavedConnections() {
  try {
    const stored = localStorage.getItem(SAVED_CONNECTIONS_KEY)
    if (stored) savedConnections.value = JSON.parse(stored)
  } catch {}
}

function saveConnection() {
  const url = activeUrl.value.trim().replace(/\/+$/, '')
  if (!url) return
  if (!savedConnections.value.includes(url)) {
    savedConnections.value.push(url)
    localStorage.setItem(SAVED_CONNECTIONS_KEY, JSON.stringify(savedConnections.value))
    ElMessage.success('Connection saved')
  }
}

async function checkHealth() {
  if (!activeUrl.value) return
  checking.value = true
  try {
    serverAlive.value = await checkServerHealth(activeUrl.value)
  } finally {
    checking.value = false
  }
}

async function detectModels() {
  if (!activeUrl.value) return
  loadingModels.value = true
  try {
    const models = await fetchServerModels(activeUrl.value)
    detectedModels.value = models.map(m => m.id)
    if (detectedModels.value.length && !selectedModel.value) {
      selectedModel.value = detectedModels.value[0]
    }
  } finally {
    loadingModels.value = false
  }
}

async function onUrlChange() {
  serverAlive.value = false
  detectedModels.value = []
  selectedModel.value = ''
  await checkHealth()
  if (serverAlive.value) await detectModels()
  // Update current conversation meta
  if (chatStore.currentId) {
    chatStore.updateConversationMeta(chatStore.currentId, {
      serverUrl: activeUrl.value,
      model: selectedModel.value,
    })
  }
}

// --- Chat ---

function newChat() {
  const url = activeUrl.value || 'http://localhost:30000'
  const model = selectedModel.value || ''
  chatStore.createConversation(url, model)
}

function deleteConv(id: string) {
  chatStore.deleteConversation(id)
}

function renderContent(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>')
}

async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || streaming.value) return

  // Ensure there's a current conversation
  if (!chatStore.currentId) {
    newChat()
  }

  chatStore.addMessage({ role: 'user', content: text, timestamp: Date.now() })
  inputText.value = ''
  streamBuffer.value = ''
  streaming.value = true

  await scrollToBottom()

  const chatMessages: { role: string; content: string }[] = []
  if (systemPrompt.value) {
    chatMessages.push({ role: 'system', content: systemPrompt.value })
  }
  for (const m of currentMessages.value) {
    chatMessages.push({ role: m.role, content: m.content })
  }

  try {
    abortController = new AbortController()
    const response = await fetch(`${activeUrl.value}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: selectedModel.value || 'default',
        messages: chatMessages,
        stream: true,
        max_tokens: 2048,
      }),
      signal: abortController.signal
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`)
    }

    const reader = response.body!.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const chunk = decoder.decode(value, { stream: true })
      for (const line of chunk.split('\n')) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6).trim()
        if (data === '[DONE]') break
        try {
          const parsed = JSON.parse(data)
          const delta = parsed.choices?.[0]?.delta?.content
          if (delta) {
            streamBuffer.value += delta
            await scrollToBottom()
          }
        } catch {}
      }
    }

    chatStore.addMessage({ role: 'assistant', content: streamBuffer.value, timestamp: Date.now() })
    streamBuffer.value = ''
  } catch (e: any) {
    if (e.name !== 'AbortError') {
      chatStore.addMessage({ role: 'assistant', content: `Error: ${e.message}`, timestamp: Date.now() })
    }
  } finally {
    streaming.value = false
    abortController = null
    await scrollToBottom()
  }
}

function stopStream() {
  abortController?.abort()
  if (streamBuffer.value) {
    chatStore.addMessage({ role: 'assistant', content: streamBuffer.value + ' [stopped]', timestamp: Date.now() })
  }
  streaming.value = false
  streamBuffer.value = ''
}

async function scrollToBottom() {
  await nextTick()
  if (messagesEl.value) {
    messagesEl.value.scrollTop = messagesEl.value.scrollHeight
  }
}

// --- Watch model selection changes ---
watch(selectedModel, (model) => {
  if (chatStore.currentId) {
    chatStore.updateConversationMeta(chatStore.currentId, { model })
  }
})

// When switching conversations, restore their URL/model
watch(() => chatStore.currentId, () => {
  const conv = chatStore.current
  if (conv) {
    if (conv.serverUrl && conv.serverUrl !== activeUrl.value) {
      activeUrl.value = conv.serverUrl
      onUrlChange()
    }
    if (conv.model) selectedModel.value = conv.model
  }
})

// --- Lifecycle ---

onMounted(async () => {
  loadSavedConnections()
  chatStore.load()
  await configStore.load()

  // Init URL from config
  const cfg = configStore.config?.server
  if (cfg?.mode === 'remote' && cfg.remoteUrl) {
    activeUrl.value = cfg.remoteUrl.replace(/\/+$/, '')
  } else {
    const host = (cfg?.host === '0.0.0.0' ? 'localhost' : cfg?.host) || 'localhost'
    const port = cfg?.port || 30000
    activeUrl.value = `http://${host}:${port}`
  }

  // Add config URL to saved list if not there
  if (activeUrl.value && !savedConnections.value.includes(activeUrl.value)) {
    savedConnections.value.unshift(activeUrl.value)
    localStorage.setItem(SAVED_CONNECTIONS_KEY, JSON.stringify(savedConnections.value))
  }

  await checkHealth()
  if (serverAlive.value) await detectModels()

  // If no conversation exists, create one
  if (!chatStore.currentId) {
    newChat()
  }

  healthInterval = setInterval(checkHealth, 10000)
})

onUnmounted(() => {
  if (healthInterval) clearInterval(healthInterval)
})
</script>

<style scoped>
.chat-page {
  display: flex;
  height: calc(100vh - 60px);
}

/* Sidebar */
.chat-sidebar {
  width: 240px;
  background: #f8f9fa;
  border-right: 1px solid #e8e8e8;
  display: flex;
  flex-direction: column;
  position: relative;
  transition: width 0.2s;
  flex-shrink: 0;
}
.chat-sidebar.collapsed { width: 0; overflow: hidden; }
.sidebar-header { padding: 12px; }
.conversation-list {
  flex: 1;
  overflow-y: auto;
  padding: 0 8px;
}
.conv-item {
  padding: 10px 12px;
  border-radius: 6px;
  cursor: pointer;
  margin-bottom: 4px;
  transition: background 0.15s;
}
.conv-item:hover { background: #ebeef5; }
.conv-item.active { background: #e6f0ff; }
.conv-title {
  font-size: 13px;
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.conv-meta {
  font-size: 11px;
  color: #909399;
  margin-top: 4px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.conv-delete { opacity: 0; transition: opacity 0.15s; }
.conv-item:hover .conv-delete { opacity: 1; }
.empty-sidebar {
  text-align: center;
  color: #c0c4cc;
  font-size: 13px;
  padding: 20px;
}
.sidebar-toggle {
  position: absolute;
  right: -16px;
  top: 50%;
  transform: translateY(-50%);
  width: 16px;
  height: 40px;
  background: #e8e8e8;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  border-radius: 0 4px 4px 0;
  font-size: 11px;
  color: #909399;
  z-index: 1;
}

/* Main area */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

/* Connection bar */
.connection-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  border-bottom: 1px solid #ebeef5;
  background: #fafafa;
  flex-wrap: wrap;
  gap: 8px;
}
.conn-left, .conn-right { display: flex; align-items: center; gap: 8px; }
.conn-status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  display: inline-block;
}
.conn-status-dot.online { background: #67c23a; box-shadow: 0 0 4px #67c23a; }
.conn-status-dot.offline { background: #f56c6c; }

/* Messages */
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.empty-hint { color: #909399; text-align: center; margin-top: 40px; }
.message {
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 8px;
}
.message.user {
  align-self: flex-end;
  background: #409eff;
  color: white;
}
.message.assistant {
  align-self: flex-start;
  background: #f5f7fa;
  color: #303133;
}
.role { font-size: 11px; font-weight: 600; margin-bottom: 6px; opacity: 0.7; }
.content :deep(pre) {
  background: rgba(0,0,0,0.1);
  padding: 8px;
  border-radius: 4px;
  overflow-x: auto;
  margin: 4px 0;
}
.content :deep(code) { font-family: monospace; font-size: 13px; }
.cursor { animation: blink 1s step-end infinite; }
@keyframes blink { 0%, 100% { opacity: 1 } 50% { opacity: 0 } }

/* Input */
.input-area {
  padding: 16px;
  border-top: 1px solid #e8e8e8;
}
.input-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 8px;
}
</style>
