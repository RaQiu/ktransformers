import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

const STORAGE_KEY = 'kt-chat-history'

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
}

export interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  serverUrl: string
  model: string
  createdAt: number
  updatedAt: number
}

export const useChatStore = defineStore('chat', () => {
  const conversations = ref<Conversation[]>([])
  const currentId = ref<string | null>(null)

  const current = computed(() =>
    conversations.value.find(c => c.id === currentId.value) || null
  )

  const sortedConversations = computed(() =>
    [...conversations.value].sort((a, b) => b.updatedAt - a.updatedAt)
  )

  function load() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const parsed = JSON.parse(stored)
        conversations.value = parsed.conversations || []
        currentId.value = parsed.currentId || null
        // validate currentId
        if (currentId.value && !conversations.value.find(c => c.id === currentId.value)) {
          currentId.value = conversations.value[0]?.id || null
        }
      }
    } catch {}
  }

  function persist() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        conversations: conversations.value,
        currentId: currentId.value,
      }))
    } catch {}
  }

  function createConversation(serverUrl: string, model: string): string {
    const id = `chat-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    const conv: Conversation = {
      id,
      title: 'New Chat',
      messages: [],
      serverUrl,
      model,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }
    conversations.value.unshift(conv)
    currentId.value = id
    persist()
    return id
  }

  function switchTo(id: string) {
    currentId.value = id
    persist()
  }

  function addMessage(msg: ChatMessage) {
    const conv = current.value
    if (!conv) return
    conv.messages.push(msg)
    conv.updatedAt = Date.now()
    // Auto-title from first user message
    if (conv.title === 'New Chat' && msg.role === 'user') {
      conv.title = msg.content.slice(0, 40) + (msg.content.length > 40 ? '...' : '')
    }
    persist()
  }

  function updateLastAssistantMessage(content: string) {
    const conv = current.value
    if (!conv) return
    const last = conv.messages[conv.messages.length - 1]
    if (last && last.role === 'assistant') {
      last.content = content
    } else {
      conv.messages.push({ role: 'assistant', content, timestamp: Date.now() })
    }
    conv.updatedAt = Date.now()
    persist()
  }

  function deleteConversation(id: string) {
    const idx = conversations.value.findIndex(c => c.id === id)
    if (idx === -1) return
    conversations.value.splice(idx, 1)
    if (currentId.value === id) {
      currentId.value = conversations.value[0]?.id || null
    }
    persist()
  }

  function clearAll() {
    conversations.value = []
    currentId.value = null
    persist()
  }

  function updateConversationMeta(id: string, meta: { serverUrl?: string; model?: string }) {
    const conv = conversations.value.find(c => c.id === id)
    if (!conv) return
    if (meta.serverUrl !== undefined) conv.serverUrl = meta.serverUrl
    if (meta.model !== undefined) conv.model = meta.model
    persist()
  }

  return {
    conversations, currentId, current, sortedConversations,
    load, persist, createConversation, switchTo,
    addMessage, updateLastAssistantMessage,
    deleteConversation, clearAll, updateConversationMeta,
  }
})
