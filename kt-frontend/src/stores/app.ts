import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useAppStore = defineStore('app', () => {
  const theme = ref<'light' | 'dark'>('light')
  const locale = ref('en')
  const sidebarCollapsed = ref(false)

  async function init() {
    const config = await window.electronAPI.config.get('general.language')
    if (config && config !== 'auto') locale.value = config
  }

  function toggleTheme() {
    theme.value = theme.value === 'light' ? 'dark' : 'light'
  }

  function toggleSidebar() {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  function setLocale(lang: string) {
    locale.value = lang
  }

  return { theme, locale, sidebarCollapsed, init, toggleTheme, toggleSidebar, setLocale }
})
