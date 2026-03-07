<template>
  <div :data-theme="theme">
    <router-view />
  </div>
</template>

<script setup lang="ts">
import { onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useAppStore } from './stores/app'
import { storeToRefs } from 'pinia'

const appStore = useAppStore()
const router = useRouter()
const { theme } = storeToRefs(appStore)

onMounted(() => {
  appStore.init()
  // Handle keyboard shortcut navigation from Electron main process (Ext 10)
  window.electronAPI.nav.onGoto((route) => {
    router.push(route)
  })
})

watch(theme, (newTheme) => {
  document.documentElement.setAttribute('data-theme', newTheme)
})
</script>

<style>
@import './styles/global.scss';
</style>
