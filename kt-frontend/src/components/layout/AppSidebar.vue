<template>
  <div class="sidebar">
    <div class="logo">
      <span v-if="!collapsed">KTransformers</span>
      <span v-else>KT</span>
    </div>
    <el-menu
      :default-active="route.path"
      router
      :collapse="collapsed"
      background-color="#001529"
      text-color="#ffffffb3"
      active-text-color="#409eff"
    >
      <el-menu-item index="/">
        <el-icon><Monitor /></el-icon>
        <span>{{ t('nav.dashboard') }}</span>
      </el-menu-item>
      <el-menu-item index="/service">
        <el-icon><Setting /></el-icon>
        <span>{{ t('nav.service') }}</span>
      </el-menu-item>
      <el-menu-item index="/models">
        <el-icon><Files /></el-icon>
        <span>{{ t('nav.models') }}</span>
      </el-menu-item>
      <el-menu-item index="/bench">
        <el-icon><DataAnalysis /></el-icon>
        <span>{{ t('nav.bench') }}</span>
      </el-menu-item>
      <el-menu-item index="/chat">
        <el-icon><ChatDotRound /></el-icon>
        <span>{{ t('nav.chat') }}</span>
        <el-badge v-if="serverRunning" value="Live" type="success" class="running-badge" />
      </el-menu-item>
      <el-menu-item index="/config">
        <el-icon><Tools /></el-icon>
        <span>{{ t('nav.config') }}</span>
      </el-menu-item>
    </el-menu>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useAppStore } from '@/stores/app'
import { useServerStore } from '@/stores/server'
import { storeToRefs } from 'pinia'
import { Monitor, Setting, Files, Tools, DataAnalysis, ChatDotRound } from '@element-plus/icons-vue'

const route = useRoute()
const { t } = useI18n()
const appStore = useAppStore()
const serverStore = useServerStore()
const collapsed = computed(() => appStore.sidebarCollapsed)
const { running: serverRunning } = storeToRefs(serverStore)
</script>

<style scoped>
.sidebar {
  height: 100%;
  background: #001529;
}
.sidebar :deep(.el-menu) {
  border-right: none;
}
.logo {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 18px;
  font-weight: bold;
}
.running-badge {
  margin-left: 8px;
}
</style>
