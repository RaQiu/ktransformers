<template>
  <el-container class="app-layout">
    <el-aside :width="sidebarWidth">
      <AppSidebar />
    </el-aside>
    <el-container>
      <el-header height="60px">
        <AppHeader />
      </el-header>
      <el-main>
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useAppStore } from '@/stores/app'
import AppSidebar from './AppSidebar.vue'
import AppHeader from './AppHeader.vue'

const appStore = useAppStore()
const sidebarWidth = computed(() => appStore.sidebarCollapsed ? '64px' : '200px')
</script>

<style scoped>
.app-layout {
  height: 100vh;
}
</style>
