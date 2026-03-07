<template>
  <div class="header">
    <el-button @click="appStore.toggleSidebar" :icon="Fold" circle />
    <div class="spacer"></div>
    <el-dropdown @command="handleLocale">
      <el-button :icon="Switch" circle />
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item command="en">English</el-dropdown-item>
          <el-dropdown-item command="zh">中文</el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
    <el-button @click="appStore.toggleTheme" :icon="theme === 'light' ? Moon : Sunny" circle />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useAppStore } from '@/stores/app'
import { Fold, Switch, Moon, Sunny } from '@element-plus/icons-vue'

const appStore = useAppStore()
const { locale } = useI18n()
const theme = computed(() => appStore.theme)

function handleLocale(lang: string) {
  appStore.setLocale(lang)
  locale.value = lang
}
</script>

<style scoped>
.header {
  display: flex;
  align-items: center;
  padding: 0 20px;
  height: 100%;
  border-bottom: 1px solid #e8e8e8;
}
.spacer {
  flex: 1;
}
</style>
