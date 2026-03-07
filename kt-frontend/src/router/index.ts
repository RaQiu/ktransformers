import { createRouter, createWebHashHistory } from 'vue-router'
import AppLayout from '@/components/layout/AppLayout.vue'
import DashboardPage from '@/pages/DashboardPage.vue'
import ServicePage from '@/pages/ServicePage.vue'
import ModelsPage from '@/pages/ModelsPage.vue'
import ConfigPage from '@/pages/ConfigPage.vue'
import WizardPage from '@/pages/WizardPage.vue'
import BenchPage from '@/pages/BenchPage.vue'
import ChatPage from '@/pages/ChatPage.vue'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    {
      path: '/wizard',
      name: 'wizard',
      component: WizardPage
    },
    {
      path: '/',
      component: AppLayout,
      children: [
        { path: '', name: 'dashboard', component: DashboardPage },
        { path: 'service', name: 'service', component: ServicePage },
        { path: 'models', name: 'models', component: ModelsPage },
        { path: 'config', name: 'config', component: ConfigPage },
        { path: 'bench', name: 'bench', component: BenchPage },
        { path: 'chat', name: 'chat', component: ChatPage }
      ]
    }
  ]
})

export default router
