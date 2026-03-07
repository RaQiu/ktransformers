import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDoctorStore = defineStore('doctor', () => {
  const results = ref<any>({})
  const running = ref(false)

  async function run() {
    running.value = true
    try {
      results.value = await window.electronAPI.doctor.run()
    } finally {
      running.value = false
    }
  }

  return { results, running, run }
})
