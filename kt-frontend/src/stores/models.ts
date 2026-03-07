import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useModelsStore = defineStore('models', () => {
  const models = ref<any[]>([])
  const selectedModel = ref<any>(null)

  async function load() {
    models.value = await window.electronAPI.models.list()
  }

  async function add(model: any) {
    await window.electronAPI.models.add(model)
    await load()
  }

  async function remove(name: string) {
    await window.electronAPI.models.remove(name)
    await load()
  }

  async function update(name: string, updates: any) {
    await window.electronAPI.models.update(name, updates)
    await load()
  }

  function select(model: any) {
    selectedModel.value = model
  }

  return { models, selectedModel, load, add, remove, update, select }
})
