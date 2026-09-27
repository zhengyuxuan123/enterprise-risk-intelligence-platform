import { defineStore } from 'pinia'
import { api } from '../api'
import { useAiTaskStore } from './aiTask'

function clearAuthStorage () {
  localStorage.removeItem('token')
  localStorage.removeItem('user')
}

export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: localStorage.getItem('token') || '',
    user: JSON.parse(localStorage.getItem('user') || 'null')
  }),
  getters: {
    permissions: state => state.user?.permissions || []
  },
  actions: {
    async login (form) {
      // A browser may switch accounts without reloading the application.
      useAiTaskStore().discardSession()
      const result = await api.login(form)
      this.token = result.token
      this.user = result
      localStorage.setItem('token', result.token)
      localStorage.setItem('user', JSON.stringify(result))
    },
    logout () {
      useAiTaskStore().discardSession()
      this.token = ''
      this.user = null
      clearAuthStorage()
    },
    has (permission) {
      return this.permissions.includes(permission)
    }
  }
})
