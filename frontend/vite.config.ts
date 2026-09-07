import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

const apiBackendUrl = process.env.VITE_API_BACKEND_URL || 'http://127.0.0.1:8001'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': apiBackendUrl,
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    css: true,
    fileParallelism: false,
    maxWorkers: 1,
  },
})
