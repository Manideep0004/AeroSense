import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      },
      '/model-info': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      },
      '/predict': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      },
      '/predict-batch': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      }
    }
  }
})
