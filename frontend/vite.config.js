import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,          // default port
    proxy: {
      // Proxy all /api requests to backend server
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // If your backend expects the /api prefix, keep it; otherwise rewrite
        // rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})