import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The frontend talks to FastAPI only through /api/*.
// In development Vite proxies those calls to the local backend so the browser
// never needs CORS during day-to-day work.
const apiProxy = {
  '/api': {
    target: process.env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000',
    changeOrigin: true,
  },
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
    proxy: apiProxy,
  },
  // Same proxy for `vite preview`, so the production build can be smoke-tested
  // against the real API locally.
  preview: {
    port: 4173,
    strictPort: false,
    proxy: apiProxy,
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
