import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API is proxied rather than called cross-origin, so the app uses plain
// relative `/api/...` paths and works identically under `vite dev` and when
// built and served from anywhere. It also keeps the backend's CORS allowlist
// from being load-bearing.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8770",
        changeOrigin: false,
      },
    },
  },
});
