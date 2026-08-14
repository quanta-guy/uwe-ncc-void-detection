import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Bound to localhost only: this is a local inspection workstation, not a service.
//
// /ollama is proxied to the local Ollama daemon rather than called directly from the
// page. Ollama restricts browser origins by default, and proxying keeps the model
// endpoint same-origin - so nothing has to be reconfigured, and no inspection data
// ever crosses a network boundary.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/ollama": {
        target: "http://127.0.0.1:11434",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/ollama/, ""),
      },
    },
  },
});
