import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Bound to localhost only: this is a local inspection workstation, not a service.
export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", port: 5173, strictPort: true },
});
