import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'monaco-lsp.html'),
      },
      output: {
        entryFileNames: 'assets/[name].js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name].[ext]'
      }
    }
  },
  server: {
    port: 3000,
    open: true
  },
  resolve: {
    alias: {
      // '@codingame/monaco-vscode-api/vscode/vs': 'node_modules/@codingame/monaco-vscode-api/vscode/src/vs',
      // '@codingame/monaco-vscode-api/vscode/src/vs/workbench/api/common/extHostTypes': 'node_modules/@codingame/monaco-vscode-api/vscode/src/vs/workbench/api/common/extHostTypes.js',
      // '@codingame/monaco-vscode-api/vscode/src/vs/base/common/errors': 'node_modules/@codingame/monaco-vscode-api/vscode/src/vs/base/common/errors.js',
    }
  }
});