// Simple Node.js WebSocket-to-stdio bridge for LSP
// Usage: node lsp-ws-bridge.js
// Requires: npm install ws

const { spawn } = require('child_process');
const WebSocket = require('ws');

const LSP_PORT = 3001; // Port for WebSocket server
const LSP_CMD = 'python';
const LSP_ARGS = ['bengali_lsp_server.py'];
const LSP_CWD = '../backend'; // Adjust if needed

const wss = new WebSocket.Server({ port: LSP_PORT });

wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  const lsp = spawn(LSP_CMD, LSP_ARGS, { cwd: LSP_CWD });

  // WebSocket -> LSP stdin
  ws.on('message', (msg) => {
    lsp.stdin.write(msg);
  });

  // LSP stdout -> WebSocket
  lsp.stdout.on('data', (data) => {
    ws.send(data);
  });

  // LSP stderr (for debugging)
  lsp.stderr.on('data', (data) => {
    console.error('LSP stderr:', data.toString());
  });

  ws.on('close', () => {
    lsp.kill();
    console.log('WebSocket client disconnected');
  });

  lsp.on('exit', (code) => {
    ws.close();
  });
});

console.log(`LSP WebSocket bridge running on ws://localhost:${LSP_PORT}`);
