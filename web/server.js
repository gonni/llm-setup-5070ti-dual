/**
 * LLM Streaming Web Application Server
 * Connects to vLLM server at http://gpu2:8000/v1
 * Zero external dependencies (Pure Node.js built-in modules)
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const PORT = parseInt(process.env.PORT, 10) || 3000;
const LLM_BASE_URL = (process.env.LLM_BASE_URL || 'http://gpu2:8000/v1').replace(/\/+$/, '');
const PUBLIC_DIR = path.join(__dirname, 'public');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
};

/**
 * Fetch helper for LLM API
 */
async function fetchLLM(endpoint, options = {}) {
  const targetUrl = `${LLM_BASE_URL}${endpoint}`;
  return fetch(targetUrl, options);
}

/**
 * Handle static file requests
 */
function serveStaticFile(req, res, pathname) {
  let filePath = path.join(PUBLIC_DIR, pathname === '/' ? 'index.html' : pathname);

  // Security: prevent directory traversal
  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403, { 'Content-Type': 'text/plain' });
    res.end('403 Forbidden');
    return;
  }

  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      // Fallback to index.html for SPA if not found
      filePath = path.join(PUBLIC_DIR, 'index.html');
      fs.stat(filePath, (fallbackErr, fallbackStats) => {
        if (fallbackErr || !fallbackStats.isFile()) {
          res.writeHead(404, { 'Content-Type': 'text/plain' });
          res.end('404 Not Found');
          return;
        }
        streamFile(res, filePath);
      });
      return;
    }
    streamFile(res, filePath);
  });
}

function streamFile(res, filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';
  res.writeHead(200, {
    'Content-Type': contentType,
    'Cache-Control': 'no-cache',
  });
  fs.createReadStream(filePath).pipe(res);
}

/**
 * Helper to read JSON request body
 */
function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', chunk => {
      body += chunk;
      if (body.length > 50 * 1024 * 1024) {
        // 50MB limit for high-res images
        reject(new Error('Payload too large (Max 50MB)'));
      }
    });
    req.on('end', () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (err) {
        reject(new Error('Invalid JSON'));
      }
    });
    req.on('error', reject);
  });
}

/**
 * Main HTTP Server
 */
const server = http.createServer(async (req, res) => {
  // CORS Headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;

  // 1. GET /api/models - List available models from LLM server
  if (pathname === '/api/models' && req.method === 'GET') {
    try {
      const response = await fetchLLM('/models');
      if (!response.ok) {
        throw new Error(`LLM server responded with HTTP ${response.status}`);
      }
      const data = await response.json();
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        success: true,
        llmBaseUrl: LLM_BASE_URL,
        models: data.data || [],
        raw: data
      }));
    } catch (err) {
      console.error('[Error] /api/models:', err.message);
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        success: false,
        error: `Failed to connect to LLM server at ${LLM_BASE_URL}/models: ${err.message}`,
        llmBaseUrl: LLM_BASE_URL,
        models: []
      }));
    }
    return;
  }

  // 2. GET /api/health - Check server & LLM connectivity
  if (pathname === '/api/health' && req.method === 'GET') {
    let llmConnected = false;
    let availableModels = [];
    try {
      const resp = await fetchLLM('/models');
      if (resp.ok) {
        const json = await resp.json();
        llmConnected = true;
        availableModels = (json.data || []).map(m => m.id);
      }
    } catch (e) {
      llmConnected = false;
    }

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: 'ok',
      timestamp: new Date().toISOString(),
      llm: {
        baseUrl: LLM_BASE_URL,
        connected: llmConnected,
        models: availableModels
      }
    }));
    return;
  }

  // 3. POST /api/chat - Streaming chat completion
  if (pathname === '/api/chat' && req.method === 'POST') {
    let body;
    try {
      body = await readJsonBody(req);
    } catch (err) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
      return;
    }

    const {
      model = 'qwen3.8:27b',
      messages = [],
      temperature = 0.7,
      max_tokens = 16384,
      stream = true,
      top_p = 0.9,
      enable_thinking = true,
    } = body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'messages array is required' }));
      return;
    }

    // Set up SSE streaming headers to the browser client
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
    });

    const abortController = new AbortController();

    // If browser client aborts connection, cancel upstream LLM request
    req.on('close', () => {
      if (!res.writableEnded) {
        abortController.abort();
      }
    });

    try {
      // Calculate safe max_tokens (Smart token estimation distinguishing text vs base64 images)
      const MAX_CONTEXT_LEN = 24576; // Match vLLM max_model_len
      let textChars = 0;
      let imageCount = 0;
      for (const msg of messages) {
        if (typeof msg.content === 'string') {
          textChars += msg.content.length;
        } else if (Array.isArray(msg.content)) {
          for (const part of msg.content) {
            if (part.type === 'text') {
              textChars += (part.text || '').length;
            } else if (part.type === 'image_url') {
              imageCount++;
            }
          }
        }
      }

      // Conservative estimation: ~1.8 chars/token + ~1500 tokens per image + safety margin
      const estimatedPromptTokens = Math.ceil(textChars / 1.8) + (imageCount * 1500) + 128;
      const maxAllowedTokens = Math.max(512, MAX_CONTEXT_LEN - estimatedPromptTokens);
      const safeMaxTokens = Math.min(parseInt(max_tokens, 10) || 16384, maxAllowedTokens);

      const llmPayload = {
        model,
        messages,
        temperature,
        max_tokens: safeMaxTokens,
        stream: true,
        top_p,
        chat_template_kwargs: {
          enable_thinking: Boolean(enable_thinking !== false),
        },
      };

      const upstreamResponse = await fetch(`${LLM_BASE_URL}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(llmPayload),
        signal: abortController.signal,
      });

      if (!upstreamResponse.ok) {
        const errorText = await upstreamResponse.text();
        res.write(`data: ${JSON.stringify({ error: `LLM Error (${upstreamResponse.status}): ${errorText}` })}\n\n`);
        res.write('data: [DONE]\n\n');
        res.end();
        return;
      }

      // Stream the response body from vLLM directly to the client
      const reader = upstreamResponse.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const textChunk = decoder.decode(value, { stream: true });
        res.write(textChunk);
      }

      res.end();
    } catch (err) {
      if (err.name === 'AbortError') {
        // Client aborted
      } else {
        console.error('[Error] /api/chat upstream stream error:', err.message);
        if (!res.writableEnded) {
          res.write(`data: ${JSON.stringify({ error: `Streaming error: ${err.message}` })}\n\n`);
          res.write('data: [DONE]\n\n');
          res.end();
        }
      }
    }
    return;
  }

  // 4. Static files
  serveStaticFile(req, res, pathname);
});

const net = require('net');

function findAvailablePort(startPort) {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        resolve(findAvailablePort(startPort + 1));
      } else {
        reject(err);
      }
    });
    srv.listen(startPort, () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
  });
}

findAvailablePort(PORT).then((freePort) => {
  server.listen(freePort, () => {
    console.log(`\n======================================================`);
    console.log(`  LLM Streaming Web Application Started!`);
    console.log(`  Web UI:         http://localhost:${freePort}`);
    console.log(`  LLM Endpoint:   ${LLM_BASE_URL}`);
    console.log(`======================================================\n`);
  });
}).catch((err) => {
  console.error('[Fatal Error] Failed to find available port:', err);
  process.exit(1);
});


