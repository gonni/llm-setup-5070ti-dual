#!/usr/bin/env python3
"""
LLM Streaming Web Application Server (Python Standard Library)
Connects to vLLM server at http://gpu2:8000/v1
Zero external dependencies (Pure Python 3 stdlib)
"""

import http.server
import json
import os
import sys
import urllib.request
import urllib.error
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = int(os.environ.get("PORT", 3000))
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://gpu2:8000/v1").rstrip("/")
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


class StreamingHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean terminal logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        pathname = parsed.path

        # 1. GET /api/models
        if pathname == "/api/models":
            try:
                req = urllib.request.Request(f"{LLM_BASE_URL}/models", headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    res_body = {
                        "success": True,
                        "llmBaseUrl": LLM_BASE_URL,
                        "models": data.get("data", []),
                        "raw": data,
                    }
                    self.wfile.write(json.dumps(res_body).encode("utf-8"))
            except Exception as e:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                res_body = {
                    "success": False,
                    "error": f"Failed to connect to LLM server: {str(e)}",
                    "llmBaseUrl": LLM_BASE_URL,
                    "models": [],
                }
                self.wfile.write(json.dumps(res_body).encode("utf-8"))
            return

        # 2. GET /api/health
        if pathname == "/api/health":
            llm_connected = False
            models = []
            try:
                req = urllib.request.Request(f"{LLM_BASE_URL}/models")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        llm_connected = True
                        models = [m.get("id") for m in data.get("data", [])]
            except Exception:
                llm_connected = False

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(
                json.dumps({
                    "status": "ok",
                    "llm": {
                        "baseUrl": LLM_BASE_URL,
                        "connected": llm_connected,
                        "models": models,
                    },
                }).encode("utf-8")
            )
            return

        # 3. Static Files
        self.serve_static(pathname)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        pathname = parsed.path

        # POST /api/chat
        if pathname == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                body = json.loads(body_bytes.decode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
                return

            # Ensure stream is True & handle enable_thinking
            body["stream"] = True
            if "enable_thinking" in body:
                enable_thinking = bool(body["enable_thinking"])
                body.setdefault("chat_template_kwargs", {})["enable_thinking"] = enable_thinking

            post_data = json.dumps(body).encode("utf-8")

            try:
                req = urllib.request.Request(
                    f"{LLM_BASE_URL}/chat/completions",
                    data=post_data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

                with urllib.request.urlopen(req) as resp:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache, no-transform")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()

                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        self.wfile.write(line)
                        self.wfile.flush()
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(f"data: {json.dumps({'error': f'LLM Error ({e.code}): {err_body}'})}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as e:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(f"data: {json.dumps({'error': f'Stream error: {str(e)}'})}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            return

        self.send_response(404)
        self.end_headers()

    def serve_static(self, pathname):
        rel_path = "index.html" if pathname == "/" else pathname.lstrip("/")
        file_path = os.path.abspath(os.path.join(PUBLIC_DIR, rel_path))

        if not file_path.startswith(PUBLIC_DIR) or not os.path.exists(file_path) or os.path.isdir(file_path):
            file_path = os.path.join(PUBLIC_DIR, "index.html")

        if not os.path.exists(file_path):
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"404 Not Found")
            return

        _, ext = os.path.splitext(file_path)
        content_type = MIME_TYPES.get(ext.lower(), "application/octet-stream")

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        with open(file_path, "rb") as f:
            self.wfile.write(f.read())


def run():
    port = PORT
    while port < PORT + 100:
        try:
            server_address = ("", port)
            httpd = HTTPServer(server_address, StreamingHTTPRequestHandler)
            print(f"\n======================================================")
            print(f"  LLM Streaming Web Application Started! (Python)")
            print(f"  Web UI:         http://localhost:{port}")
            print(f"  LLM Endpoint:   {LLM_BASE_URL}")
            print(f"======================================================\n")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer shutting down.")
                httpd.server_close()
            break
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"[Warning] Port {port} in use, trying {port + 1}...")
                port += 1
            else:
                raise e


if __name__ == "__main__":
    run()
