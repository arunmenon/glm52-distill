#!/usr/bin/env python3
"""mock_vast.py — local stand-in for the Vast API, for sweep certification.

Serves the three endpoints the conductor uses, with state controlled by a
JSON file so smoke scenarios can inject failures without touching real
billing:

  GET /users/current/     -> {"credit": <state.credit>}
  GET /instances/<id>/    -> {"instances": {actual_status, ssh_host,
                              ssh_port, public_ipaddr, ports, dph_total}}
  PUT /instances/<id>/    -> records {"state": ...} into state.stops

State file (default /tmp/mock_vast_state.json) fields:
  credit: float          — what /users/current returns
  dph_total: float       — instance hourly price
  ssh_host, ssh_port     — where the "box" really is (the real GPU box)
  fail_mode: null|"http500"|"error_body"|"no_credit"
  stops: []              — appended on every PUT

Usage:
  python3 mock_vast.py --port 8642 --state /tmp/mock_vast_state.json
  VAST_API_BASE=http://127.0.0.1:8642 VAST_API_KEY=mock python3 08_sweep.py ...
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

STATE = Path("/tmp/mock_vast_state.json")


def state() -> dict:
    return json.loads(STATE.read_text())


def save(d: dict):
    STATE.write_text(json.dumps(d, indent=1))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        s = state()
        if s.get("fail_mode") == "http500":
            self._json(500, {"detail": "injected failure"})
            return
        if s.get("fail_mode") == "error_body":
            self._json(200, {"error": "injected 200-with-error"})
            return
        if self.path.startswith("/users/current"):
            if s.get("fail_mode") == "no_credit":
                self._json(200, {})
                return
            self._json(200, {"credit": s["credit"]})
        elif self.path.startswith("/instances/"):
            self._json(200, {"instances": {
                "actual_status": s.get("actual_status", "running"),
                "ssh_host": s["ssh_host"], "ssh_port": s["ssh_port"],
                "public_ipaddr": s["ssh_host"],
                "ports": {"22/tcp": [{"HostPort": s["ssh_port"]}]},
                "dph_total": s["dph_total"]}})
        else:
            self._json(404, {"error": "unknown path"})

    def do_PUT(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        s = state()
        s.setdefault("stops", []).append(body)
        if body.get("state") == "stopped":
            s["actual_status"] = "stopped"
        save(s)
        self._json(200, {"success": True})


def main():
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--state", default=str(STATE))
    args = ap.parse_args()
    STATE = Path(args.state)
    if not STATE.exists():
        save({"credit": 50.0, "dph_total": 0.80,
              "ssh_host": "127.0.0.1", "ssh_port": 22,
              "fail_mode": None, "stops": []})
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
