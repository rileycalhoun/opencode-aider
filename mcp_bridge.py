#!/usr/bin/env python3
"""mcp_bridge: read-only tools for ACP sessions (stdio MCP server).

Declared in session/new so the model calls reads NATIVELY (structured args,
server-side execution) instead of narrating fenced blocks. WRITES stay with
the client (fenced protocol + approvals) — this bridge can never mutate.

Roots: /home/opencode/projects, /tmp/agent-shared. Everything else refused.
Outputs capped (length + matches) with truncation markers.
"""
import fnmatch
import json
import os
import re
import sys

ROOTS = ["/home/opencode/projects", "/tmp/agent-shared"]
MAX_CHARS = 20000
MAX_MATCHES = 50


def _resolve(path):
    # Agents address the shared tree as /projects/... (container mount);
    # on the host that is /home/opencode/projects/...
    if path == "/projects" or path.startswith("/projects/"):
        path = "/home/opencode/projects" + path[len("/projects"):]
    real = os.path.realpath(path)
    for root in ROOTS:
        if real == root or real.startswith(root + os.sep):
            return real
    raise ValueError("outside allowed roots: %s" % path)


TOOLS = [
    {"name": "file_read",
     "description": "Read a text file under the project roots. Prefer offset/limit for large files.",
     "inputSchema": {"type": "object",
                     "properties": {"path": {"type": "string"},
                                    "offset": {"type": "integer"},
                                    "limit": {"type": "integer"}},
                     "required": ["path"]}},
    {"name": "glob",
     "description": "List files matching a glob pattern under a root dir.",
     "inputSchema": {"type": "object",
                     "properties": {"pattern": {"type": "string"},
                                    "root": {"type": "string"}},
                     "required": ["pattern"]}},
    {"name": "grep",
     "description": "Search file contents for a regex. Returns file:line hits.",
     "inputSchema": {"type": "object",
                     "properties": {"pattern": {"type": "string"},
                                    "root": {"type": "string"},
                                    "include": {"type": "string"}},
                     "required": ["pattern"]}},
]


def t_file_read(a):
    p = _resolve(a["path"])
    off = int(a.get("offset") or 0)
    lim = int(a.get("limit") or 200)
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    sel = lines[off:off + lim]
    out = "".join(sel)
    if off + lim < len(lines):
        out += "\n[... %d more lines, use offset=%d]" % (len(lines) - off - lim, off + lim)
    return out[:MAX_CHARS]


def t_glob(a):
    root = _resolve(a.get("root") or ROOTS[0])
    pat = a["pattern"]
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        if ".git" in dirpath.split(os.sep) or "node_modules" in dirpath.split(os.sep):
            continue
        for fn in files:
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(fn, pat):
                hits.append(rel)
                if len(hits) >= MAX_MATCHES:
                    return "\n".join(hits) + "\n[... capped at %d]" % MAX_MATCHES
    return "\n".join(hits) if hits else "(no matches)"


def t_grep(a):
    root = _resolve(a.get("root") or ROOTS[0])
    rx = re.compile(a["pattern"])
    inc = a.get("include") or "*"
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        parts = dirpath.split(os.sep)
        if ".git" in parts or "node_modules" in parts or "target" in parts:
            continue
        for fn in files:
            if not fnmatch.fnmatch(fn, inc):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            hits.append("%s:%d:%s" % (
                                os.path.relpath(fp, root), i, line.strip()[:200]))
                            if len(hits) >= MAX_MATCHES:
                                return "\n".join(hits) + "\n[... capped]"
            except (OSError, UnicodeError):
                continue
    return "\n".join(hits) if hits else "(no matches)"


HANDLERS = {"file_read": t_file_read, "glob": t_glob, "grep": t_grep}


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        mid = msg.get("id")
        method = msg.get("method", "")
        try:
            if method == "initialize":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "shimreads", "version": "1.0"}}})
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
            elif method == "tools/call":
                params = msg.get("params") or {}
                name = params.get("name", "")
                fn = HANDLERS.get(name)
                if fn is None:
                    raise ValueError("unknown tool: " + name)
                out = fn(params.get("arguments") or {})
                send({"jsonrpc": "2.0", "id": mid, "result": {"content": [
                    {"type": "text", "text": out[:MAX_CHARS]}]}})
            elif mid is not None:
                send({"jsonrpc": "2.0", "id": mid, "result": {}})
        except Exception as e:
            if mid is not None:
                send({"jsonrpc": "2.0", "id": mid,
                      "error": {"code": -32603, "message": str(e)[:300]}})


main()
