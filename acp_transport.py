"""ACP transport: turns via a long-lived `opencode acp` process.

Replaces per-turn `opencode run` subprocesses, which buffer the whole
response and emit it as one JSON event at turn end. ACP streams
`agent_message_chunk` deltas live, so text flows during generation.

Event contract (identical to the old run_opencode pump, so collect() is
untouched downstream): yields ('text', delta) | ('alien_tool', name).

Mapping notes, all observed live against opencode 1.18.33:
- agent_message_chunk content {type: text, text} -> ("text", delta).
  Deltas arrive already sliced; no cumulative diffing needed.
- tool_call / tool_call_update frames are ALWAYS alien here: we declare no
  MCP servers, and fenced client tools travel as text, so any ACP-native
  tool call is necessarily an opencode built-in. Title only; results stay
  suppressed exactly like the CLI path.
- Inbound requests from the agent (none observed across all probes) get a
  logged MethodNotFound denial: fail-visible beats hang-forever.
- HOME isolation: the ACP process runs under an isolated HOME holding only
  auth.json. Ambient host config (MCP servers, plugins) otherwise leaks
  undeclared tools into every session.
"""
import asyncio
import json
import os

OPENCODE_BIN = os.environ.get("OPENCODE_CLI", "/usr/local/bin/opencode")
ACP_HOME = "/home/opencode/.cache/acp-home"
ACP_AUTH_SRC = "/home/opencode/.local/share/opencode/auth.json"
IDLE_TIMEOUT = 300  # mirrors old PER_READ_TIMEOUT: abort a silent turn
TURN_TIMEOUT = 1200  # hard ceiling per turn (20 min)


class AcpError(Exception):
    pass


def ensure_acp_home():
    """Isolated HOME holding only auth.json. Same user, same 0600 perms."""
    dst_dir = os.path.join(ACP_HOME, ".local", "share", "opencode")
    os.makedirs(dst_dir, mode=0o700, exist_ok=True)
    dst = os.path.join(dst_dir, "auth.json")
    try:
        with open(ACP_AUTH_SRC, "rb") as f:
            wanted = f.read()
    except OSError as e:
        raise AcpError("cannot read CLI auth for ACP home: %s" % e)
    try:
        with open(dst, "rb") as f:
            if f.read() == wanted:
                return
    except OSError:
        pass
    with open(dst, "wb") as f:
        f.write(wanted)
    os.chmod(dst, 0o600)


class AcpProcess:
    """One long-lived `opencode acp` proc; sessions multiplexed on it."""

    def __init__(self):
        self.proc = None
        self._write_lock = None
        self._pending = {}
        self._sessions = {}
        self._id_seq = 0

    def _ensure_loop_primitives(self):
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()

    async def ensure_alive(self):
        self._ensure_loop_primitives()
        if self.proc is not None and self.proc.returncode is None:
            return
        ensure_acp_home()
        env = dict(os.environ)
        env["HOME"] = ACP_HOME
        self.proc = await asyncio.create_subprocess_exec(
            OPENCODE_BIN, "acp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=ACP_HOME, env=env,
            limit=4 * 1024 * 1024,
        )
        self._pending = {}
        self._sessions = {}
        asyncio.ensure_future(self._reader_loop())

    def _fail_all(self, err):
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(err)
        self._pending = {}

    async def _reader_loop(self):
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line.decode("utf-8", "replace"))
                except Exception:
                    continue
                await self._dispatch(obj)
        except Exception:
            pass
        finally:
            self._fail_all(AcpError("acp process died"))
            self.proc = None

    async def _dispatch(self, obj):
        if not isinstance(obj, dict):
            return
        if "id" in obj and ("result" in obj or "error" in obj):
            fut = self._pending.pop(obj["id"], None)
            if fut is not None and not fut.done():
                fut.set_result(obj)
            return
        params = obj.get("params") or {}
        if obj.get("method") == "session/update":
            q = self._sessions.get(params.get("sessionId"))
            if q is not None:
                await q.put(params.get("update") or {})
            return
        if "id" in obj and obj.get("method"):
            await self._send_raw({
                "jsonrpc": "2.0", "id": obj["id"],
                "error": {"code": -32601,
                          "message": "Method not found: %s" % obj.get("method")},
            })

    async def _send_raw(self, obj):
        async with self._write_lock:
            self.proc.stdin.write((json.dumps(obj) + "\n").encode())
            await self.proc.stdin.drain()

    async def request(self, method, params, timeout=60):
        await self.ensure_alive()
        self._id_seq += 1
        mid = self._id_seq
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._pending[mid] = fut
        await self._send_raw({"jsonrpc": "2.0", "id": mid,
                              "method": method, "params": params})
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(mid, None)

    def session_queue(self, sid):
        q = asyncio.Queue()
        self._sessions[sid] = q
        return q

    def forget_session(self, sid):
        self._sessions.pop(sid, None)


_ACP = AcpProcess()


async def acp_turn_events(model, prompt_text, workdir=ACP_HOME):
    """Yield ('text', delta) | ('alien_tool', name) for one model turn."""
    acp = _ACP
    await acp.ensure_alive()

    resp = await acp.request("session/new",
                             {"cwd": workdir, "mcpServers": []})
    sid = (resp.get("result") or {}).get("sessionId")
    if not sid:
        raise AcpError("session/new gave no sessionId: %s"
                       % json.dumps(resp)[:200])
    try:
        await acp.request("session/set_config_option",
                          {"sessionId": sid, "configId": "model",
                           "value": "opencode/" + model})
        updates = acp.session_queue(sid)
        prompt_task = asyncio.ensure_future(acp.request(
            "session/prompt",
            {"sessionId": sid, "prompt": [
                {"type": "text", "text": prompt_text}]},
            timeout=TURN_TIMEOUT))
        pending_get = asyncio.ensure_future(updates.get())
        try:
            while True:
                done, _ = await asyncio.wait(
                    [pending_get, prompt_task],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=IDLE_TIMEOUT)
                if not done:
                    raise AcpError("acp turn idle too long")
                if prompt_task in done:
                    try:
                        resp = prompt_task.result()
                    except Exception as e:
                        raise AcpError("prompt failed: %s" % e)
                    if isinstance(resp, dict) and resp.get("error"):
                        raise AcpError("prompt refused: %s"
                                       % json.dumps(resp["error"])[:200])
                    return
                update = pending_get.result()
                pending_get = asyncio.ensure_future(updates.get())
                kind = update.get("sessionUpdate")
                if kind == "agent_message_chunk":
                    content = update.get("content") or {}
                    if (content.get("type") == "text"
                            and content.get("text")):
                        yield ("text", content["text"])
                elif kind in ("tool_call", "tool_call_update"):
                    yield ("alien_tool",
                           update.get("title") or "unknown")
                # ignore: available_commands_update, usage_update, etc.
        finally:
            if not prompt_task.done():
                prompt_task.cancel()
            try:
                await acp.request("session/close", {"sessionId": sid},
                                  timeout=10)
            except Exception:
                pass
    finally:
        acp.forget_session(sid)
