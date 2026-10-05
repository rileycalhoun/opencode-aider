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

Hardening (each item traces to a verified audit finding, 2026-10-04):
- Agent stderr is captured to a bounded ring and dumped on turn failure.
  It used to go to DEVNULL, which blinded every diagnosis.
- Dropped frames (sessionId miss, malformed lines, correlation miss) are
  logged with counters instead of vanishing silently.
- Leaked per-turn waiters are cancelled; GeneratorExit closes the session
  promptly instead of orphaning it.
- Session queues are woken on process death so turns fail fast instead of
  hanging to the idle timeout.
- Idle timeout message matches the retry classifier, so one transient stall
  retries instead of failing once and permanently.
- usage_update frames are logged at 50K-token steps: a progress heartbeat
  that distinguishes reasoning-in-progress from dead air.
- The shared process recycles after MAX_TURNS_PER_PROC turns (only when no
  turn is active), bounding leaked server-side session state.
"""
import asyncio
import collections
import json
import os

OPENCODE_BIN = os.environ.get("OPENCODE_CLI", "/usr/local/bin/opencode")
ACP_HOME = "/home/opencode/.cache/acp-home"
ACP_AUTH_SRC = "/home/opencode/.local/share/opencode/auth.json"
IDLE_TIMEOUT = 300  # base: abort a turn silent this long (matches retry classifier)
IDLE_TIMEOUT_FIRST = 600  # pre-first-frame: giant prompts are slow to start
IDLE_TIMEOUT_TOOL = 900  # tool inflight: ACP sends nothing during execution
TURN_TIMEOUT = 1200  # hard ceiling per turn (20 min)
MAX_TURNS_PER_PROC = 25  # recycle the shared proc past this many turns
STDERR_RING = 200  # agent stderr lines kept for failure dumps
USAGE_LOG_STEP = 50000  # log a heartbeat every this many tokens


class AcpError(Exception):
    pass


def _log(msg):
    print("[acp] %s" % msg, flush=True)


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
        self._spawn_lock = None
        self._pending = {}
        self._sessions = {}
        self._id_seq = 0
        self._turns = 0
        self._active = 0
        self._stderr_ring = collections.deque(maxlen=STDERR_RING)
        self._drop_session = 0
        self._drop_parse = 0
        self._drop_correlate = 0

    def _ensure_loop_primitives(self):
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        if self._spawn_lock is None:
            self._spawn_lock = asyncio.Lock()

    async def ensure_alive(self):
        self._ensure_loop_primitives()
        async with self._spawn_lock:
            if self.proc is not None and self.proc.returncode is None:
                if self._turns >= MAX_TURNS_PER_PROC and self._active == 0:
                    _log("recycling ACP proc after %d turns" % self._turns)
                    await self._terminate_proc()
                else:
                    return
            ensure_acp_home()
            env = dict(os.environ)
            env["HOME"] = ACP_HOME
            self.proc = await asyncio.create_subprocess_exec(
                OPENCODE_BIN, "acp",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=ACP_HOME, env=env,
                limit=4 * 1024 * 1024,
            )
            self._pending = {}
            self._sessions = {}
            self._turns = 0
            self._stderr_ring.clear()
            _log("ACP proc spawned pid=%s" % self.proc.pid)
            asyncio.ensure_future(self._reader_loop())
            asyncio.ensure_future(self._stderr_loop())

    async def _terminate_proc(self):
        proc, self.proc = self.proc, None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _fail_all(self, err):
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(err)
        self._pending = {}
        dead = {"__acp_dead__": True}
        for q in list(self._sessions.values()):
            try:
                q.put_nowait(dead)
            except Exception:
                pass

    async def _stderr_loop(self):
        try:
            while True:
                line = await self.proc.stderr.readline()
                if not line:
                    break
                self._stderr_ring.append(
                    line.decode("utf-8", "replace").rstrip()[:300])
        except Exception:
            pass

    def dump_stderr(self, tail=15):
        lines = list(self._stderr_ring)[-tail:]
        return lines if lines else ["(agent stderr empty)"]

    async def _reader_loop(self):
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line.decode("utf-8", "replace"))
                except Exception:
                    self._drop_parse += 1
                    _log("dropped malformed stdout line (total %d)"
                         % self._drop_parse)
                    continue
                await self._dispatch(obj)
        except Exception as e:
            _log("reader loop ended: %s" % e)
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
            else:
                self._drop_correlate += 1
                _log("dropped response with no waiter "
                     "(total %d): %s" % (self._drop_correlate,
                                        json.dumps(obj)[:160]))
            return
        params = obj.get("params") or {}
        if obj.get("method") == "session/update":
            q = self._sessions.get(params.get("sessionId"))
            if q is not None:
                await q.put(params.get("update") or {})
            else:
                self._drop_session += 1
                _log("dropped update for unknown session "
                     "(total %d)" % self._drop_session)
            return
        if "id" in obj and obj.get("method"):
            _log("denying unexpected inbound %s" % obj.get("method"))
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
    acp._active += 1
    acp._turns += 1
    last_usage_logged = 0
    turn_start = asyncio.get_event_loop().time()
    frames_seen = 0
    last_frame_kind = "none"
    tool_active = False  # set on tool_call, cleared on text (ACP is silent mid-tool)
    try:
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
                    if tool_active:
                        idle = IDLE_TIMEOUT_TOOL
                    elif frames_seen == 0:
                        idle = IDLE_TIMEOUT_FIRST
                    else:
                        idle = IDLE_TIMEOUT
                    done, _ = await asyncio.wait(
                        [pending_get, prompt_task],
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=idle)
                    if not done:
                        raise AcpError(
                            "acp turn timeout: idle too long without frames "
                            "(idle %.0fs, elapsed %.0fs, frames %d, last %s, tool_active %s)"
                            % (idle, asyncio.get_event_loop().time() - turn_start,
                               frames_seen, last_frame_kind, tool_active))
                    if prompt_task in done:
                        try:
                            resp = prompt_task.result()
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            raise AcpError("prompt failed: %s" % e)
                        if isinstance(resp, dict) and resp.get("error"):
                            raise AcpError("prompt refused: %s"
                                           % json.dumps(resp["error"])[:200])
                        return
                    update = pending_get.result()
                    pending_get = asyncio.ensure_future(updates.get())
                    frames_seen += 1
                    if isinstance(update, dict) and update.get("__acp_dead__"):
                        raise AcpError("acp process died mid-turn (aborted)")
                    kind = update.get("sessionUpdate")
                    last_frame_kind = kind
                    if kind == "agent_message_chunk":
                        content = update.get("content") or {}
                        if (content.get("type") == "text"
                                and content.get("text")):
                            tool_active = False  # model spoke: prior tool done
                            yield ("text", content["text"])
                    elif kind in ("tool_call", "tool_call_update"):
                        tool_active = True
                        yield ("alien_tool",
                               update.get("title") or "unknown")
                    elif kind == "usage_update":
                        try:
                            used = int(update.get("used") or 0)
                        except Exception:
                            used = 0
                        if used - last_usage_logged >= USAGE_LOG_STEP:
                            last_usage_logged = used
                            _log("turn progress: %dk tokens so far" % (used // 1000))
                    # ignore: available_commands_update, etc.
            finally:
                if not pending_get.done():
                    pending_get.cancel()
                if not prompt_task.done():
                    # Abnormal exit (disconnect/timeout/cancel): tell the
                    # server to STOP WORK, not just stop listening. Cancelling
                    # our wait-future alone leaves the turn running blind.
                    prompt_task.cancel()
                    try:
                        await acp.request("session/cancel",
                                          {"sessionId": sid}, timeout=10)
                    except Exception:
                        pass
                try:
                    await acp.request("session/close", {"sessionId": sid},
                                      timeout=10)
                except Exception:
                    pass
        finally:
            acp.forget_session(sid)
    finally:
        acp._active -= 1
