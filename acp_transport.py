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
ACP_HOME = os.environ.get("ACP_HOME_DIR",
                           "/home/opencode/.cache/acp-home")
ACP_AUTH_SRC = "/home/opencode/.local/share/opencode/auth.json"
IDLE_TIMEOUT = 300  # base: abort a turn silent this long (matches retry classifier)
IDLE_TIMEOUT_FIRST = 600  # pre-first-frame: giant prompts are slow to start
IDLE_TIMEOUT_TOOL = 900  # tool inflight: ACP sends nothing during execution
TURN_TIMEOUT = 1200  # hard ceiling per turn (20 min)
MAX_TURNS_PER_PROC = 25  # recycle the shared proc past this many turns
STDERR_RING = 200  # agent stderr lines kept for failure dumps
USAGE_LOG_STEP = 50000  # log a heartbeat every this many tokens
MCP_BRIDGE_NAME = "shimreads"  # read-only tools; server executes, no roundtrip
MCP_BRIDGE_CMD = "/home/opencode/opencode-compat-shim/.venv/bin/python"
MCP_BRIDGE_SCRIPT = "/home/opencode/opencode-compat-shim-fork/mcp_bridge.py"


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
            clear_affinity()  # sids die with the proc
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
        proc = self.proc
        try:
            while True:
                line = await proc.stderr.readline()
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
        proc = self.proc
        try:
            while True:
                line = await proc.stdout.readline()
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
            if self.proc is proc:
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

# Session affinity: key -> {"sid", "sent", "last", "busy", "model"}.
# Reuses one ACP session across turns of the same conversation so follow-up
# prompts send ONLY new messages instead of re-sending 100k+ chars.
# Safety: busy sessions are never shared (concurrent turn goes fresh);
# idle >30min evicted; any turn failure drops the mapping (fresh next time);
# proc recycle clears the map (sids die with the proc).
_AFFINITY = {}
_AFFINITY_IDLE = 1800
_AFFINITY_MAX = 32
_AFF_ORPHANS = []  # sids to close+forget (evicted while idle/busy)
_AFF_USE = {}  # sid -> max turn tokens seen (server-side accumulation)
AFF_USE_CAP = 250000  # fresh session past this many accumulated tokens


def affinity_key(key):
    return key


def peek_sent(key):
    e = _AFFINITY.get(key)
    if not e:
        return None
    import time as _t
    if _t.monotonic() - e["last"] > _AFFINITY_IDLE:
        _AFFINITY.pop(key, None)
        _AFF_ORPHANS.append(e["sid"])  # idle-evicted: must be closed
        return None
    import time as _t
    if _t.monotonic() - e["last"] > _AFFINITY_IDLE:
        return None
    if e["busy"]:
        return None
    return e["sent"]


def mark_sent(key, sid, count, model, pfx=None):
    import time as _t
    prev = _AFFINITY.get(key)
    if prev is not None and prev.get("sid") != sid:
        _AFF_ORPHANS.append(prev["sid"])  # displaced mapping: close it
    while len(_AFFINITY) >= _AFFINITY_MAX and key not in _AFFINITY:
        oldest = min(_AFFINITY, key=lambda k: _AFFINITY[k]["last"])
        old = _AFFINITY.pop(oldest, None)
        if old is not None:
            _AFF_ORPHANS.append(old["sid"])
    _AFFINITY[key] = {"sid": sid, "sent": count, "last": _t.monotonic(),
                      "busy": False, "model": model, "pfx": pfx}


def drop_affinity(key):
    e = _AFFINITY.pop(key, None)
    if e is not None:
        _AFF_ORPHANS.append(e["sid"])


async def reap_orphans(acp):
    while _AFF_ORPHANS:
        sid = _AFF_ORPHANS.pop(0)
        try:
            await acp.request("session/close", {"sessionId": sid},
                              timeout=10)
        except Exception:
            pass
        acp.forget_session(sid)


def clear_affinity():
    _AFFINITY.clear()


_ACP = AcpProcess()


async def acp_turn_events(model, prompt_text, workdir=ACP_HOME,
                          progress_cb=None, affinity=None,
                          turn_timeout=TURN_TIMEOUT):
    """Yield ('text', delta) | ('alien_tool', name) for one model turn.
    progress_cb, when given, fires on usage milestones so callers can
    heartbeat the client instead of sitting silent."""
    acp = _ACP
    await acp.ensure_alive()
    acp._active += 1
    acp._turns += 1
    last_usage_logged = 0
    # affinity: (key, delta_text, full_count). Fresh session when key unknown,
    # busy, idle, or model-mismatched server state unclear -> fall back safe.
    aff_key = aff_delta = aff_count = aff_pfx = None
    aff_entry = None
    bridged_ids = set()
    await reap_orphans(acp)
    if affinity:
        aff_key, aff_delta, aff_count = affinity[0], affinity[1], affinity[2]
        aff_pfx = affinity[3] if len(affinity) > 3 else None
        aff_entry = _AFFINITY.get(aff_key)
        if aff_entry is not None:
            import time as _t
            if (aff_entry["busy"]
                    or _t.monotonic() - aff_entry["last"] > _AFFINITY_IDLE
                    or aff_entry["sid"] not in acp._sessions
                    or (aff_pfx is not None
                        and aff_entry.get("pfx") is not None
                        and aff_entry["pfx"] != aff_pfx)
                    or _AFF_USE.get(aff_entry["sid"], 0) > AFF_USE_CAP):
                # Busy, idle, gone, foreign conversation, or overgrown:
                # never reuse. Orphan the sid for reaping.
                _AFF_ORPHANS.append(aff_entry["sid"])
                drop_affinity(aff_key)
                aff_entry = None
        if aff_entry is not None:
            aff_entry["busy"] = True
    turn_start = asyncio.get_event_loop().time()
    frames_seen = 0
    last_frame_kind = "none"
    tool_active = False  # set on tool_call, cleared on text (ACP is silent mid-tool)
    shared = aff_entry is not None
    sid = aff_entry["sid"] if shared else None
    ok = True
    abnormal = True  # safe default: early failure forgets session state
    try:
        if not shared:
            # No MCP servers, ever: only fenced client tools exist here.
            # (MCP_BRIDGE_* kept for debugging; do not re-enable without
            # need. Server-side tools bypass client approvals.)
            resp = await acp.request("session/new",
                                     {"cwd": workdir, "mcpServers": []})
            sid = (resp.get("result") or {}).get("sessionId")
            if not sid:
                raise AcpError("session/new gave no sessionId: %s"
                               % json.dumps(resp)[:200])
        else:
            _log("affinity hit %s: delta %dch (saved full resend)"
                 % (aff_key, len(aff_delta or "")))
        try:
            # Register the queue FIRST: frames arriving between session/new
            # and set_config had nowhere to land (the 2/turn unknown-session
            # drops in the log).
            updates = acp.session_queue(sid)
            if not shared or aff_entry.get("model") != model:
                await acp.request("session/set_config_option",
                                  {"sessionId": sid, "configId": "model",
                                   "value": "opencode/" + model})
            send_text = prompt_text
            if shared and aff_delta:
                send_text = aff_delta  # follow-up: only what's new
            # Empty delta: re-send full text into the SAME session (re-sync),
            # keep the mapping.
            prompt_task = asyncio.ensure_future(acp.request(
                "session/prompt",
                {"sessionId": sid, "prompt": [
                    {"type": "text", "text": send_text}]},
                timeout=turn_timeout))
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
                            abnormal = True
                            raise AcpError("prompt failed: %s" % e)
                        if isinstance(resp, dict) and resp.get("error"):
                            abnormal = True
                            raise AcpError("prompt refused: %s"
                                           % json.dumps(resp["error"])[:200])
                        # Linger-drain: trailing frames already queued (final
                        # text chunks) must not be dropped with the return.
                        try:
                            while True:
                                extra = await asyncio.wait_for(
                                    updates.get(), timeout=1.5)
                                frames_seen += 1
                                if isinstance(extra, dict):
                                    ek = extra.get("sessionUpdate")
                                    if ek == "agent_message_chunk":
                                        ec = extra.get("content") or {}
                                        if (ec.get("type") == "text"
                                                and ec.get("text")):
                                            yield ("text", ec["text"])
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                        abnormal = False
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
                        title = update.get("title") or ""
                        call_id = update.get("toolCallId") or ""
                        if call_id and call_id in bridged_ids:
                            bridged = True
                        else:
                            bridged = (title == MCP_BRIDGE_NAME
                                       or title.startswith(MCP_BRIDGE_NAME + "_"))
                            if bridged and call_id:
                                bridged_ids.add(call_id)
                            elif not title and not bridged:
                                _log("untitled tool frame: %s"
                                     % json.dumps(update)[:200])
                        if bridged:
                            # Server-side read (file_read/glob/grep): executed
                            # synchronously, result already in-frame. Forward
                            # as progress so the client sees movement.
                            if progress_cb is not None:
                                try:
                                    res = progress_cb("read %s" % title)
                                    if res is not None:
                                        await res
                                except (ConnectionResetError,
                                        asyncio.CancelledError):
                                    raise
                                except Exception:
                                    pass
                        else:
                            raise AcpError(
                                "alien_tool_frame:%s: server executed a native "
                                "tool; results never reach the client"
                                % (title or "unknown"))
                    elif kind == "usage_update":
                        try:
                            used = int(update.get("used") or 0)
                        except Exception:
                            used = 0
                        if used:
                            _AFF_USE[sid] = max(_AFF_USE.get(sid, 0), used)
                        if used - last_usage_logged >= USAGE_LOG_STEP:
                            last_usage_logged = used
                            _log("turn progress: %dk tokens so far" % (used // 1000))
                            if progress_cb is not None:
                                try:
                                    res = progress_cb(
                                        "%dk tokens so far" % (used // 1000))
                                    if res is not None:
                                        await res
                                except (ConnectionResetError,
                                        asyncio.CancelledError):
                                    raise
                                except Exception:
                                    pass
                    # ignore: available_commands_update, etc.
            finally:
                if not pending_get.done():
                    pending_get.cancel()
                if abnormal:
                    # Abnormal exit: tell the server to STOP WORK.
                    prompt_task.cancel()
                    try:
                        await acp.request("session/cancel",
                                          {"sessionId": sid}, timeout=10)
                    except Exception:
                        pass
                if abnormal:
                    # Failed turn: poison would linger server-side. Close and
                    # forget so the next turn starts clean -- but only if the
                    # mapping is still ours (a concurrent fallback may have
                    # published a replacement).
                    if shared and (_AFFINITY.get(aff_key) or {}).get(
                            "sid") == sid:
                        drop_affinity(aff_key)
                    try:
                        await acp.request("session/close", {"sessionId": sid},
                                          timeout=10)
                    except Exception:
                        pass
                elif aff_key is not None:
                    # Healthy turn with a conversation key: keep the session,
                    # record how many messages the server has seen.
                    mark_sent(aff_key, sid, aff_count, model, aff_pfx)
                else:
                    # No key (direct collect callers): one-shot, close it.
                    try:
                        await acp.request("session/close", {"sessionId": sid},
                                          timeout=10)
                    except Exception:
                        pass
        finally:
            if aff_entry is not None:
                aff_entry["busy"] = False
            if abnormal or aff_key is None:
                acp.forget_session(sid)
    finally:
        acp._active -= 1
