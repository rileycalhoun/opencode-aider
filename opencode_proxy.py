#!/usr/bin/env python3
"""OpenAI-shaped SSE proxy that tunnels OpenCode free-tier models via its CLI.

OpenCode's free tier is locked behind desktop-app identity: bare HTTP gets 403.
The only authenticated transport is the signed CLI, which speaks a custom
per-call JSON protocol. This proxy translates OpenAI <-> that protocol.

Two hard problems this solves, both verified empirically (2026-09-28):

1. WinError 206 ("filename or extension is too long").
   The prompt used to ride in argv, and Windows caps a command line at 32,767
   characters. Any real conversation blows that instantly. The prompt now goes
   in over STDIN, which has no such limit. (`--file` does NOT work on its own:
   the CLI rejects it with "You must provide a message".)

2. Tool calling.
   The shim used to drop the OpenAI `tools` array entirely, so models behind it
   could not drive client tools and would hallucinate results. The full tool
   catalogue is now serialised into the prompt, the model emits a fenced
   tool-call block, and the shim parses it back into real OpenAI `tool_calls`
   frames. The client executes the tools; the shim never does.

   The CLI injects its OWN tools (bash/read/write) that do not exist in the client
   and cannot be disabled in this build (custom agent definitions in
   opencode.json / agent/*.md are not loaded by opencode-cli 2.0.11). Those
   `tool_use` events are therefore SUPPRESSED: never forwarded to the client, and
   the model is told up front it has no built-in tools. If it reaches for one
   anyway, the shim injects a corrective nudge instead of leaking an alien
   tool call.
"""

import asyncio, json, os, uuid, argparse, re, time
from acp_transport import AcpError, acp_turn_events
from aiohttp import web

OPENCODE_CLI = os.environ.get(
    "OPENCODE_CLI", "/usr/local/bin/opencode"
)
# Alien tool containment now comes from the ACP isolated HOME
# (see acp_transport.ensure_acp_home), not a scratch cwd.
# Transient CLI failures worth a retry: the step watchdog aborts (often after a
# permission prompt stalled), read timeouts, connection resets.
TRANSIENT_RE = re.compile(r"interrupt|abort|timeout|econn|reset|timed out", re.I)
NARRATE_RE = re.compile(r"tool protocol|fenced block|only.*block|emit.*block", re.I)
MAX_PROMPT_CHARS = 60000  # ceiling for flattened convo; head elided past this
TOOL_DESC_LIMIT = 200  # per-tool description chars kept in catalogue
DEFAULT_MODEL = "longcat-2.5-preview-free"

FREE_MODELS = [
    "longcat-2.5-preview-free",
    "space-bunny-free",
    "mimo-v2.6-flash-free",
    "ling-3.0-flash-fin-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
    "muse-spark-1.3-contributor-free",
]
DEAD_MODELS = {
    "mimo-v2.5-free": "Model retired by OpenCode (provider.no-route).",
    "deepseek-v4-flash-free": "Model retired by OpenCode (provider.no-route).",
    "jev-1.13-free": "System One model - not a chat model, unreachable via CLI.",
}
ALIASES = {
    "longcat": "longcat-2.5-preview-free", "longcat-2.5": "longcat-2.5-preview-free",
    "bunny": "space-bunny-free", "space-bunny": "space-bunny-free",
    "mimo": "mimo-v2.6-flash-free", "ling": "ling-3.0-flash-fin-free",
    "nemotron": "nemotron-3-ultra-free", "nemotron-3-ultra": "nemotron-3-ultra-free",
    "muse-spark": "muse-spark-1.3-contributor-free",
    "muse-spark-1.3": "muse-spark-1.3-contributor-free",
    "spark": "muse-spark-1.3-contributor-free",
}

ZEN_API_URL = "https://opencode.ai/zen/v1"
ZEN_KEY_FILE = "/home/opencode/.local/share/opencode/auth.json"

# Free models verified to serve over bare HTTPS with the account key (live
# probe, not docs). Everything else in FREE_MODELS needs the CLI transport.
DIRECT_MODELS = {
    "space-bunny-free",
}


def load_zen_key():
    """Account API key for direct calls. Same user, same 0600 file the CLI
    itself reads; never logged, never echoed. Returns None when unavailable,
    in which case the direct route reports disabled instead of failing oddly."""
    try:
        import json as _json
        with open(ZEN_KEY_FILE) as _f:
            return _json.load(_f)["opencode"]["key"]
    except Exception:
        return None


OPEN_FENCE = "<tool_call>"
CLOSE_FENCE = "</tool_call>"

PROTOCOL = """\
# TOOL PROTOCOL (read carefully)

You do NOT have any built-in tools. Anything you think you can do with a
built-in shell, file reader or editor does not exist here, and its output is
discarded. You have exactly the tools listed below, provided by an external
harness that executes them for you and returns the results.

To call one or more tools, output ONLY this fenced block, with no other text:

<tool_call>
{"name": "TOOL_NAME", "arguments": {"KEY": "VALUE"}}
</tool_call>

Rules:
- The block must be the entire response. No prose before or after it.
- `arguments` must be a JSON object matching that tool's `parameters` schema.
- To call several tools at once, put one JSON object per line inside the single
  block. They are executed in parallel.
- Use EXACT tool names as written below.
- When you already have everything you need, reply with plain prose and NO block.
- For READS use the native shimreads_file_read / shimreads_glob / shimreads_grep tools (fast, structured arguments). Project files live under /home/opencode/projects/ (agents call it /projects/).

Completed example (real call — copy this shape exactly, with a real tool name):

<tool_call>
{"name": "get_items", "arguments": {}}
</tool_call>
"""

PROTOCOL_SPARK = PROTOCOL + """\
Names like todowrite, task, bash, read, skill, websearch are NOT yours. They do not exist here. Emitting one fails the turn outright — use the fenced block above with a listed tool instead."""


def resolve_model(name):
    name = (name or DEFAULT_MODEL).strip()
    for p in ("opencode-go/", "opencode/"):
        if name.startswith(p):
            name = name[len(p):]
    return ALIASES.get(name.lower(), name)


# Undertaker lockdown: only free-tier models may pass. Anything else returns
# 404 without ever touching the CLI, so a caller can never spend paid Zen
# credit through this proxy.
def allow_model(requested):
    model = resolve_model(requested)
    if model in DEAD_MODELS:
        return False, model
    if model not in FREE_MODELS:
        return False, model
    return True, model


# ── tool catalogue ───────────────────────────────────────────────────────────

def render_tools(tools):
    """Serialise the OpenAI tools array into a compact prompt catalogue."""
    if not tools:
        return ""
    out = ["# AVAILABLE TOOLS"]
    for t in tools:
        fn = t.get("function") or {}
        out.append(f"## {fn.get('name')}")
        if fn.get("description"):
            desc = fn["description"].strip()
            if len(desc) > TOOL_DESC_LIMIT:
                desc = desc[:TOOL_DESC_LIMIT] + "\u2026"
            out.append(desc)
        schema = fn.get("parameters") or {}
        props = schema.get("properties") or {}
        if props:
            out.append("parameters (JSON Schema):")
            out.append(json.dumps(
                {"type": schema.get("type", "object"),
                 "properties": props,
                 "required": schema.get("required", [])},
                ensure_ascii=False, separators=(",", ":")))
        out.append("")
    return "\n".join(out)


def _content_to_text(content):
    if isinstance(content, list):
        bits = []
        for p in content:
            if not isinstance(p, dict):
                continue
            if p.get("type") in ("image_url", "image"):
                bits.append("[image supplied; this transport is text-only]")
            else:
                bits.append(p.get("text") or "")
        return " ".join(b for b in bits if b)
    return content or ""


def flatten_history(msgs):
    """Flatten the whole client conversation into one prompt.

    The CLI is a fresh process per call, so context must ride in the prompt.
    Tool calls and their results are labelled so the model can chain them.
    """
    lines = []
    for m in msgs:
        role = m.get("role", "user")
        if role == "tool":
            name = m.get("name") or m.get("tool_call_id") or "tool"
            lines.append(f"[tool result: {name}]\n{_content_to_text(m.get('content'))}")
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            try:
                args = json.dumps(json.loads(args)) if isinstance(args, str) else args
            except Exception:
                pass
            lines.append(
                f"[tool call: {fn.get('name')}]\narguments: {json.dumps(args, ensure_ascii=False)}")
        text = _content_to_text(m.get("content"))
        if not text:
            continue
        if role == "system":
            lines.append(f"[system]\n{text}")
        elif role == "assistant":
            lines.append(f"[assistant]\n{text}")
        else:
            lines.append(f"[user]\n{text}")
    return "\n\n".join(lines).strip()


def tool_choice_directive(body):
    """Extra instruction when the caller constrains tool use.

    Returns (render_catalogue, directive). The proxy previously ignored
    tool_choice entirely, so a client demanding a specific function call
    could get prose instead with no recourse.
    """
    tc = body.get("tool_choice")
    if tc is None or tc == "auto":
        return True, "", None
    if tc == "none":
        return False, "", None
    name = None
    if isinstance(tc, dict):
        if tc.get("type") == "function":
            fn = tc.get("function") or {}
            name = fn.get("name")
        else:
            name = tc.get("name")
    if not name:
        return True, "", None
    return True, (
        "[system] The caller requires a call to `%s` this turn. Respond "
        "with ONLY the tool-call block for that tool, no prose before or "
        "after it." % name), name


def affinity_key_for(model, body, header_key=None):
    """Stable conversation key: explicit header wins, else hash of
    model + system + first user message. Collisions across DISTINCT live
    conversations fall back safe (busy sessions go fresh)."""
    if header_key:
        return "x:" + header_key[:64]
    import hashlib as _hl
    msgs = body.get("messages", [])
    sys_text = " ".join(
        _content_to_text(m.get("content")) for m in msgs
        if m.get("role") == "system")[:2000]
    first_user = next((m for m in msgs if m.get("role") == "user"), {})
    first_text = _content_to_text(first_user.get("content"))[:2000]
    return "h:" + _hl.sha1(
        (model + "|" + sys_text + "|" + first_text).encode()).hexdigest()[:16]


def build_prompt(body):
    msgs = body.get("messages", [])
    tools = body.get("tools") or []
    render_catalogue, directive, forced = tool_choice_directive(body)
    if forced:
        # Forced turn: render ONLY the demanded tool. Full 67-tool catalogue
        # buries the directive and burns ~30-100k chars.
        tools = [t for t in tools
                 if ((t.get("function") or {}).get("name") == forced)] or tools
    parts = []
    if tools and render_catalogue:
        model_name = (body.get("model") or "").lower()
        proto = PROTOCOL_SPARK if "muse-spark" in model_name else PROTOCOL
        parts.append(proto)
        parts.append(render_tools(tools))
    if directive:
        parts.append(directive)
    convo = flatten_history(msgs)
    if len(convo) > MAX_PROMPT_CHARS:
        # Keep the head (usually system prompt) + the recent tail.
        head, tail = convo[:4000], convo[-(MAX_PROMPT_CHARS - 4000):]
        convo = ("[shim: %d older chars elided]\n" % (len(convo) - MAX_PROMPT_CHARS)
                 + head + "\n\n...\n\n" + tail)
    parts.append(convo if convo else "Hello")
    return "\n\n".join(p for p in parts if p)


# ── tool-call stream splitting ───────────────────────────────────────────────

def _partial_suffix_len(buf, fence):
    """Longest k < len(fence) such that buf ends with fence[:k]."""
    maxk = min(len(fence) - 1, len(buf))
    for k in range(maxk, 0, -1):
        if buf.endswith(fence[:k]):
            return k
    return 0


class Splitter:
    """Split model text into prose and fenced tool-call blocks, streaming-safe.

    Yields ('text', s) / ('tool', s). Holds back only enough trailing text that
    a fence delimiter cannot straddle a chunk boundary, so prose still streams.
    """

    def __init__(self):
        self.buf = ""
        self.in_tool = False

    def feed(self, new):
        self.buf += new
        while self.buf:
            if self.in_tool:
                i = self.buf.find(CLOSE_FENCE)
                if i >= 0:
                    if i:
                        yield ("tool", self.buf[:i])
                    self.buf = self.buf[i + len(CLOSE_FENCE):]
                    self.in_tool = False
                    continue
                keep = _partial_suffix_len(self.buf, CLOSE_FENCE)
                if len(self.buf) > keep:
                    cut = len(self.buf) - keep
                    yield ("tool", self.buf[:cut])
                    self.buf = self.buf[cut:]
                return
            i = self.buf.find(OPEN_FENCE)
            if i >= 0:
                if i:
                    yield ("text", self.buf[:i])
                self.buf = self.buf[i + len(OPEN_FENCE):]
                self.in_tool = True
                continue
            keep = _partial_suffix_len(self.buf, OPEN_FENCE)
            if len(self.buf) > keep:
                cut = len(self.buf) - keep
                yield ("text", self.buf[:cut])
                self.buf = self.buf[cut:]
            return

    def flush(self):
        """Emit whatever is left when the model stops."""
        if self.buf:
            yield ("tool" if self.in_tool else "text", self.buf)
            self.buf = ""
        self.in_tool = False


def parse_tool_block(raw):
    """Parse the inside of a tool fence into a list of OpenAI tool calls."""
    raw = raw.strip()
    if not raw:
        return []
    obj = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        objs = []
        for line in raw.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        obj = objs or None
    if obj is None:
        return []
    items = obj if isinstance(obj, list) else [obj]
    calls = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name") or it.get("tool") or it.get("function")
        if isinstance(name, dict):
            name = name.get("name")
        if not name:
            continue
        args = it.get("arguments", it.get("parameters", it.get("args", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}
        if not isinstance(args, dict):
            args = {"value": args}
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": str(name),
                         "arguments": json.dumps(args, ensure_ascii=False)},
        })
    return calls


# ── CLI transport ────────────────────────────────────────────────────────────

class ShimError(Exception):
    pass


# ACP transport lives in acp_transport.py (multiplexed sessions on one
# long-lived `opencode acp` process, streaming agent_message_chunk
# deltas live). The per-turn subprocess pump it replaces is gone;
# collect() below is unchanged apart from its event source.


async def collect(model, prompt, stream_cb=None, log=None,
                  progress_cb=None, affinity=None):
    """Run one turn. Returns (text, tool_calls, alien_tool_names).

    Logs time-to-first-text so a slow-prefill turn is distinguishable from a
    wedged one: previously the only log lines were turn start and turn end,
    so minutes of legitimate model compute looked identical to a hang.
    """
    t0 = time.monotonic()
    first_at = None
    def note_text():
        nonlocal first_at
        if first_at is None:
            first_at = time.monotonic() - t0
            if log:
                log("first text after %.1fs" % first_at)

    # NOTE (was docstring tail, kept as comment during repair):
    # If the model reached for one of the CLI's own tools, the turn is retried
    # once with a corrective nudge appended to the prompt. Transient CLI deaths
    # (step watchdog abort, timeout, connection reset) are also retried: without
    # this the partial text of a killed turn is masked as a complete answer and
    # the client sees the turn 'finish' with no tool calls.

    alien = []
    corrected = False

    async def progress(msg):
        if progress_cb is None:
            return
        try:
            res = progress_cb(msg)
            if res is not None:
                await res
        except Exception:
            pass

    for attempt in range(3):
        sp = Splitter()
        text_parts, tool_raw = [], []
        alien = []
        gen = acp_turn_events(model, prompt, progress_cb=progress,
                                affinity=affinity)
        try:
            async for ev, payload in gen:
                if ev == "text":
                    for kind, seg in sp.feed(payload):
                        if kind == "text":
                            text_parts.append(seg)
                            note_text()
                            if stream_cb:
                                await stream_cb(seg)
                        else:
                            tool_raw.append(seg)
                elif ev == "alien_tool":
                    alien.append(payload)
                    await progress("executing %s\u2026" % payload)
        except (ShimError, AcpError) as e:
            if attempt < 2 and TRANSIENT_RE.search(str(e)):
                if stream_cb:
                    await stream_cb(f"\n[shim: turn transport failed ({e}); retrying turn]\n")
                continue
            raise
        except (ConnectionResetError, asyncio.CancelledError):
            # Client went away (stop button / disconnect): close the turn
            # generator so its finally sends session/cancel + session/close.
            # Without this the server runs the full 20min ceiling blind.
            try:
                await gen.aclose()
            except Exception:
                pass
            raise
        for kind, seg in sp.flush():
            if kind == "text":
                text_parts.append(seg)
                note_text()
                if stream_cb:
                    await stream_cb(seg)
            else:
                tool_raw.append(seg)

        calls = parse_tool_block("".join(tool_raw))
        if not calls and not alien and attempt < 2 and NARRATE_RE.search("".join(text_parts)):
            # Model talked about the protocol instead of using it. Correct it.
            prompt = (prompt + "\n\n[system] Your last response talked about the tool "
                      "protocol instead of using it. Do not explain or narrate. Either "
                      "emit ONLY the fenced <tool_call> block, or reply with plain "
                      "prose and no block.")
            corrected = True
            continue
        if calls or not alien or attempt == 2:
            return "".join(text_parts), calls, alien, attempt + 1
        if corrected:
            # Already spent the one correction on narration; hand the alien
            # turn back so the CLIENT re-plans instead of burning full resends.
            return "".join(text_parts), calls, alien, attempt + 1
        corrected = True
        # The model tried a tool that does not exist in the client. Correct it.
        prompt = (prompt + "\n\n[system] Your last attempt tried to use a built-in "
                  "tool (" + ", ".join(sorted(set(alien))) + "). That tool does not "
                  "exist here and its output is discarded. Use the tool protocol "
                  "above with one of the listed tools, or answer in plain text.")
    return "", [], [], 3


# ── SSE plumbing ─────────────────────────────────────────────────────────────

def chunk(cid, model, delta=None, finish=None):
    d = {}
    if delta is not None:
        # The OpenAI wire format requires delta to be an OBJECT. A bare
        # string here passes Python fine but explodes client-side schema
        # validation (Vercel AI SDK zod: "expected object, received string"),
        # turning a readable proxy error into an inscrutable union error.
        # Coerce once, centrally, so no error path can emit a bad frame.
        if isinstance(delta, str):
            delta = {"content": delta}
        d["delta"] = delta
    if finish:
        d["finish_reason"] = finish
    return json.dumps({"id": cid, "object": "chat.completion.chunk",
                       "created": 0, "model": model,
                       "choices": [{"index": 0, **d}]})


def tool_call_chunks(cid, model, calls):
    """OpenAI streaming shape for tool calls: index, id, name, then arguments."""
    out = []
    for i, c in enumerate(calls):
        out.append(json.dumps({
            "id": cid, "object": "chat.completion.chunk", "created": 0, "model": model,
            "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": i, "id": c["id"], "type": "function",
                "function": {"name": c["function"]["name"], "arguments": ""}}]}}]}))
        out.append(json.dumps({
            "id": cid, "object": "chat.completion.chunk", "created": 0, "model": model,
            "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": i,
                "function": {"arguments": c["function"]["arguments"]}}]}}]}))
    return out


def final_body(cid, model, text, calls, shim=None):
    msg = {"role": "assistant", "content": text or None}
    if calls:
        msg["tool_calls"] = calls
    return {"id": cid, "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "shim": shim or {}}


# ── direct Zen passthrough ───────────────────────────

async def handle_direct(request, body, model):
    """Relay to the Zen OpenAI-compatible endpoint untouched.

    Tools, tool_choice, temperature, max_tokens and every other OpenAI field
    pass through verbatim: tool calling is native here, so none of the fenced
    text protocol applies. Streaming chunks are forwarded byte-for-byte as
    they arrive (810 chunks over 37s measured), which is the true incremental
    streaming the CLI transport cannot provide.
    """
    import aiohttp as _aiohttp
    key = load_zen_key()
    if not key:
        return web.json_response(
            {"error": {"message": "direct route unavailable: account key unreadable",
                       "type": "server_error", "code": "no_key"}}, status=503)
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + key}
    timeout = _aiohttp.ClientTimeout(total=600, sock_read=300)
    try:
        session = _aiohttp.ClientSession(timeout=timeout)
        upstream = await session.post(
            ZEN_API_URL + "/chat/completions",
            headers=headers, json=body)
    except Exception as e:
        return web.json_response(
            {"error": {"message": "zen unreachable: %s" % e,
                       "type": "server_error"}}, status=502)
    if body.get("stream"):
        resp = web.StreamResponse(status=upstream.status, headers={
            "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
            "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        await resp.prepare(request)
        try:
            async for chunk_bytes in upstream.content.iter_any():
                if chunk_bytes:
                    await resp.write(chunk_bytes)
            await resp.write_eof()
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            await upstream.release()
            await session.close()
        print("[proxy] direct %s streamed done" % model, flush=True)
        return resp
    try:
        payload = await upstream.json()
    except Exception as e:
        return web.json_response(
            {"error": {"message": "zen bad reply: %s" % e,
                       "type": "server_error"}}, status=502)
    finally:
        await upstream.release()
        await session.close()
    return web.json_response(payload, status=upstream.status)


# ── handlers ─────────────────────────────────────────────────────────────────

async def handle_chat(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": {"message": "invalid JSON body"}}, status=400)

    requested = body.get("model") or DEFAULT_MODEL
    model = resolve_model(requested)
    ok, model = allow_model(requested)
    if not ok:
        if model in DEAD_MODELS:
            msg, code = DEAD_MODELS[model], "model_dead"
        else:
            msg, code = ("model '%s' is not served by this proxy; free-tier models only" % requested, "model_not_found")
        return web.json_response(
            {"error": {"message": msg, "type": "model_not_found",
                       "code": code, "param": "model"}}, status=404)

    if model in DIRECT_MODELS:
        print("[proxy] %s direct %s tools=%d" % (
            model, "stream" if body.get("stream") else "block",
            len(body.get("tools") or [])), flush=True)
        return await handle_direct(request, body, model)
    stream = bool(body.get("stream"))
    ntools = len(body.get("tools") or [])
    prompt = build_prompt(body)
    from acp_transport import peek_sent as _peek
    _akey = affinity_key_for(
        model, body, request.headers.get("X-Shim-Session"))
    _sent = _peek(_akey)
    if _sent is not None:
        _delta = flatten_history(body.get("messages", [])[_sent:])
        affinity = (_akey, _delta, len(body.get("messages", [])))
    else:
        affinity = (_akey, "", len(body.get("messages", [])))
    print(f"[proxy] {requested}->{model} {len(prompt)}ch "
          f"{'stream' if stream else 'block'} tools={ntools}", flush=True)
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]

    if not stream:
        try:
            text, calls, alien, attempts = await collect(
                model, prompt,
                log=lambda m: print(f"[proxy] {cid} {m}", flush=True),
                affinity=affinity)
            return web.json_response(final_body(cid, model, text, calls,
                                               shim={"prompt_chars": len(prompt),
                                                     "alien": alien,
                                                     "attempts": attempts}))
        except Exception as e:
            print(f"[proxy] ERROR: {e}", flush=True)
            return web.json_response({"error": {"message": str(e)}}, status=502)

    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "Connection": "keep-alive", "X-Accel-Buffering": "no"})
    await resp.prepare(request)

    async def send(payload):
        await resp.write(f"data: {payload}\n\n".encode())

    try:
        await send(chunk(cid, model, delta={"role": "assistant", "content": ""}))

        async def on_text(seg):
            await send(chunk(cid, model, delta={"content": seg}))

        async def on_progress(msg):
            await send(": [shim: %s]" % msg)

        text, calls, alien, attempts = await collect(
            model, prompt, stream_cb=on_text, progress_cb=on_progress,
            log=lambda m: print(f"[proxy] {cid} {m}", flush=True),
            affinity=affinity)
        for payload in tool_call_chunks(cid, model, calls):
            await send(payload)
        await send(": [shim: prompt=%dch alien=%s attempts=%d]"
                   % (len(prompt), ",".join(sorted(set(alien))) or "none",
                      attempts))
        await send(chunk(cid, model, finish="tool_calls" if calls else "stop"))
        await send("[DONE]")
        await resp.write_eof()
        print(f"[proxy] -> done text={len(text)}ch tools={len(calls)} alien={alien}", flush=True)
    except asyncio.TimeoutError:
        await send(chunk(cid, model, delta="[proxy] timeout waiting for opencode"))
        await send(chunk(cid, model, finish="stop"))
        await send("[DONE]")
        await resp.write_eof()
    except Exception as e:
        print(f"[proxy] STREAM ERROR: {e}", flush=True)
        try:
            await send(chunk(cid, model, delta=f"[proxy error] {e}"))
            await send(chunk(cid, model, finish="stop"))
            await send("[DONE]")
            await resp.write_eof()
        except Exception:
            pass
    return resp


async def handle_models(request):
    return web.json_response({"object": "list", "data": [
        {"id": m, "object": "model", "created": 0, "owned_by": "opencode"}
        for m in FREE_MODELS] + [
        {"id": a, "object": "model", "created": 0, "owned_by": "opencode-alias"}
        for a in ALIASES]})


async def health(request):
    return web.json_response({
        "status": "ok", "default_model": DEFAULT_MODEL, "free_models": FREE_MODELS,
        "dead_models": sorted(DEAD_MODELS), "transport": "stdin",
        "tool_bridge": True, "cli": OPENCODE_CLI, "cli_exists": os.path.exists(OPENCODE_CLI),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18788)
    args = ap.parse_args()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat)
    app.router.add_post("/chat/completions", handle_chat)
    print(f"[proxy] streaming :{args.port} (ACP transport)", flush=True)
    print(f"[proxy] cli exists: {os.path.exists(OPENCODE_CLI)}", flush=True)
    host = os.environ.get("OC_SHIM_HOST", "127.0.0.1")
    web.run_app(app, host=host, port=args.port)


if __name__ == "__main__":
    main()
