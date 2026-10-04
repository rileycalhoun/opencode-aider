# opencode-aider

An OpenAI-compatible SSE proxy that serves OpenCode's free-tier models to
**Aider**, **AiderDesk**, and any other OpenAI-compatible client.

Most of OpenCode free tier is locked behind desktop-app identity: bare
HTTP requests get rejected (403), so this proxy drives the authenticated
`opencode` CLI and translates its output to OpenAI SSE. Two transports,
routed per model automatically. `space-bunny-free` serves over bare HTTPS
with the account key, so requests for it skip local processes entirely and
relay natively -- true incremental streaming plus native `tool_calls`, no
fence protocol involved. Every other free model goes over ACP: one
long-lived `opencode acp` process hosts multiplexed sessions (one per HTTP
turn), `session/set_config_option` selects the model, and
`agent_message_chunk` deltas relay live as SSE -- genuine incremental
streaming, verified at 301 chunks over 8 seconds. Client tools still travel
as a fenced text protocol parsed back into `tool_calls`; opencode-native
tool calls are suppressed exactly as before.

Forked from
[ArcticWinterSturm/opencode-compat-shim](https://github.com/ArcticWinterSturm/opencode-compat-shim)
(Hermes-oriented, Windows-only). This fork is client-neutral, runs on Linux,
and locks the model surface to free-tier models so callers can never spend
paid credit through it.

## Requirements

- An installed `opencode` CLI, authenticated (`opencode auth login`), with
  access to at least one free-tier model. Verify with:
  `opencode run --model opencode/space-bunny-free "reply OK"`
- Python 3.10+ with `aiohttp` (`pip install aiohttp` or `uv pip install aiohttp)

## Run it

```bash
python opencode_proxy.py --port 18788

# Test
curl http://127.0.0.1:18788/health
curl -X POST http://127.0.0.1:18788/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"space-bunny-free","messages":[{"role":"user","content":"hi"}]}'
```

Environment overrides (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `OPENCODE_CLI` | `/usr/local/bin/opencode` | Path to the authenticated CLI binary |
| `OC_SHIM_CWD` | `/home/opencode/.cache/oc-shim-cwd` | Scratch dir the CLI runs in (its own file tools stay here, never your repo) |
| `OC_SHIM_HOST` | `127.0.0.1` | Bind address. Bind a LAN/Tailscale IP to serve other machines; never `0.0.0.0` on an untrusted network — the proxy has no auth of its own |

Example as a systemd service (runs as an unprivileged user, Tailscale-only):

```ini
[Service]
Type=simple
User=opencode
WorkingDirectory=/home/opencode
Environment=HOME=/home/opencode
Environment=OC_SHIM_HOST=100.121.228.23
ExecStart=/home/opencode/opencode-aider/.venv/bin/python /home/opencode/opencode-aider/opencode_proxy.py --port 18788
Restart=on-failure
```

## Use it with Aider (terminal)

Point Aider at the proxy as a generic OpenAI-compatible endpoint. The API key
value is ignored by the proxy; it just has to be non-empty:

```bash
export OPENAI_API_BASE=http://127.0.0.1:18788/v1
export OPENAI_API_KEY=none
aider --model openai/space-bunny-free
```

Or per-invocation:

```bash
aider --model openai/space-bunny-free \
  --openai-api-base http://127.0.0.1:18788/v1 \
  --api-key openai=none
```

Aider's function calling works: the proxy advertises your `tools` to the
model through a fenced protocol block and translates the model's calls back
into standard OpenAI `tool_calls` frames, which Aider executes. `tool_choice`
is honored — `"none"` strips the catalogue, a named `{"type":"function",
"function":{"name": ...}}` forces that call.

## Use it with AiderDesk

Add a custom OpenAI-compatible provider (Settings → Providers, or directly in
the providers config):

- Type: `openai-compatible`
- Base URL: `http://<proxy-host>:18788/v1` (use the machine's LAN or Tailscale
  address when AiderDesk runs in Docker — `localhost` inside a container is
  the container itself)
- API key: `none`
- Models: any of the free-tier IDs below (e.g. `space-bunny-free`)

Then select the provider on any agent profile. The proxy appears to AiderDesk
as a normal OpenAI-compatible backend, so streaming, tool use, and token
tracking behave as usual. (Reported usage will show 0 tokens — the proxy does
not meter; the spend is zero because the models are free.)

## Models

`GET /v1/models` lists what's served. Free-tier IDs (plus short aliases like
`bunny`, `spark`, `longcat`, `mimo`, `ling`, `nemotron`):

- `space-bunny-free`, `longcat-2.5-preview-free`, `mimo-v2.6-flash-free`,
  `ling-3.0-flash-fin-free`, `nemotron-3-ultra-free`,
  `nemotron-3.5-lightning-free`, `muse-spark-1.3-contributor-free`

Anything else returns `404 model_not_found` **without touching the CLI** —
this is deliberate. An earlier version passed unknown model names straight
through as `opencode/<name>`, which meant a client typo (or worse) could
invoke a paid model on your Zen account. Retired models return
`503 model_dead` with the reason.

Free-tier availability shifts without notice (models get disabled per
account and retired upstream). If a listed model 403s, check it directly
first: `opencode run --model opencode/<id> "hi"`. If the CLI itself is
refused, the model is gone for your account — remove it from your client
config, not from this proxy.

## How tool calling works

OpenAI `tools` arrays cannot cross the CLI boundary, so the proxy serialises
your tool catalogue into the prompt with a strict fenced protocol
(`<tool_call>...</tool_call>`, one JSON object per line for parallel calls)
and parses the model's fenced replies back into `tool_calls` frames. The
model is told up front it has no built-in tools.

The CLI injects its *own* tools (bash/read/write) that do not exist in your
client and cannot be disabled in 1.18.x. Those `tool_use` events are
suppressed — never forwarded — and constrained to the scratch `OC_SHIM_CWD`.
If the model reaches for one anyway, the turn is retried once with a
corrective nudge. Transient CLI deaths (watchdog aborts, timeouts, resets)
are retried up to 3 times per turn; a genuinely dead turn surfaces as a 502,
never as a silent empty answer.

## Security posture

- **No auth on the proxy.** Bind it to loopback or a trusted tailnet address.
  Anyone who can reach it can use the free models behind it.
- **Free-tier only, enforced before the CLI runs.** Unknown and paid model
  names are rejected with 404. This is the spend control — do not remove it.
- **Prompts cross a process boundary** (stdin to the CLI) and conversation
  history is flattened to text each turn, including prior tool calls and
  results labelled as such. Do not send secrets you would not paste into the
  model directly.
- **No request logging of bodies.** The proxy logs model name, prompt length,
  and tool counts to stdout, never message content.

## Tests

```bash
python test_units.py        # offline: stream splitting, tool parsing, argv cap
```

`test_tools.py` and `test_multistep.py` exercise a live server
(`http://127.0.0.1:18788` by default). Test paths that hardcode a machine
location were left as found upstream except the module loader, which now
resolves `opencode_proxy.py` relative to the test file.

## Differences from upstream

- Client-neutral tool protocol (`<tool_call>`, no Hermes naming anywhere the
  model can see) and honored `tool_choice` (`none` strips the catalogue, a
  named function forces the call).
- Free-model allowlist enforced pre-CLI (upstream passes any name through).
- Linux paths via environment (`OPENCODE_CLI`, `OC_SHIM_CWD`, `OC_SHIM_HOST`)
  instead of hardcoded Windows locations.
- `muse-spark-1.3-contributor-free` (+ aliases) added; retired models return
  503 with reasons instead of hanging.
- Test module loader resolves relative to the test file (upstream hardcodes
  a Windows path).
