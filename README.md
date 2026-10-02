<div align="center">

# vertex-proxy
*OpenAI-compatible proxy for Google Vertex AI (Gemini) models*

[Features](#features) • [Quick start](#quick-start) • [Configuration](#configuration) • [API](#api)

</div>

A lightweight Rust proxy that exposes Google Vertex AI (Gemini) models through
the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat).
It speaks OpenAI on one side and Vertex AI `generateContent` on the other, so
existing OpenAI clients work with Vertex models without changes.

## Features

- ⚡ **OpenAI-compatible API** - `/v1/chat/completions` and `/v1/models` endpoints
- 🚀 **Streaming** - SSE responses with `stream: true`
- 🔧 **Function calling** - multi-step tool loops with `tools` / `tool_choice`
- 🧠 **Gemini 3 thinking** - thought-signature round-trip via tool-call IDs
- 🖼️ **Image support** - base64 data-URL images as inline data
- 🔑 **OAuth auth** - gcloud ADC with automatic token refresh

## Quick start

### 1. Authenticate

```sh
gcloud auth application-default login \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

> [!NOTE]
> Use a *user* account ADC (`gcloud auth application-default login`), not a
> service-account key. The proxy uses the OAuth refresh-token flow.

### 2. Build and run

```sh
cargo build --release
VERTEX_LOCATION=global ./target/release/vertex-proxy
```

Listens on `127.0.0.1:8000` by default.

### 3. Try it

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Explain TCP in one sentence."}]
  }'
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `PORT` / `VERTEX_PROXY_PORT` | `8000` | Port to listen on |
| `VERTEX_PROXY_HOST` | `127.0.0.1` | Bind address |
| `VERTEX_LOCATION` | `global` | Vertex AI location (e.g. `global`, `europe-west1`) |
| `VERTEX_PROJECT_ID` | ADC project | Google Cloud project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | `~/.config/gcloud/application_default_credentials.json` | Path to ADC file |
| `RUST_LOG` | `vertex_proxy=info,tower_http=info` | Log filter |

## API

### Chat completions

`POST /v1/chat/completions` (also `/chat/completions`) accepts standard OpenAI
requests: `model`, `messages`, `stream`, `tools`, `tool_choice`, `temperature`,
`top_p`, `max_tokens` / `max_completion_tokens`, and `stop`.

Model names may be prefixed with `vertex_ai/`, `google/`, or `vertex/`; the
prefix is stripped automatically.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-3.8-flash",
    "stream": true,
    "messages": [{"role": "user", "content": "Write a haiku about proxies."}]
  }'
```

### Function calling

OpenAI-style tools map to Vertex `functionDeclarations`. Multi-step tool loops
are supported: the proxy encodes Vertex thought signatures into the returned
tool-call IDs, so Gemini 3's reasoning state is preserved across turns.

```json
{
  "model": "gemini-3.8-flash",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": { "city": { "type": "string" } }
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

### Models

`GET /v1/models` returns the list of supported models.

## How it works

1. Receives an OpenAI Chat Completions request.
2. Converts messages to Vertex `contents` (roles `user` / `model`, tool results
   as `functionResponse`, tool calls as `functionCall`).
3. Authenticates with a cached OAuth token and forwards to
   `aiplatform.googleapis.com`.
4. Converts the response (or SSE stream) back to OpenAI format.

> [!TIP]
> Tool-call IDs carry the thought signature as
> `<id>__<function>__thought__<signature>`, so the signature round-trips
> through any OpenAI client that echoes tool-call IDs back.

## Project layout

```
src/
  main.rs     Server setup, routes, env config
  handler.rs  HTTP handlers: chat completions, model list, health
  convert.rs  OpenAI <-> Vertex request/response conversion
  types.rs    Shared request/response types
  auth.rs     gcloud ADC loading and OAuth token refresh
```