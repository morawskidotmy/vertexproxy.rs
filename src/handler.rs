use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;
use futures::StreamExt;
use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, warn};

use crate::auth::AuthManager;
use crate::convert::*;
use crate::types::*;

#[derive(Clone)]
pub struct AppState {
    pub auth: AuthManager,
    pub http_client: reqwest::Client,
    /// Dynamically selected Vertex AI location (lowest latency), or pinned
    /// via VERTEX_LOCATION.
    pub region: Arc<RwLock<String>>,
    /// Regions observed not to serve a requested model, mapped to when they
    /// were excluded. Exclusions expire after 5 minutes so regions that host
    /// other (regional) models can be re-selected later.
    pub failed_regions: Arc<RwLock<HashMap<String, Instant>>>,
    pub deepseek_endpoint_id: Option<String>,
    pub deepseek_location: String,
    pub deepseek_model: String,
}

pub async fn health() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/plain; charset=utf-8")],
        "I'm alive!",
    )
}

pub async fn list_models() -> impl IntoResponse {
    let models = vec![
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "deepseek-v4.1-flash",
    ];

    let data = models
        .into_iter()
        .map(|m| ModelEntry {
            id: m.to_string(),
            object: "model",
            owned_by: if m.starts_with("deepseek") { "deepseek-ai" } else { "google" },
        })
        .collect();

    Json(ModelListResponse {
        object: "list",
        data,
    })
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let model = clean_model_name(&req.model).to_string();

    // DeepSeek models (Model Garden custom endpoints) are served via the
    // OpenAI-compatible /chat/completions path on the deployed endpoint, so
    // forward the request through with no format conversion.
    if model == "deepseek-v4.1-flash" || model.starts_with("deepseek-") {
        return handle_deepseek_completion(&state, &req, &model).await;
    }

    let raw_roles: Vec<&str> = req.messages.iter().map(|m| m.role.as_str()).collect();
    let last_info = req.messages.last().map(|m| {
        let content_type = m.content.as_ref().map(|c| match c {
            serde_json::Value::String(_) => "string",
            serde_json::Value::Array(_) => "array",
            serde_json::Value::Null => "null",
            _ => "other",
        }).unwrap_or("none");
        format!(
            "role={} content={} tool_calls={} tool_call_id={}",
            m.role,
            content_type,
            m.tool_calls.as_ref().map(|t| t.len()).unwrap_or(0),
            m.tool_call_id.as_ref().map(|s| s.as_str()).unwrap_or("none")
        )
    }).unwrap_or_else(|| "no-messages".to_string());
    tracing::info!(
        "chat request: model={} stream={} roles={:?} tools={} last=[{}]",
        model, req.stream, raw_roles, req.tools.as_ref().map(|t| t.len()).unwrap_or(0), last_info
    );

    let vertex_req = match build_vertex_request(&req) {
        Ok(r) => r,
        Err(err) => {
            warn!("Invalid chat completion request: {}", err);
            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
                "error": {
                    "message": err,
                    "type": "invalid_request_error",
                    "code": 400
                }
            }))).into_response();
        }
    };

    let token = match state.auth.get_token().await {
        Ok(t) => t,
        Err(err) => {
            error!("Failed to obtain OAuth token: {}", err);
            return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": {
                    "message": format!("Auth failure: {}", err),
                    "type": "authentication_error",
                    "code": 503
                }
            }))).into_response();
        }
    };

    let project_id = state.auth.project_id();
    let location = state.region.read().await.clone();

    if req.stream {
        let resp = match send_with_region_fallback(
            &state,
            &token,
            &project_id,
            &location,
            &model,
            ":streamGenerateContent?alt=sse",
            &serde_json::to_value(&vertex_req).unwrap_or_default(),
        )
        .await
        {
            Ok(r) => r,
            Err((code, err_json)) => {
                error!("Request to Vertex AI failed: {}", err_json);
                return (code, Json(err_json)).into_response();
            }
        };

        let status = resp.status();
        if !status.is_success() {
            let err_text = resp.text().await.unwrap_or_default();
            if status == StatusCode::BAD_REQUEST {
                error!(
                    "Vertex AI 400 request dump: model={} roles={:?} built_req={}",
                    model,
                    req.messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
                    serde_json::to_string(&vertex_req).unwrap_or_default()
                );
            }
            error!("Vertex AI returned error HTTP {}: {}", status, err_text);
            let code = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            return (code, Json(serde_json::json!({
                "error": {
                    "message": err_text,
                    "type": "vertex_error",
                    "code": status.as_u16()
                }
            }))).into_response();
        }

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(64);
        let stream_id = format!("chatcmpl-vertex-{}", created);
        let model_name = model.clone();

        tokio::spawn(async move {
            let mut byte_stream = resp.bytes_stream();
            let mut line_buffer = String::new();
            let mut had_tool_calls = false;
            let mut role_sent = false;

            while let Some(chunk_res) = byte_stream.next().await {
                let chunk = match chunk_res {
                    Ok(b) => b,
                    Err(e) => {
                        error!("Error reading Vertex stream: {}", e);
                        break;
                    }
                };

                let text = String::from_utf8_lossy(&chunk);
                line_buffer.push_str(&text);

                while let Some(idx) = line_buffer.find('\n') {
                    let line = line_buffer[..idx].trim().to_string();
                    line_buffer.drain(..=idx);

                    if line.is_empty() || !line.starts_with("data:") {
                        continue;
                    }

                    let payload = line.strip_prefix("data:").unwrap_or("").trim();
                    if payload.is_empty() {
                        continue;
                    }

                    if let Ok(vertex_chunk) = serde_json::from_str::<VertexResponse>(payload) {
                        for cand in vertex_chunk.candidates {
                            if let Some(content) = cand.content {
                                for part in content.parts {
                                    // Send role on the very first delta if not sent yet
                                    let role_opt = if !role_sent {
                                        role_sent = true;
                                        Some("assistant")
                                    } else {
                                        None
                                    };

                                    if let Some(txt) = part.text {
                                        if !txt.is_empty() || role_opt.is_some() {
                                            let is_thought = part.thought.unwrap_or(false);
                                            let (content, reasoning_content) = if is_thought {
                                                (None, if txt.is_empty() { None } else { Some(txt) })
                                            } else {
                                                (if txt.is_empty() { None } else { Some(txt) }, None)
                                            };
                                            let chunk_resp = OpenAiChunkResponse {
                                                id: stream_id.clone(),
                                                object: "chat.completion.chunk",
                                                created,
                                                model: model_name.clone(),
                                                choices: vec![OpenAiChunkChoice {
                                                    index: 0,
                                                    delta: OpenAiChunkDelta {
                                                        role: role_opt,
                                                        content,
                                                        reasoning_content,
                                                        tool_calls: None,
                                                    },
                                                    finish_reason: None,
                                                }],
                                            };
                                            if let Ok(json_str) = serde_json::to_string(&chunk_resp) {
                                                let event = format!("data: {}\n\n", json_str);
                                                if tx.send(Ok(Bytes::from(event))).await.is_err() {
                                                    return;
                                                }
                                            }
                                        }
                                    }

                                    if let Some(fc) = part.function_call {
                                        had_tool_calls = true;
                                        let raw_id = fc.id.unwrap_or_else(|| format!("call_{}", fc.name));
                                        let call_id = if let Some(sig) = part.thought_signature {
                                            format!("{}__{}__thought__{}", raw_id, fc.name, sig)
                                        } else {
                                            raw_id
                                        };

                                        let args_str = serde_json::to_string(&fc.args).unwrap_or_else(|_| "{}".to_string());
                                        let chunk_resp = OpenAiChunkResponse {
                                            id: stream_id.clone(),
                                            object: "chat.completion.chunk",
                                            created,
                                            model: model_name.clone(),
                                            choices: vec![OpenAiChunkChoice {
                                                index: 0,
                                                delta: OpenAiChunkDelta {
                                                    role: role_opt,
                                                    content: None,
                                                    reasoning_content: None,
                                                    tool_calls: Some(vec![OpenAiChunkToolCall {
                                                        index: 0,
                                                        id: Some(call_id),
                                                        tool_type: Some("function"),
                                                        function: OpenAiFunctionCall {
                                                            name: fc.name,
                                                            arguments: args_str,
                                                        },
                                                    }]),
                                                },
                                                finish_reason: None,
                                            }],
                                        };
                                        if let Ok(json_str) = serde_json::to_string(&chunk_resp) {
                                            let event = format!("data: {}\n\n", json_str);
                                            if tx.send(Ok(Bytes::from(event))).await.is_err() {
                                                return;
                                            }
                                        }
                                    }
                                }
                            }

                            if let Some(reason) = cand.finish_reason {
                                let finish_reason = if had_tool_calls || reason == "TOOL_CALL" {
                                    Some("tool_calls".to_string())
                                } else {
                                    match reason.as_str() {
                                        "STOP" => Some("stop".to_string()),
                                        "MAX_TOKENS" => Some("length".to_string()),
                                        "SAFETY" => Some("content_filter".to_string()),
                                        other => Some(other.to_lowercase()),
                                    }
                                };

                                let chunk_resp = OpenAiChunkResponse {
                                    id: stream_id.clone(),
                                    object: "chat.completion.chunk",
                                    created,
                                    model: model_name.clone(),
                                    choices: vec![OpenAiChunkChoice {
                                        index: 0,
                                        delta: OpenAiChunkDelta::default(),
                                        finish_reason,
                                    }],
                                };
                                if let Ok(json_str) = serde_json::to_string(&chunk_resp) {
                                    let event = format!("data: {}\n\n", json_str);
                                    if tx.send(Ok(Bytes::from(event))).await.is_err() {
                                        return;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // End of stream
            let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
        });

        let body_stream = ReceiverStream::new(rx);
        let body = Body::from_stream(body_stream);

        Response::builder()
            .header(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"))
            .header(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"))
            .header(header::CONNECTION, HeaderValue::from_static("keep-alive"))
            .body(body)
            .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Stream response error").into_response())
    } else {
        let resp = match send_with_region_fallback(
            &state,
            &token,
            &project_id,
            &location,
            &model,
            ":generateContent",
            &serde_json::to_value(&vertex_req).unwrap_or_default(),
        )
        .await
        {
            Ok(r) => r,
            Err((code, err_json)) => {
                error!("Request to Vertex AI failed: {}", err_json);
                return (code, Json(err_json)).into_response();
            }
        };

        let status = resp.status();
        if !status.is_success() {
            let err_text = resp.text().await.unwrap_or_default();
            if status == StatusCode::BAD_REQUEST {
                error!(
                    "Vertex AI 400 request dump: model={} roles={:?} built_req={}",
                    model,
                    req.messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
                    serde_json::to_string(&vertex_req).unwrap_or_default()
                );
            }
            error!("Vertex AI returned error HTTP {}: {}", status, err_text);
            let code = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            return (code, Json(serde_json::json!({
                "error": {
                    "message": err_text,
                    "type": "vertex_error",
                    "code": status.as_u16()
                }
            }))).into_response();
        }

        let vertex_resp: VertexResponse = match resp.json().await {
            Ok(r) => r,
            Err(err) => {
                error!("Failed to parse Vertex AI response: {}", err);
                return (StatusCode::BAD_GATEWAY, Json(serde_json::json!({
                    "error": {
                        "message": format!("Vertex AI JSON parse error: {}", err),
                        "type": "vertex_error",
                        "code": 502
                    }
                }))).into_response();
            }
        };

        let openai_resp = vertex_response_to_openai(vertex_resp, &model, created);
        Json(openai_resp).into_response()
    }
}

/// URL builder for a Gemini publisher model in a given location.
pub(crate) fn build_gemini_url(project_id: &str, location: &str, model: &str, suffix: &str) -> String {
    let host = if location == "global" {
        "aiplatform.googleapis.com".to_string()
    } else {
        format!("{}-aiplatform.googleapis.com", location)
    };
    format!(
        "https://{}/v1/projects/{}/locations/{}/publishers/google/models/{}{}",
        host, project_id, location, model, suffix
    )
}

/// POSTs a generateContent request, transparently retrying via the `global`
/// location if the auto-selected region does not host the requested model.
/// Passes the upstream error through if the retry also fails.
async fn send_with_region_fallback(
    state: &Arc<AppState>,
    token: &str,
    project_id: &str,
    location: &str,
    model: &str,
    suffix: &str,
    body: &serde_json::Value,
) -> Result<reqwest::Response, (StatusCode, serde_json::Value)> {
    let mut attempt = location.to_string();
    for _ in 0..2 {
        let url = build_gemini_url(project_id, &attempt, model, suffix);
        let resp = state
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .header("X-Goog-User-Project", project_id)
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|err| {
                error!("Request to Vertex AI failed: {}", err);
                (
                    StatusCode::BAD_GATEWAY,
                    serde_json::json!({
                        "error": {
                            "message": format!("Vertex AI network error: {}", err),
                            "type": "vertex_error",
                            "code": 502
                        }
                    }),
                )
            })?;

        if resp.status() == StatusCode::NOT_FOUND && attempt != "global" {
            warn!(
                "Model {} not found in region {}, excluding it and retrying via global",
                model, attempt
            );
            state.failed_regions
                .write()
                .await
                .insert(attempt.clone(), Instant::now());
            // Immediately re-probe so the next requests pick the next-best
            // region instead of paying the 404 detour repeatedly.
            let probe_state = state.clone();
            let probe_target = state.region.clone();
            tokio::spawn(async move {
                crate::probe_and_select(&probe_state, &probe_target).await;
            });
            attempt = "global".to_string();
            continue;
        }
        return Ok(resp);
    }
    unreachable!()
}

/// Forwards an OpenAI-compatible chat completion request to a DeepSeek model
/// deployed from Model Garden to a Vertex AI endpoint (vLLM container).
/// No format conversion is needed: the endpoint already speaks OpenAI's schema.
async fn handle_deepseek_completion(
    state: &Arc<AppState>,
    req: &ChatCompletionRequest,
    model: &str,
) -> Response {
    let endpoint_id = match state.deepseek_endpoint_id {
        Some(ref e) => e.clone(),
        None => {
            warn!("DeepSeek request but no endpoint configured: {}", model);
            return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": {
                    "message": "DeepSeek endpoint not configured. Set VERTEX_DEEPSEEK_ENDPOINT_ID to the endpoint ID of the model deployed from Model Garden (e.g. 123456789), and VERTEX_DEEPSEEK_LOCATION to its region (default us-central1).",
                    "type": "invalid_request_error",
                    "code": 503
                }
            }))).into_response();
        }
    };

    let token = match state.auth.get_token().await {
        Ok(t) => t,
        Err(err) => {
            error!("Failed to obtain OAuth token: {}", err);
            return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": {
                    "message": format!("Auth failure: {}", err),
                    "type": "authentication_error",
                    "code": 503
                }
            }))).into_response();
        }
    };

    let project_id = state.auth.project_id();
    let url = format!(
        "https://aiplatform.googleapis.com/v1/projects/{}/locations/{}/endpoints/{}/chat/completions",
        project_id, state.deepseek_location, endpoint_id
    );

    // Keep the client's fields, but override the served model name so it
    // matches the name the vLLM deployment was started with.
    let mut body = match serde_json::to_value(req) {
        Ok(v) => v,
        Err(err) => {
            error!("Failed to serialize DeepSeek request: {}", err);
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": {
                    "message": format!("Request serialization error: {}", err),
                    "type": "invalid_request_error",
                    "code": 500
                }
            }))).into_response();
        }
    };
    if let Some(obj) = body.as_object_mut() {
        obj.insert("model".to_string(), serde_json::json!(state.deepseek_model));
    }

    tracing::info!(
        "deepseek request: model={} stream={} -> {}",
        model,
        req.stream,
        url
    );

    let resp = match state
        .http_client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("X-Goog-User-Project", project_id)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(err) => {
            error!("Request to DeepSeek endpoint failed: {}", err);
            return (StatusCode::BAD_GATEWAY, Json(serde_json::json!({
                "error": {
                    "message": format!("DeepSeek endpoint network error: {}", err),
                    "type": "vertex_error",
                    "code": 502
                }
            }))).into_response();
        }
    };

    if req.stream {
        return forward_sse(resp).await;
    }

    let status = resp.status();
    if !status.is_success() {
        let err_text = resp.text().await.unwrap_or_default();
        error!("DeepSeek endpoint returned HTTP {}: {}", status, err_text);
        return (
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            Json(serde_json::json!({
                "error": {
                    "message": err_text,
                    "type": "vertex_error",
                    "code": status.as_u16()
                }
            })),
        )
            .into_response();
    }

    // Pass the OpenAI-format JSON payload through unchanged.
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_static("application/json"));
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(err) => {
            error!("Failed to read DeepSeek response: {}", err);
            return (StatusCode::BAD_GATEWAY, Json(serde_json::json!({
                "error": {
                    "message": format!("DeepSeek endpoint read error: {}", err),
                    "type": "vertex_error",
                    "code": 502
                }
            }))).into_response();
        }
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from(bytes))
        .unwrap_or_else(|_| {
            (StatusCode::INTERNAL_SERVER_ERROR, "DeepSeek response error").into_response()
        })
}

/// Byte-for-byte passthrough of the upstream SSE stream (vLLM/OpenAI format).
async fn forward_sse(resp: reqwest::Response) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(64);
    tokio::spawn(async move {
        let mut byte_stream = resp.bytes_stream();
        while let Some(chunk_res) = byte_stream.next().await {
            match chunk_res {
                Ok(b) => {
                    if tx.send(Ok(b)).await.is_err() {
                        return;
                    }
                }
                Err(e) => {
                    error!("Error reading DeepSeek stream: {}", e);
                    break;
                }
            }
        }
    });
    let body_stream = ReceiverStream::new(rx);
    let body = Body::from_stream(body_stream);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"))
        .header(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"))
        .header(header::CONNECTION, HeaderValue::from_static("keep-alive"))
        .body(body)
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Stream response error").into_response())
}
