use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;
use futures::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, warn};

use crate::auth::AuthManager;
use crate::convert::*;
use crate::types::*;

#[derive(Clone)]
pub struct AppState {
    pub auth: AuthManager,
    pub location: String,
    pub http_client: reqwest::Client,
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
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ];

    let data = models
        .into_iter()
        .map(|m| ModelEntry {
            id: m.to_string(),
            object: "model",
            owned_by: "google",
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
    let location = &state.location;

    let base_url = format!(
        "https://aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}",
        project_id, location, model
    );

    if req.stream {
        let stream_url = format!("{}:streamGenerateContent?alt=sse", base_url);
        let resp = match state
            .http_client
            .post(&stream_url)
            .header("Authorization", format!("Bearer {}", token))
            .header("X-Goog-User-Project", project_id)
            .header("Content-Type", "application/json")
            .json(&vertex_req)
            .send()
            .await
        {
            Ok(r) => r,
            Err(err) => {
                error!("Request to Vertex AI failed: {}", err);
                return (StatusCode::BAD_GATEWAY, Json(serde_json::json!({
                    "error": {
                        "message": format!("Vertex AI network error: {}", err),
                        "type": "vertex_error",
                        "code": 502
                    }
                }))).into_response();
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
                    line_buffer = line_buffer[idx + 1..].to_string();

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
                                            let chunk_resp = OpenAiChunkResponse {
                                                id: stream_id.clone(),
                                                object: "chat.completion.chunk",
                                                created,
                                                model: model_name.clone(),
                                                choices: vec![OpenAiChunkChoice {
                                                    index: 0,
                                                    delta: OpenAiChunkDelta {
                                                        role: role_opt,
                                                        content: if txt.is_empty() { None } else { Some(txt) },
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
        let generate_url = format!("{}:generateContent", base_url);
        let resp = match state
            .http_client
            .post(&generate_url)
            .header("Authorization", format!("Bearer {}", token))
            .header("X-Goog-User-Project", project_id)
            .header("Content-Type", "application/json")
            .json(&vertex_req)
            .send()
            .await
        {
            Ok(r) => r,
            Err(err) => {
                error!("Request to Vertex AI failed: {}", err);
                return (StatusCode::BAD_GATEWAY, Json(serde_json::json!({
                    "error": {
                        "message": format!("Vertex AI network error: {}", err),
                        "type": "vertex_error",
                        "code": 502
                    }
                }))).into_response();
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
