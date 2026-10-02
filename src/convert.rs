use std::collections::HashMap;
use crate::types::*;

pub fn clean_model_name(raw: &str) -> &str {
    let s = raw.strip_prefix("vertex_ai/").unwrap_or(raw);
    let s = s.strip_prefix("google/").unwrap_or(s);
    let s = s.strip_prefix("vertex/").unwrap_or(s);
    s
}

/// Vertex AI's OpenAPI parser rejects JSON Schema keywords that OpenAI clients
/// commonly include (e.g. `$schema`, `exclusiveMinimum`, `multipleOf`). Strip
/// them recursively so tool schemas are accepted.
fn sanitize_schema(value: &mut serde_json::Value) {
    const STRIP_KEYS: &[&str] = &[
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "$comment",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "const",
    ];

    match value {
        serde_json::Value::Object(map) => {
            for key in STRIP_KEYS {
                map.remove(*key);
            }
            for (_, v) in map.iter_mut() {
                sanitize_schema(v);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr.iter_mut() {
                sanitize_schema(v);
            }
        }
        _ => {}
    }
}

pub fn parse_content_parts(content: &Option<serde_json::Value>) -> Vec<VertexPart> {
    let mut parts = Vec::new();
    let val = match content {
        Some(v) => v,
        None => return parts,
    };

    match val {
        serde_json::Value::String(s) => {
            if !s.is_empty() {
                parts.push(VertexPart {
                    text: Some(s.clone()),
                    thought_signature: None,
                    function_call: None,
                    function_response: None,
                    inline_data: None,
                });
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let Some(obj) = item.as_object() {
                    let item_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    if item_type == "text" {
                        if let Some(text) = obj.get("text").and_then(|t| t.as_str()) {
                            if !text.is_empty() {
                                parts.push(VertexPart {
                                    text: Some(text.to_string()),
                                    thought_signature: None,
                                    function_call: None,
                                    function_response: None,
                                    inline_data: None,
                                });
                            }
                        }
                    } else if item_type == "image_url" {
                        if let Some(image_url_obj) = obj.get("image_url").and_then(|u| u.as_object()) {
                            if let Some(url) = image_url_obj.get("url").and_then(|u| u.as_str()) {
                                if url.starts_with("data:") {
                                    if let Some((header, b64)) = url.split_once(',') {
                                        let mime = header
                                            .split(';')
                                            .next()
                                            .and_then(|h| h.strip_prefix("data:"))
                                            .unwrap_or("image/png");
                                        parts.push(VertexPart {
                                            text: None,
                                            thought_signature: None,
                                            function_call: None,
                                            function_response: None,
                                            inline_data: Some(VertexInlineData {
                                                mime_type: mime.to_string(),
                                                data: b64.to_string(),
                                            }),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        other => {
            let s = other.to_string();
            if !s.is_empty() {
                parts.push(VertexPart {
                    text: Some(s),
                    thought_signature: None,
                    function_call: None,
                    function_response: None,
                    inline_data: None,
                });
            }
        }
    }
    parts
}



pub fn build_vertex_request(req: &ChatCompletionRequest) -> Result<VertexRequest, String> {
    let mut system_parts = Vec::new();
    let mut raw_contents: Vec<VertexContent> = Vec::new();

    // Map tool_call_id to function name
    let mut tool_id_to_name: HashMap<String, String> = HashMap::new();
    for msg in &req.messages {
        if msg.role == "assistant" {
            if let Some(ref tcs) = msg.tool_calls {
                for tc in tcs {
                    tool_id_to_name.insert(tc.id.clone(), tc.function.name.clone());
                    // Also store the stripped id if it contains __thought__
                    if let Some((clean_id, _)) = tc.id.split_once("__thought__") {
                        tool_id_to_name.insert(clean_id.to_string(), tc.function.name.clone());
                    }
                }
            }
        }
    }

    for msg in &req.messages {
        if msg.role == "system" {
            system_parts.extend(parse_content_parts(&msg.content));
            continue;
        }

        if msg.role == "tool" {
            let name = if let Some(ref n) = msg.name {
                n.clone()
            } else if let Some(ref tid) = msg.tool_call_id {
                if let Some(mapped) = tool_id_to_name.get(tid) {
                    mapped.clone()
                } else if let Some((prefix, _)) = tid.split_once("__thought__") {
                    if let Some(mapped) = tool_id_to_name.get(prefix) {
                        mapped.clone()
                    } else if let Some(n) = prefix.split("__").nth(1) {
                        n.to_string()
                    } else {
                        prefix.strip_prefix("call_").unwrap_or(prefix).to_string()
                    }
                } else if let Some(n) = tid.split("__").nth(1) {
                    n.to_string()
                } else {
                    tid.strip_prefix("call_").unwrap_or(tid).to_string()
                }
            } else {
                "tool".to_string()
            };

            let response_val = match &msg.content {
                Some(serde_json::Value::Object(map)) => serde_json::Value::Object(map.clone()),
                Some(serde_json::Value::String(s)) => {
                    // Try parsing string as JSON object
                    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(s) {
                        serde_json::Value::Object(map)
                    } else {
                        serde_json::json!({ "result": s })
                    }
                }
                Some(val) => serde_json::json!({ "result": val }),
                None => serde_json::json!({ "result": "" }),
            };

            let part = VertexPart {
                text: None,
                thought_signature: None,
                function_call: None,
                function_response: Some(VertexFunctionResponse {
                    name,
                    response: response_val,
                }),
                inline_data: None,
            };

            raw_contents.push(VertexContent {
                role: "user".to_string(),
                parts: vec![part],
            });
            continue;
        }

        let role = if msg.role == "assistant" { "model" } else { "user" };
        let mut parts = Vec::new();

        if msg.role == "assistant" {
            if let Some(ref tcs) = msg.tool_calls {
                for tc in tcs {
                    let mut thought_sig: Option<String> = None;
                    let mut clean_call_id = tc.id.clone();

                    if let Some((left, sig)) = tc.id.split_once("__thought__") {
                        thought_sig = Some(sig.to_string());
                        if let Some((orig_id, _)) = left.split_once("__") {
                            clean_call_id = orig_id.to_string();
                        } else {
                            clean_call_id = left.to_string();
                        }
                    }

                    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                        .unwrap_or_else(|_| serde_json::json!({}));

                    parts.push(VertexPart {
                        text: None,
                        thought_signature: thought_sig,
                        function_call: Some(VertexFunctionCall {
                            name: tc.function.name.clone(),
                            args,
                            id: Some(clean_call_id),
                        }),
                        function_response: None,
                        inline_data: None,
                    });
                }
            }
        }

        let content_parts = parse_content_parts(&msg.content);
        parts.extend(content_parts);

        if !parts.is_empty() {
            raw_contents.push(VertexContent {
                role: role.to_string(),
                parts,
            });
        }
    }

    if raw_contents.is_empty() {
        return Err("No messages to send".to_string());
    }

    fn has_function_response(c: &VertexContent) -> bool {
        c.parts.iter().any(|p| p.function_response.is_some())
    }

    fn has_text(c: &VertexContent) -> bool {
        c.parts.iter().any(|p| p.text.is_some())
    }

    // Merge consecutive messages of the same role. A user turn containing a
    // functionResponse (tool result) must NOT be merged with a following user
    // turn containing fresh text: Vertex rejects a single user content that
    // mixes functionResponse with text ("Requests ending with a model turn are
    // not supported."). Keep them as separate user turns instead.
    let mut contents: Vec<VertexContent> = Vec::new();
    for content in raw_contents {
        if let Some(last) = contents.last_mut() {
            if last.role == content.role {
                let mixes_tool_result_with_text =
                    (has_function_response(last) && has_text(&content))
                        || (has_text(last) && has_function_response(&content));
                if !mixes_tool_result_with_text {
                    last.parts.extend(content.parts);
                    continue;
                }
            }
        }
        contents.push(content);
    }

    // Vertex requires the conversation to end with a user turn. If it ends with
    // a model turn (e.g. the client passed back a trailing assistant message),
    // append a minimal user turn so the model can continue generating.
    if let Some(last) = contents.last() {
        if last.role == "model" {
            contents.push(VertexContent {
                role: "user".to_string(),
                parts: vec![VertexPart {
                    text: Some("Continue.".to_string()),
                    thought_signature: None,
                    function_call: None,
                    function_response: None,
                    inline_data: None,
                }],
            });
            tracing::info!("appended user continuation turn after trailing model turn");
        }
    }

    let final_roles: Vec<&str> = contents.iter().map(|c| c.role.as_str()).collect();
    tracing::info!("vertex contents roles: {:?}", final_roles);

    let system_instruction = if !system_parts.is_empty() {
        Some(VertexSystemInstruction { parts: system_parts })
    } else {
        None
    };

    let tools = if let Some(ref ts) = req.tools {
        let decls: Vec<VertexFunctionDeclaration> = ts
            .iter()
            .map(|t| {
                let mut parameters = t.function.parameters.clone();
                if let Some(ref mut params) = parameters {
                    sanitize_schema(params);
                }
                VertexFunctionDeclaration {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    parameters,
                }
            })
            .collect();

        if !decls.is_empty() {
            Some(vec![VertexTool { function_declarations: decls }])
        } else {
            None
        }
    } else {
        None
    };

    let tool_config = if let Some(ref tc) = req.tool_choice {
        if let Some(s) = tc.as_str() {
            match s {
                "none" => Some(serde_json::json!({
                    "functionCallingConfig": { "mode": "NONE" }
                })),
                "auto" => Some(serde_json::json!({
                    "functionCallingConfig": { "mode": "AUTO" }
                })),
                "required" | "any" => Some(serde_json::json!({
                    "functionCallingConfig": { "mode": "ANY" }
                })),
                _ => None,
            }
        } else if let Some(obj) = tc.as_object() {
            if let Some(func_obj) = obj.get("function").and_then(|f| f.as_object()) {
                if let Some(name) = func_obj.get("name").and_then(|n| n.as_str()) {
                    Some(serde_json::json!({
                        "functionCallingConfig": {
                            "mode": "ANY",
                            "allowedFunctionNames": [name]
                        }
                    }))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let max_tokens = req.max_completion_tokens.or(req.max_tokens);
    let stop_sequences = req.stop.as_ref().map(|s| match s {
        StopCondition::Single(st) => vec![st.clone()],
        StopCondition::Multiple(vec) => vec.clone(),
    });

    let generation_config = if req.temperature.is_some()
        || max_tokens.is_some()
        || req.top_p.is_some()
        || stop_sequences.is_some()
    {
        Some(VertexGenerationConfig {
            temperature: req.temperature,
            max_output_tokens: max_tokens,
            top_p: req.top_p,
            stop_sequences,
        })
    } else {
        None
    };

    Ok(VertexRequest {
        contents,
        system_instruction,
        tools,
        tool_config,
        generation_config,
    })
}

pub fn vertex_response_to_openai(
    vertex_resp: VertexResponse,
    model: &str,
    created: u64,
) -> OpenAiChatCompletionResponse {
    let mut choices = Vec::new();

    for (idx, cand) in vertex_resp.candidates.into_iter().enumerate() {
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();

        if let Some(content) = cand.content {
            for part in content.parts {
                if let Some(txt) = part.text {
                    text_parts.push(txt);
                }
                if let Some(fc) = part.function_call {
                    let raw_id = fc.id.unwrap_or_else(|| format!("call_{}", fc.name));
                    let call_id = if let Some(sig) = part.thought_signature {
                        format!("{}__{}__thought__{}", raw_id, fc.name, sig)
                    } else {
                        raw_id
                    };

                    let args_str = serde_json::to_string(&fc.args).unwrap_or_else(|_| "{}".to_string());
                    tool_calls.push(OpenAiResponseToolCall {
                        id: call_id,
                        tool_type: "function",
                        function: OpenAiFunctionCall {
                            name: fc.name,
                            arguments: args_str,
                        },
                    });
                }
            }
        }

        let content_str = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join(""))
        };

        let finish_reason = if !tool_calls.is_empty() {
            Some("tool_calls".to_string())
        } else {
            match cand.finish_reason.as_deref() {
                Some("STOP") => Some("stop".to_string()),
                Some("MAX_TOKENS") => Some("length".to_string()),
                Some("SAFETY") => Some("content_filter".to_string()),
                Some(other) => Some(other.to_lowercase()),
                None => Some("stop".to_string()),
            }
        };

        choices.push(OpenAiChoice {
            index: idx as u32,
            message: OpenAiResponseMessage {
                role: "assistant",
                content: content_str,
                tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
            },
            finish_reason,
        });
    }

    if choices.is_empty() {
        choices.push(OpenAiChoice {
            index: 0,
            message: OpenAiResponseMessage {
                role: "assistant",
                content: Some(String::new()),
                tool_calls: None,
            },
            finish_reason: Some("stop".to_string()),
        });
    }

    let usage = if let Some(meta) = vertex_resp.usage_metadata {
        OpenAiUsage {
            prompt_tokens: meta.prompt_token_count.unwrap_or(0),
            completion_tokens: meta.candidates_token_count.unwrap_or(0),
            total_tokens: meta.total_token_count.unwrap_or(0),
        }
    } else {
        OpenAiUsage::default()
    };

    OpenAiChatCompletionResponse {
        id: format!("chatcmpl-vertex-{}", created),
        object: "chat.completion",
        created,
        model: model.to_string(),
        choices,
        usage,
    }
}
