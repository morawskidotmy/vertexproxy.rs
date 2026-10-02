use std::collections::HashMap;
use crate::types::*;

pub fn clean_model_name(raw: &str) -> &str {
    let s = raw.strip_prefix("vertex_ai/").unwrap_or(raw);
    let s = s.strip_prefix("google/").unwrap_or(s);
    let s = s.strip_prefix("vertex/").unwrap_or(s);
    let s = s.strip_prefix("deepseek-ai/").unwrap_or(s);
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

fn looks_like_base64(s: &str) -> bool {
    let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    clean.len() >= 64 && clean.chars().all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=')
}

pub fn parse_image_data(val: &serde_json::Value) -> Option<VertexInlineData> {
    // 1. Direct inlineData / inline_data object
    let inline_obj = val.get("inlineData").or_else(|| val.get("inline_data"));
    if let Some(inline) = inline_obj {
        if let (Some(mime), Some(data)) = (
            inline.get("mimeType").or_else(|| inline.get("mime_type")).and_then(|m| m.as_str()),
            inline.get("data").and_then(|d| d.as_str()),
        ) {
            let clean: String = data.chars().filter(|c| !c.is_whitespace()).collect();
            if !clean.is_empty() {
                return Some(VertexInlineData {
                    mime_type: mime.to_string(),
                    data: clean,
                });
            }
        }
    }

    // 2. Anthropic source object: { "source": { "type": "base64", "media_type": "...", "data": "..." } }
    if let Some(source) = val.get("source") {
        if let (Some(mime), Some(data)) = (
            source.get("media_type").or_else(|| source.get("mime_type")).and_then(|m| m.as_str()),
            source.get("data").and_then(|d| d.as_str()),
        ) {
            let clean: String = data.chars().filter(|c| !c.is_whitespace()).collect();
            if !clean.is_empty() {
                return Some(VertexInlineData {
                    mime_type: mime.to_string(),
                    data: clean,
                });
            }
        }
    }

    // 3. String URL or data URI from image_url, image, url, or the value itself
    let url_str = val.get("image_url")
        .and_then(|u| u.as_str().or_else(|| u.get("url").and_then(|v| v.as_str())))
        .or_else(|| val.get("image").and_then(|u| u.as_str().or_else(|| u.get("url").and_then(|v| v.as_str()))))
        .or_else(|| val.get("url").and_then(|u| u.as_str()))
        .or_else(|| val.as_str());

    if let Some(url) = url_str {
        if url.starts_with("data:") {
            if let Some((header, b64)) = url.split_once(',') {
                let mime = header
                    .split(';')
                    .next()
                    .and_then(|h| h.strip_prefix("data:"))
                    .unwrap_or("image/png")
                    .trim();
                let clean: String = b64.chars().filter(|c| !c.is_whitespace()).collect();
                if !clean.is_empty() {
                    return Some(VertexInlineData {
                        mime_type: if mime.is_empty() { "image/png".to_string() } else { mime.to_string() },
                        data: clean,
                    });
                }
            }
        } else if looks_like_base64(url) {
            let clean: String = url.chars().filter(|c| !c.is_whitespace()).collect();
            return Some(VertexInlineData {
                mime_type: "image/png".to_string(),
                data: clean,
            });
        }
    }

    None
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
                if s.starts_with("data:image/") || s.starts_with("data:application/pdf") {
                    if let Some(inline) = parse_image_data(val) {
                        parts.push(VertexPart {
                            text: None,
                            thought_signature: None,
                            function_call: None,
                            function_response: None,
                            inline_data: Some(inline),
                        });
                        return parts;
                    }
                }
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
                    } else if item_type == "image_url" || item_type == "image" || item_type == "input_image" {
                        if let Some(inline) = parse_image_data(item) {
                            parts.push(VertexPart {
                                text: None,
                                thought_signature: None,
                                function_call: None,
                                function_response: None,
                                inline_data: Some(inline),
                            });
                        }
                    } else if let Some(inline) = parse_image_data(item) {
                        parts.push(VertexPart {
                            text: None,
                            thought_signature: None,
                            function_call: None,
                            function_response: None,
                            inline_data: Some(inline),
                        });
                    } else if let Some(text) = obj.get("text").and_then(|t| t.as_str()) {
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
                } else if let Some(inline) = parse_image_data(item) {
                    parts.push(VertexPart {
                        text: None,
                        thought_signature: None,
                        function_call: None,
                        function_response: None,
                        inline_data: Some(inline),
                    });
                }
            }
        }
        other => {
            if let Some(inline) = parse_image_data(other) {
                parts.push(VertexPart {
                    text: None,
                    thought_signature: None,
                    function_call: None,
                    function_response: None,
                    inline_data: Some(inline),
                });
            } else {
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

            let mut tool_images = Vec::new();
            if let Some(serde_json::Value::Array(arr)) = &msg.content {
                for item in arr {
                    if let Some(inline) = parse_image_data(item) {
                        tool_images.push(inline);
                    }
                }
            }

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

            for inline in tool_images {
                raw_contents.push(VertexContent {
                    role: "user".to_string(),
                    parts: vec![VertexPart {
                        text: None,
                        thought_signature: None,
                        function_call: None,
                        function_response: None,
                        inline_data: Some(inline),
                    }],
                });
            }
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

        // Filter out empty text parts
        parts.retain(|p| {
            p.inline_data.is_some()
                || p.function_call.is_some()
                || p.function_response.is_some()
                || p.text.as_ref().map(|t| !t.trim().is_empty()).unwrap_or(false)
        });

        let is_last_msg = msg as *const _ == req.messages.last().map(|m| m as *const _).unwrap_or(std::ptr::null());
        if parts.is_empty() && is_last_msg && msg.role == "user" {
            parts.push(VertexPart {
                text: Some("Continue.".to_string()),
                thought_signature: None,
                function_call: None,
                function_response: None,
                inline_data: None,
            });
        }

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

    fn has_non_function_response(c: &VertexContent) -> bool {
        c.parts.iter().any(|p| p.function_response.is_none())
    }

    // Merge consecutive messages of the same role. A user turn containing a
    // functionResponse (tool result) must NOT be merged with a user turn
    // containing text or images: Vertex rejects a single user content that
    // mixes functionResponse with other parts ("Requests ending with a model turn are
    // not supported."). Keep them as separate user turns instead.
    let mut contents: Vec<VertexContent> = Vec::new();
    for content in raw_contents {
        if let Some(last) = contents.last_mut() {
            if last.role == content.role {
                let mixes_tool_result =
                    (has_function_response(last) && has_non_function_response(&content))
                        || (has_non_function_response(last) && has_function_response(&content));
                if !mixes_tool_result {
                    last.parts.extend(content.parts);
                    continue;
                }
            }
        }
        contents.push(content);
    }

    // Vertex requires the conversation to end with a valid user turn with content.
    // If it ends with a model turn, or the user turn has no valid parts, guarantee
    // a valid user continuation so Vertex never fails with:
    // "Requests ending with a model turn are not supported."
    let needs_continuation = match contents.last() {
        None => true,
        Some(last) => {
            if last.role != "user" {
                true
            } else {
                let has_valid_content = last.parts.iter().any(|p| {
                    if let Some(ref t) = p.text {
                        !t.trim().is_empty()
                    } else {
                        p.inline_data.is_some() || p.function_response.is_some() || p.function_call.is_some()
                    }
                });
                !has_valid_content
            }
        }
    };

    if needs_continuation {
        if let Some(last) = contents.last_mut() {
            if last.role == "user" && !has_function_response(last) {
                last.parts.retain(|p| {
                    p.inline_data.is_some()
                        || p.function_response.is_some()
                        || p.function_call.is_some()
                        || p.text.as_ref().map(|t| !t.trim().is_empty()).unwrap_or(false)
                });
                last.parts.push(VertexPart {
                    text: Some("Continue.".to_string()),
                    thought_signature: None,
                    function_call: None,
                    function_response: None,
                    inline_data: None,
                });
            } else {
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
            }
        } else {
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
        }
        tracing::info!("ensured conversation ends with a valid non-empty user turn");
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

    let model_lower = req.model.to_lowercase();
    let is_flash = model_lower.contains("flash");
    let is_pro = model_lower.contains("pro");

    // 1. Check reasoning_effort
    let effort_budget = req.reasoning_effort.as_deref().and_then(|e| match e.to_lowercase().as_str() {
        "none" => Some(0),
        "low" => Some(1024),
        "medium" => Some(4096),
        "high" => Some(8192),
        _ => None,
    });

    // 2. Check thinking parameter (OpenAI / Anthropic format)
    let explicit_budget = effort_budget
        .or_else(|| {
            req.thinking.as_ref().and_then(|v| match v {
                serde_json::Value::Object(map) => {
                    if map.get("type").and_then(|t| t.as_str()) == Some("disabled") {
                        Some(0)
                    } else {
                        map.get("thinkingBudget")
                            .or_else(|| map.get("budget_tokens"))
                            .and_then(|b| b.as_i64().map(|n| n as i32))
                    }
                }
                serde_json::Value::Number(n) => n.as_i64().map(|n| n as i32),
                _ => None,
            })
        })
        .or(req.thinking_budget);

    // 3. Check environment variable VERTEX_THINKING_BUDGET
    let env_budget = std::env::var("VERTEX_THINKING_BUDGET")
        .ok()
        .and_then(|s| s.parse::<i32>().ok());

    let final_budget: Option<i32> = if let Some(b) = explicit_budget {
        Some(b)
    } else if let Some(b) = env_budget {
        if b < 0 {
            None
        } else {
            Some(b)
        }
    } else if is_flash {
        // Flash models are built to be fast.
        // Google defaults them to thinking mode, adding 3-5 seconds of silent latency.
        // Default thinkingBudget to 0 for all Flash models so they stream in sub-second time.
        Some(0)
    } else {
        None
    };

    let thinking_config = if let Some(mut budget) = final_budget {
        if is_pro && budget < 128 {
            if budget == 0 {
                None // Pro models reject 0 with HTTP 400
            } else {
                budget = 128;
                Some(VertexThinkingConfig {
                    thinking_budget: budget,
                    include_thoughts: Some(true),
                })
            }
        } else if budget > 0 {
            Some(VertexThinkingConfig {
                thinking_budget: budget,
                include_thoughts: Some(true),
            })
        } else {
            Some(VertexThinkingConfig {
                thinking_budget: 0,
                include_thoughts: None,
            })
        }
    } else {
        // Dynamic thinking on Gemini models:
        // Vertex AI requires includeThoughts: true in thinkingConfig to actually emit the thoughts!
        Some(VertexThinkingConfig {
            thinking_budget: -1,
            include_thoughts: Some(true),
        })
    };

    let generation_config = if req.temperature.is_some()
        || max_tokens.is_some()
        || req.top_p.is_some()
        || stop_sequences.is_some()
        || thinking_config.is_some()
    {
        Some(VertexGenerationConfig {
            temperature: req.temperature,
            max_output_tokens: max_tokens,
            top_p: req.top_p,
            stop_sequences,
            thinking_config,
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
