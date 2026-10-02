use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stop: Option<StopCondition>,
    #[serde(default)]
    pub tools: Option<Vec<OpenAiTool>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub thinking: Option<serde_json::Value>,
    #[serde(default)]
    pub thinking_budget: Option<i32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StopCondition {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAiFunctionDef,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiFunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAiFunctionCall,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiFunctionCall {
    pub name: String,
    pub arguments: String,
}

// Vertex AI structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexRequest {
    pub contents: Vec<VertexContent>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "systemInstruction")]
    pub system_instruction: Option<VertexSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<VertexTool>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "toolConfig")]
    pub tool_config: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "generationConfig")]
    pub generation_config: Option<VertexGenerationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexContent {
    pub role: String,
    pub parts: Vec<VertexPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexSystemInstruction {
    pub parts: Vec<VertexPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "thoughtSignature")]
    pub thought_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "functionCall")]
    pub function_call: Option<VertexFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "functionResponse")]
    pub function_response: Option<VertexFunctionResponse>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "inlineData")]
    pub inline_data: Option<VertexInlineData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexFunctionCall {
    pub name: String,
    pub args: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexFunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexInlineData {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexTool {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<VertexFunctionDeclaration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexFunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxOutputTokens")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "stopSequences")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinkingConfig")]
    pub thinking_config: Option<VertexThinkingConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexThinkingConfig {
    #[serde(rename = "thinkingBudget")]
    pub thinking_budget: i32,
}

// Vertex responses

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct VertexResponse {
    #[serde(default)]
    pub candidates: Vec<VertexCandidate>,
    #[serde(default, rename = "usageMetadata")]
    pub usage_metadata: Option<VertexUsageMetadata>,
    #[serde(default, rename = "modelVersion")]
    pub model_version: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VertexCandidate {
    #[serde(default)]
    pub content: Option<VertexContentCandidate>,
    #[serde(default, rename = "finishReason")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct VertexContentCandidate {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub parts: Vec<VertexPartResponse>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VertexPartResponse {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub thought: Option<bool>,
    #[serde(default, rename = "thoughtSignature")]
    pub thought_signature: Option<String>,
    #[serde(default, rename = "functionCall")]
    pub function_call: Option<VertexFunctionCallResponse>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VertexFunctionCallResponse {
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Value,
    #[serde(default)]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VertexUsageMetadata {
    #[serde(default, rename = "promptTokenCount")]
    pub prompt_token_count: Option<u32>,
    #[serde(default, rename = "candidatesTokenCount")]
    pub candidates_token_count: Option<u32>,
    #[serde(default, rename = "totalTokenCount")]
    pub total_token_count: Option<u32>,
}

// OpenAI responses

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiChoice>,
    pub usage: OpenAiUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChoice {
    pub index: u32,
    pub message: OpenAiResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiResponseMessage {
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiResponseToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: &'static str,
    pub function: OpenAiFunctionCall,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct OpenAiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// Streaming chunk responses

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChunkResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiChunkChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChunkChoice {
    pub index: u32,
    pub delta: OpenAiChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct OpenAiChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiChunkToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChunkToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub tool_type: Option<&'static str>,
    pub function: OpenAiFunctionCall,
}

// Model list response

#[derive(Debug, Clone, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}
