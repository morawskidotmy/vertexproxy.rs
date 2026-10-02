use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::Deserialize;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

#[derive(Debug, Deserialize)]
struct AdcFile {
    client_id: Option<String>,
    client_secret: Option<String>,
    refresh_token: Option<String>,
    quota_project_id: Option<String>,
    #[serde(rename = "type")]
    cred_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: Option<u64>,
}

#[derive(Clone)]
struct CachedToken {
    access_token: String,
    expires_at: Instant,
}

#[derive(Clone)]
pub struct AuthManager {
    client_id: String,
    client_secret: String,
    refresh_token: String,
    project_id: String,
    cached_token: Arc<RwLock<Option<CachedToken>>>,
    http_client: reqwest::Client,
}

impl AuthManager {
    pub fn new(
        custom_project_id: Option<String>,
        custom_adc_path: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let adc_path = if let Some(p) = custom_adc_path {
            p.to_path_buf()
        } else if let Ok(env_path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
            PathBuf::from(env_path)
        } else {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".config/gcloud/application_default_credentials.json")
        };

        if !adc_path.exists() {
            return Err(format!(
                "gcloud ADC file not found at: {}. Please run: gcloud auth application-default login",
                adc_path.display()
            ).into());
        }

        let content = std::fs::read_to_string(&adc_path)?;
        let adc: AdcFile = serde_json::from_str(&content)?;

        if adc.cred_type.as_deref() == Some("service_account") {
            warn!("Service account ADC found. For best compatibility, user ADC from 'gcloud auth application-default login' is recommended.");
        }

        let client_id = adc.client_id.ok_or("ADC is missing client_id")?;
        let client_secret = adc.client_secret.ok_or("ADC is missing client_secret")?;
        let refresh_token = adc.refresh_token.ok_or("ADC is missing refresh_token")?;

        let project_id = custom_project_id
            .or_else(|| std::env::var("VERTEX_PROJECT_ID").ok())
            .or(adc.quota_project_id)
            .unwrap_or_else(|| "project-d86e1c0a-02e9-4a72-8ff".to_string());

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_nodelay(true)
            .build()?;

        info!(
            "ADC loaded successfully from {}. Using project_id: {}",
            adc_path.display(),
            project_id
        );

        Ok(Self {
            client_id,
            client_secret,
            refresh_token,
            project_id,
            cached_token: Arc::new(RwLock::new(None)),
            http_client,
        })
    }

    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    pub async fn get_token(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Fast path: read lock
        {
            let guard = self.cached_token.read().await;
            if let Some(ref tok) = *guard {
                if Instant::now() + Duration::from_secs(60) < tok.expires_at {
                    return Ok(tok.access_token.clone());
                }
            }
        }

        // Slow path: write lock + refresh
        let mut guard = self.cached_token.write().await;
        // Double check after acquiring write lock
        if let Some(ref tok) = *guard {
            if Instant::now() + Duration::from_secs(60) < tok.expires_at {
                return Ok(tok.access_token.clone());
            }
        }

        info!("Refreshing Google Cloud OAuth2 access token...");
        let params = [
            ("client_id", self.client_id.as_str()),
            ("client_secret", self.client_secret.as_str()),
            ("refresh_token", self.refresh_token.as_str()),
            ("grant_type", "refresh_token"),
        ];

        let resp = self
            .http_client
            .post("https://oauth2.googleapis.com/token")
            .form(&params)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!("Failed to refresh OAuth token: status {} body {}", status, body);
            return Err(format!("OAuth token refresh failed (HTTP {}): {}", status, body).into());
        }

        let token_resp: TokenResponse = resp.json().await?;
        let expires_in = token_resp.expires_in.unwrap_or(3600);
        let expires_at = Instant::now() + Duration::from_secs(expires_in);

        *guard = Some(CachedToken {
            access_token: token_resp.access_token.clone(),
            expires_at,
        });

        info!("OAuth2 token refreshed successfully, valid for {}s", expires_in);
        Ok(token_resp.access_token)
    }
}
