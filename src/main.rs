mod auth;
mod convert;
mod handler;
mod types;

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::routing::{get, post};
use axum::Router;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::auth::AuthManager;
use crate::handler::{chat_completions, health, list_models, AppState};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "vertex_proxy=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let host = std::env::var("VERTEX_PROXY_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("PORT")
        .or_else(|_| std::env::var("VERTEX_PROXY_PORT"))
        .unwrap_or_else(|_| "8000".to_string())
        .parse()
        .expect("Valid port number");

    let location = std::env::var("VERTEX_LOCATION").unwrap_or_else(|_| "global".to_string());
    let custom_project = std::env::var("VERTEX_PROJECT_ID").ok();

    // Auto-select the lowest-latency Vertex region unless VERTEX_LOCATION is
    // pinned by the user.
    let region_state: Arc<RwLock<String>> = Arc::new(RwLock::new(location.clone()));

    // DeepSeek (Model Garden custom endpoint) configuration
    let deepseek_endpoint_id = std::env::var("VERTEX_DEEPSEEK_ENDPOINT_ID")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let deepseek_location =
        std::env::var("VERTEX_DEEPSEEK_LOCATION").unwrap_or_else(|_| "us-central1".to_string());
    let deepseek_model = std::env::var("VERTEX_DEEPSEEK_MODEL")
        .unwrap_or_else(|_| "deepseek-v4.1-flash".to_string());

    info!("Starting Vertex Proxy (Rust) on {}:{}...", host, port);

    let auth = AuthManager::new(custom_project, None)?;
    info!("Target Vertex AI project: {}", auth.project_id());
    if let Some(endpoint) = &deepseek_endpoint_id {
        info!(
            "DeepSeek endpoint configured: {} (location {}, model {})",
            endpoint, deepseek_location, deepseek_model
        );
    } else {
        info!("DeepSeek endpoint not configured (set VERTEX_DEEPSEEK_ENDPOINT_ID to enable deepseek-v4.1-flash).");
    }

    // Warm up OAuth token in background and keep it refreshed
    let auth_clone = auth.clone();
    tokio::spawn(async move {
        loop {
            match auth_clone.get_token().await {
                Ok(_) => {
                    // Refresh 10 minutes before typical 1-hour expiry
                    tokio::time::sleep(Duration::from_secs(50 * 60)).await;
                }
                Err(e) => {
                    tracing::error!("Background token refresh failed: {}", e);
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            }
        }
    });

    let http_client = reqwest::Client::builder()
        .pool_max_idle_per_host(256)
        .pool_idle_timeout(Duration::from_secs(120))
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .tcp_nodelay(true)
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(600))
        .build()?;

    let state = Arc::new(AppState {
        auth,
        http_client,
        region: region_state,
        failed_regions: Arc::new(RwLock::new(HashMap::new())),
        deepseek_endpoint_id,
        deepseek_location,
        deepseek_model,
    });

    // Continuously probe Vertex locations and pick the lowest latency one
    // that actually serves the model (unless VERTEX_LOCATION is pinned).
    if std::env::var("VERTEX_LOCATION").is_err() {
        let probe_state = state.clone();
        let probe_target = state.region.clone();
        tokio::spawn(async move {
            // Probe immediately, then re-probe every 60s.
            loop {
                probe_and_select(&probe_state, &probe_target).await;
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        });
    } else {
        info!(
            "VERTEX_LOCATION is pinned: auto region selection disabled (using {})",
            state.region.read().await.clone()
        );
    }

    let app = Router::new()
        .route("/", get(health))
        .route("/health", get(health))
        .route("/health/liveliness", get(health))
        .route("/v1/models", get(list_models))
        .route("/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Vertex Proxy listening on http://{}", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Vertex Proxy server stopped.");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

/// Candidate Vertex AI locations for the latency prober.
const VERTEX_REGIONS: &[&str] = &[
    "global",
    "europe-central2",
    "europe-west1",
    "europe-west3",
    "europe-west4",
    "us-central1",
    "us-east1",
    "us-west1",
];

/// Measures the round-trip to a region's aiplatform endpoint using a DNS +
/// TCP-connect probe (fast, free, and robust under load).
async fn probe_latency(region: &str) -> Option<f64> {
    let host = if region == "global" {
        "aiplatform.googleapis.com".to_string()
    } else {
        format!("{}-aiplatform.googleapis.com", region)
    };

    let t0 = Instant::now();
    let addr = tokio::net::lookup_host((host.as_str(), 443))
        .await
        .ok()?
        .next()?;
    let _stream = tokio::net::TcpStream::connect(addr).await.ok()?;
    Some(t0.elapsed().as_secs_f64() * 1000.0)
}

/// Excluded regions stay out of the candidate pool for this long (30 min).
const REGION_FAIL_COOLDOWN: Duration = Duration::from_secs(1800);

/// Probes all candidate regions concurrently and stores the fastest one,
/// skipping regions known not to serve the requested models.
pub(crate) async fn probe_and_select(
    state: &Arc<AppState>,
    region_state: &Arc<RwLock<String>>,
) {
    // Compute the currently-excluded regions (recent failures only).
    let now = Instant::now();
    let excluded: HashSet<String> = {
        let failed = state.failed_regions.read().await;
        failed
            .iter()
            .filter(|(_, when)| now.duration_since(**when) < REGION_FAIL_COOLDOWN)
            .map(|(region, _)| region.clone())
            .collect()
    };

    let futures: Vec<_> = VERTEX_REGIONS
        .iter()
        .filter(|region| !excluded.contains(**region))
        .map(|region| {
            let region = region.to_string();
            async move {
                match tokio::time::timeout(
                    Duration::from_secs(3),
                    probe_latency(&region),
                )
                .await
                {
                    Ok(Some(ms)) => Some((region, ms)),
                    _ => None,
                }
            }
        })
        .collect();

    let results: Vec<Option<(String, f64)>> = futures::future::join_all(futures).await;
    let best = results
        .into_iter()
        .flatten()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((name, ms)) = best {
        let mut guard = region_state.write().await;
        if *guard != name {
            info!(
                "Auto-selected lowest-latency Vertex region: {} ({:.0}ms)",
                name, ms
            );
            *guard = name.clone();
        }
    } else {
        tracing::warn!("Region probe returned no reachable location; keeping current region");
    }
}
