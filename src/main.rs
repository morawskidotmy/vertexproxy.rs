mod auth;
mod convert;
mod handler;
mod types;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::routing::{get, post};
use axum::Router;
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

    info!("Starting Vertex Proxy (Rust) on {}:{}...", host, port);

    let auth = AuthManager::new(custom_project, None)?;
    info!("Target Vertex AI project: {}", auth.project_id());
    info!("Target Vertex AI location: {}", location);

    // Warm up OAuth token in background
    let auth_clone = auth.clone();
    tokio::spawn(async move {
        if let Err(e) = auth_clone.get_token().await {
            tracing::error!("Initial token acquisition failed: {}", e);
        } else {
            info!("Initial Google OAuth token acquired and ready.");
        }
    });

    let http_client = reqwest::Client::builder()
        .pool_max_idle_per_host(64)
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_nodelay(true)
        .timeout(Duration::from_secs(600))
        .build()?;

    let state = Arc::new(AppState {
        auth,
        location,
        http_client,
    });

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
