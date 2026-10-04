"""The HTTP API: `dearflow-gateway`."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig
from dearflow.gateway.deps import Gateway, resolve_api_key
from dearflow.gateway.routers import runs, threads
from dearflow.gateway.run_manager import RunManager
from dearflow.models import ModelFactory
from dearflow.persistence import Persistence


def create_app(app_config: AppConfig, *, model_factory: ModelFactory | None = None, persistence: Persistence | None = None) -> FastAPI:
    """Build the API. Storage follows `persistence.enabled`; pass `persistence` to override (tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = persistence or Persistence.from_config(app_config.persistence)
        factory = LeadAgentFactory(app_config, model_factory=model_factory, checkpointer=store.checkpointer)
        app.state.gateway = Gateway(
            config=app_config,
            persistence=store,
            factory=factory,
            runs=RunManager(retain_seconds=app_config.gateway.retain_seconds),
            api_key=resolve_api_key(app_config),
        )
        try:
            yield
        finally:
            factory.flush_memory()  # save pending memory updates
            if persistence is None:
                store.close()

    app = FastAPI(title="DearFlow Gateway", lifespan=lifespan)
    if app_config.gateway.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=app_config.gateway.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["Content-Location"],
        )

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        """Liveness: the process is up."""
        return {"status": "healthy", "service": "dearflow-gateway"}

    @app.get("/health/ready", tags=["health"])
    async def ready() -> dict:
        """Readiness: the database answers."""
        app.state.gateway.persistence.database.execute("SELECT 1")
        return {"status": "ready"}

    app.include_router(threads.router)
    app.include_router(runs.router)
    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the DearFlow HTTP gateway.")
    parser.add_argument("--config", default=os.environ.get("DEARFLOW_CONFIG", "config.yaml"))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)

    config = AppConfig.from_yaml(args.config)
    uvicorn.run(create_app(config), host=args.host or config.gateway.host, port=args.port or config.gateway.port)
    return 0
