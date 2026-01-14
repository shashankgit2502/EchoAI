"""
Workflow Orchestrator API
Microservice-ready FastAPI application

Architecture:
- Service Layer: Wraps core business logic
- Internal APIs: Component-to-component communication
- External APIs: User-facing endpoints
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

# Import external API routers
from app.api.routes import (
    health,
    validate,
    workflow,
    agent,
    runtime,
    visualize,
    telemetry
)

# Import internal API routers
from app.api.internal import (
    validator as internal_validator,
    workflow as internal_workflow,
    agent as internal_agent,
    runtime as internal_runtime,
    storage as internal_storage,
    visualize as internal_visualize
)

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Dynamic Workflow-Centric Multi-Agent System Builder - Microservice Ready"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# CUSTOM EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for request validation errors
    Provides clearer error messages for API consumers
    """
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        message = error["msg"]
        error_type = error["type"]

        errors.append({
            "field": field,
            "message": message,
            "type": error_type
        })

    logger.warning(f"Validation error on {request.url.path}: {errors}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Request validation failed. Please check the required fields and formats.",
            "details": errors
        }
    )


# ============================================================================
# INCLUDE EXTERNAL API ROUTERS (User-facing)
# ============================================================================

app.include_router(health.router)
app.include_router(validate.router)
app.include_router(workflow.router)
app.include_router(agent.router)
app.include_router(runtime.router)
app.include_router(visualize.router)
app.include_router(telemetry.router)


# ============================================================================
# INCLUDE INTERNAL API ROUTERS (Component-to-component)
# ============================================================================

app.include_router(internal_validator.router)
app.include_router(internal_workflow.router)
app.include_router(internal_agent.router)
app.include_router(internal_runtime.router)
app.include_router(internal_storage.router)
app.include_router(internal_visualize.router)


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "architecture": "microservice-ready",
        "endpoints": {
            "external": [
                "/health",
                "/validate",
                "/workflow",
                "/agent",
                "/runtime",
                "/visualize",
                "/telemetry"
            ],
            "internal": [
                "/api/internal/validator",
                "/api/internal/workflow",
                "/api/internal/agent",
                "/api/internal/runtime",
                "/api/internal/storage",
                "/api/internal/visualize"
            ]
        }
    }


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Storage path: {settings.STORAGE_BASE_PATH}")
    logger.info(f"Checkpointing: {settings.ENABLE_CHECKPOINTING}")
    logger.info(f"Architecture: Microservice-ready (Service Layer + Internal APIs)")
    logger.info("All service boundaries enforced")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info(f"Shutting down {settings.APP_NAME}")


# ============================================================================
# MAIN (for running directly)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
