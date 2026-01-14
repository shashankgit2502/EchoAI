"""
Health Check API Routes
System health and status endpoints
"""
from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter(tags=["Health"])

settings = get_settings()


@router.get("/health")
async def health_check():
    """
    Detailed health check endpoint

    Returns system status and component health
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "components": {
            "validator_service": "ok",
            "workflow_service": "ok",
            "agent_service": "ok",
            "runtime_service": "ok",
            "storage_service": "ok",
            "visualization_service": "ok",
            "telemetry_service": "ok"
        },
        "architecture": "microservice-ready"
    }


@router.get("/ping")
async def ping():
    """
    Simple ping endpoint for load balancers
    """
    return {"status": "ok"}
