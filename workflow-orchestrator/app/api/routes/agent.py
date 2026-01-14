"""
External Agent API Routes
User-facing agent management endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from app.schemas.api_models import (
    AgentDefinition,
    ListAgentsResponse
)
from app.services.agent_service import get_agent_service

router = APIRouter(prefix="/agent", tags=["Agent"])

# Get service instance
agent_service = get_agent_service()


@router.get("/list", response_model=ListAgentsResponse)
def list_agents() -> ListAgentsResponse:
    """
    List all registered agents
    """
    try:
        return agent_service.list_agents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}", response_model=Optional[AgentDefinition])
def get_agent(agent_id: str) -> Optional[AgentDefinition]:
    """
    Get agent by ID
    """
    try:
        agent = agent_service.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{agent_id}")
def delete_agent(agent_id: str) -> dict:
    """
    Delete agent from registry
    """
    try:
        success = agent_service.delete_agent(agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"success": True, "agent_id": agent_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
