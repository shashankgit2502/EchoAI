"""
Internal Agent API
Component-to-component agent operations
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from app.schemas.api_models import (
    CreateAgentRequest,
    CreateAgentResponse,
    ValidateAgentPermissionsRequest,
    ValidateAgentPermissionsResponse,
    ListAgentsResponse,
    AgentDefinition
)
from app.services.agent_service import get_agent_service

router = APIRouter(prefix="/api/internal/agent", tags=["Internal-Agent"])

# Get service instance
agent_service = get_agent_service()


@router.post("/create", response_model=CreateAgentResponse)
async def create_agent(request: CreateAgentRequest) -> CreateAgentResponse:
    """
    Internal API: Create runtime agent

    Used by runtime component during execution
    """
    try:
        result = await agent_service.create_runtime_agent(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-permissions", response_model=ValidateAgentPermissionsResponse)
def validate_permissions(request: ValidateAgentPermissionsRequest) -> ValidateAgentPermissionsResponse:
    """
    Internal API: Validate agent-to-agent permissions

    Used by runtime component before agent-to-agent calls
    """
    try:
        result = agent_service.validate_permissions(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=ListAgentsResponse)
def list_agents() -> ListAgentsResponse:
    """
    Internal API: List all agents
    """
    try:
        result = agent_service.list_agents()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}", response_model=Optional[AgentDefinition])
def get_agent(agent_id: str) -> Optional[AgentDefinition]:
    """
    Internal API: Get agent by ID
    """
    try:
        result = agent_service.get_agent(agent_id)
        if not result:
            raise HTTPException(status_code=404, detail="Agent not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
