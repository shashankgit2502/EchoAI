from fastapi import APIRouter, Depends, HTTPException
from echolib.di import container
from echolib.security import user_context
from echolib.types import *

from echolib.services import AgentService
from echolib.types import AgentTemplate, Agent

router = APIRouter(prefix='/agents', tags=['AgentApi'])

def svc() -> AgentService:
    return container.resolve('agent.service')

# ==================== EXISTING ROUTES (UNCHANGED) ====================

@router.post('/create/prompt')
async def create_prompt(prompt: str, template: AgentTemplate):
    """Simple agent creation from prompt (existing API)."""
    return svc().createFromPrompt(prompt, template).model_dump()

@router.post('/create/card')
async def create_card(cardJSON: dict, template: AgentTemplate):
    """Canvas-based agent creation (existing API)."""
    return svc().createFromCanvasCard(cardJSON, template).model_dump()

@router.post('/validate')
async def validate(agent: Agent):
    """Simple agent validation (existing API)."""
    return svc().validateA2A(agent).model_dump()

@router.get('/list')
async def list_agents():
    """List all agents (existing API)."""
    return [a.model_dump() for a in svc().listAgents()]

# ==================== NEW ORCHESTRATOR ROUTES ====================

# Agent Registry
@router.post('/register')
async def register_agent(agent: dict):
    """Register a new agent in the registry."""
    try:
        registry = container.resolve('agent.registry')
        result = registry.register_agent(agent)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/{agent_id}')
async def get_agent(agent_id: str):
    """Get agent by ID from registry."""
    try:
        registry = container.resolve('agent.registry')
        agent = registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/registry/list')
async def list_registered_agents():
    """List all agents in registry."""
    try:
        registry = container.resolve('agent.registry')
        agents = registry.list_agents()
        return {"agents": agents, "count": len(agents)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put('/{agent_id}')
async def update_agent(agent_id: str, updates: dict):
    """Update an existing agent."""
    try:
        registry = container.resolve('agent.registry')
        updated = registry.update_agent(agent_id, updates)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete('/{agent_id}')
async def delete_agent(agent_id: str):
    """Delete an agent from registry."""
    try:
        registry = container.resolve('agent.registry')
        registry.delete_agent(agent_id)
        return {"message": "Agent deleted", "agent_id": agent_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/role/{role}')
async def get_agents_by_role(role: str):
    """Get agents by role."""
    try:
        registry = container.resolve('agent.registry')
        agents = registry.get_agents_by_role(role)
        return {"role": role, "agents": agents, "count": len(agents)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Agent Factory
@router.post('/instantiate/{agent_id}')
async def instantiate_agent(agent_id: str, bind_tools: bool = True):
    """Create runtime agent instance from definition."""
    try:
        registry = container.resolve('agent.registry')
        factory = container.resolve('agent.factory')

        agent_def = registry.get_agent(agent_id)
        if not agent_def:
            raise HTTPException(status_code=404, detail="Agent not found")

        instance = factory.create_agent(agent_def, bind_tools=bind_tools)
        return instance
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/instantiate/batch')
async def instantiate_agents_batch(agent_ids: list):
    """Create multiple runtime agent instances."""
    try:
        registry = container.resolve('agent.registry')
        factory = container.resolve('agent.factory')

        agent_defs = registry.get_agents_for_workflow(agent_ids)
        instances = factory.create_agents_for_workflow(agent_defs)

        return {"instances": instances, "count": len(instances)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Agent Permissions
@router.post('/permissions/check')
async def check_permission(
    caller_id: str,
    target_id: str,
    workflow: dict,
    agents: dict
):
    """Check if caller can communicate with target agent."""
    try:
        permissions = container.resolve('agent.permissions')
        allowed = permissions.can_call_agent(
            caller_id=caller_id,
            target_id=target_id,
            workflow=workflow,
            agent_registry=agents
        )
        return {
            "caller_id": caller_id,
            "target_id": target_id,
            "allowed": allowed
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/permissions/validate')
async def validate_permissions(workflow: dict, agents: dict):
    """Validate all permissions in a workflow."""
    try:
        permissions = container.resolve('agent.permissions')
        errors = permissions.validate_workflow_permissions(workflow, agents)
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/permissions/targets/{agent_id}')
async def get_allowed_targets(agent_id: str, workflow: dict, agents: dict):
    """Get list of agents that the given agent can call."""
    try:
        permissions = container.resolve('agent.permissions')
        targets = permissions.get_allowed_targets(
            agent_id=agent_id,
            workflow=workflow,
            agent_registry=agents
        )
        return {
            "agent_id": agent_id,
            "allowed_targets": targets
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
