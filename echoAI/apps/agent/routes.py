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
async def create_prompt(request: dict):
    """
    Create or update agent from prompt with template matching.

    Accepts a JSON body with:
    - prompt (str, required): Natural language description of the desired agent.
    - agent_id (str, optional): If provided, treats this as an UPDATE request.
    - name (str, optional): Override agent name.
    - icon (str, optional): Override icon.
    - role (str, optional): Override role.
    - description (str, optional): Override description.
    - tools (list, optional): Override tools list.
    - variables (list, optional): Override variables.
    - settings (dict, optional): Override settings.

    Response includes "action" field indicating what happened:
    - "CREATE_AGENT": New agent was created
    - "UPDATE_AGENT": Existing agent was updated (when agent_id provided)
    - "AGENT_EXISTS": Similar agent already exists (can be configured/modified)

    The service will:
    1. If agent_id provided: Update existing agent, preserving name/ID
    2. Analyze intent from the prompt.
    3. Check for existing similar agents (returns AGENT_EXISTS if found).
    4. Match against predefined templates.
    5. Build from template if matched, else use LLM generation.
    6. Register the agent in the registry.
    7. Return the full agent definition with action type.
    """
    try:
        prompt = request.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        agent_id = request.get("agent_id")
        service = svc()

        # UPDATE MODE: If agent_id is provided, treat as update request
        if agent_id:
            try:
                result = service.updateFromPrompt(agent_id, prompt)
                return result
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        # CREATE MODE: Check for existing similar agents first
        # Analyze intent to check for similar agents
        intent = service._analyze_intent(prompt)
        existing_match = service._check_existing_agents(intent, prompt)

        if existing_match:
            # Return existing agent info without creating new one
            return existing_match

        # Build AgentTemplate from request overrides
        template = AgentTemplate(
            name=request.get("name", ""),
            icon=request.get("icon"),
            role=request.get("role"),
            description=request.get("description"),
            prompt=request.get("system_prompt"),
            tools=request.get("tools"),
            variables=request.get("variables"),
            settings=request.get("settings"),
        )

        agent = service.createFromPrompt(prompt, template)

        # Return with action field
        response = agent.model_dump()
        response["action"] = "CREATE_AGENT"
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

# Agent Design
@router.post('/design/prompt')
async def design_agent_from_prompt(request: dict):
    """
    Design agent from natural language prompt.

    Accepts a JSON body with:
    - prompt (str, required): Natural language description
    - agent_id (str, optional): If provided, updates existing agent
    - model (str, optional): LLM model to use
    - icon (str, optional): Agent icon
    - tools (list, optional): Tools list
    - variables (list, optional): Variables list

    Response includes "action" field:
    - "CREATE_AGENT": New agent was designed
    - "UPDATE_AGENT": Existing agent was updated
    - "AGENT_EXISTS": Similar agent/template already exists (can be configured/modified)
    """
    try:
        designer = container.resolve('agent.designer')
        registry = container.resolve('agent.registry')

        # Extract request fields
        user_prompt = request.get("prompt", "")
        if not user_prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        agent_id = request.get("agent_id")

        # UPDATE MODE: If agent_id provided, update existing agent
        if agent_id:
            existing_agent = registry.get_agent(agent_id)
            if not existing_agent:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

            # Use update method that preserves name/ID
            updated_agent = designer.update_from_prompt(
                existing_agent=existing_agent,
                user_prompt=user_prompt
            )

            # Save updates to registry
            registry.update_agent(agent_id, updated_agent)

            return {
                "action": "UPDATE_AGENT",
                "agent_id": agent_id,
                "agent_name": updated_agent.get("name"),
                "agent": updated_agent
            }

        # CREATE MODE: Check for existing similar agents/templates first
        service = container.resolve('agent.service')
        intent = service._analyze_intent(user_prompt)
        existing_match = service._check_existing_agents(intent, user_prompt)

        if existing_match:
            # Return existing agent/template info without creating new one
            return existing_match

        # No existing match found - proceed to design new agent
        default_model = request.get("model", "mistral-nemo-12b")
        icon = request.get("icon", "")
        tools = request.get("tools", [])
        variables = request.get("variables", [])

        # Design agent
        agent = designer.design_from_prompt(
            user_prompt=user_prompt,
            default_model=default_model,
            icon=icon,
            tools=tools,
            variables=variables
        )

        # Register agent automatically
        registry.register_agent(agent)

        return {
            "action": "CREATE_AGENT",
            "agent": agent
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@router.get('/registry/master-list')
async def get_master_agent_list():
    """Get master agent list for workflow builder display."""
    try:
        registry = container.resolve('agent.registry')
        return registry.get_master_list()
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

@router.patch('/{agent_id}/schema')
async def update_agent_io_schema(agent_id: str, request: dict):
    """Update agent's input/output schema (for workflow integration)."""
    try:
        registry = container.resolve('agent.registry')

        input_schema = request.get("input_schema")
        output_schema = request.get("output_schema")

        updates = {}
        if input_schema is not None:
            updates["input_schema"] = input_schema
        if output_schema is not None:
            updates["output_schema"] = output_schema

        if not updates:
            raise HTTPException(status_code=400, detail="Provide input_schema or output_schema")

        updated = registry.update_agent(agent_id, updates)
        return {
            "agent_id": agent_id,
            "input_schema": updated.get("input_schema", []),
            "output_schema": updated.get("output_schema", [])
        }
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

# ==================== AGENT TEMPLATES ====================

@router.get('/templates/all')
async def get_all_agent_templates():
    """
    Get all agent templates (static + created agents).
    Combines static templates from JSON file with agents from registry.
    """
    try:
        import json
        from pathlib import Path

        registry = container.resolve('agent.registry')

        # Load static templates
        templates_path = Path(__file__).parent.parent / "storage" / "agent_templates.json"
        static_templates = []

        if templates_path.exists():
            with open(templates_path, encoding='utf-8') as f:
                data = json.load(f)
                static_templates = data.get("templates", [])

        # Get created agents from registry
        master_list = registry.get_master_list()
        created_agents = master_list.get("agents", [])

        # Format created agents as templates
        created_templates = []
        for agent in created_agents:
            created_templates.append({
                "name": agent.get("name", "Unnamed Agent"),
                "icon": agent.get("icon", "🤖"),
                "description": agent.get("description", ""),
                "role": agent.get("role", ""),
                "agent_id": agent.get("agent_id"),
                "source": "created"
            })

        return {
            "templates": static_templates,
            "created": created_templates,
            "total_templates": len(static_templates),
            "total_created": len(created_templates),
            "total": len(static_templates) + len(created_templates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/templates/static')
async def get_static_templates():
    """Get only static agent templates from JSON file."""
    try:
        import json
        from pathlib import Path

        templates_path = Path(__file__).parent.parent / "storage" / "agent_templates.json"

        if not templates_path.exists():
            return {"templates": [], "count": 0}

        with open(templates_path, encoding='utf-8') as f:
            data = json.load(f)
            templates = data.get("templates", [])

        return {
            "templates": templates,
            "count": len(templates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== INTENT CLASSIFICATION (LLM-based) ====================

@router.post('/classify-intent')
async def classify_user_intent(request: dict):
    """
    Classify user intent using LLM reasoning.

    This endpoint enables natural language understanding for conversational flows.
    Instead of pattern matching, it uses LLM to understand user intent.

    Request body:
    {
        "context": "name_confirmation" | "refinement" | "tool_selection" | "general",
        "suggested_value": "FinGenius Pro",  // The value being confirmed/modified
        "user_message": "oh yes this name is great",
        "conversation_history": [...]  // Optional: previous messages for context
    }

    Response:
    {
        "intent": "CONFIRMATION" | "MODIFICATION" | "REJECTION" | "CLARIFICATION",
        "confidence": 0.95,
        "reasoning": "User expressed approval...",
        "extracted_value": null | "new value if modification"
    }
    """
    try:
        service = container.resolve('agent.service')

        context = request.get("context", "general")
        suggested_value = request.get("suggested_value", "")
        user_message = request.get("user_message", "")
        conversation_history = request.get("conversation_history", [])

        if not user_message:
            raise HTTPException(status_code=400, detail="user_message is required")

        result = service.classify_user_intent(
            context=context,
            suggested_value=suggested_value,
            user_message=user_message,
            conversation_history=conversation_history
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
