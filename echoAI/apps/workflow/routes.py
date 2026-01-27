from fastapi import APIRouter, Depends, HTTPException, Query, Body, Request
from echolib.di import container
from echolib.security import user_context
from echolib.types import *

from echolib.services import WorkflowService
from echolib.types import (
    Agent,
    Workflow,
    WorkflowValidationRequest,
    SaveFinalRequest,
    CloneWorkflowRequest,
    ExecuteWorkflowRequest
)

router = APIRouter(prefix='/workflows', tags=['WorkflowApi'])

def svc() -> WorkflowService:
    return container.resolve('workflow.service')

# ==================== EXISTING ROUTES (UNCHANGED) ====================

@router.post('/create/prompt')
async def create_prompt(prompt: str, agents: list[Agent]):
    """Simple workflow creation (existing API)."""
    return svc().createFromPrompt(prompt, agents).model_dump()

@router.post('/create/canvas')
async def create_canvas(canvasJSON: dict):
    """Canvas-based workflow creation (existing API)."""
    return svc().createFromCanvas(canvasJSON).model_dump()

@router.post('/validate')
async def validate(workflow: Workflow):
    """Simple workflow validation (existing API)."""
    return svc().validate(workflow).model_dump()

# ==================== NEW ORCHESTRATOR ROUTES ====================

# Workflow Design
@router.post('/design/prompt')
async def design_from_prompt(
    request: Request,
    prompt: str = Query(None, description="Natural language prompt for workflow design")
):
    """
    Design workflow from natural language prompt.
    Returns draft workflow + agent definitions.

    Accepts TWO formats:
    1. Query parameters: POST /workflows/design/prompt?prompt=...
    2. Request body: POST /workflows/design/prompt with {"prompt": "...", "default_llm": {...}}

    Query parameters take precedence over body if both are provided.
    """
    try:
        # Determine prompt and default_llm from either query params or body
        final_prompt = prompt  # from query param
        final_llm = None

        # If no query param, try to parse body
        if not final_prompt:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    final_prompt = body.get("prompt")
                    final_llm = body.get("default_llm")
            except Exception:
                # No valid JSON body, that's okay if we have query params
                pass

        if not final_prompt:
            raise HTTPException(
                status_code=400,
                detail="Prompt is required. Provide via query parameter (?prompt=...) or request body ({\"prompt\": \"...\"})"
            )

        designer = container.resolve('workflow.designer')
        workflow, agents = designer.design_from_prompt(final_prompt, final_llm)
        return {
            "workflow": workflow,
            "agents": agents
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        raise HTTPException(status_code=400, detail=error_detail)

@router.post('/build')
async def build_workflow_manual(request: dict):
    """
    Build workflow manually with agent I/O schema specification.
    Allows reusing existing agents and defining their I/O schemas for this workflow.

    Request format:
    {
      "workflow": { ... },
      "agent_schemas": {
        "agt_001": {
          "input_schema": ["sales_data"],
          "output_schema": ["analysis"]
        },
        "agt_002": {
          "input_schema": ["analysis"],
          "output_schema": ["report"]
        }
      },
      "update_base_agents": false  // Optional: update base agent definitions
    }
    """
    try:
        workflow = request.get("workflow")
        agent_schemas = request.get("agent_schemas", {})
        update_base = request.get("update_base_agents", False)

        if not workflow:
            raise HTTPException(status_code=400, detail="workflow field required")

        # If update_base_agents is True, update the agent registry
        if update_base and agent_schemas:
            registry = container.resolve('agent.registry')
            for agent_id, schemas in agent_schemas.items():
                try:
                    registry.update_agent(agent_id, schemas)
                except Exception as e:
                    print(f"Warning: Failed to update agent {agent_id}: {e}")

        # Store agent schema overrides in workflow metadata
        if agent_schemas:
            if "metadata" not in workflow:
                workflow["metadata"] = {}
            workflow["metadata"]["agent_schemas"] = agent_schemas

        return {"workflow": workflow, "agent_schemas": agent_schemas}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Validation
@router.post('/validate/draft')
async def validate_draft(req: WorkflowValidationRequest):
    """Validate draft workflow (sync only, before HITL)."""
    try:
        validator = container.resolve('workflow.validator')
        result = validator.validate_draft(req.workflow, req.agents)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/validate/final')
async def validate_final(req: WorkflowValidationRequest):
    """Validate workflow after HITL (full async validation)."""
    try:
        validator = container.resolve('workflow.validator')
        result = await validator.validate_final(req.workflow, req.agents)

        if result.is_valid():
            # Mark as validated
            req.workflow["status"] = "validated"
            req.workflow["validation"] = {
                "validated_at": "placeholder_timestamp",  # TODO: Add real timestamp
                "validation_hash": "placeholder_hash"  # TODO: Add real hash
            }

        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Draft Listing
@router.get('/draft/list')
async def list_draft_workflows():
    """List all workflows in draft folder."""
    try:
        storage = container.resolve('workflow.storage')
        workflows = storage.list_draft_workflows()
        return {"workflows": workflows, "total": len(workflows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Storage
@router.post('/temp/save')
async def save_temp(workflow: dict):
    """Save workflow as temp for testing."""
    try:
        if workflow.get("status") != "validated":
            raise HTTPException(
                status_code=400,
                detail="Workflow must be validated before saving as temp"
            )

        storage = container.resolve('workflow.storage')
        result = storage.save_workflow(workflow, state="temp")
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/{workflow_id}/temp')
async def load_temp(workflow_id: str):
    """Load temp workflow."""
    try:
        storage = container.resolve('workflow.storage')
        workflow = storage.load_workflow(workflow_id=workflow_id, state="temp")
        return workflow
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Temp workflow not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete('/{workflow_id}/temp')
async def delete_temp(workflow_id: str):
    """Delete temp workflow."""
    try:
        storage = container.resolve('workflow.storage')
        storage.delete_workflow(workflow_id, state="temp")
        return {"message": "Temp workflow deleted", "workflow_id": workflow_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/final/save')
async def save_final(req: SaveFinalRequest):
    """Save workflow as final (versioned, immutable)."""
    try:
        storage = container.resolve('workflow.storage')
        result = storage.save_final_workflow(req.workflow)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/{workflow_id}/final/{version}')
async def load_final(workflow_id: str, version: str):
    """Load specific final version."""
    try:
        storage = container.resolve('workflow.storage')
        workflow = storage.load_workflow(
            workflow_id=workflow_id,
            state="final",
            version=version
        )
        return workflow
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow version not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/{workflow_id}/versions')
async def list_versions(workflow_id: str):
    """List all final versions of a workflow."""
    try:
        storage = container.resolve('workflow.storage')
        versions = storage.list_versions(workflow_id)
        if not versions:
            raise HTTPException(status_code=404, detail="No versions found")
        return {
            "workflow_id": workflow_id,
            "versions": versions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/clone')
async def clone_final(req: CloneWorkflowRequest):
    """Clone final workflow to draft for editing."""
    try:
        storage = container.resolve('workflow.storage')
        cloned = storage.clone_final_to_draft(
            workflow_id=req.workflow_id,
            from_version=req.from_version
        )
        return {
            "message": "Workflow cloned to draft",
            "workflow_id": req.workflow_id,
            "base_version": req.from_version
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Final workflow not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Load workflow by ID (tries draft → temp → final)
@router.get('/{workflow_id}')
async def load_workflow(workflow_id: str):
    """Load a workflow by ID, searching across draft/temp/final states."""
    try:
        storage = container.resolve('workflow.storage')
        agent_registry = container.resolve('agent.registry')

        workflow = None
        for state in ["draft", "temp", "final"]:
            try:
                workflow = storage.load_workflow(workflow_id=workflow_id, state=state)
                if workflow:
                    break
            except (FileNotFoundError, Exception):
                continue

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Load associated agents
        agent_ids = workflow.get("agents", [])
        agents = agent_registry.get_agents_for_workflow(agent_ids)

        return {
            "workflow": workflow,
            "agents": agents
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Execution
@router.post('/execute')
async def execute_workflow(req: ExecuteWorkflowRequest):
    """Execute workflow (test or final mode)."""
    try:
        executor = container.resolve('workflow.executor')
        result = executor.execute_workflow(
            workflow_id=req.workflow_id,
            execution_mode=req.mode,
            version=req.version,
            input_payload=req.input_payload
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Visualization
@router.get('/{workflow_id}/graph')
async def get_graph(workflow_id: str, state: str = "temp"):
    """Get graph representation of workflow."""
    try:
        graph_mapper = container.resolve('workflow.graph_mapper')
        graph = graph_mapper.get_graph(workflow_id, state)
        return graph
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==================== NODE MAPPER ROUTES ====================

@router.post('/canvas/to-backend')
async def canvas_to_backend(request: dict):
    """
    Convert frontend canvas to backend workflow format.

    Request:
    {
      "canvas_nodes": [...],
      "connections": [...],
      "workflow_name": "Optional explicit name"
    }
    """
    try:
        node_mapper = container.resolve('workflow.node_mapper')

        canvas_nodes = request.get("canvas_nodes", [])
        connections = request.get("connections", [])
        workflow_name = request.get("workflow_name")

        if not canvas_nodes:
            raise HTTPException(status_code=400, detail="canvas_nodes required")

        workflow, agents = node_mapper.map_frontend_to_backend(
            canvas_nodes=canvas_nodes,
            connections=connections,
            workflow_name=workflow_name,
            auto_generate_name=True
        )

        return {
            "workflow": workflow,
            "agents": agents
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/backend/to-canvas')
async def backend_to_canvas(request: dict):
    """
    Convert backend workflow to frontend canvas format.

    Request:
    {
      "workflow": {...},
      "agents": {...}
    }
    """
    try:
        node_mapper = container.resolve('workflow.node_mapper')

        workflow = request.get("workflow")
        agents = request.get("agents")

        if not workflow or not agents:
            raise HTTPException(
                status_code=400,
                detail="Both workflow and agents required"
            )

        canvas_nodes, connections = node_mapper.map_backend_to_frontend(
            workflow=workflow,
            agents_dict=agents
        )

        return {
            "canvas_nodes": canvas_nodes,
            "connections": connections,
            "workflow_name": workflow.get("name", "Untitled Workflow")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/canvas/save')
async def save_canvas_workflow(request: dict):
    """
    Save canvas workflow directly (converts + validates + saves).

    Request:
    {
      "canvas_nodes": [...],
      "connections": [...],
      "workflow_name": "Optional",
      "save_as": "draft|temp",
      "workflow_id": "Optional - preserves existing ID if provided"
    }
    """
    try:
        node_mapper = container.resolve('workflow.node_mapper')
        validator = container.resolve('workflow.validator')
        storage = container.resolve('workflow.storage')
        agent_registry = container.resolve('agent.registry')

        # Convert to backend format (preserve existing workflow_id if provided)
        workflow, agents = node_mapper.map_frontend_to_backend(
            canvas_nodes=request.get("canvas_nodes", []),
            connections=request.get("connections", []),
            workflow_name=request.get("workflow_name"),
            execution_model=request.get("execution_model"),
            workflow_id=request.get("workflow_id")
        )

        # Validate
        validation_result = validator.validate_draft(workflow, agents)
        if not validation_result.is_valid():
            return {
                "success": False,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings
            }

        # Register agents
        for agent_id, agent in agents.items():
            agent["agent_id"] = agent_id
            agent_registry.register_agent(agent)

        # Save workflow
        save_as = request.get("save_as", "draft")
        if save_as == "temp":
            workflow["status"] = "validated"

        result = storage.save_workflow(workflow, state=save_as)

        return {
            "success": True,
            "workflow_id": workflow["workflow_id"],
            "workflow_name": workflow["name"],
            "state": save_as,
            **result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WORKFLOW CHAT (RUNTIME TESTING) ====================

@router.post('/chat/start')
async def start_chat_session(request: dict):
    """
    Start a new chat session for workflow testing.

    Request:
    {
      "workflow_id": "wf_xxx",
      "mode": "test",  // or "workflow_mode" for backward compatibility
      "version": null,
      "initial_context": {}  // or "context" for backward compatibility
    }
    """
    try:
        from .runtime.chat_session import ChatSessionManager

        session_manager = ChatSessionManager()

        workflow_id = request.get("workflow_id")
        # Accept both 'mode' and 'workflow_mode' for backward compatibility
        mode = request.get("mode") or request.get("workflow_mode", "test")
        version = request.get("version")
        # Accept both 'initial_context' and 'context' for backward compatibility
        initial_context = request.get("initial_context") or request.get("context", {})

        if not workflow_id:
            raise HTTPException(status_code=400, detail="workflow_id required")

        # Create session
        session = session_manager.create_session(
            workflow_id=workflow_id,
            workflow_mode=mode,
            workflow_version=version,
            initial_context=initial_context
        )

        return {
            "session_id": session.session_id,
            "workflow_id": session.workflow_id,
            "mode": session.workflow_mode,
            "created_at": session.created_at.isoformat(),
            "message": "Chat session started. Send messages to test workflow."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/chat/send')
async def send_chat_message(request: dict):
    """
    Send message and execute workflow.

    Request:
    {
      "session_id": "session_xxx",
      "message": "Analyze this feedback: Great product!",
      "execute_workflow": true
    }
    """
    try:
        from .runtime.chat_session import ChatSessionManager

        session_manager = ChatSessionManager()
        executor = container.resolve('workflow.executor')

        session_id = request.get("session_id")
        message = request.get("message")
        execute_workflow = request.get("execute_workflow", True)

        if not session_id or not message:
            raise HTTPException(
                status_code=400,
                detail="session_id and message required"
            )

        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Add user message
        session_manager.add_user_message(session_id, message)

        # Execute workflow if requested
        if execute_workflow:
            # Execute workflow with message as input
            result = executor.execute_workflow(
                workflow_id=session.workflow_id,
                execution_mode=session.workflow_mode,
                version=session.workflow_version,
                input_payload={
                    "message": message,
                    "context": session.context
                }
            )

            # Extract response from workflow output
            output = result.get("output", {})

            # Try to extract meaningful response from output
            if "message" in output:
                response_text = output["message"]
            elif "messages" in output and output["messages"]:
                # Get last message if messages array exists
                last_msg = output["messages"][-1]
                if isinstance(last_msg, dict):
                    response_text = last_msg.get("content", str(last_msg))
                else:
                    response_text = str(last_msg)
            else:
                # Fallback: return stringified output
                response_text = f"Workflow executed: {str(output)}"

            run_id = result.get("run_id")

            # Add assistant response
            session_manager.add_assistant_message(
                session_id=session_id,
                message=response_text,
                run_id=run_id,
                metadata={"execution_status": result.get("status")}
            )

            return {
                "session_id": session_id,
                "run_id": run_id,
                "status": result.get("status"),
                "response": response_text,
                "execution_result": result
            }
        else:
            # Just acknowledge message
            return {
                "session_id": session_id,
                "status": "message_added",
                "message": "Message added to session"
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/chat/history/{session_id}')
async def get_chat_history(session_id: str):
    """
    Get chat history for a session.

    Returns:
    {
      "session_id": "...",
      "workflow_id": "...",
      "messages": [...],
      "context": {...}
    }
    """
    try:
        from .runtime.chat_session import ChatSessionManager

        session_manager = ChatSessionManager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return session.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/chat/sessions')
async def list_chat_sessions(workflow_id: str = None, limit: int = 50):
    """
    List chat sessions.

    Query params:
    - workflow_id: Filter by workflow
    - limit: Max sessions to return
    """
    try:
        from .runtime.chat_session import ChatSessionManager

        session_manager = ChatSessionManager()
        sessions = session_manager.list_sessions(
            workflow_id=workflow_id,
            limit=limit
        )

        return {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "workflow_id": s.workflow_id,
                    "mode": s.workflow_mode,
                    "message_count": len(s.messages),
                    "created_at": s.created_at.isoformat(),
                    "last_activity": s.last_activity.isoformat()
                }
                for s in sessions
            ],
            "count": len(sessions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete('/chat/{session_id}')
async def delete_chat_session(session_id: str):
    """
    Delete a chat session.
    """
    try:
        from .runtime.chat_session import ChatSessionManager

        session_manager = ChatSessionManager()
        session_manager.delete_session(session_id)

        return {
            "success": True,
            "session_id": session_id,
            "message": "Session deleted"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== WORKFLOW SAVE (AFTER TESTING) ====================

@router.post('/save-final')
async def save_tested_workflow(request: dict):
    """
    Save workflow as final after testing.

    Request:
    {
      "workflow_id": "wf_xxx",
      "version": "1.0",
      "notes": "Tested and approved"
    }
    """
    try:
        storage = container.resolve('workflow.storage')

        workflow_id = request.get("workflow_id")
        version = request.get("version", "1.0")
        notes = request.get("notes", "")

        if not workflow_id:
            raise HTTPException(status_code=400, detail="workflow_id required")

        # Load temp workflow
        workflow = storage.load_workflow(
            workflow_id=workflow_id,
            state="temp"
        )

        # Update version and notes
        workflow["version"] = version
        workflow["metadata"]["save_notes"] = notes
        workflow["status"] = "validated"

        # Save as final
        result = storage.save_final_workflow(workflow)

        return {
            "success": True,
            "workflow_id": workflow_id,
            "version": version,
            "path": result["path"],
            "message": "Workflow saved as final"
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Temp workflow not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HITL (HUMAN-IN-THE-LOOP) ====================

@router.get('/hitl/status/{run_id}')
async def get_hitl_status(run_id: str):
    """
    Get HITL status for a run.

    Returns:
    {
      "run_id": "run_xxx",
      "state": "waiting_for_human",
      "blocked_at": "agent_id",
      "allowed_actions": ["approve", "reject", "modify", "defer"],
      "has_pending_review": true
    }
    """
    try:
        from .runtime.hitl import HITLManager

        hitl = HITLManager()
        status = hitl.get_status(run_id)

        return status

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/hitl/context/{run_id}')
async def get_hitl_context(run_id: str):
    """
    Get full HITL context for decision-making.

    Returns:
    {
      "workflow_id": "wf_xxx",
      "run_id": "run_xxx",
      "blocked_at": "agent_id",
      "agent_output": {...},
      "tools_used": [...],
      "execution_metrics": {...},
      "state_snapshot": {...},
      "previous_decisions": [...]
    }
    """
    try:
        from .runtime.hitl import HITLManager

        hitl = HITLManager()
        context = hitl.get_context(run_id)

        if not context:
            raise HTTPException(status_code=404, detail="HITL context not found")

        return context.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/hitl/approve')
async def approve_hitl(request: dict):
    """
    Approve workflow execution at HITL checkpoint.

    Request:
    {
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "rationale": "Looks good, proceed"
    }

    Response:
    {
      "action": "approve",
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "status": "approved",
      "can_resume": true
    }
    """
    try:
        from .runtime.hitl import HITLManager

        hitl = HITLManager()

        run_id = request.get("run_id")
        actor = request.get("actor", "unknown")
        rationale = request.get("rationale")

        if not run_id:
            raise HTTPException(status_code=400, detail="run_id required")

        result = hitl.approve(run_id=run_id, actor=actor, rationale=rationale)

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/hitl/reject')
async def reject_hitl(request: dict):
    """
    Reject workflow execution at HITL checkpoint.

    Request:
    {
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "rationale": "Output doesn't meet requirements"
    }

    Response:
    {
      "action": "reject",
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "status": "rejected",
      "can_resume": false,
      "terminated": true
    }
    """
    try:
        from .runtime.hitl import HITLManager

        hitl = HITLManager()

        run_id = request.get("run_id")
        actor = request.get("actor", "unknown")
        rationale = request.get("rationale")

        if not run_id:
            raise HTTPException(status_code=400, detail="run_id required")

        result = hitl.reject(run_id=run_id, actor=actor, rationale=rationale)

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/hitl/modify')
async def modify_hitl(request: dict):
    """
    Modify workflow/agent configuration at HITL checkpoint (TEMP workflows only).

    Request:
    {
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "changes": {
        "agent_id": "agt_001",
        "llm": {"temperature": 0.5},
        "prompt": "Updated prompt"
      },
      "rationale": "Need higher temperature for creativity"
    }

    Response:
    {
      "action": "modify",
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "status": "modified",
      "changes": {...},
      "validation_required": true,
      "can_resume": false
    }
    """
    try:
        from .runtime.hitl import HITLManager

        hitl = HITLManager()

        run_id = request.get("run_id")
        actor = request.get("actor", "unknown")
        changes = request.get("changes", {})
        rationale = request.get("rationale")

        if not run_id:
            raise HTTPException(status_code=400, detail="run_id required")

        if not changes:
            raise HTTPException(status_code=400, detail="changes required")

        result = hitl.modify(
            run_id=run_id,
            actor=actor,
            changes=changes,
            rationale=rationale
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/hitl/defer')
async def defer_hitl(request: dict):
    """
    Defer HITL decision (postpone approval).

    Request:
    {
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "defer_until": "2026-01-20T10:00:00Z",
      "rationale": "Need to review with team"
    }

    Response:
    {
      "action": "defer",
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "status": "deferred",
      "defer_until": "2026-01-20T10:00:00Z",
      "can_resume": false
    }
    """
    try:
        from .runtime.hitl import HITLManager
        from datetime import datetime

        hitl = HITLManager()

        run_id = request.get("run_id")
        actor = request.get("actor", "unknown")
        defer_until_str = request.get("defer_until")
        rationale = request.get("rationale")

        if not run_id:
            raise HTTPException(status_code=400, detail="run_id required")

        defer_until = None
        if defer_until_str:
            defer_until = datetime.fromisoformat(defer_until_str.replace('Z', '+00:00'))

        result = hitl.defer(
            run_id=run_id,
            actor=actor,
            defer_until=defer_until,
            rationale=rationale
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/hitl/pending')
async def list_pending_hitl_reviews():
    """
    List all workflows waiting for HITL review.

    Response:
    [
      {
        "run_id": "run_xxx",
        "workflow_id": "wf_xxx",
        "blocked_at": "agent_id",
        "created_at": "2026-01-17T10:00:00Z"
      }
    ]
    """
    try:
        from .runtime.hitl import HITLManager

        hitl = HITLManager()
        pending = hitl.list_pending_reviews()

        return {"pending_reviews": pending, "count": len(pending)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/hitl/decisions/{run_id}')
async def get_hitl_decisions(run_id: str):
    """
    Get HITL decision audit trail for a run.

    Response:
    [
      {
        "decision_id": "hitl_dec_xxx",
        "run_id": "run_xxx",
        "action": "approve",
        "actor": "user@example.com",
        "timestamp": "2026-01-17T10:05:00Z",
        "rationale": "Looks good"
      }
    ]
    """
    try:
        from .runtime.hitl import HITLManager

        hitl = HITLManager()
        decisions = hitl.get_decisions(run_id)

        return {
            "run_id": run_id,
            "decisions": [d.to_dict() for d in decisions],
            "count": len(decisions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
