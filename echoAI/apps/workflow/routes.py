from fastapi import APIRouter, Depends, HTTPException
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
async def design_from_prompt(prompt: str, default_llm: dict = None):
    """
    Design workflow from natural language prompt.
    Returns draft workflow + agent definitions.
    """
    try:
        designer = container.resolve('workflow.designer')
        workflow, agents = designer.design_from_prompt(prompt, default_llm)
        return {
            "workflow": workflow,
            "agents": agents
        }
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
