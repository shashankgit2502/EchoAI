"""
Workflow Service
Service layer wrapper for workflow design, compilation, and lifecycle operations
Provides clean interface for internal API communication

Uses LangGraph v1 patterns (2025):
- StateGraph compilation
- Async execution (ainvoke, astream)
- Durable execution with checkpointing
- Human-in-the-loop built-in
"""
from typing import Optional, Tuple
from app.schemas.api_models import (
    UserRequest,
    AgentSystemDesign,
    DomainAnalysis,
    CompileWorkflowRequest,
    CompileWorkflowResponse,
    ModifyWorkflowRequest,
    VersionWorkflowRequest,
    VersionWorkflowResponse
)
from app.workflow.designer import WorkflowDesigner
from app.workflow.compiler import WorkflowCompiler
from app.workflow.versioning import bump_version, parse_version
from app.core.config import get_settings
from app.services.llm_provider import get_llm_provider
from app.core.logging import get_logger

logger = get_logger(__name__)


class WorkflowService:
    """
    Service layer for workflow operations

    Enforces service boundaries:
    - No direct imports in API layer
    - DTO-based request/response
    - Async + idempotent
    - Ready for microservice extraction

    Responsibilities:
    - Workflow design (LLM-based)
    - Workflow modification (HITL)
    - Workflow compilation (JSON → LangGraph StateGraph)
    - Version management

    LangGraph v1 Integration:
    - Uses StateGraph for compilation
    - Supports async execution (ainvoke/astream)
    - Checkpointing for durable execution
    - Built-in HITL support
    """

    def __init__(self):
        """Initialize workflow service with LangGraph v1 components"""
        self._designer = WorkflowDesigner()
        self._compiler = WorkflowCompiler()
        self._settings = get_settings()
        self._llm_provider = get_llm_provider()
        logger.info("WorkflowService initialized (LangGraph v1)")

    async def design_from_user_request(
        self,
        user_request: UserRequest
    ) -> Tuple[AgentSystemDesign, DomainAnalysis, str]:
        """
        Design workflow from natural language request

        Uses LangChain v1 LLMs with LangGraph orchestration

        Args:
            user_request: User's natural language request

        Returns:
            Tuple of (agent_system, domain_analysis, meta_prompt_used)
        """
        logger.info(f"Designing workflow from user request")

        agent_system, analysis, meta_prompt = await self._designer.design_from_user_request(
            user_request
        )

        logger.info(f"Workflow designed: {agent_system.system_name}")
        return agent_system, analysis, meta_prompt

    async def modify_agent_system(
        self,
        request: ModifyWorkflowRequest
    ) -> AgentSystemDesign:
        """
        Modify existing agent system (HITL)

        Supports human-in-the-loop workflow modifications

        Args:
            request: Modification request with agent system and instructions

        Returns:
            Modified AgentSystemDesign
        """
        logger.info(f"Modifying agent system: {request.agent_system.system_name}")

        model_override = self._extract_model_override(request.modification_request)
        if model_override:
            logger.info(f"Applying model override: {model_override}")
            return self._apply_model_override(request.agent_system, model_override)

        modified_system = await self._designer.modify_agent_system(
            request.agent_system,
            request.modification_request
        )

        logger.info(f"Agent system modified: {modified_system.system_name}")
        return modified_system

    def _extract_model_override(self, message: str) -> Optional[str]:
        """
        Detect explicit model change requests in plain language.

        Returns:
            Model ID if detected, otherwise None.
        """
        if not message:
            return None

        lower = message.lower()
        if "model" not in lower and "llm" not in lower and "nvidia" not in lower:
            return None

        # Look for exact model IDs from catalog
        for model_meta in self._llm_provider.list_available_models():
            if model_meta.id.lower() in lower:
                return model_meta.id

        # Heuristic for NVIDIA/Nemotron request
        if "nvidia" in lower or "nemotron" in lower:
            return self._settings.DEFAULT_LLM_MODEL

        return None

    def _apply_model_override(
        self,
        agent_system: AgentSystemDesign,
        model_id: str
    ) -> AgentSystemDesign:
        """Apply model override to all agents while preserving other settings."""
        for agent in agent_system.agents:
            agent.llm_config.model = model_id
        return agent_system

    async def compile_workflow(
        self,
        request: CompileWorkflowRequest
    ) -> CompileWorkflowResponse:
        """
        Compile workflow JSON to LangGraph StateGraph

        LangGraph v1 compilation:
        - Creates StateGraph with custom state
        - Adds nodes for each agent
        - Adds edges (sequential/parallel/conditional)
        - Compiles to executable graph

        Args:
            request: Compilation request with agent system

        Returns:
            CompileWorkflowResponse with compilation status
        """
        logger.info(f"Compiling workflow to LangGraph StateGraph: {request.agent_system.system_name}")

        try:
            # Select workflow to compile
            workflow_name = request.workflow_name
            if workflow_name is None:
                # Use first workflow if not specified
                if not request.agent_system.workflows:
                    return CompileWorkflowResponse(
                        success=False,
                        workflow_name="",
                        graph_compiled=False,
                        error="No workflows found in agent system"
                    )
                workflow_name = request.agent_system.workflows[0].name

            # Compile to LangGraph StateGraph (v1)
            # Returns compiled graph ready for execution with:
            # - ainvoke() for single execution
            # - astream() for streaming execution
            # - Checkpointing support for durable execution
            compiled_graph = await self._compiler.compile(
                request.agent_system,
                workflow_name
            )

            logger.info(f"Workflow compiled to StateGraph successfully: {workflow_name}")

            return CompileWorkflowResponse(
                success=True,
                workflow_name=workflow_name,
                graph_compiled=True,
                error=None
            )

        except Exception as e:
            logger.error(f"Workflow compilation failed: {e}", exc_info=True)
            return CompileWorkflowResponse(
                success=False,
                workflow_name=request.workflow_name or "",
                graph_compiled=False,
                error=str(e)
            )

    def bump_workflow_version(
        self,
        request: VersionWorkflowRequest
    ) -> VersionWorkflowResponse:
        """
        Bump workflow version

        Args:
            request: Version bump request

        Returns:
            VersionWorkflowResponse with new version
        """
        logger.info(f"Bumping version for {request.workflow_id}: {request.current_version}")

        major, minor = parse_version(request.current_version)
        new_version = bump_version(request.current_version, request.bump_type)

        logger.info(f"Version bumped: {request.current_version} → {new_version}")

        return VersionWorkflowResponse(
            workflow_id=request.workflow_id,
            old_version=request.current_version,
            new_version=new_version
        )

    async def validate_compilation(
        self,
        agent_system: AgentSystemDesign
    ) -> bool:
        """
        Quick check if agent system can be compiled to LangGraph

        Args:
            agent_system: System to validate

        Returns:
            True if compilable to StateGraph, False otherwise
        """
        try:
            if not agent_system.workflows:
                return False

            # Try compiling first workflow to StateGraph
            await self._compiler.compile(
                agent_system,
                agent_system.workflows[0].name
            )
            return True

        except Exception as e:
            logger.warning(f"Compilation validation failed: {e}")
            return False


# ============================================================================
# SINGLETON INSTANCE (optional, or use dependency injection)
# ============================================================================

_workflow_service: Optional[WorkflowService] = None


def get_workflow_service() -> WorkflowService:
    """
    Get singleton workflow service instance

    Returns:
        WorkflowService instance with LangGraph v1 support
    """
    global _workflow_service

    if _workflow_service is None:
        _workflow_service = WorkflowService()

    return _workflow_service
