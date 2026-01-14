"""
Process Service - Unified Message Processing with Intent Detection

This service handles intent detection and routes to appropriate actions.
It serves as the single entry point for frontend messages.

NOTE: This is a NEW service that does not modify existing services.
      Existing endpoints continue to work as before.
"""
import re
from typing import Optional, Dict, Any
from app.core.logging import get_logger
from app.schemas.api_models import (
    ProcessMessageRequest,
    ProcessMessageResponse,
    AgentSystemDesign,
    UserRequest,
    ModifyWorkflowRequest,
    ExecuteWorkflowRequest,
    SaveWorkflowRequest,
    ValidateAgentSystemRequest
)

logger = get_logger(__name__)


class IntentDetector:
    """
    Detects user intent from message text.
    Uses word boundary matching and keyword prioritization.
    """

    # Modification keywords (checked FIRST to avoid conflicts like "tester" matching "test")
    MODIFY_KEYWORDS = [
        'add', 'adding', 'remove', 'removing', 'insert', 'inserting',
        'delete', 'deleting', 'modify', 'modifying', 'change', 'changing',
        'edit', 'editing', 'update', 'updating', 'replace', 'replacing',
        'rename', 'renaming', 'include', 'including', 'append', 'appending',
        'create new agent', 'add agent', 'remove agent', 'more agent',
        'new agent', 'another agent', 'additional agent',
        # Workflow structure modification keywords
        'make it', 'make this', 'convert to', 'switch to', 'turn into',
        'parallel workflow', 'sequential workflow', 'hierarchical workflow',
        'make parallel', 'make sequential', 'make hierarchical',
        'use parallel', 'use sequential', 'use hierarchical'
    ]

    # Test keywords (use word boundary to avoid "tester" matching)
    TEST_KEYWORDS = ['test', 'try', 'execute', 'run']

    # Save keywords
    SAVE_KEYWORDS = ['save', 'finalize', 'finish', 'complete', 'done']

    @classmethod
    def detect(cls, message: str, has_workflow: bool = False, pending_modification: bool = False) -> str:
        """
        Detect intent from user message.

        Args:
            message: User's message text
            has_workflow: Whether a workflow currently exists
            pending_modification: Whether user clicked "Modify" button (force modification intent)

        Returns:
            Intent string: 'generate', 'modify', 'test', 'save', 'execute'
        """
        # Priority 1: If pending_modification flag is set, treat as modification
        if pending_modification and has_workflow:
            logger.info(f"Intent: modify (pending_modification flag set)")
            return 'modify'

        lower = message.lower()

        # Priority 2: Check for modification keywords (before test to avoid conflicts)
        for keyword in cls.MODIFY_KEYWORDS:
            if keyword in lower:
                if has_workflow:
                    logger.info(f"Intent: modify (keyword: {keyword})")
                    return 'modify'

        # Priority 3: Check for test keywords with word boundary
        for keyword in cls.TEST_KEYWORDS:
            if cls._match_word(lower, keyword):
                if has_workflow:
                    logger.info(f"Intent: test (keyword: {keyword})")
                    return 'test'

        # Priority 4: Check for save keywords
        for keyword in cls.SAVE_KEYWORDS:
            if cls._match_word(lower, keyword):
                if has_workflow:
                    logger.info(f"Intent: save (keyword: {keyword})")
                    return 'save'

        # Priority 5: If workflow exists and none of above, treat as execute (chat with workflow)
        if has_workflow:
            logger.info("Intent: execute (workflow exists, no specific keyword)")
            return 'execute'

        # Default: Generate new workflow
        logger.info("Intent: generate (no workflow, creating new)")
        return 'generate'

    @staticmethod
    def _match_word(text: str, word: str) -> bool:
        """Match whole word only (avoid 'tester' matching 'test')"""
        pattern = rf'\b{re.escape(word)}\b'
        return bool(re.search(pattern, text, re.IGNORECASE))


class ProcessService:
    """
    Unified service for processing user messages.
    Handles intent detection and routes to appropriate existing services.
    """

    def __init__(self):
        # Import existing services (lazy import to avoid circular dependencies)
        from app.services.workflow_service import WorkflowService
        from app.services.validator_service import ValidatorService
        from app.services.runtime_service import RuntimeService
        from app.services.storage_service import StorageService

        self.workflow_service = WorkflowService()
        self.validator_service = ValidatorService()
        self.runtime_service = RuntimeService()
        self.storage_service = StorageService()

    async def process_message(self, request: ProcessMessageRequest) -> ProcessMessageResponse:
        """
        Process user message and route to appropriate action.

        Args:
            request: ProcessMessageRequest containing message and context

        Returns:
            ProcessMessageResponse with action result
        """
        logger.info(f"Processing message: {request.message[:100]}...")

        try:
            # Detect intent
            has_workflow = request.agent_system is not None
            intent = IntentDetector.detect(
                message=request.message,
                has_workflow=has_workflow,
                pending_modification=request.pending_modification
            )

            # Route to appropriate handler
            if intent == 'generate':
                return await self._handle_generate(request)
            elif intent == 'modify':
                return await self._handle_modify(request)
            elif intent == 'test':
                return await self._handle_test(request)
            elif intent == 'execute':
                return await self._handle_execute(request)
            elif intent == 'save':
                return await self._handle_save(request)
            else:
                return ProcessMessageResponse(
                    action="error",
                    intent_detected=intent,
                    success=False,
                    error=f"Unknown intent: {intent}",
                    message="I couldn't understand what you want to do."
                )

        except Exception as e:
            logger.error(f"Process message failed: {e}", exc_info=True)
            return ProcessMessageResponse(
                action="error",
                intent_detected="error",
                success=False,
                error=str(e),
                message=f"An error occurred: {str(e)}"
            )

    async def _handle_generate(self, request: ProcessMessageRequest) -> ProcessMessageResponse:
        """Handle workflow generation"""
        logger.info("Handling: generate new workflow")

        try:
            # Use existing workflow service
            user_request = UserRequest(request=request.message)
            agent_system, analysis, meta_prompt = await self.workflow_service.design_from_user_request(user_request)

            # Auto-validate the generated workflow
            validation_request = ValidateAgentSystemRequest(
                agent_system=agent_system,
                mode='draft'
            )
            validation = await self.validator_service.validate_agent_system(validation_request)

            return ProcessMessageResponse(
                action="generate",
                intent_detected="generate",
                success=True,
                agent_system=agent_system.model_dump(),
                validation=validation.model_dump(),
                message=f"Created '{agent_system.system_name}' with {len(agent_system.agents)} agents."
            )

        except Exception as e:
            logger.error(f"Generate failed: {e}")
            return ProcessMessageResponse(
                action="error",
                intent_detected="generate",
                success=False,
                error=str(e),
                message=f"Failed to generate workflow: {str(e)}"
            )

    async def _handle_modify(self, request: ProcessMessageRequest) -> ProcessMessageResponse:
        """Handle workflow modification"""
        logger.info("Handling: modify existing workflow")

        if not request.agent_system:
            return ProcessMessageResponse(
                action="error",
                intent_detected="modify",
                success=False,
                error="No workflow to modify",
                message="There's no workflow to modify. Please create one first."
            )

        try:
            # Convert dict to AgentSystemDesign
            current_system = AgentSystemDesign(**request.agent_system)

            # Use existing workflow service modify method
            modify_request = ModifyWorkflowRequest(
                agent_system=current_system,
                modification_request=request.message
            )
            modified_system = await self.workflow_service.modify_agent_system(modify_request)

            # Auto-validate after modification
            validation_request = ValidateAgentSystemRequest(
                agent_system=modified_system,
                mode='draft'
            )
            validation = await self.validator_service.validate_agent_system(validation_request)

            return ProcessMessageResponse(
                action="modify",
                intent_detected="modify",
                success=True,
                agent_system=modified_system.model_dump(),
                validation=validation.model_dump(),
                message=f"Workflow modified. Now has {len(modified_system.agents)} agents."
            )

        except Exception as e:
            logger.error(f"Modify failed: {e}")
            return ProcessMessageResponse(
                action="error",
                intent_detected="modify",
                success=False,
                error=str(e),
                message=f"Failed to modify workflow: {str(e)}"
            )

    async def _handle_test(self, request: ProcessMessageRequest) -> ProcessMessageResponse:
        """Handle workflow testing"""
        logger.info("Handling: test workflow")

        if not request.agent_system:
            return ProcessMessageResponse(
                action="error",
                intent_detected="test",
                success=False,
                error="No workflow to test",
                message="There's no workflow to test. Please create one first."
            )

        try:
            current_system = AgentSystemDesign(**request.agent_system)

            # Validate before testing
            validation_request = ValidateAgentSystemRequest(
                agent_system=current_system,
                mode='final'
            )
            validation = await self.validator_service.validate_agent_system(validation_request)

            if not validation.valid:
                return ProcessMessageResponse(
                    action="error",
                    intent_detected="test",
                    success=False,
                    validation=validation.model_dump(),
                    error="Validation failed",
                    message="Workflow validation failed. Please fix errors before testing."
                )

            # Save as temp before executing
            save_request = SaveWorkflowRequest(
                workflow_id=current_system.system_name,
                agent_system=current_system,
                state='temp'
            )
            self.storage_service.save_temp(save_request)

            # Execute in test mode
            execute_request = ExecuteWorkflowRequest(
                workflow_id=current_system.system_name,
                execution_mode='test',
                input_payload={},
                thread_id=request.thread_id
            )
            execution_result = await self.runtime_service.execute_workflow(execute_request)

            return ProcessMessageResponse(
                action="test",
                intent_detected="test",
                success=True,
                agent_system=current_system.model_dump(),
                execution_result=execution_result.model_dump(),
                run_id=execution_result.run_id,
                thread_id=execution_result.thread_id,
                validation=validation.model_dump(),
                message="Workflow tested successfully!"
            )

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return ProcessMessageResponse(
                action="error",
                intent_detected="test",
                success=False,
                error=str(e),
                message=f"Failed to test workflow: {str(e)}"
            )

    async def _handle_execute(self, request: ProcessMessageRequest) -> ProcessMessageResponse:
        """Handle workflow execution with user input (chat with workflow)"""
        logger.info("Handling: execute workflow with input")

        if not request.agent_system:
            return ProcessMessageResponse(
                action="error",
                intent_detected="execute",
                success=False,
                error="No workflow to execute",
                message="There's no workflow to execute. Please create one first."
            )

        try:
            current_system = AgentSystemDesign(**request.agent_system)

            # Save as temp before executing
            save_request = SaveWorkflowRequest(
                workflow_id=current_system.system_name,
                agent_system=current_system,
                state='temp'
            )
            self.storage_service.save_temp(save_request)

            # Execute with user message as input
            execute_request = ExecuteWorkflowRequest(
                workflow_id=current_system.system_name,
                execution_mode=request.execution_mode or 'test',
                input_payload={'message': request.message, 'user_input': request.message},
                thread_id=request.thread_id
            )
            execution_result = await self.runtime_service.execute_workflow(execute_request)

            return ProcessMessageResponse(
                action="execute",
                intent_detected="execute",
                success=True,
                agent_system=current_system.model_dump(),
                execution_result=execution_result.model_dump(),
                run_id=execution_result.run_id,
                thread_id=execution_result.thread_id,
                message="Execution complete."
            )

        except Exception as e:
            logger.error(f"Execute failed: {e}")
            return ProcessMessageResponse(
                action="error",
                intent_detected="execute",
                success=False,
                error=str(e),
                message=f"Failed to execute workflow: {str(e)}"
            )

    async def _handle_save(self, request: ProcessMessageRequest) -> ProcessMessageResponse:
        """Handle workflow save as final"""
        logger.info("Handling: save workflow as final")

        if not request.agent_system:
            return ProcessMessageResponse(
                action="error",
                intent_detected="save",
                success=False,
                error="No workflow to save",
                message="There's no workflow to save. Please create one first."
            )

        try:
            current_system = AgentSystemDesign(**request.agent_system)

            # Save as final
            save_request = SaveWorkflowRequest(
                workflow_id=current_system.system_name,
                agent_system=current_system,
                state='final',
                version=current_system.metadata.get('version', '1.0') if current_system.metadata else '1.0'
            )
            save_result = self.storage_service.save_final(save_request)

            return ProcessMessageResponse(
                action="save",
                intent_detected="save",
                success=True,
                agent_system=current_system.model_dump(),
                message=f"Workflow saved as version {save_result.version}!"
            )

        except Exception as e:
            logger.error(f"Save failed: {e}")
            return ProcessMessageResponse(
                action="error",
                intent_detected="save",
                success=False,
                error=str(e),
                message=f"Failed to save workflow: {str(e)}"
            )


# Singleton instance
process_service = ProcessService()
