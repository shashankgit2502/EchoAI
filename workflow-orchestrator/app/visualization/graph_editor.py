"""
Graph Editor
Applies UI graph edits back to workflow JSON
Enables bidirectional sync between UI and backend
"""
from typing import Dict, Any, List, Optional
from copy import deepcopy

from app.schemas.api_models import AgentSystemDesign, WorkflowStep
from app.workflow.graph_builder import GraphNode, GraphEdge
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# EDIT OPERATIONS
# ============================================================================

class EditOperation:
    """
    Base class for edit operations
    """
    def apply(self, workflow: AgentSystemDesign) -> AgentSystemDesign:
        """Apply edit to workflow"""
        raise NotImplementedError


class AddNodeEdit(EditOperation):
    """
    Add new agent node
    """
    def __init__(self, node_data: Dict[str, Any]):
        self.node_data = node_data

    def apply(self, workflow: AgentSystemDesign) -> AgentSystemDesign:
        """Add agent to workflow"""
        logger.info(f"Adding node: {self.node_data.get('id')}")

        # Create new agent definition from node data
        from app.schemas.api_models import AgentDefinition, LLMConfig, AgentPermissions

        agent = AgentDefinition(
            id=self.node_data["id"],
            role=self.node_data.get("role", "agent"),
            system_prompt=self.node_data.get("system_prompt", ""),
            responsibilities=self.node_data.get("responsibilities", []),
            llm_config=LLMConfig(**self.node_data.get("llm_config", {
                "model": "claude-sonnet-4-5-20250929",
                "temperature": 0.7,
                "max_tokens": 4000
            })),
            tools=self.node_data.get("tools", []),
            permissions=AgentPermissions(**self.node_data.get("permissions", {
                "can_delegate": True,
                "max_tool_calls": 10
            })),
            is_master=self.node_data.get("is_master", False)
        )

        # Add to workflow
        workflow_copy = deepcopy(workflow)
        workflow_copy.agents.append(agent)

        return workflow_copy


class RemoveNodeEdit(EditOperation):
    """
    Remove agent node
    """
    def __init__(self, node_id: str):
        self.node_id = node_id

    def apply(self, workflow: AgentSystemDesign) -> AgentSystemDesign:
        """Remove agent from workflow"""
        logger.info(f"Removing node: {self.node_id}")

        workflow_copy = deepcopy(workflow)

        # Remove agent
        workflow_copy.agents = [a for a in workflow_copy.agents if a.id != self.node_id]

        # Remove from workflow steps
        if workflow_copy.workflows:
            for wf in workflow_copy.workflows:
                wf.steps = [s for s in wf.steps if s.agent_id != self.node_id]

        return workflow_copy


class UpdateNodeEdit(EditOperation):
    """
    Update node properties
    """
    def __init__(self, node_id: str, updates: Dict[str, Any]):
        self.node_id = node_id
        self.updates = updates

    def apply(self, workflow: AgentSystemDesign) -> AgentSystemDesign:
        """Update agent properties"""
        logger.info(f"Updating node: {self.node_id}")

        workflow_copy = deepcopy(workflow)

        # Find and update agent
        for agent in workflow_copy.agents:
            if agent.id == self.node_id:
                # Update fields
                for key, value in self.updates.items():
                    if hasattr(agent, key):
                        setattr(agent, key, value)
                break

        return workflow_copy


class AddEdgeEdit(EditOperation):
    """
    Add connection between agents
    """
    def __init__(self, source: str, target: str, edge_type: str = "default"):
        self.source = source
        self.target = target
        self.edge_type = edge_type

    def apply(self, workflow: AgentSystemDesign) -> AgentSystemDesign:
        """Add edge to workflow"""
        logger.info(f"Adding edge: {self.source} -> {self.target}")

        workflow_copy = deepcopy(workflow)

        # Add workflow step if not exists
        if workflow_copy.workflows:
            wf = workflow_copy.workflows[0]

            # Check if step already exists
            existing = any(s.agent_id == self.target for s in wf.steps)

            if not existing:
                # Create new step
                new_step = WorkflowStep(
                    agent_id=self.target,
                    action=f"Execute {self.target}",
                    inputs={},
                    outputs={}
                )
                wf.steps.append(new_step)

        return workflow_copy


class RemoveEdgeEdit(EditOperation):
    """
    Remove connection between agents
    """
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target

    def apply(self, workflow: AgentSystemDesign) -> AgentSystemDesign:
        """Remove edge from workflow"""
        logger.info(f"Removing edge: {self.source} -> {self.target}")

        workflow_copy = deepcopy(workflow)

        # This is complex - may need to reorganize workflow steps
        # For now, just remove the step if it was the only connection

        return workflow_copy


class MoveNodeEdit(EditOperation):
    """
    Move node position (cosmetic, doesn't affect workflow logic)
    """
    def __init__(self, node_id: str, position: Dict[str, float]):
        self.node_id = node_id
        self.position = position

    def apply(self, workflow: AgentSystemDesign) -> AgentSystemDesign:
        """Update node position metadata"""
        logger.debug(f"Moving node: {self.node_id} to {self.position}")

        workflow_copy = deepcopy(workflow)

        # Store position in metadata
        if not workflow_copy.metadata.get("node_positions"):
            workflow_copy.metadata["node_positions"] = {}

        workflow_copy.metadata["node_positions"][self.node_id] = self.position

        return workflow_copy


# ============================================================================
# GRAPH EDITOR
# ============================================================================

class GraphEditor:
    """
    Applies UI graph edits back to workflow JSON

    Enables bidirectional editing between UI and backend

    Usage:
        editor = GraphEditor()

        # Apply single edit
        updated_workflow = editor.apply_edit(workflow, AddNodeEdit(node_data))

        # Apply batch edits
        updated_workflow = editor.apply_edits(workflow, [edit1, edit2, edit3])

        # Parse edit from UI event
        edit = editor.parse_ui_edit(ui_event)
        updated_workflow = editor.apply_edit(workflow, edit)
    """

    def __init__(self):
        """Initialize graph editor"""
        logger.info("Graph editor initialized")

    def apply_edit(
        self,
        workflow: AgentSystemDesign,
        edit: EditOperation
    ) -> AgentSystemDesign:
        """
        Apply single edit to workflow

        Args:
            workflow: Current workflow
            edit: Edit operation

        Returns:
            Updated workflow
        """
        logger.debug(f"Applying edit: {type(edit).__name__}")

        try:
            updated_workflow = edit.apply(workflow)

            # Validate after edit
            from app.validator.validator import AgentSystemValidator

            validator = AgentSystemValidator(enable_async_validation=False)
            result = validator.validate_sync_only(updated_workflow)

            if not result.valid:
                logger.warning(f"Edit resulted in invalid workflow: {result.error_count} errors")
                # Return original workflow if validation fails
                return workflow

            return updated_workflow

        except Exception as e:
            logger.error(f"Failed to apply edit: {e}", exc_info=True)
            return workflow

    def apply_edits(
        self,
        workflow: AgentSystemDesign,
        edits: List[EditOperation]
    ) -> AgentSystemDesign:
        """
        Apply batch of edits

        Args:
            workflow: Current workflow
            edits: List of edit operations

        Returns:
            Updated workflow
        """
        logger.info(f"Applying {len(edits)} edits")

        current_workflow = workflow

        for edit in edits:
            current_workflow = self.apply_edit(current_workflow, edit)

        return current_workflow

    def parse_ui_edit(self, ui_event: Dict[str, Any]) -> Optional[EditOperation]:
        """
        Parse UI event into edit operation

        Args:
            ui_event: UI event data

        Returns:
            EditOperation or None

        UI Event format:
            {
                "type": "addNode" | "removeNode" | "updateNode" | "addEdge" | "removeEdge" | "moveNode",
                "data": { ... event-specific data }
            }
        """
        event_type = ui_event.get("type")
        data = ui_event.get("data", {})

        if event_type == "addNode":
            return AddNodeEdit(node_data=data)

        elif event_type == "removeNode":
            return RemoveNodeEdit(node_id=data["node_id"])

        elif event_type == "updateNode":
            return UpdateNodeEdit(
                node_id=data["node_id"],
                updates=data.get("updates", {})
            )

        elif event_type == "addEdge":
            return AddEdgeEdit(
                source=data["source"],
                target=data["target"],
                edge_type=data.get("type", "default")
            )

        elif event_type == "removeEdge":
            return RemoveEdgeEdit(
                source=data["source"],
                target=data["target"]
            )

        elif event_type == "moveNode":
            return MoveNodeEdit(
                node_id=data["node_id"],
                position=data["position"]
            )

        else:
            logger.warning(f"Unknown edit type: {event_type}")
            return None

    def validate_edit(
        self,
        workflow: AgentSystemDesign,
        edit: EditOperation
    ) -> bool:
        """
        Validate edit without applying

        Args:
            workflow: Current workflow
            edit: Edit operation

        Returns:
            True if edit is valid
        """
        try:
            # Apply edit to copy
            updated_workflow = edit.apply(deepcopy(workflow))

            # Validate
            from app.validator.validator import AgentSystemValidator

            validator = AgentSystemValidator(enable_async_validation=False)
            result = validator.validate_sync_only(updated_workflow)

            return result.valid

        except Exception as e:
            logger.error(f"Edit validation failed: {e}")
            return False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_ui_edits(
    workflow: AgentSystemDesign,
    ui_events: List[Dict[str, Any]]
) -> AgentSystemDesign:
    """
    Convenience function to apply UI edits

    Args:
        workflow: Current workflow
        ui_events: List of UI events

    Returns:
        Updated workflow
    """
    editor = GraphEditor()

    # Parse events into edits
    edits = []
    for event in ui_events:
        edit = editor.parse_ui_edit(event)
        if edit:
            edits.append(edit)

    # Apply all edits
    return editor.apply_edits(workflow, edits)


def create_edit_from_diff(
    old_workflow: AgentSystemDesign,
    new_workflow: AgentSystemDesign
) -> List[EditOperation]:
    """
    Generate edit operations from workflow diff

    Args:
        old_workflow: Original workflow
        new_workflow: Modified workflow

    Returns:
        List of edit operations
    """
    edits = []

    # Compare agents
    old_agent_ids = {a.id for a in old_workflow.agents}
    new_agent_ids = {a.id for a in new_workflow.agents}

    # Added agents
    for agent_id in new_agent_ids - old_agent_ids:
        agent = next(a for a in new_workflow.agents if a.id == agent_id)
        edits.append(AddNodeEdit(agent.dict()))

    # Removed agents
    for agent_id in old_agent_ids - new_agent_ids:
        edits.append(RemoveNodeEdit(agent_id))

    # Modified agents
    for agent_id in old_agent_ids & new_agent_ids:
        old_agent = next(a for a in old_workflow.agents if a.id == agent_id)
        new_agent = next(a for a in new_workflow.agents if a.id == agent_id)

        # Compare and create update edit
        updates = {}
        if old_agent.role != new_agent.role:
            updates["role"] = new_agent.role
        if old_agent.system_prompt != new_agent.system_prompt:
            updates["system_prompt"] = new_agent.system_prompt

        if updates:
            edits.append(UpdateNodeEdit(agent_id, updates))

    return edits
