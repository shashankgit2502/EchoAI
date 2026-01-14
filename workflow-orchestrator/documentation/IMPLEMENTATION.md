# Workflow Orchestrator - Implementation Guide

This document provides a comprehensive understanding of the Workflow Orchestrator system, including:
1. **File Reference Guide** - What each file does
2. **Execution Flow Examples** - Step-by-step traces for Sequential, Parallel, and Hierarchical workflows

---

# Part 1: File Reference Guide

## Project Structure Overview

```
workflow-orchestrator/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── api/                       # API layer (HTTP endpoints)
│   │   ├── routes/                # External user-facing APIs
│   │   └── internal/              # Internal component-to-component APIs
│   ├── core/                      # Core configuration and utilities
│   ├── schemas/                   # JSON schemas and Pydantic models
│   ├── services/                  # Business logic services
│   ├── workflow/                  # Workflow design and compilation
│   ├── validator/                 # Validation rules and orchestration
│   ├── runtime/                   # LangGraph execution engine
│   ├── storage/                   # Filesystem persistence
│   ├── agents/                    # Agent management
│   ├── tools/                     # MCP tool integration
│   └── utils/                     # Utility functions
```

---

## 1. Entry Point

### `app/main.py`
**Purpose**: FastAPI application entry point and router configuration.

**Key Responsibilities**:
- Creates FastAPI application instance
- Configures CORS middleware
- Registers all API routers (external and internal)
- Handles application startup/shutdown events
- Custom exception handlers for validation errors

**Key Functions**:
| Function | Line | Description |
|----------|------|-------------|
| `root()` | 128 | Root endpoint returning API info |
| `startup_event()` | 162 | Application startup handler |
| `shutdown_event()` | 172 | Application shutdown handler |
| `validation_exception_handler()` | 69 | Custom validation error handler |

---

## 2. API Layer

### `app/api/routes/workflow.py`
**Purpose**: User-facing workflow management endpoints.

**Key Endpoints**:
| Endpoint | Method | Line | Description |
|----------|--------|------|-------------|
| `/workflow/design` | POST | 36 | Design workflow from natural language |
| `/workflow/modify` | POST | 74 | Modify existing workflow (HITL) |
| `/workflow/process` | POST | 273 | **Unified message processing** (recommended) |
| `/workflow/save/draft` | POST | 92 | Save as draft |
| `/workflow/save/temp` | POST | 101 | Save as temp (validated) |
| `/workflow/save/final` | POST | 127 | Save as final (immutable) |
| `/workflow/load` | POST | 153 | Load workflow from storage |
| `/workflow/clone` | POST | 167 | Clone final to draft for editing |
| `/workflow/list` | GET | 255 | List all workflows |
| `/workflow/delete/draft/{id}` | DELETE | 194 | Delete draft workflow |

### `app/api/routes/runtime.py`
**Purpose**: Workflow execution endpoints.

**Key Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/runtime/execute` | POST | Execute workflow (test or final mode) |
| `/runtime/execute/streaming` | POST | Stream execution updates |
| `/runtime/resume` | POST | Resume paused workflow (HITL) |

### `app/api/routes/validate.py`
**Purpose**: Validation endpoints.

**Key Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/validate/draft` | POST | Validate draft workflow |
| `/validate/final` | POST | Validate for final save |
| `/validate/quick` | POST | Quick sync-only validation |

### `app/api/routes/visualize.py`
**Purpose**: Workflow visualization endpoints.

**Key Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/visualize/graph` | POST | Get workflow as graph (nodes/edges) |

---

## 3. Services Layer

### `app/services/process_service.py`
**Purpose**: Unified message processing with intent detection. **This is the main entry point for frontend messages.**

**Key Classes**:

#### `IntentDetector` (Lines 27-108)
Detects user intent from message text using keyword matching.

| Intent | Keywords | Description |
|--------|----------|-------------|
| `generate` | (default) | Create new workflow |
| `modify` | add, remove, modify, change, make it parallel... | Modify existing workflow |
| `test` | test, try, execute, run | Test workflow |
| `save` | save, finalize, finish, done | Save as final |
| `execute` | (when workflow exists) | Execute with user input |

#### `ProcessService` (Lines 111-436)
Routes messages to appropriate handlers.

| Method | Line | Description |
|--------|------|-------------|
| `process_message()` | 129 | Main entry - detects intent and routes |
| `_handle_generate()` | 180 | Creates new workflow via designer |
| `_handle_modify()` | 215 | Modifies existing workflow |
| `_handle_test()` | 265 | Validates and executes in test mode |
| `_handle_execute()` | 337 | Executes workflow with user input |
| `_handle_save()` | 391 | Saves workflow as final |

### `app/services/workflow_service.py`
**Purpose**: Workflow design, modification, and compilation operations.

**Key Class**: `WorkflowService`

| Method | Line | Description |
|--------|------|-------------|
| `design_from_user_request()` | 64 | Design workflow from natural language |
| `modify_agent_system()` | 88 | Modify existing workflow (HITL) |
| `compile_workflow()` | 153 | Compile JSON to LangGraph |
| `bump_workflow_version()` | 216 | Increment version number |

### `app/services/storage_service.py`
**Purpose**: Workflow persistence operations.

**Key Class**: `StorageService`

| Method | Line | Description |
|--------|------|-------------|
| `save_draft()` | 44 | Save as editable draft |
| `save_temp()` | 66 | Save as validated temp |
| `save_final()` | 88 | Save as immutable final (auto-deletes draft) |
| `load_workflow()` | 124 | Load from any state |
| `clone_final_to_draft()` | 186 | Clone final for editing |
| `delete_workflow()` | 208 | Delete draft/temp |

### `app/services/validator_service.py`
**Purpose**: Validation orchestration service wrapper.

### `app/services/runtime_service.py`
**Purpose**: Workflow execution service wrapper.

### `app/services/meta_prompt_generator.py`
**Purpose**: Analyzes user requests and generates structured meta-prompts.

**Key Class**: `MetaPromptGenerator`

| Method | Line | Description |
|--------|------|-------------|
| `analyze_request()` | 38 | Extract domain, entities, operations |
| `generate_meta_prompt()` | 170 | Create structured prompt for designer |
| `generate()` | 334 | Complete pipeline: analyze + generate |

### `app/services/llm_provider.py`
**Purpose**: LLM provider abstraction and model catalog.

---

## 4. Workflow Module

### `app/workflow/designer.py`
**Purpose**: LLM-based workflow design from meta-prompts.

**Key Class**: `WorkflowDesigner`

| Method | Line | Description |
|--------|------|-------------|
| `design_from_user_request()` | 71 | Complete pipeline: request → design |
| `design_from_meta_prompt()` | 101 | Send meta-prompt to LLM, parse response |
| `modify_agent_system()` | 422 | Targeted modification of existing system |
| `_apply_targeted_changes()` | 583 | Apply changes without full regeneration |
| `_generate_workflow_steps()` | 744 | Generate steps for pattern type |
| `_fix_agent_permissions()` | 310 | Fix common LLM mistakes |
| `_extract_json()` | 217 | Extract JSON from LLM response |

### `app/workflow/compiler.py`
**Purpose**: Compiles workflow JSON to executable LangGraph StateGraph.

**Key Class**: `WorkflowCompiler`

| Method | Line | Description |
|--------|------|-------------|
| `compile()` | 118 | Main compilation entry point |
| `_build_sequential()` | 479 | Build A → B → C → END topology |
| `_build_parallel()` | 512 | Build parallel with synchronization |
| `_build_hierarchical()` | 645 | Build master-worker-master topology |
| `_create_agent_node()` | 390 | Create LangGraph node for agent |
| `_initialize_agents()` | 292 | Initialize LLM instances |

**Key Type**: `WorkflowState` (Lines 64-87)
```python
class WorkflowState(TypedDict):
    messages: List[Dict]      # Conversation history
    current_agent: str        # Currently executing agent
    agent_outputs: Dict       # Outputs from all agents
    workflow_input: Dict      # Original input
    workflow_output: Dict     # Final output
    error: str               # Error message if any
    metadata: Dict           # Additional metadata
```

### `app/workflow/versioning.py`
**Purpose**: Version management utilities.

| Function | Description |
|----------|-------------|
| `bump_version()` | Increment version (major/minor) |
| `parse_version()` | Parse version string |

---

## 5. Validator Module

### `app/validator/validator.py`
**Purpose**: Main validation orchestrator.

**Key Class**: `AgentSystemValidator`

| Method | Line | Description |
|--------|------|-------------|
| `validate()` | 53 | Full validation (sync + async) |
| `validate_quick()` | 122 | Quick sync-only validation |
| `validate_sync_only()` | 136 | Synchronous validation only |

### `app/validator/sync_rules.py`
**Purpose**: Synchronous validation rules (schema, topology, references).

| Function | Line | Description |
|----------|------|-------------|
| `run_all_sync_validations()` | 431 | Run all sync validations |
| `validate_agent_uniqueness()` | 28 | Check unique agent IDs |
| `validate_tool_references()` | 54 | Check tool references (warning) |
| `validate_workflow_references()` | 86 | Check workflow step references |
| `validate_topology()` | 115 | Check for cycles, coordinator patterns |
| `validate_hierarchical()` | 287 | Check master agent requirements |
| `validate_system_limits()` | 390 | Check agent/tool/step limits |

### `app/validator/async_rules.py`
**Purpose**: Asynchronous validation rules (MCP, LLM availability).

| Function | Description |
|----------|-------------|
| `run_all_async_validations()` | Run all async checks |
| `validate_mcp_tools()` | Check MCP tool availability |
| `validate_llm_availability()` | Check LLM provider status |

### `app/validator/errors.py`
**Purpose**: Validation error types and result structures.

**Key Classes**:
- `ValidationError` - Individual validation error
- `ValidationResult` - Collection of errors/warnings

### `app/validator/retry.py`
**Purpose**: Retry and timeout utilities for async validation.

---

## 6. Runtime Module

### `app/runtime/executor.py`
**Purpose**: LangGraph workflow execution engine.

**Key Class**: `WorkflowExecutor`

| Method | Line | Description |
|--------|------|-------------|
| `execute()` | 44 | Execute workflow (returns final result) |
| `execute_streaming()` | 139 | Stream execution updates |
| `resume_execution()` | 215 | Resume paused workflow (HITL) |
| `_load_agent_system()` | 304 | Load from temp/final based on mode |
| `_compile_workflow()` | 315 | Compile to LangGraph |
| `_extract_output()` | 411 | Extract final output from state |

**Key Class**: `ExecutionManager` (Lines 470-564)
| Method | Description |
|--------|-------------|
| `execute_batch()` | Execute multiple workflows in parallel |
| `get_execution_history()` | Get past executions |

### `app/runtime/guards.py`
**Purpose**: Execution guards (cost, timeout, step limits).

### `app/runtime/checkpoints.py`
**Purpose**: State persistence for HITL support.

### `app/runtime/hitl.py`
**Purpose**: Human-in-the-loop interrupt handling.

---

## 7. Storage Module

### `app/storage/filesystem.py`
**Purpose**: JSON file storage with lifecycle states.

**Key Class**: `WorkflowStorage`

| Method | Line | Description |
|--------|------|-------------|
| `save_draft()` | 40 | Save to draft/ directory |
| `save_temp()` | 78 | Save to temp/ directory |
| `save_final()` | 119 | Save to final/ with version |
| `load_draft()` | 174 | Load from draft/ |
| `load_temp()` | 179 | Load from temp/ |
| `load_final()` | 184 | Load from final/ (latest or specific version) |
| `clone_final_to_draft()` | 209 | Clone for editing |
| `list_versions()` | 236 | List all final versions |
| `delete_draft()` | 261 | Delete draft file |
| `delete_temp()` | 270 | Delete temp file |
| `archive_version()` | 279 | Move final to archive/ |
| `list_all_workflows()` | 357 | List all workflows with metadata |

**Directory Structure**:
```
app/storage/workflows/
├── draft/           # Editable drafts
├── temp/            # Validated, for testing
├── final/           # Immutable, versioned
└── archive/         # Archived old versions
```

---

## 8. Core Module

### `app/core/config.py`
**Purpose**: Application configuration and settings.

**Key Class**: `Settings`
- `APP_NAME`, `APP_VERSION`
- `DEFAULT_LLM_MODEL`, `DEFAULT_LLM_TEMPERATURE`
- `STORAGE_BASE_PATH`, `WORKFLOWS_PATH`, `AGENTS_PATH`
- API keys: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`

### `app/core/constants.py`
**Purpose**: Enums, limits, and workflow states.

**Key Classes**:
- `WorkflowState` - Enum: DRAFT, TEMP, FINAL
- `CommunicationPattern` - Enum: SEQUENTIAL, PARALLEL, HIERARCHICAL, etc.
- `SystemLimits` - Max agents, tools, steps

### `app/core/logging.py`
**Purpose**: Logging configuration.

### `app/core/telemetry.py`
**Purpose**: OpenTelemetry instrumentation.

---

## 9. Schemas Module

### `app/schemas/api_models.py`
**Purpose**: Pydantic models for request/response DTOs.

**Key Models**:
| Model | Description |
|-------|-------------|
| `UserRequest` | Natural language workflow request |
| `AgentSystemDesign` | Complete agent system definition |
| `AgentDefinition` | Single agent specification |
| `WorkflowDefinition` | Workflow with steps |
| `WorkflowStep` | Individual workflow step |
| `ProcessMessageRequest` | Unified process endpoint request |
| `ProcessMessageResponse` | Unified process endpoint response |
| `ExecuteWorkflowRequest` | Execution request |
| `ExecutionStatus` | Execution status/result |
| `ValidationResponse` | Validation result |
| `SaveWorkflowRequest` | Save request |
| `SaveWorkflowResponse` | Save result |

### JSON Schemas
- `app/schemas/agent_schema.json` - Agent definition schema
- `app/schemas/workflow_schema.json` - Workflow definition schema
- `app/schemas/tool_schema.json` - Tool definition schema
- `app/schemas/graph_schema.json` - Visualization graph schema

---

## 10. Agents Module

### `app/agents/factory.py`
**Purpose**: Agent instantiation at runtime.

### `app/agents/registry.py`
**Purpose**: Agent definition storage and retrieval.

### `app/agents/permissions.py`
**Purpose**: Agent permission rules for hierarchical patterns.

---

## 11. Tools Module

### `app/tools/registry.py`
**Purpose**: Tool registry and catalog.

### `app/tools/mcp_client.py`
**Purpose**: MCP (Model Context Protocol) client wrapper.

### `app/tools/health.py`
**Purpose**: MCP server health checks.

---

## 12. Utils Module

### `app/utils/ids.py`
**Purpose**: ID generation utilities.

### `app/utils/json_utils.py`
**Purpose**: JSON handling utilities.

### `app/utils/time.py`
**Purpose**: Time and date utilities.

---

# Part 2: Execution Flow - Sequential Workflow

## Example Input
**User Request**: "Create a workflow that writes a blog post. First research the topic, then write the content, finally edit and proofread."

## Step-by-Step Execution Trace

### Step 1: Frontend Sends Request
**Location**: `index.html` → `sendMessage()` function

```javascript
// Frontend calls the unified process endpoint
fetch('/workflow/process', {
  method: 'POST',
  body: JSON.stringify({
    message: "Create a workflow that writes a blog post...",
    agent_system: null,  // No existing workflow
    pending_modification: false
  })
})
```

### Step 2: API Endpoint Receives Request
**File**: `app/api/routes/workflow.py`
**Function**: `process_message()` (Line 273)

```python
@router.post("/process", response_model=ProcessMessageResponse)
async def process_message(request: ProcessMessageRequest) -> ProcessMessageResponse:
    from app.services.process_service import process_service
    result = await process_service.process_message(request)
    return result
```

### Step 3: Intent Detection
**File**: `app/services/process_service.py`
**Function**: `IntentDetector.detect()` (Line 55)

```
Input: "Create a workflow that writes a blog post..."
has_workflow: False (no existing workflow)
```

**Logic**:
1. Check MODIFY_KEYWORDS - No match
2. Check TEST_KEYWORDS - No match
3. Check SAVE_KEYWORDS - No match
4. No workflow exists → Intent: `generate`

**Output**: `"generate"`

### Step 4: Route to Generate Handler
**File**: `app/services/process_service.py`
**Function**: `_handle_generate()` (Line 180)

```python
async def _handle_generate(self, request: ProcessMessageRequest):
    # Create user request DTO
    user_request = UserRequest(request=request.message)

    # Call workflow service
    agent_system, analysis, meta_prompt = await self.workflow_service.design_from_user_request(user_request)
```

### Step 5: Workflow Service Initiates Design
**File**: `app/services/workflow_service.py`
**Function**: `design_from_user_request()` (Line 64)

```python
async def design_from_user_request(self, user_request: UserRequest):
    agent_system, analysis, meta_prompt = await self._designer.design_from_user_request(user_request)
    return agent_system, analysis, meta_prompt
```

### Step 6: Meta-Prompt Generation (Step 1 of 2-Step LLM)
**File**: `app/services/meta_prompt_generator.py`
**Function**: `generate()` (Line 334)

```python
async def generate(self, user_request: UserRequest) -> MetaPromptResponse:
    # Step 1: Analyze request
    analysis = await self.analyze_request(user_request)

    # Step 2: Generate meta-prompt
    meta_prompt = await self.generate_meta_prompt(analysis)

    return MetaPromptResponse(analysis=analysis, meta_prompt=meta_prompt)
```

#### Step 6a: Analyze Request
**Function**: `analyze_request()` (Line 38)

**LLM Call**: Sends user request to LLM with analysis prompt

**Output** (DomainAnalysis):
```json
{
  "domain": "content_creation",
  "entities": ["blog_post", "topic", "content"],
  "operations": ["research", "write", "edit", "proofread"],
  "data_sources": ["user_input", "web_search"],
  "output_requirements": ["blog_post"],
  "temporal_requirements": "on-demand",
  "suggested_patterns": ["sequential"],
  "complexity_score": 5
}
```

#### Step 6b: Generate Meta-Prompt
**Function**: `generate_meta_prompt()` (Line 170)

Creates structured prompt with:
- Domain analysis
- Agent specifications requirements
- Tool specifications requirements
- Workflow specifications requirements
- JSON schema template

### Step 7: Agent System Design (Step 2 of 2-Step LLM)
**File**: `app/workflow/designer.py`
**Function**: `design_from_meta_prompt()` (Line 101)

```python
async def design_from_meta_prompt(self, meta_prompt: str, analysis: DomainAnalysis):
    # Send meta-prompt to designer LLM
    response = await self.designer_llm.ainvoke(design_prompt.format_messages(...))

    # Extract and parse JSON
    agent_system_json = self._extract_json(response.content)

    # Fix common LLM mistakes
    agent_system_json = self._fix_agent_permissions(agent_system_json)

    # Parse to Pydantic model
    agent_system = AgentSystemDesign(**agent_system_json)

    return agent_system
```

**LLM Output** (Sequential Workflow):
```json
{
  "system_name": "blog_content_creation_system",
  "description": "Multi-agent system for creating blog posts",
  "domain": "content_creation",
  "communication_pattern": "sequential",
  "agents": [
    {
      "id": "research_agent",
      "name": "Research Agent",
      "role": "Topic Research",
      "system_prompt": "You are a research specialist...",
      "tools": ["web_search"],
      "llm_config": {"model": "...", "temperature": 0.3},
      "is_master": false
    },
    {
      "id": "writer_agent",
      "name": "Writer Agent",
      "role": "Content Writer",
      "system_prompt": "You are a professional content writer...",
      "tools": [],
      "llm_config": {"model": "...", "temperature": 0.7},
      "is_master": false
    },
    {
      "id": "editor_agent",
      "name": "Editor Agent",
      "role": "Content Editor",
      "system_prompt": "You are an expert editor and proofreader...",
      "tools": [],
      "llm_config": {"model": "...", "temperature": 0.2},
      "is_master": false
    }
  ],
  "workflows": [
    {
      "name": "blog_creation_workflow",
      "communication_pattern": "sequential",
      "steps": [
        {"agent_id": "research_agent", "action": "research"},
        {"agent_id": "writer_agent", "action": "write"},
        {"agent_id": "editor_agent", "action": "edit"}
      ]
    }
  ]
}
```

### Step 8: Auto-Validation
**File**: `app/services/process_service.py`
**Function**: `_handle_generate()` (Line 189)

```python
# Auto-validate the generated workflow
validation_request = ValidateAgentSystemRequest(agent_system=agent_system, mode='draft')
validation = await self.validator_service.validate_agent_system(validation_request)
```

**File**: `app/validator/validator.py`
**Function**: `validate()` (Line 53)

```python
async def validate(self, agent_system: AgentSystemDesign):
    # Step 1: Synchronous validation
    sync_errors = run_all_sync_validations(agent_system)

    # Step 2: Asynchronous validation (optional)
    async_errors = await run_all_async_validations(agent_system)

    return ValidationResult.create_invalid(all_errors)
```

**File**: `app/validator/sync_rules.py`
**Function**: `run_all_sync_validations()` (Line 431)

Validation checks performed:
1. `validate_agent_uniqueness()` - ✅ All agent IDs unique
2. `validate_tool_references()` - ⚠️ Warning for web_search (resolved via MCP)
3. `validate_workflow_references()` - ✅ All agent_ids exist
4. `validate_topology()` - ✅ No cycles in sequential workflow
5. `validate_hierarchical()` - ✅ Skipped (not hierarchical)
6. `validate_system_limits()` - ✅ Within limits

### Step 9: Return Response to Frontend
**File**: `app/services/process_service.py`
**Function**: `_handle_generate()` (Line 196)

```python
return ProcessMessageResponse(
    action="generate",
    intent_detected="generate",
    success=True,
    agent_system=agent_system.model_dump(),
    validation=validation.model_dump(),
    message=f"Created 'blog_content_creation_system' with 3 agents."
)
```

### Step 10: User Approves → Save as Temp
**Location**: `index.html` → User clicks "Approve"

**API Call**: `POST /workflow/save/temp`

**File**: `app/api/routes/workflow.py`
**Function**: `save_temp()` (Line 101)

```python
@router.post("/save/temp")
async def save_temp(request: SaveWorkflowRequest):
    # Validate before saving
    validation = await validator_service.validate_agent_system(...)

    # Save to temp
    return storage_service.save_temp(request)
```

**File**: `app/services/storage_service.py`
**Function**: `save_temp()` (Line 66)

```python
def save_temp(self, request: SaveWorkflowRequest):
    result = self._storage.save_temp(request.workflow_id, request.agent_system)
    return result
```

**File**: `app/storage/filesystem.py`
**Function**: `save_temp()` (Line 78)

Saves to: `app/storage/workflows/temp/blog_content_creation_system.temp.json`

### Step 11: User Tests Workflow
**Location**: `index.html` → User types "test"

**API Call**: `POST /workflow/process`
```json
{
  "message": "test",
  "agent_system": {...},
  "pending_modification": false
}
```

**Intent Detection**: `test` (matched by TEST_KEYWORDS)

**File**: `app/services/process_service.py`
**Function**: `_handle_test()` (Line 265)

```python
async def _handle_test(self, request: ProcessMessageRequest):
    # Validate
    validation = await self.validator_service.validate_agent_system(...)

    # Save as temp
    self.storage_service.save_temp(save_request)

    # Execute in test mode
    execute_request = ExecuteWorkflowRequest(
        workflow_id=current_system.system_name,
        execution_mode='test',
        input_payload={},
        thread_id=request.thread_id
    )
    execution_result = await self.runtime_service.execute_workflow(execute_request)
```

### Step 12: Workflow Execution
**File**: `app/runtime/executor.py`
**Function**: `execute()` (Line 44)

```python
async def execute(self, request: ExecuteWorkflowRequest) -> ExecutionStatus:
    # Generate run_id and thread_id
    run_id = str(uuid.uuid4())
    thread_id = request.thread_id or str(uuid.uuid4())

    # Load agent system from TEMP
    agent_system = self._load_agent_system(request)  # Line 304

    # Compile to LangGraph
    compiled_workflow = self._compile_workflow(agent_system, enable_checkpointing=True)

    # Create initial state
    initial_state = self._create_initial_state(request.input_payload)

    # Execute with checkpointing
    final_state = await self._execute_with_checkpointing(compiled_workflow, initial_state, thread_id, run_id)
```

### Step 13: LangGraph Compilation
**File**: `app/workflow/compiler.py`
**Function**: `compile()` (Line 118)

```python
def compile(self, agent_system: AgentSystemDesign, workflow_name: str, enable_checkpointing: bool):
    # Select workflow
    workflow = self._select_workflow(agent_system, workflow_name)

    # Determine pattern (sequential)
    effective_pattern = workflow.communication_pattern  # "sequential"

    # Initialize LLMs for all agents
    self._initialize_agents(agent_system.agents)

    # Build StateGraph
    graph = StateGraph(WorkflowState)

    # Add agent nodes
    for step in workflow.steps:
        agent = self._get_agent_by_id(agent_system, step.agent_id)
        node_func = self._create_agent_node(agent, agent_system)
        graph.add_node(agent.id, node_func)

    # Build sequential topology
    self._build_sequential(graph, workflow, agent_system)

    # Set entry point
    graph.set_entry_point(workflow.steps[0].agent_id)  # "research_agent"

    # Compile with checkpointing
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled
```

### Step 14: Build Sequential Topology
**File**: `app/workflow/compiler.py`
**Function**: `_build_sequential()` (Line 479)

```python
def _build_sequential(self, graph: StateGraph, workflow: WorkflowDefinition, agent_system: AgentSystemDesign):
    # Build: research_agent -> writer_agent -> editor_agent -> END

    steps = workflow.steps
    for i in range(len(steps) - 1):
        current_node = steps[i].agent_id
        next_node = steps[i + 1].agent_id
        graph.add_edge(current_node, next_node)
        # Edge: research_agent -> writer_agent
        # Edge: writer_agent -> editor_agent

    # Last step to END
    graph.add_edge(steps[-1].agent_id, END)
    # Edge: editor_agent -> END
```

**Resulting Graph**:
```
research_agent → writer_agent → editor_agent → END
```

### Step 15: Agent Execution
**File**: `app/workflow/compiler.py`
**Function**: `_create_agent_node()` → `agent_node()` (Line 408)

For each agent in sequence:

**Agent 1: research_agent**
```python
def agent_node(state: WorkflowState):
    # Get workflow input
    workflow_input = state.get("workflow_input", {})

    # Build messages
    messages = [
        SystemMessage(content=agent.system_prompt),
        HumanMessage(content=f"Workflow input:\n{workflow_input}")
    ]

    # Call LLM
    response = agent_llm.invoke(messages)

    # Return updated state
    return {
        "agent_outputs": {"research_agent": response.content},
        "current_agent": "research_agent",
        "error": None
    }
```

**Agent 2: writer_agent**
- Receives research_agent's output in context
- Writes blog content based on research

**Agent 3: editor_agent**
- Receives all previous outputs
- Edits and proofreads the content

### Step 16: Extract Final Output
**File**: `app/runtime/executor.py`
**Function**: `_extract_output()` (Line 411)

```python
def _extract_output(self, final_state: WorkflowState) -> Dict[str, Any]:
    # Check for errors
    if final_state.get("error"):
        return {"error": final_state["error"], "all_outputs": agent_outputs}

    # Get agent outputs
    agent_outputs = final_state.get("agent_outputs", {})
    # {
    #   "research_agent": "Research findings...",
    #   "writer_agent": "Blog post content...",
    #   "editor_agent": "Edited final blog post..."
    # }

    # Return last agent's output
    return {
        "result": agent_outputs["editor_agent"],
        "all_outputs": agent_outputs
    }
```

### Step 17: Return Execution Result
**File**: `app/services/process_service.py`
**Function**: `_handle_test()` (Line 315)

```python
return ProcessMessageResponse(
    action="test",
    intent_detected="test",
    success=True,
    agent_system=current_system.model_dump(),
    execution_result=execution_result.model_dump(),
    run_id=execution_result.run_id,
    thread_id=execution_result.thread_id,
    message="Workflow tested successfully!"
)
```

---

# Part 3: Execution Flow - Parallel Workflow

## Example Input
**User Request**: "Create a workflow that compares Python, JavaScript, and Rust programming languages simultaneously."

## Key Differences from Sequential

### Step 7: Agent System Design Output
**File**: `app/workflow/designer.py`

**LLM Output** (Parallel Workflow):
```json
{
  "system_name": "programming_language_comparison_system",
  "communication_pattern": "parallel",
  "agents": [
    {
      "id": "python_analyst",
      "role": "Python Language Analyst",
      "is_master": false
    },
    {
      "id": "javascript_analyst",
      "role": "JavaScript Language Analyst",
      "is_master": false
    },
    {
      "id": "rust_analyst",
      "role": "Rust Language Analyst",
      "is_master": false
    }
  ],
  "workflows": [
    {
      "name": "language_comparison_workflow",
      "communication_pattern": "parallel",
      "steps": [
        {
          "agent_id": "python_analyst",
          "action": "analyze",
          "parallel_with": ["javascript_analyst", "rust_analyst"]
        },
        {
          "agent_id": "javascript_analyst",
          "action": "analyze",
          "parallel_with": ["python_analyst", "rust_analyst"]
        },
        {
          "agent_id": "rust_analyst",
          "action": "analyze",
          "parallel_with": ["python_analyst", "javascript_analyst"]
        }
      ]
    }
  ]
}
```

### Step 8: Topology Validation
**File**: `app/validator/sync_rules.py`
**Function**: `validate_topology()` (Line 115)

```python
def validate_topology(system: AgentSystemDesign):
    # Skip cycle checking for PARALLEL system patterns
    if system.communication_pattern == CommunicationPattern.PARALLEL:
        return errors  # No cycle check needed
```

### Step 14: Build Parallel Topology
**File**: `app/workflow/compiler.py`
**Function**: `_build_parallel()` (Line 512)

```python
def _build_parallel(self, graph: StateGraph, workflow: WorkflowDefinition, agent_system: AgentSystemDesign):
    steps = workflow.steps

    for i, step in enumerate(steps):
        if step.parallel_with:
            # All parallel steps point to END
            parallel_agents = [step.agent_id] + step.parallel_with

            for agent_id in parallel_agents:
                graph.add_edge(agent_id, END)
```

**Resulting Graph** (Fan-out pattern):
```
           ┌─→ python_analyst ─────┐
ENTRY ─────┼─→ javascript_analyst ─┼─→ END
           └─→ rust_analyst ───────┘
```

### Step 15: Parallel Agent Execution
**File**: `app/workflow/compiler.py`

All three agents execute **concurrently**:
- `python_analyst` analyzes Python
- `javascript_analyst` analyzes JavaScript
- `rust_analyst` analyzes Rust

**State Merging** (Lines 32-38):
```python
def merge_dicts(left: Dict, right: Dict) -> Dict:
    """Merge two dictionaries - used for agent_outputs"""
    return {**left, **right}
```

Each agent's output is merged into the shared `agent_outputs` dict:
```python
{
    "python_analyst": "Python analysis...",
    "javascript_analyst": "JavaScript analysis...",
    "rust_analyst": "Rust analysis..."
}
```

### Step 16: Extract Combined Output
**File**: `app/runtime/executor.py`
**Function**: `_extract_output()` (Line 411)

```python
# For parallel, return all outputs
agent_outputs = final_state.get("agent_outputs", {})
return {"all_outputs": agent_outputs}
```

**Final Output**:
```json
{
  "all_outputs": {
    "python_analyst": "Python: High-level, readable, great for ML...",
    "javascript_analyst": "JavaScript: Browser native, async, versatile...",
    "rust_analyst": "Rust: Memory safe, fast, steep learning curve..."
  }
}
```

---

# Part 4: Execution Flow - Hierarchical Workflow

## Example Input
**User Request**: "Create a workflow with a project manager that coordinates a frontend developer, backend developer, and QA tester."

## Key Differences from Sequential/Parallel

### Step 7: Agent System Design Output
**File**: `app/workflow/designer.py`

**LLM Output** (Hierarchical Workflow):
```json
{
  "system_name": "software_development_system",
  "communication_pattern": "hierarchical",
  "agents": [
    {
      "id": "project_manager",
      "role": "Project Coordinator",
      "is_master": true,
      "permissions": {
        "can_call_agents": ["frontend_dev", "backend_dev", "qa_tester"],
        "can_delegate": true
      }
    },
    {
      "id": "frontend_dev",
      "role": "Frontend Developer",
      "is_master": false
    },
    {
      "id": "backend_dev",
      "role": "Backend Developer",
      "is_master": false
    },
    {
      "id": "qa_tester",
      "role": "QA Tester",
      "is_master": false
    }
  ],
  "workflows": [
    {
      "name": "development_workflow",
      "communication_pattern": "hierarchical",
      "steps": [
        {"agent_id": "project_manager", "action": "coordinate"},
        {"agent_id": "frontend_dev", "action": "develop", "parallel_with": ["backend_dev"]},
        {"agent_id": "backend_dev", "action": "develop", "parallel_with": ["frontend_dev"]},
        {"agent_id": "qa_tester", "action": "test"},
        {"agent_id": "project_manager", "action": "aggregate"}
      ]
    }
  ]
}
```

### Step 7a: Fix Agent Permissions
**File**: `app/workflow/designer.py`
**Function**: `_fix_agent_permissions()` (Line 310)

```python
def _fix_agent_permissions(self, system_json: dict):
    communication_pattern = system_json.get("communication_pattern")  # "hierarchical"

    # CRITICAL: For hierarchical, ensure at least one master agent
    if communication_pattern == "hierarchical" and len(master_agents) == 0:
        # Find coordinator by naming patterns
        coordinator_keywords = ["coordinator", "orchestrator", "master", "manager"]
        for agent in agents:
            if any(keyword in agent["id"].lower() for keyword in coordinator_keywords):
                agent["is_master"] = True
                break

    # Ensure master can call all sub-agents
    if agent.get("is_master", False):
        permissions["can_call_agents"] = [aid for aid in all_agent_ids if aid != agent["id"]]
        permissions["can_delegate"] = True
```

### Step 8: Hierarchical Validation
**File**: `app/validator/sync_rules.py`
**Function**: `validate_hierarchical()` (Line 287)

```python
def validate_hierarchical(system: AgentSystemDesign):
    if system.communication_pattern != CommunicationPattern.HIERARCHICAL:
        return errors

    # Count master agents
    master_agents = [agent for agent in system.agents if agent.is_master]

    if len(master_agents) == 0:
        errors.append(schema_error(
            message="Hierarchical pattern requires at least one master agent"
        ))
    elif len(master_agents) > 1:
        errors.append(schema_error(
            message="Multiple master agents found. Should have exactly one."
        ))

    return errors
```

### Step 14: Build Hierarchical Topology
**File**: `app/workflow/compiler.py`
**Function**: `_build_hierarchical()` (Line 645)

```python
def _build_hierarchical(self, graph: StateGraph, workflow: WorkflowDefinition, agent_system: AgentSystemDesign):
    # Find master agent
    master_agent = next((a for a in agent_system.agents if a.is_master), None)

    # Get worker agents
    worker_agents = self._get_ordered_workers(workflow, agent_system, master_agent.id)
    # ["frontend_dev", "backend_dev", "qa_tester"]

    # Create TWO distinct nodes for master: delegate and aggregate
    master_delegate_id = f"{master_agent.id}_delegate"  # "project_manager_delegate"
    master_aggregate_id = f"{master_agent.id}_aggregate"  # "project_manager_aggregate"

    # Add master nodes with phase-specific logic
    delegate_node = self._create_hierarchical_master_node(master_agent, agent_system, "delegate")
    aggregate_node = self._create_hierarchical_master_node(master_agent, agent_system, "aggregate")
    graph.add_node(master_delegate_id, delegate_node)
    graph.add_node(master_aggregate_id, aggregate_node)

    # Set entry point to delegation phase
    self._hierarchical_entry = master_delegate_id

    # Build edges
    self._build_hierarchical_edges(graph, workflow, master_delegate_id, master_aggregate_id, worker_agents, master_agent.id)

    # aggregate -> END
    graph.add_edge(master_aggregate_id, END)
```

### Step 14a: Analyze Parallel Groups
**File**: `app/workflow/compiler.py`
**Function**: `_analyze_parallel_groups()` (Line 776)

```python
def _analyze_parallel_groups(self, worker_steps: List, worker_agents: List[str]):
    # From steps:
    # frontend_dev (parallel_with: [backend_dev])
    # backend_dev (parallel_with: [frontend_dev])
    # qa_tester (no parallel_with)

    groups = [
        {"agents": ["frontend_dev", "backend_dev"], "parallel": True},
        {"agents": ["qa_tester"], "parallel": False}
    ]
    return groups
```

### Step 14b: Build Hierarchical Edges
**File**: `app/workflow/compiler.py`
**Function**: `_build_hierarchical_edges()` (Line 706)

```python
def _build_hierarchical_edges(self, graph, workflow, delegate_id, aggregate_id, worker_agents, master_id):
    execution_groups = self._analyze_parallel_groups(worker_steps, worker_agents)
    # [
    #   {"agents": ["frontend_dev", "backend_dev"], "parallel": True},
    #   {"agents": ["qa_tester"], "parallel": False}
    # ]

    # First group: delegate -> parallel workers
    for agent_id in ["frontend_dev", "backend_dev"]:
        graph.add_edge(delegate_id, agent_id)

    # Connect groups
    for agent in ["frontend_dev", "backend_dev"]:
        graph.add_edge(agent, "qa_tester")

    # Last group -> aggregate
    graph.add_edge("qa_tester", aggregate_id)
```

**Resulting Graph**:
```
                              ┌─→ frontend_dev ─┐
project_manager_delegate ────┤                 ├─→ qa_tester ─→ project_manager_aggregate ─→ END
                              └─→ backend_dev ──┘
```

### Step 15: Hierarchical Master Node Execution
**File**: `app/workflow/compiler.py`
**Function**: `_create_hierarchical_master_node()` (Line 565)

**Delegate Phase**:
```python
def master_node(state: WorkflowState):
    phase_instruction = "You are starting the workflow. Review the input and prepare context for the worker agents..."

    messages = [
        SystemMessage(content=f"{agent.system_prompt}\n\n{phase_instruction}"),
        HumanMessage(content=f"Original request:\n{workflow_input}")
    ]

    response = agent_llm.invoke(messages)

    return {
        "agent_outputs": {"project_manager_delegate": response.content},
        "current_agent": "project_manager_delegate"
    }
```

**Worker Execution** (frontend_dev, backend_dev run in parallel):
- Each receives delegate's output as context
- Each produces their own output

**Aggregate Phase**:
```python
def master_node(state: WorkflowState):
    phase_instruction = "All worker agents have completed. Review ALL their outputs and compile a comprehensive FINAL result..."

    # Build context from ALL previous outputs
    context = "\n\n".join([
        f"=== Output from {agent_id} ===\n{output}"
        for agent_id, output in agent_outputs.items()
    ])

    messages = [
        SystemMessage(content=f"{agent.system_prompt}\n\n{phase_instruction}"),
        HumanMessage(content=f"Previous agent outputs:\n{context}")
    ]

    response = agent_llm.invoke(messages)

    return {
        "agent_outputs": {"project_manager_aggregate": response.content},
        "current_agent": "project_manager_aggregate"
    }
```

### Step 16: Extract Final Output
**File**: `app/runtime/executor.py`

```python
agent_outputs = {
    "project_manager_delegate": "Project plan and task assignments...",
    "frontend_dev": "Frontend implementation details...",
    "backend_dev": "Backend API implementation...",
    "qa_tester": "Test results and findings...",
    "project_manager_aggregate": "FINAL: Project completed successfully with all components integrated..."
}

# Return aggregate (last) output
return {
    "result": agent_outputs["project_manager_aggregate"],
    "all_outputs": agent_outputs
}
```

---

# Summary: Key Function Call Chains

## Generate New Workflow
```
Frontend: sendMessage()
    ↓
API: /workflow/process (workflow.py:273)
    ↓
ProcessService.process_message() (process_service.py:129)
    ↓
IntentDetector.detect() → "generate" (process_service.py:55)
    ↓
ProcessService._handle_generate() (process_service.py:180)
    ↓
WorkflowService.design_from_user_request() (workflow_service.py:64)
    ↓
MetaPromptGenerator.generate() (meta_prompt_generator.py:334)
    ├── analyze_request() → DomainAnalysis
    └── generate_meta_prompt() → meta_prompt string
    ↓
WorkflowDesigner.design_from_meta_prompt() (designer.py:101)
    ├── LLM call to generate JSON
    ├── _extract_json() → dict
    ├── _fix_agent_permissions() → fixed dict
    └── AgentSystemDesign(**dict)
    ↓
ValidatorService.validate_agent_system() (validator_service.py)
    ↓
AgentSystemValidator.validate() (validator.py:53)
    ├── run_all_sync_validations() (sync_rules.py:431)
    └── run_all_async_validations() (async_rules.py)
    ↓
Return ProcessMessageResponse
```

## Execute Workflow
```
Frontend: "test" message
    ↓
API: /workflow/process
    ↓
IntentDetector.detect() → "test"
    ↓
ProcessService._handle_test() (process_service.py:265)
    ↓
RuntimeService.execute_workflow() (runtime_service.py)
    ↓
WorkflowExecutor.execute() (executor.py:44)
    ├── _load_agent_system() → load from temp/final
    ├── _compile_workflow() → LangGraph StateGraph
    │       ↓
    │   WorkflowCompiler.compile() (compiler.py:118)
    │       ├── _initialize_agents() → create LLM instances
    │       ├── _create_agent_node() → create node functions
    │       └── _build_sequential/parallel/hierarchical() → add edges
    │
    └── _execute_with_checkpointing() → run graph
            ↓
        compiled_workflow.ainvoke(initial_state)
            ↓
        [Agent nodes execute in order based on topology]
            ↓
        _extract_output() → final result
```

## Modify Workflow
```
Frontend: "make it parallel"
    ↓
IntentDetector.detect() → "modify"
    ↓
ProcessService._handle_modify() (process_service.py:215)
    ↓
WorkflowService.modify_agent_system() (workflow_service.py:88)
    ↓
WorkflowDesigner.modify_agent_system() (designer.py:422)
    ├── LLM call with targeted modification prompt
    ├── _extract_json() → changes dict
    ├── _apply_targeted_changes() (designer.py:583)
    │       ├── Apply system-level changes
    │       ├── Apply agent modifications (add/modify/remove)
    │       ├── Apply workflow modifications
    │       ├── Sync communication_pattern
    │       └── _generate_workflow_steps() if agents changed
    └── _fix_agent_permissions()
    ↓
Auto-validate and return
```

## Save Workflow Lifecycle
```
DRAFT (editable):
    POST /workflow/save/draft
        → StorageService.save_draft()
        → WorkflowStorage.save_draft()
        → app/storage/workflows/draft/{id}.draft.json

TEMP (validated, testable):
    POST /workflow/save/temp
        → Validate first
        → StorageService.save_temp()
        → app/storage/workflows/temp/{id}.temp.json

FINAL (immutable, versioned):
    POST /workflow/save/final
        → Validate first
        → StorageService.save_final()
        → Auto-delete draft and temp
        → app/storage/workflows/final/{id}.v{version}.final.json
```

---

# Glossary

| Term | Description |
|------|-------------|
| **Agent** | LLM-powered component with specific role and tools |
| **Workflow** | Sequence of steps defining agent execution order |
| **Communication Pattern** | How agents interact: sequential, parallel, hierarchical |
| **HITL** | Human-in-the-Loop - pause for human approval |
| **StateGraph** | LangGraph data structure for workflow execution |
| **Checkpointing** | Saving execution state for resumption |
| **Meta-Prompt** | Structured prompt generated from user request |
| **MCP** | Model Context Protocol - tool integration standard |
| **DRAFT** | Editable workflow state |
| **TEMP** | Validated, testable workflow state |
| **FINAL** | Immutable, versioned production workflow |
