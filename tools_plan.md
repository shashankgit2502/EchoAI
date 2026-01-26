# EchoAI Tool System Architecture Plan

**Document Version**: 1.0
**Created**: 2026-01-26
**Status**: READY FOR IMPLEMENTATION

---

## 1. Problem Statement

### Current State Analysis

After thorough code analysis, the current tool system has the following critical gaps:

1. **ToolService is a Placeholder** (`echolib/services.py:44-47`)
   ```python
   def invokeTool(self, name: str, args: dict) -> ToolResult:
       if name not in self._tools:
           raise ValueError('tool not found')
       return ToolResult(name=name, output={'echo': args})  # <-- JUST ECHOES
   ```

2. **ToolDef Lacks Essential Fields** (`echolib/types.py:33-36`)
   - Only has `name` and `description`
   - NO input/output schema
   - NO execution logic
   - NO tool type classification

3. **AgentFactory Has Placeholder Tool Binding** (`apps/agent/factory/factory.py:120-142`)
   - `_bind_tools()` creates fake bindings
   - TODO comment: "Implement actual MCP tool binding"
   - Tools are never actually invoked during workflow execution

4. **CrewAI Adapter Ignores Tools** (`apps/workflow/crewai_adapter.py`)
   - Creates CrewAI agents without tool bindings
   - No `tools` parameter passed to `Agent()` constructor

5. **Tools Are Not Separated From Core**
   - Existing tool implementations in `Tools I made/` folder
   - Not discoverable or registrable by the system
   - No standardized interface

6. **MCP Connector Not Integrated As Tool**
   - MCP routes exist (`/connectors/mcp/invoke`)
   - But cannot be used as a tool by agents
   - Separate invocation path

### Business Impact

- Agents cannot use any actual tools during workflow execution
- Sequential workflows pass context but cannot execute code, search web, read files, etc.
- The workflow orchestrator is "orchestrating" LLM-only agents
- Users cannot add custom tools or MCP-based tools to agents

---

## 1.5 Current Frontend-to-Backend Tool Flow (As Implemented Today)

### Overview

The system already has a UI for tool selection and a data path from frontend to backend. However, the final execution step is missing. Understanding this existing flow is critical to ensure we don't break it.

### Step 1: User Adds Tool in UI (`workflow_builder_ide.html`)

**Location**: Lines 824-900, 2667-2734

When a user clicks the "+ Tool" button on an Agent node, they see a dropdown with tool types:

```javascript
// Tool types available in UI
const toolTypes = {
    'tools': 'Third-party Tool',      // Generic external tools
    'code': 'Code Execution',          // Python/JS code execution
    'subworkflow': 'Subworkflow',      // Nested workflow
    'subworkflow_deployment': 'Subworkflow Deployment',
    'mcp_server': 'MCP Server'         // MCP-based tools
};
```

**addTool() function** (line 2667):
```javascript
const addTool = (toolType) => {
    selected.value.config.tools.push({
        id: Date.now(),
        name: toolNames[toolType],    // e.g., "Third-party Tool"
        type: toolType,               // e.g., "tools", "mcp_server"
        enabled: true,
        config: {
            provider: '',             // For third-party: slack, github, etc.
            action: '',               // e.g., send_message
            credentials: '',
            language: 'python',       // For code execution
            code: '',
            timeout: 30,
            workflowId: '',           // For subworkflow
            serverId: '',             // For MCP
            serverUrl: '',
            operationType: 'tool',
            operationName: '',
            description: ''
        }
    });
};
```

**Result**: Tool is stored in the node's `config.tools` array.

### Step 2: User Configures Tool (Tool Modal)

**Location**: Lines 1460-1615

User clicks on a tool to configure it. The modal shows different fields based on `tool.type`:

- **tools (Third-party)**: provider dropdown (Slack, GitHub, Jira, etc.), action, credentials
- **code**: language, code textarea, timeout
- **mcp_server**: serverId, serverUrl, operationType, operationName

**saveToolConfig()** (line 2729):
```javascript
const saveToolConfig = () => {
    selected.value.config.tools[toolConfigModal.value.toolIndex] = toolConfigModal.value.tool;
    closeToolConfig();
};
```

### Step 3: Canvas Saved to Backend

**Location**: Lines 3151-3193, API call at 1732-1754

When user saves workflow, the entire canvas (including tools) is sent:

```javascript
// workflowAPI.saveCanvas() - POST /workflows/canvas/save
body: JSON.stringify({
    canvas_nodes: canvasNodes,     // Each node has config.tools array
    connections: connections,
    workflow_name: workflowName,
    save_as: saveAs,
    execution_model: executionModel
})
```

### Step 4: Backend NodeMapper Processes Tools

**Location**: `apps/workflow/visualization/node_mapper.py:206-317, 354-378`

`_convert_node_to_agent()` extracts tools from node config:

```python
# For Agent/Subagent/Prompt nodes (line 250-252)
if node_type in ["Agent", "Subagent", "Prompt"]:
    agent["llm"] = self._extract_llm_config(config)
    agent["tools"] = self._resolve_tools(config.get("tools", []))  # <-- TOOLS EXTRACTED HERE
```

`_resolve_tools()` converts tool names to IDs:

```python
def _resolve_tools(self, frontend_tools: List[Dict[str, Any]]) -> List[str]:
    tool_ids = []
    for tool in frontend_tools:
        tool_name = tool.get("name", "")
        if self.tool_registry:
            tool_id = self.tool_registry.get_tool_id_by_name(tool_name)
            if tool_id:
                tool_ids.append(tool_id)
        else:
            # CURRENT BEHAVIOR: Falls back to name-based placeholder
            tool_ids.append(tool_name.lower().replace(" ", "_"))
    return tool_ids
```

**Current Gap**: `self.tool_registry` is `None` because it's never initialized.

### Step 5: Agent Stored with Tools

**Location**: `apps/agent/registry/registry.py`

Agent is saved to JSON file with tools array:

```json
{
    "agent_id": "agt_xxx",
    "name": "Code Executor Agent",
    "role": "Autonomous AI agent",
    "tools": ["third-party_tool", "code_execution"],  // <-- Placeholder names
    "llm": {...},
    "metadata": {...}
}
```

### Step 6: Workflow Compilation (WHERE IT BREAKS)

**Location**: `apps/workflow/designer/compiler.py`, `apps/workflow/crewai_adapter.py`

During compilation, `WorkflowCompiler` creates LangGraph nodes using `CrewAIAdapter`:

```python
# crewai_adapter.py - create_sequential_agent_node() (line 345-517)
def sequential_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # ...
    agent = Agent(
        role=agent_config.get("role"),
        goal=agent_config.get("goal"),
        backstory=agent_config.get("description"),
        allow_delegation=False,
        llm=self._get_llm_for_agent(agent_config),
        verbose=True
        # NOTE: tools=[] MISSING! <-- THIS IS THE GAP
    )
```

**The tools array from agent_config is IGNORED**. No tools are bound to the CrewAI agent.

### Step 7: Workflow Execution (No Tool Invocation)

**Location**: `apps/workflow/runtime/executor.py`

When workflow executes:
1. LangGraph calls each agent node
2. CrewAI agent runs with LLM only (no tools)
3. Agent produces output based solely on LLM reasoning
4. Output passed to next agent via `crew_result`

**What's Missing**:
- CrewAI Agent doesn't have tools bound
- No tool invocation happens
- Tool output never enriches agent response

### Current Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CURRENT TOOL FLOW (INCOMPLETE)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FRONTEND (workflow_builder_ide.html)                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. User clicks "+ Tool" on Agent node                                │   │
│  │ 2. Selects tool type (tools/code/mcp_server/etc)                    │   │
│  │ 3. Configures tool (provider, action, credentials)                   │   │
│  │ 4. Tool stored in node.config.tools[]                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ POST /workflows/canvas/save                                          │   │
│  │ Body: { canvas_nodes: [...], connections: [...] }                    │   │
│  │ Each canvas_node has: { config: { tools: [...] } }                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  BACKEND                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ NodeMapper._convert_node_to_agent()                                  │   │
│  │ → _resolve_tools(config.tools) → ["tool_name_as_id"]                │   │
│  │ → Agent saved with tools: ["third-party_tool"]                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ WorkflowCompiler → CrewAIAdapter                                     │   │
│  │ → Creates CrewAI Agent(role, goal, llm)                              │   │
│  │ → tools=[] NOT PASSED ❌                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Workflow Execution                                                   │   │
│  │ → Agent runs with LLM only (no tools)                                │   │
│  │ → Output = LLM reasoning only                                        │   │
│  │ → crew_result passed to next agent                                   │   │
│  │ → NO TOOL INVOCATION ❌                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What This Plan Will Fix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROPOSED TOOL FLOW (COMPLETE)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FRONTEND (NO CHANGES REQUIRED)                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Same as current - UI flow remains unchanged                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  BACKEND (ENHANCED)                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ NodeMapper._resolve_tools() → Uses ToolRegistry ✅                   │   │
│  │ → Returns actual tool_ids (not placeholder names)                    │   │
│  │ → Agent saved with tools: ["tool_calculator", "tool_web_search"]     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ CrewAIAdapter (ENHANCED)                                             │   │
│  │ → Reads agent_config.tools                                           │   │
│  │ → Fetches ToolDef from ToolRegistry                                  │   │
│  │ → Creates CrewAI tool wrappers                                       │   │
│  │ → Agent(role, goal, llm, tools=[...]) ✅                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Workflow Execution (ENHANCED)                                        │   │
│  │ → Agent has tools bound                                              │   │
│  │ → LLM decides when to invoke tool                                    │   │
│  │ → ToolExecutor.invoke(tool_id, input) → REAL EXECUTION ✅            │   │
│  │ → Tool output merged into agent response                             │   │
│  │ → crew_result = tool_output + agent_reasoning                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### UI Tool Type to Backend Execution Mapping

| UI Tool Type | Frontend Config | Backend Resolution | Execution Method |
|--------------|-----------------|-------------------|------------------|
| `tools` (Third-party) | provider, action, credentials | ToolRegistry lookup by name | ToolExecutor._execute_local() or _execute_api() |
| `code` | language, code, timeout | Built-in code executor tool | ToolExecutor._execute_local() with code_executor |
| `mcp_server` | serverId, serverUrl, operationName | MCP connector lookup | ToolExecutor._execute_mcp() → POST /connectors/mcp/invoke |
| `subworkflow` | workflowId, inputMapping | Not a tool (workflow reference) | Handled separately by compiler |

### Constraints from Current UI Behavior

1. **Tool structure must remain compatible**:
   ```javascript
   tool = {
       id: number,
       name: string,
       type: "tools" | "code" | "mcp_server" | ...,
       enabled: boolean,
       config: { ... }
   }
   ```

2. **NodeMapper must continue to receive tools via `config.tools`**

3. **Backend must handle all tool types the UI supports**

4. **tool_registry lookup must support matching by name (for UI-entered names)**

---

## 2. Goals

### Primary Goals

1. **Real Tool Execution**: Tools must actually execute when invoked
2. **Schema Enforcement**: Input/output validation via JSON Schema
3. **Agent Integration**: Agents decide when to invoke tools during workflow execution
4. **Tool-Enriched Output**: Agent output includes tool results + agent reasoning
5. **Sequential Workflow Preservation**: Agent-to-agent data flow must remain intact
6. **MCP Transparency**: MCP tools work identically to local tools from agent perspective

### Secondary Goals

1. **Pluggable Architecture**: Add new tools without modifying core code
2. **Discovery Mechanism**: System discovers tools from external folder
3. **Per-Agent Tool Assignment**: Each agent has its own tool(s)
4. **Cost/Performance Tracking**: Track tool invocation metrics
5. **Graceful Degradation**: Handle tool failures without crashing workflow

### Non-Goals (Out of Scope)

- Tool marketplace/sharing
- Tool versioning
- Tool permissions per user
- Real-time tool hot-reloading
- Tool sandboxing/isolation

---

## 3. Constraints (Derived From Existing Code)

### DO NOT BREAK

1. **Route Signatures** (apps/tool/routes.py)
   - `POST /register` - must accept `ToolDef`
   - `GET /list` - must return tool references
   - `POST /invoke/{name}` - must execute tool and return result

2. **Workflow Execution Flow** (apps/workflow/runtime/executor.py)
   - `initial_state` structure with `user_input`, `original_user_input`, `messages`
   - `compiled_graph.invoke(initial_state, config)` pattern
   - `crew_result` state key for agent-to-agent data passing

3. **Agent Definition Schema** (apps/agent/registry/registry.py)
   - `agent_id`, `name`, `role`, `description`, `tools` (list of tool IDs)
   - `input_schema`, `output_schema` for workflow state mapping

4. **CrewAI Integration** (apps/workflow/crewai_adapter.py)
   - LangGraph owns topology
   - CrewAI executes within nodes only
   - State flows: LangGraph → CrewAI → LangGraph

5. **Dependency Injection** (echolib/di.py)
   - Services resolved via `container.resolve('service.name')`
   - Registration in `container.py` files

### MUST MAINTAIN

- Backward compatibility with existing workflows (tools=[] still works)
- JSON-based persistence (for now, DB later)
- Existing test suite must pass
- No breaking changes to Pydantic models

---

## 4. Architecture Design

### 4.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TOOL SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Tool        │    │  Tool        │    │  Tool                │  │
│  │  Discovery   │───▶│  Registry    │◀───│  Persistence         │  │
│  │  Service     │    │  (Manager)   │    │  (JSON → DB later)   │  │
│  └──────────────┘    └──────┬───────┘    └──────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                    TOOL EXECUTOR                                ││
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                ││
│  │  │ Local Tool │  │ MCP Tool   │  │ API Tool   │                ││
│  │  │ Executor   │  │ Executor   │  │ Executor   │                ││
│  │  └────────────┘  └────────────┘  └────────────┘                ││
│  └────────────────────────────────────────────────────────────────┘│
│                             │                                       │
│                             ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                 AGENT INTEGRATION LAYER                         ││
│  │  - Tool binding during workflow compilation                     ││
│  │  - Tool invocation decision (LLM-based)                         ││
│  │  - Tool output → Agent reasoning → Combined output              ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Tool Definition Schema (Enhanced)

```python
# File: echolib/types.py (EXTEND existing ToolDef)

class ToolType(str, Enum):
    LOCAL = "local"       # Python function in tools folder
    MCP = "mcp"           # MCP connector endpoint
    API = "api"           # Direct HTTP API call
    CREWAI = "crewai"     # CrewAI-native tool

class ToolDef(BaseModel):
    # Existing fields
    name: str
    description: str

    # NEW fields
    tool_id: str                              # Unique identifier (tool_xxx)
    tool_type: ToolType = ToolType.LOCAL      # How to execute
    input_schema: Dict[str, Any]              # JSON Schema for input validation
    output_schema: Dict[str, Any]             # JSON Schema for output validation

    # Execution config (depends on tool_type)
    execution_config: Dict[str, Any] = {}
    # LOCAL: {"module": "calculator.service", "class": "CalculatorService", "method": "calculate"}
    # MCP: {"connector_id": "mcp_xxx", "endpoint": "/invoke"}
    # API: {"url": "https://...", "method": "POST", "headers": {...}}

    # Metadata
    version: str = "1.0"
    tags: List[str] = []
    status: str = "active"                    # active, deprecated, disabled
    metadata: Dict[str, Any] = {}
```

### 4.3 Tool Registry (Manager)

```python
# File: echoAI/apps/tool/registry.py (NEW)

class ToolRegistry:
    """
    Central registry for all tools.
    Single source of truth for tool definitions.
    """

    def __init__(self, storage_dir: Path, discovery_dirs: List[Path] = None):
        self.storage_dir = storage_dir
        self.discovery_dirs = discovery_dirs or []
        self._cache: Dict[str, ToolDef] = {}
        self._load_all()

    # === CRUD Operations ===

    def register(self, tool: ToolDef) -> Dict[str, str]:
        """Register a new tool or update existing."""
        # Validate input/output schemas
        # Save to storage
        # Update cache
        pass

    def get(self, tool_id: str) -> Optional[ToolDef]:
        """Get tool by ID."""
        pass

    def list_all(self) -> List[ToolDef]:
        """List all registered tools."""
        pass

    def list_by_type(self, tool_type: ToolType) -> List[ToolDef]:
        """List tools by type."""
        pass

    def delete(self, tool_id: str) -> None:
        """Delete a tool."""
        pass

    # === Discovery ===

    def discover_local_tools(self) -> List[ToolDef]:
        """
        Scan discovery_dirs for tool manifests.
        Each tool folder should have a tool_manifest.json
        """
        pass

    # === Agent Binding ===

    def get_tools_for_agent(self, tool_ids: List[str]) -> List[ToolDef]:
        """Get tools assigned to an agent."""
        pass

    # === Validation ===

    def validate_tool_input(self, tool_id: str, input_data: Dict) -> bool:
        """Validate input against tool's input_schema."""
        pass
```

### 4.4 Tool Executor

```python
# File: echoAI/apps/tool/executor.py (NEW)

class ToolExecutor:
    """
    Executes tools based on their type.
    Handles input validation, execution, and output formatting.
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._local_instances: Dict[str, Any] = {}  # Cache for local tool instances

    async def invoke(
        self,
        tool_id: str,
        input_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> ToolResult:
        """
        Execute a tool and return the result.

        Args:
            tool_id: Tool identifier
            input_data: Input data for the tool
            context: Optional context (agent_id, workflow_id, etc.)

        Returns:
            ToolResult with output data
        """
        tool = self.registry.get(tool_id)
        if not tool:
            raise ValueError(f"Tool '{tool_id}' not found")

        # Validate input
        self._validate_input(tool, input_data)

        # Execute based on type
        if tool.tool_type == ToolType.LOCAL:
            result = await self._execute_local(tool, input_data)
        elif tool.tool_type == ToolType.MCP:
            result = await self._execute_mcp(tool, input_data)
        elif tool.tool_type == ToolType.API:
            result = await self._execute_api(tool, input_data)
        else:
            raise ValueError(f"Unknown tool type: {tool.tool_type}")

        # Validate output
        self._validate_output(tool, result)

        return ToolResult(
            name=tool.name,
            tool_id=tool_id,
            output=result,
            metadata={"execution_context": context}
        )

    async def _execute_local(self, tool: ToolDef, input_data: Dict) -> Dict:
        """Execute a local Python tool."""
        config = tool.execution_config
        module_path = config["module"]
        class_name = config["class"]
        method_name = config["method"]

        # Dynamic import and execution
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        # Get or create instance
        cache_key = f"{module_path}.{class_name}"
        if cache_key not in self._local_instances:
            self._local_instances[cache_key] = cls()

        instance = self._local_instances[cache_key]
        method = getattr(instance, method_name)

        # Execute (handle sync and async)
        if asyncio.iscoroutinefunction(method):
            return await method(input_data)
        else:
            return method(input_data)

    async def _execute_mcp(self, tool: ToolDef, input_data: Dict) -> Dict:
        """Execute tool via MCP connector."""
        config = tool.execution_config
        connector_id = config["connector_id"]

        # Call MCP connector's invoke endpoint
        # Uses existing ConnectorManager internally
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://localhost:8000/connectors/mcp/invoke",
                json={
                    "connector_id": connector_id,
                    "payload": input_data
                }
            )
            return response.json()

    async def _execute_api(self, tool: ToolDef, input_data: Dict) -> Dict:
        """Execute tool via direct HTTP API."""
        config = tool.execution_config
        url = config["url"]
        method = config.get("method", "POST")
        headers = config.get("headers", {})

        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                json=input_data,
                headers=headers
            )
            return response.json()
```

### 4.5 Agent Tools Folder Structure

```
echoAI/
└── AgentTools/                           # INSIDE echoAI folder
    ├── __init__.py                       # Makes it a Python package
    │
    ├── calculator/
    │   ├── __init__.py
    │   ├── tool_manifest.json            # Tool definition
    │   ├── service.py                    # CalculatorService class
    │   └── models.py
    │
    ├── math/
    │   ├── __init__.py
    │   ├── arithmetic.py
    │   ├── statistics.py
    │   └── linear_algebra.py
    │
    ├── web_search/
    │   ├── __init__.py
    │   ├── tool_manifest.json
    │   ├── service.py                    # WebSearchService class
    │   ├── models.py
    │   ├── interfaces.py
    │   └── providers/
    │       ├── __init__.py
    │       ├── google.py
    │       ├── bing.py
    │       └── duckduckgo.py
    │
    ├── file_reader/
    │   ├── __init__.py
    │   ├── tool_manifest.json
    │   ├── service.py                    # FileReaderService class
    │   ├── models.py
    │   ├── registry.py
    │   ├── embeddings.py
    │   ├── vector_store.py
    │   ├── summarizer.py
    │   ├── stream_utils.py
    │   ├── parsers/
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── pdf_parser.py
    │   │   ├── json_parser.py
    │   │   └── xml_parser.py
    │   └── csv_capability/
    │       ├── __init__.py
    │       ├── csv_agent.py
    │       ├── csv_summarizer.py
    │       └── stream_handler.py
    │
    ├── code_generator/
    │   ├── __init__.py
    │   ├── tool_manifest.json
    │   └── service.py                    # CodeGeneratorService class
    │
    └── code_reviewer/
        ├── __init__.py
        ├── tool_manifest.json
        └── service.py                    # CodeReviewerService class
```

**Note**: This structure mirrors the existing `Tools I made/` folder. During Phase 4, we will:
1. Rename `Tools I made/` to `AgentTools/`
2. Add `tool_manifest.json` to each tool folder
3. Update import paths as needed

### 4.6 Tool Manifest Schema

```json
// tools/calculator/tool_manifest.json
{
  "tool_id": "tool_calculator",
  "name": "Calculator",
  "description": "Performs mathematical calculations including arithmetic, statistics, and linear algebra",
  "tool_type": "local",
  "version": "1.0",
  "tags": ["math", "calculation", "utility"],

  "input_schema": {
    "type": "object",
    "properties": {
      "operation": {
        "type": "string",
        "enum": ["add", "subtract", "multiply", "divide", "sqrt", "mean", "median"]
      },
      "values": {
        "type": "array",
        "items": {"type": "number"}
      },
      "precision": {
        "type": "integer",
        "default": 2
      }
    },
    "required": ["operation", "values"]
  },

  "output_schema": {
    "type": "object",
    "properties": {
      "operation": {"type": "string"},
      "result": {"type": ["number", "array"]}
    },
    "required": ["operation", "result"]
  },

  "execution_config": {
    "module": "AgentTools.calculator.service",
    "class": "CalculatorService",
    "method": "calculate"
  }
}
```

### 4.7 Agent Integration (Critical)

#### A. Tool Binding During Workflow Compilation

```python
# File: apps/workflow/crewai_adapter.py (MODIFY)

def create_sequential_agent_node(self, agent_config: Dict[str, Any]) -> Callable:
    """Create LangGraph node with tool-enabled CrewAI agent."""

    def sequential_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        from crewai import Agent, Task, Crew

        # ... existing code for context extraction ...

        # NEW: Bind tools to agent
        tool_executor = self._get_tool_executor()
        tool_ids = agent_config.get("tools", [])
        crewai_tools = []

        for tool_id in tool_ids:
            tool_def = tool_executor.registry.get(tool_id)
            if tool_def:
                # Create CrewAI-compatible tool wrapper
                crewai_tool = self._create_crewai_tool_wrapper(tool_def, tool_executor)
                crewai_tools.append(crewai_tool)

        # Create agent WITH tools
        agent = Agent(
            role=agent_config.get("role"),
            goal=agent_config.get("goal"),
            backstory=agent_config.get("description"),
            tools=crewai_tools,  # <-- NOW ACTUALLY BINDS TOOLS
            llm=self._get_llm_for_agent(agent_config),
            verbose=True
        )

        # ... rest of execution ...

    return sequential_agent_node


def _create_crewai_tool_wrapper(self, tool_def: ToolDef, executor: ToolExecutor):
    """Create a CrewAI-compatible tool from our ToolDef."""
    from crewai.tools import BaseTool
    from pydantic import Field

    # Dynamically create tool class
    class DynamicTool(BaseTool):
        name: str = tool_def.name
        description: str = tool_def.description

        def _run(self, **kwargs) -> str:
            # Synchronous wrapper around async executor
            import asyncio
            result = asyncio.run(executor.invoke(tool_def.tool_id, kwargs))
            return json.dumps(result.output)

    return DynamicTool()
```

#### B. Agent Tool Invocation Decision

The agent decides when to invoke a tool based on:

1. **Task Requirements**: The task description mentions the tool capability
2. **Input Availability**: Required tool inputs are present in state
3. **LLM Reasoning**: The LLM decides tool is needed for the task

CrewAI handles this automatically when tools are bound to agents. The agent's LLM will:
- See the available tools and their descriptions
- Decide which tool(s) to call based on the task
- Format the tool input
- Receive and interpret the tool output
- Incorporate tool output into the final response

#### C. Tool Output → Agent Reasoning → Combined Output

```
Sequential Workflow Example:
===========================

[User Input]: "Write a Python function to calculate factorial and test it"

[Agent 1: Code Executor]
├── Receives: user_input
├── Has Tool: code_executor_tool
├── LLM Reasoning: "I need to write code. Let me use my code execution tool."
├── Tool Invocation: code_executor_tool.execute({"code": "def factorial(n)...", "language": "python"})
├── Tool Output: {"status": "success", "output": "120", "execution_time": "0.02s"}
├── Agent Output: "I wrote and executed a factorial function. Here's the code: ... The test returned 120."
└── State Update: crew_result = "Code: def factorial(n)... Test output: 120"

[Agent 2: Code Reviewer]
├── Receives: crew_result (from Agent 1)
├── Has Tool: code_review_tool
├── LLM Reasoning: "I need to review this code. Let me analyze it."
├── Tool Invocation: code_review_tool.review({"code": "def factorial(n)...", "criteria": ["correctness", "style"]})
├── Tool Output: {"score": 85, "issues": ["No docstring", "Could use recursion guard"], "suggestions": [...]}
├── Agent Output: "I reviewed the code. Score: 85/100. Issues: No docstring, missing recursion guard."
└── State Update: crew_result = "Review: Score 85/100. Issues: ..."

[Agent 3: Code Tester]
├── Receives: crew_result (from Agent 2), original_user_input
├── Has Tool: code_tester_tool
├── LLM Reasoning: "I need to test this code thoroughly."
├── Tool Invocation: code_tester_tool.test({"code": "def factorial(n)...", "test_cases": [0, 1, 5, 10]})
├── Tool Output: {"passed": 4, "failed": 0, "coverage": "100%", "results": [...]}
├── Agent Output: "All tests passed. Coverage: 100%. Results: ..."
└── State Update: crew_result = "Tests: 4/4 passed, 100% coverage"

[Final Output]: Combined result from all agents with tool outputs integrated
```

### 4.8 MCP as a Tool Type

MCP connectors become tools transparently:

```json
// Registered via POST /tools/register or auto-discovered
{
  "tool_id": "tool_mcp_github",
  "name": "GitHub MCP",
  "description": "Interact with GitHub via MCP connector",
  "tool_type": "mcp",
  "input_schema": {
    "type": "object",
    "properties": {
      "action": {"type": "string", "enum": ["list_repos", "create_issue", "get_pr"]},
      "params": {"type": "object"}
    },
    "required": ["action"]
  },
  "output_schema": {
    "type": "object"
  },
  "execution_config": {
    "connector_id": "mcp_github_connector_001",
    "endpoint": "/invoke"
  }
}
```

When invoked, ToolExecutor calls:
```python
POST http://localhost:8000/connectors/mcp/invoke
{
  "connector_id": "mcp_github_connector_001",
  "payload": { ... }
}
```

The agent never knows it's using MCP. Same interface as local tools.

### 4.9 Persistence Layer

```python
# File: echoAI/apps/tool/storage.py (NEW)

class ToolStorage:
    """
    Persistence layer for tools.
    Currently JSON-based, designed for easy DB migration.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = storage_dir / "tool_index.json"

    def save_tool(self, tool: ToolDef) -> str:
        """Save tool to JSON file."""
        file_path = self.storage_dir / f"{tool.tool_id}.json"
        with open(file_path, 'w') as f:
            json.dump(tool.model_dump(), f, indent=2)
        self._update_index(tool)
        return str(file_path)

    def load_tool(self, tool_id: str) -> Optional[ToolDef]:
        """Load tool from JSON file."""
        file_path = self.storage_dir / f"{tool_id}.json"
        if not file_path.exists():
            return None
        with open(file_path) as f:
            data = json.load(f)
        return ToolDef(**data)

    def load_all(self) -> List[ToolDef]:
        """Load all tools."""
        tools = []
        for file_path in self.storage_dir.glob("tool_*.json"):
            with open(file_path) as f:
                data = json.load(f)
            tools.append(ToolDef(**data))
        return tools

    def delete_tool(self, tool_id: str) -> None:
        """Delete tool file."""
        file_path = self.storage_dir / f"{tool_id}.json"
        if file_path.exists():
            file_path.unlink()
        self._remove_from_index(tool_id)

    def _update_index(self, tool: ToolDef) -> None:
        """Update master index file."""
        index = self._load_index()
        index["tools"][tool.tool_id] = {
            "name": tool.name,
            "tool_type": tool.tool_type,
            "status": tool.status
        }
        self._save_index(index)

    # Migration hook for future DB integration
    def migrate_to_db(self, db_connection):
        """Future: Migrate JSON data to database."""
        pass
```

### 4.10 Updated Routes

```python
# File: echoAI/apps/tool/routes.py (MODIFY - keep signatures, change implementation)

from fastapi import APIRouter, Depends, HTTPException
from echolib.di import container
from echolib.types import ToolDef, ToolResult

router = APIRouter(prefix='/tools', tags=['ToolApi'])

def registry() -> ToolRegistry:
    return container.resolve('tool.registry')

def executor() -> ToolExecutor:
    return container.resolve('tool.executor')

# EXISTING ROUTE - same signature, new implementation
@router.post('/register')
async def register(tool: ToolDef):
    """Register a new tool."""
    try:
        result = registry().register(tool)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# EXISTING ROUTE - same signature, new implementation
@router.get('/list')
async def list_tools():
    """List all registered tools."""
    tools = registry().list_all()
    return [{"tool_id": t.tool_id, "name": t.name, "tool_type": t.tool_type} for t in tools]

# EXISTING ROUTE - same signature, new implementation
@router.post('/invoke/{name}')
async def invoke(name: str, args: dict):
    """Invoke a tool by name."""
    try:
        # Find tool by name (for backward compatibility)
        tools = registry().list_all()
        tool = next((t for t in tools if t.name == name), None)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")

        result = await executor().invoke(tool.tool_id, args)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# NEW ROUTES

@router.post('/invoke/id/{tool_id}')
async def invoke_by_id(tool_id: str, args: dict):
    """Invoke a tool by ID."""
    try:
        result = await executor().invoke(tool_id, args)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/{tool_id}')
async def get_tool(tool_id: str):
    """Get tool definition by ID."""
    tool = registry().get(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
    return tool.model_dump()

@router.post('/discover')
async def discover_tools():
    """Trigger tool discovery from external folders."""
    discovered = registry().discover_local_tools()
    return {"discovered": len(discovered), "tools": [t.tool_id for t in discovered]}

@router.get('/agent/{agent_id}')
async def get_agent_tools(agent_id: str):
    """Get tools assigned to an agent."""
    agent_registry = container.resolve('agent.registry')
    agent = agent_registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    tool_ids = agent.get("tools", [])
    tools = registry().get_tools_for_agent(tool_ids)
    return {"agent_id": agent_id, "tools": [t.model_dump() for t in tools]}
```

---

## 5. Container/DI Updates

```python
# File: echoAI/apps/tool/container.py (MODIFY)

from echolib.di import container
from pathlib import Path

# Import new components
from .registry import ToolRegistry
from .executor import ToolExecutor
from .storage import ToolStorage

# Configuration
TOOLS_STORAGE_DIR = Path(__file__).parent.parent / "storage" / "tools"
TOOLS_DISCOVERY_DIRS = [
    Path(__file__).parent.parent.parent / "AgentTools"  # echoAI/AgentTools folder
]

# Initialize storage
_storage = ToolStorage(TOOLS_STORAGE_DIR)

# Initialize registry with discovery
_registry = ToolRegistry(
    storage=_storage,
    discovery_dirs=TOOLS_DISCOVERY_DIRS
)

# Initialize executor
_executor = ToolExecutor(registry=_registry)

# Register in DI container
container.register('tool.storage', lambda: _storage)
container.register('tool.registry', lambda: _registry)
container.register('tool.executor', lambda: _executor)

# Keep backward compatibility with ToolService
from echolib.services import ToolService
_tool_service = ToolService()
container.register('tool.service', lambda: _tool_service)
```

---

## 6. Why This Approach

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Modify existing ToolService directly** | Minimal code changes | Breaks encapsulation, mixes concerns | Rejected |
| **CrewAI-native tools only** | Simple integration | Limits tool types, locks to CrewAI | Rejected |
| **Separate microservice for tools** | Clean separation | Over-engineering, latency overhead | Rejected |
| **Adapter pattern with registry** | Clean, extensible, supports all tool types | More initial code | **SELECTED** |

### Why Selected Approach is Better

1. **Separation of Concerns**
   - ToolRegistry: Discovery and management
   - ToolExecutor: Execution logic
   - ToolStorage: Persistence
   - Each can be modified independently

2. **Extensibility**
   - Add new tool types by adding executor methods
   - No modification to core agent/workflow code

3. **MCP Transparency**
   - Agents don't know if a tool is local or MCP
   - Same interface for all tool types

4. **Backward Compatibility**
   - Existing routes unchanged
   - Existing ToolService still works
   - Workflows without tools continue to work

5. **Future-Proof**
   - JSON storage can migrate to DB
   - Tool versioning can be added
   - Tool permissions can be layered on

---

## 7. Implementation Sequence

### Phase 1: Foundation (Days 1-3)
1. Enhance ToolDef model in types.py
2. Create ToolStorage class
3. Create ToolRegistry class
4. Update container.py for new components

### Phase 2: Execution (Days 4-6)
5. Create ToolExecutor class
6. Implement local tool execution
7. Implement MCP tool execution
8. Update routes.py with new implementations

### Phase 3: Agent Integration (Days 7-9)
9. Update CrewAI adapter for tool binding
10. Create CrewAI tool wrapper factory
11. Test tool invocation in sequential workflow
12. Test tool output propagation between agents

### Phase 4: AgentTools Setup (Days 10-12)
13. Move existing tools to external folder
14. Create tool_manifest.json for each tool
15. Implement tool discovery
16. Test discovery and registration

### Phase 5: Testing & Polish (Days 13-14)
17. Unit tests for all new components
18. Integration tests for tool-enabled workflows
19. Documentation
20. Performance optimization

---

## 8. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `echolib/types.py` | MODIFY | Extend ToolDef, add ToolType enum, enhance ToolResult |
| `apps/tool/container.py` | MODIFY | Register new components |
| `apps/tool/routes.py` | MODIFY | Update implementations, add new routes |
| `apps/tool/registry.py` | CREATE | ToolRegistry class |
| `apps/tool/executor.py` | CREATE | ToolExecutor class |
| `apps/tool/storage.py` | CREATE | ToolStorage class |
| `apps/workflow/crewai_adapter.py` | MODIFY | Add tool binding logic |
| `apps/agent/factory/factory.py` | MODIFY | Real tool binding |
| `echoAI/AgentTools/` | MODIFY | Rename from `Tools I made/`, add tool manifests |
| `tests/test_tool_system.py` | CREATE | Comprehensive tests |

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing workflows | Low | High | Backward compatibility layer, comprehensive testing |
| Tool execution failures | Medium | Medium | Graceful error handling, fallback to LLM-only |
| MCP connector unavailability | Medium | Medium | Timeout handling, retry logic |
| Performance degradation | Low | Medium | Async execution, caching |
| Tool input validation failures | Medium | Low | Clear error messages, schema documentation |

---

## 10. Success Criteria

1. **Functional**
   - Tools execute with real results (not just echoed)
   - Agents invoke tools during workflow execution
   - Tool output is incorporated into agent responses
   - Sequential workflow data flow preserved

2. **Non-Functional**
   - Tool invocation < 5s (local), < 30s (MCP)
   - No breaking changes to existing API
   - All existing tests pass

3. **Verification**
   - Demo: 3-agent sequential workflow with one tool each
   - All tools in "Tools I made" folder are registered and invocable
   - MCP connector can be used as a tool

---

## Appendix A: Tool Manifest Examples

### Calculator Tool

```json
{
  "tool_id": "tool_calculator",
  "name": "Calculator",
  "description": "Performs mathematical operations",
  "tool_type": "local",
  "version": "1.0",
  "tags": ["math", "calculation"],
  "input_schema": {
    "type": "object",
    "properties": {
      "operation": {"type": "string"},
      "values": {"type": "array", "items": {"type": "number"}}
    },
    "required": ["operation", "values"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "operation": {"type": "string"},
      "result": {}
    }
  },
  "execution_config": {
    "module": "AgentTools.calculator.service",
    "class": "CalculatorService",
    "method": "calculate"
  }
}
```

### MCP GitHub Tool

```json
{
  "tool_id": "tool_mcp_github",
  "name": "GitHub MCP",
  "description": "GitHub operations via MCP",
  "tool_type": "mcp",
  "version": "1.0",
  "tags": ["github", "vcs", "mcp"],
  "input_schema": {
    "type": "object",
    "properties": {
      "action": {"type": "string"},
      "params": {"type": "object"}
    },
    "required": ["action"]
  },
  "output_schema": {"type": "object"},
  "execution_config": {
    "connector_id": "mcp_github_001",
    "endpoint": "/invoke"
  }
}
```

---

## Appendix B: Agent Tool Binding Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WORKFLOW COMPILATION                                  │
│                                                                         │
│  Workflow JSON ──▶ WorkflowCompiler ──▶ LangGraph StateGraph            │
│                          │                                              │
│                          │ For each agent:                              │
│                          ▼                                              │
│                  ┌───────────────────┐                                  │
│                  │ Create Agent Node  │                                  │
│                  └─────────┬─────────┘                                  │
│                            │                                             │
│                            │ agent_config.tools = ["tool_calc", "tool_x"]│
│                            ▼                                             │
│                  ┌───────────────────┐                                  │
│                  │ CrewAIAdapter.    │                                  │
│                  │ create_sequential │                                  │
│                  │ _agent_node()     │                                  │
│                  └─────────┬─────────┘                                  │
│                            │                                             │
│                            │ For each tool_id:                          │
│                            ▼                                             │
│                  ┌───────────────────┐                                  │
│                  │ ToolRegistry.get()│──▶ ToolDef                       │
│                  └─────────┬─────────┘                                  │
│                            │                                             │
│                            ▼                                             │
│                  ┌───────────────────┐                                  │
│                  │ _create_crewai_   │                                  │
│                  │ tool_wrapper()    │──▶ CrewAI Tool                   │
│                  └─────────┬─────────┘                                  │
│                            │                                             │
│                            ▼                                             │
│                  ┌───────────────────┐                                  │
│                  │ CrewAI Agent(     │                                  │
│                  │   tools=[...]     │  ◀── Tools bound to agent        │
│                  │ )                 │                                  │
│                  └───────────────────┘                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix C: Sequential Workflow With Tools - Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      RUNTIME EXECUTION                                    │
│                                                                          │
│  initial_state = {                                                       │
│    "user_input": "Write factorial function and test it",                 │
│    "original_user_input": "Write factorial function and test it",        │
│    "messages": []                                                        │
│  }                                                                       │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ AGENT 1: Code Executor (tools: [tool_code_executor])                 ││
│  │                                                                       ││
│  │  Input: state["user_input"]                                          ││
│  │                                                                       ││
│  │  LLM Reasoning: "I need to write and execute code"                   ││
│  │                                                                       ││
│  │  ┌─────────────────────────────────────────────────────────────┐    ││
│  │  │ TOOL INVOCATION                                               │    ││
│  │  │  tool_code_executor.execute({                                 │    ││
│  │  │    "code": "def factorial(n): ...",                          │    ││
│  │  │    "language": "python"                                       │    ││
│  │  │  })                                                           │    ││
│  │  │                                                               │    ││
│  │  │  Returns: {"status": "success", "output": "120"}              │    ││
│  │  └─────────────────────────────────────────────────────────────┘    ││
│  │                                                                       ││
│  │  Agent combines: tool output + own analysis                          ││
│  │                                                                       ││
│  │  Output: crew_result = "Code written and tested. Output: 120"        ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                           │                                              │
│                           ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ AGENT 2: Code Reviewer (tools: [tool_code_reviewer])                 ││
│  │                                                                       ││
│  │  Input: state["crew_result"] (from Agent 1)                          ││
│  │         state["original_user_input"]                                  ││
│  │                                                                       ││
│  │  LLM Reasoning: "I received code, need to review it"                 ││
│  │                                                                       ││
│  │  ┌─────────────────────────────────────────────────────────────┐    ││
│  │  │ TOOL INVOCATION                                               │    ││
│  │  │  tool_code_reviewer.review({                                  │    ││
│  │  │    "code": "def factorial(n): ...",                          │    ││
│  │  │    "criteria": ["correctness", "style", "efficiency"]         │    ││
│  │  │  })                                                           │    ││
│  │  │                                                               │    ││
│  │  │  Returns: {"score": 85, "issues": [...], "suggestions": [...]}│    ││
│  │  └─────────────────────────────────────────────────────────────┘    ││
│  │                                                                       ││
│  │  Agent combines: tool output + own review notes                      ││
│  │                                                                       ││
│  │  Output: crew_result = "Review complete. Score: 85. Issues: ..."     ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                           │                                              │
│                           ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ AGENT 3: Code Tester (tools: [tool_code_tester])                     ││
│  │                                                                       ││
│  │  Input: state["crew_result"] (from Agent 2)                          ││
│  │         state["original_user_input"]                                  ││
│  │                                                                       ││
│  │  LLM Reasoning: "I need to test this code thoroughly"                ││
│  │                                                                       ││
│  │  ┌─────────────────────────────────────────────────────────────┐    ││
│  │  │ TOOL INVOCATION                                               │    ││
│  │  │  tool_code_tester.test({                                      │    ││
│  │  │    "code": "def factorial(n): ...",                          │    ││
│  │  │    "test_cases": [{"input": [5], "expected": 120}, ...]       │    ││
│  │  │  })                                                           │    ││
│  │  │                                                               │    ││
│  │  │  Returns: {"passed": 5, "failed": 0, "coverage": "100%"}      │    ││
│  │  └─────────────────────────────────────────────────────────────┘    ││
│  │                                                                       ││
│  │  Agent combines: tool output + final summary                         ││
│  │                                                                       ││
│  │  Output: crew_result = "All tests passed. 100% coverage."            ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                           │                                              │
│                           ▼                                              │
│  final_state = {                                                         │
│    "crew_result": "All tests passed. 100% coverage.",                   │
│    "original_user_input": "Write factorial function and test it",        │
│    "messages": [agent1_msg, agent2_msg, agent3_msg]                      │
│  }                                                                       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

**END OF PLAN**
