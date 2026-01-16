# Agent Builder Implementation Summary

## ✅ What Was Implemented

### 1. **Updated Agent Schema**
**File:** `echolib/schemas/agent_schema.json`

**New Fields Added:**
- `icon` (string) - Emoji icon for agent (e.g., "🔬")
- `prompt` (string) - System prompt for agent behavior
- `model` (string) - Model identifier (e.g., "mistral-nemo-12b")
- `variables` (array) - Runtime variables with name, type, defaultValue
- `settings` (object) - Contains temperature, max_token, top_p, max_iteration

**Backward Compatibility:**
- Kept `llm` field as deprecated for existing agents
- Changed required fields to minimum: agent_id, name, role, tools

### 2. **LLM Provider Configuration**
**File:** `llm_provider.json`

Models configured:
- `mistral-nemo-12b` (Ollama)
- `gpt-4o-mini` (OpenAI)
- `gpt-4o` (OpenAI)
- `claude-sonnet-4-5` (Anthropic)
- `claude-opus-4-5` (Anthropic)

Each model has:
- `id` - Unique identifier
- `provider` - ollama/openai/anthropic
- `base_url` - API endpoint
- `model_name` - Actual model name to pass to API

### 3. **Agent Designer Service**
**File:** `apps/agent/designer/agent_designer.py`

**Features:**
- Designs agents from natural language prompts using LLM
- Reads model config from `llm_provider.json`
- Supports all configured LLM providers
- Auto-generates agent spec (name, role, description, prompt, I/O schema)
- Falls back to basic structure if LLM fails

**Key Methods:**
- `design_from_prompt()` - Main entry point
- `_design_with_llm()` - Uses LLM to analyze prompt
- `_get_llm_client()` - Resolves model from llm_provider.json

### 4. **Master Agent List**
**File:** `apps/storage/agents/ai_agents.json`

**Structure:**
```json
{
  "agents": [
    {
      "agent_id": "agt_xxx",
      "name": "Research Analyst",
      "icon": "🔬",
      "role": "...",
      "description": "..."
    }
  ]
}
```

**Purpose:**
- Lists all agents for Workflow Builder display
- Lightweight summaries (not full agent definitions)
- Updated automatically when agents are created/deleted

### 5. **Updated Agent Registry**
**File:** `apps/agent/registry/registry.py`

**New Features:**
- Saves individual agent JSON files
- Updates `ai_agents.json` master list automatically
- Loads agents from storage on startup
- Skips `ai_agents.json` when loading individual agents

**New Methods:**
- `_load_master_list()` - Load master list
- `_save_master_list()` - Save master list
- `_update_master_list_add()` - Add agent to master list
- `_update_master_list_remove()` - Remove agent from master list
- `get_master_list()` - Get master list for workflow builder

### 6. **New API Endpoints**

#### **POST** `/agents/design/prompt`
Create agent from natural language prompt.

**Request:**
```json
{
  "prompt": "Create an agent that analyzes customer feedback",
  "model": "mistral-nemo-12b",
  "icon": "🔬",
  "tools": ["Web Search", "Web Fetch"],
  "variables": [
    {
      "name": "research_topic",
      "type": "string",
      "defaultValue": ""
    }
  ]
}
```

**Response:**
```json
{
  "agent": {
    "agent_id": "agt_xxx",
    "name": "Customer Feedback Analyzer",
    "icon": "🔬",
    "role": "Sentiment Analysis",
    "description": "...",
    "prompt": "You are...",
    "model": "mistral-nemo-12b",
    "tools": ["Web Search"],
    "variables": [...],
    "settings": {
      "temperature": 0.7,
      "max_token": 2000,
      "top_p": 0.9,
      "max_iteration": 5
    }
  }
}
```

#### **GET** `/agents/registry/master-list`
Get master agent list for workflow builder.

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "agt_001",
      "name": "Research Analyst",
      "icon": "🔬",
      "role": "Data Analysis",
      "description": "..."
    }
  ]
}
```

### 7. **Workflow Designer Integration**
**File:** `apps/workflow/designer/designer.py`

**Changes:**
- Injects `agent_registry` into constructor
- Saves all workflow agents to registry automatically
- Agents from workflow creation now appear in master list

**Behavior:**
- When workflow is created from prompt
- Each agent is saved individually to `storage/agents/{agent_id}.json`
- Master list is updated
- Workflow Builder can now see and reuse these agents

### 8. **Storage Structure**

```
echoAI/
├── llm_provider.json              ← Model configurations
├── apps/
│   └── storage/
│       └── agents/
│           ├── ai_agents.json     ← Master list
│           ├── agt_001.json       ← Individual agents
│           ├── agt_002.json
│           └── agt_003.json
```

---

## 🔧 How to Test

### 1. **Start the Server**
```bash
cd echoAI
pip install -r requirements.txt
uvicorn apps.gateway.main:app --reload --port 8000
```

### 2. **Create Agent from Prompt**
```bash
curl -X POST http://localhost:8000/agents/design/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create an agent that does research from web for a stock, performs technical and fundamental analysis and suggests whether to buy, sell or hold",
    "model": "mistral-nemo-12b",
    "icon": "📈",
    "tools": ["Web Search", "Web Fetch", "Document Analysis"],
    "variables": [
      {
        "name": "stock_symbol",
        "type": "string",
        "defaultValue": "AAPL"
      },
      {
        "name": "analysis_depth",
        "type": "string",
        "defaultValue": "comprehensive"
      }
    ]
  }'
```

### 3. **Get Master Agent List** (for Workflow Builder)
```bash
curl http://localhost:8000/agents/registry/master-list
```

### 4. **Get Individual Agent**
```bash
curl http://localhost:8000/agents/agt_xxx
```

### 5. **Create Workflow** (agents auto-saved)
```bash
curl -X POST http://localhost:8000/workflows/design/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a workflow for stock analysis"
  }'
```

Check `apps/storage/agents/` - you'll see individual agent JSON files created!

---

## 🔗 Integration with Frontend

### Agent Builder Page

**When user clicks "Create Agent":**
```javascript
const response = await fetch('/agents/design/prompt', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: userPrompt,
    icon: selectedIcon,
    tools: selectedTools,
    variables: definedVariables,
    model: selectedModel
  })
});

const { agent } = await response.json();
// Agent is now created and registered
```

### Workflow Builder Page

**Load available agents:**
```javascript
const response = await fetch('/agents/registry/master-list');
const { agents } = await response.json();

// Display agents in dropdown/grid
agents.forEach(agent => {
  console.log(agent.name, agent.icon, agent.role);
});
```

**Use agent in workflow:**
```javascript
// Agent is already registered
// Just reference by agent_id in workflow
const workflow = {
  agents: ["agt_001", "agt_002"],
  // ...
};
```

---

## ✅ What Works Now

1. ✅ Create agents from prompt in Agent Builder
2. ✅ Agents saved individually to JSON files
3. ✅ Master list automatically updated
4. ✅ Workflow Builder can fetch all available agents
5. ✅ Workflow creation also saves agents individually
6. ✅ Model configuration separated from agent definition
7. ✅ Backward compatibility maintained

---

## 🎯 Key Benefits

1. **Separation of Concerns:**
   - Agent definition separate from workflow
   - Agents can be reused across workflows
   - Model config centralized in `llm_provider.json`

2. **Frontend Integration:**
   - Agent Builder creates standalone agents
   - Workflow Builder lists all available agents
   - Clean API endpoints

3. **Storage Efficiency:**
   - Individual agent files for easy editing
   - Master list for quick overview
   - No duplication of agent definitions

4. **Flexibility:**
   - Change model without changing agent
   - Update agent independently
   - Version control friendly (individual JSON files)

---

## 📝 Next Steps (Optional)

1. **Frontend UI:**
   - Build Agent Builder page (`type-agent_builder.html`)
   - Add agent dropdown in Workflow Builder
   - Agent editing UI

2. **Model Selection:**
   - UI dropdown to select from available models
   - Model settings customization

3. **Tool Selection:**
   - UI for selecting tools from available MCP tools
   - Tool configuration interface

4. **Testing:**
   - End-to-end test: Agent Builder → Workflow Builder
   - Test agent reusability
   - Test master list updates

---

## 🐛 Troubleshooting

**Agent not appearing in master list:**
- Check `apps/storage/agents/ai_agents.json`
- Verify agent was registered (check `agt_xxx.json` exists)
- Check backend logs for errors

**Model not working:**
- Check `llm_provider.json` has correct configuration
- Verify Ollama/OpenAI is running and accessible
- Check environment variables (if using OpenAI)

**Workflow agents not saving:**
- Verify `agent_registry` is injected into `WorkflowDesigner`
- Check `apps/workflow/container.py` initialization
- Check backend logs for registration errors

---

## 📂 Files Modified/Created

**Created:**
1. `llm_provider.json`
2. `apps/agent/designer/agent_designer.py`
3. `apps/agent/designer/__init__.py`
4. `apps/storage/agents/ai_agents.json`
5. `AGENT_BUILDER_IMPLEMENTATION.md` (this file)

**Modified:**
1. `echolib/schemas/agent_schema.json`
2. `apps/agent/registry/registry.py`
3. `apps/agent/routes.py`
4. `apps/agent/container.py`
5. `apps/workflow/designer/designer.py`
6. `apps/workflow/container.py`

**Total:** 11 files

---

## ✨ Summary

Everything you requested is implemented:
- ✅ Prompt-based agent creation
- ✅ New agent schema with icon, prompt, variables, settings
- ✅ Model field instead of llm provider
- ✅ Individual agent JSON files
- ✅ Master agent list (`ai_agents.json`)
- ✅ Agents from workflows also saved individually
- ✅ Workflow Builder integration ready
- ✅ No code revamped, only updated/extended
- ✅ Backward compatibility maintained

**Ready for frontend integration!** 🚀

---

## 📊 Input/Output Schema Logic

### What Are They?

**Input Schema:** List of data keys the agent **expects to receive** from previous agent/workflow.

**Output Schema:** List of data keys the agent **will produce** and pass to next agent.

### Example Flow

```
Agent 1 (Data Fetcher):
  input_schema: []
  output_schema: ["sales_data"]

Agent 2 (Analyzer):
  input_schema: ["sales_data"]      ← Must match Agent 1 output
  output_schema: ["analysis_result"]

Agent 3 (Reporter):
  input_schema: ["analysis_result"] ← Must match Agent 2 output
  output_schema: ["final_report"]
```

**Purpose:** Ensures data flows correctly between agents in workflows.

---

## ✨ NEW: I/O Schema Workflow

### Agent Builder Behavior (UPDATED)

**Agents created in Agent Builder now have EMPTY I/O schemas:**
```json
{
  "agent_id": "agt_001",
  "name": "Stock Analyzer",
  "input_schema": [],   ← Empty by default
  "output_schema": []   ← Empty by default
}
```

### Workflow Builder Behavior (NEW)

**When reusing agents in Workflow Builder, define I/O schemas based on workflow data flow:**

#### Option 1: Update Base Agent Definition

```bash
PATCH /agents/agt_001/schema
{
  "input_schema": ["stock_data"],
  "output_schema": ["analysis"]
}
```

This updates the agent permanently in the registry.

#### Option 2: Define I/O Schemas Per Workflow (Recommended)

```bash
POST /workflows/build
{
  "workflow": {
    "workflow_id": "wf_001",
    "agents": ["agt_001", "agt_002"],
    "connections": [{"from": "agt_001", "to": "agt_002"}]
  },
  "agent_schemas": {
    "agt_001": {
      "input_schema": ["stock_symbol"],
      "output_schema": ["technical_analysis", "fundamental_analysis"]
    },
    "agt_002": {
      "input_schema": ["technical_analysis", "fundamental_analysis"],
      "output_schema": ["recommendation"]
    }
  },
  "update_base_agents": false
}
```

**Benefits:**
- ✅ Same agent can have different I/O schemas in different workflows
- ✅ Base agent definition stays flexible
- ✅ Workflow-specific data flow
- ✅ Optional: Set `update_base_agents: true` to update registry

---

## 🔄 Complete Workflow: Agent Builder → Workflow Builder

### Step 1: Create Agent in Agent Builder

```bash
POST /agents/design/prompt
{
  "prompt": "Create agent for stock analysis",
  "icon": "📈",
  "tools": ["Web Search", "Financial API"]
}
```

**Response:**
```json
{
  "agent": {
    "agent_id": "agt_stock_001",
    "name": "Stock Analyzer",
    "input_schema": [],    ← Empty
    "output_schema": []    ← Empty
  }
}
```

### Step 2: Get Available Agents in Workflow Builder

```bash
GET /agents/registry/master-list
```

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "agt_stock_001",
      "name": "Stock Analyzer",
      "icon": "📈"
    }
  ]
}
```

### Step 3: Build Workflow with I/O Schemas

```bash
POST /workflows/build
{
  "workflow": {
    "workflow_id": "wf_stock_pipeline",
    "name": "Stock Analysis Pipeline",
    "agents": ["agt_stock_001", "agt_reporter_001"],
    "connections": [
      {"from": "agt_stock_001", "to": "agt_reporter_001"}
    ]
  },
  "agent_schemas": {
    "agt_stock_001": {
      "input_schema": ["stock_symbol"],
      "output_schema": ["analysis_result"]
    },
    "agt_reporter_001": {
      "input_schema": ["analysis_result"],
      "output_schema": ["final_report"]
    }
  }
}
```

### Step 4: Validate Workflow

```bash
POST /workflows/validate/draft
```

Validator checks: `agt_stock_001.output_schema` matches `agt_reporter_001.input_schema` ✅

---

## 📝 New API Endpoints

### **PATCH** `/agents/{agent_id}/schema`
Update agent's input/output schema.

**Request:**
```json
{
  "input_schema": ["customer_data"],
  "output_schema": ["sentiment_score", "key_themes"]
}
```

**Response:**
```json
{
  "agent_id": "agt_001",
  "input_schema": ["customer_data"],
  "output_schema": ["sentiment_score", "key_themes"]
}
```

### **POST** `/workflows/build`
Build workflow manually with agent I/O schema specification.

**Request:**
```json
{
  "workflow": { ... },
  "agent_schemas": {
    "agt_001": {
      "input_schema": [...],
      "output_schema": [...]
    }
  },
  "update_base_agents": false
}
```

**Parameters:**
- `workflow` - Workflow definition
- `agent_schemas` - I/O schemas per agent (optional)
- `update_base_agents` - If true, updates base agent definitions (default: false)

---

## 🎯 Best Practices

### Agent Builder
✅ Create agents with generic/empty I/O schemas
✅ Focus on agent's core functionality
✅ Let Workflow Builder define data flow

### Workflow Builder
✅ Define I/O schemas based on workflow data flow
✅ Use `agent_schemas` in `/workflows/build`
✅ Keep `update_base_agents: false` for flexibility
✅ Only set `update_base_agents: true` if agent is workflow-specific

---

## ✅ Implementation Complete

**What's Implemented:**
1. ✅ Agents created with empty I/O schemas in Agent Builder
2. ✅ Endpoint to update agent I/O schema (`PATCH /agents/{agent_id}/schema`)
3. ✅ Workflow builder endpoint with I/O schema specification (`POST /workflows/build`)
4. ✅ Optional base agent update when building workflows
5. ✅ Workflow-specific I/O schema storage in metadata
6. ✅ Validation ensures data flow correctness

**Files Modified:**
1. `apps/agent/designer/agent_designer.py` - Empty I/O schemas by default
2. `apps/agent/routes.py` - Added PATCH endpoint for I/O schema
3. `apps/workflow/routes.py` - Added POST /workflows/build endpoint


What Was Done

  1. Agent Builder Creates Empty I/O Schemas
  - Agents now have input_schema: [] and output_schema: [] by default
  - No assumptions about data flow

  2. New Endpoint: Update Agent I/O Schema
  PATCH /agents/{agent_id}/schema
  {
    "input_schema": ["stock_data"],
    "output_schema": ["analysis"]
  }

  3. New Endpoint: Build Workflow with I/O Schemas
  POST /workflows/build
  {
    "workflow": {...},
    "agent_schemas": {
      "agt_001": {
        "input_schema": ["input_data"],
        "output_schema": ["processed_data"]
      }
    },
    "update_base_agents": false  // Optional
  }

  How It Works

  Agent Builder:
  1. Create agent → I/O schemas are empty []
  2. Agent saved to registry

  Workflow Builder:
  1. Fetch agents from master list
  2. User defines I/O schemas based on workflow data flow
  3. Two options:
    - Option A: Update base agent permanently (PATCH /agents/{id}/schema)
    - Option B: Define per-workflow (recommended - POST /workflows/build)

  Benefits:
  - ✅ Same agent, different I/O schemas in different workflows
  - ✅ Base agent stays flexible
  - ✅ Workflow-specific data flow
  - ✅ Validator ensures Agent1.output = Agent2.input

  Files Modified

  1. apps/agent/designer/agent_designer.py - Empty schemas by default
  2. apps/agent/routes.py - Added PATCH /agents/{id}/schema
  3. apps/workflow/routes.py - Added POST /workflows/build

---

## 🔄 NODE MAPPER LAYER IMPLEMENTATION

### Overview
The Node Mapper provides bidirectional conversion between the frontend canvas (workflow_builder_ide.html) and backend workflow schema. It handles all 16 node types while preserving visual layout and connection information.

### What Was Implemented

#### 1. **Core Mapper Service**
**File:** `apps/workflow/visualization/node_mapper.py`

**Key Features:**
- Bidirectional mapping: Frontend ↔ Backend
- Handles all 16 node types (Start, End, Agent, Conditional, Loop, etc.)
- Preserves UI layout (x, y, icon, color) in metadata
- Auto-generates workflow names (explicit or from first agent)
- Validates Start node must be first
- Auto-layout fallback for missing positions
- Connection preservation for arrow rendering

**Node Types Supported:**
```
Entry/Exit: Start, End
Agents: Agent, Subagent
LLM: Prompt
Logic: Conditional, Loop, Map
Quality: Self-Review, HITL
Integration: API, MCP Server
Code: Code Execution, Template
Resilience: Failsafe
Utility: Merge
```

#### 2. **Key Mapping Functions**

**Frontend → Backend:**
```python
map_frontend_to_backend(canvas_nodes, connections, workflow_name)
→ Returns: (workflow_dict, agents_dict)
```

**Backend → Frontend:**
```python
map_backend_to_frontend(workflow, agents_dict)
→ Returns: (canvas_nodes, connections)
```

#### 3. **Node Type Handling**

Each frontend node type maps to a backend agent with `metadata.node_type`:

```json
{
  "agent_id": "agt_001",
  "name": "Ticket Analyzer",
  "role": "Autonomous AI agent",
  "metadata": {
    "node_type": "Agent",
    "ui_layout": {
      "x": 300,
      "y": 200,
      "icon": "🔶",
      "color": "#f59e0b"
    }
  }
}
```

**Special Node Handling:**
- **Start**: Defines workflow.state_schema inputs
- **End**: Defines expected outputs
- **HITL**: Adds to workflow.human_in_loop.review_points
- **Conditional**: Stores branches in metadata
- **Loop/Map**: Stores iteration config in metadata
- **API/MCP**: Stores integration config in metadata

#### 4. **Connection Preservation**

Connections include ID for arrow rendering:
```json
{
  "id": "conn_001",
  "from": "agt_001",
  "to": "agt_002",
  "condition": "optional condition string"
}
```

This ensures arrows persist when workflows are loaded from backend.

#### 5. **Workflow Name Support**

**Updated Workflow Schema:**
```json
{
  "workflow_id": "wf_xxx",
  "name": "Customer Support Workflow",  ← NEW FIELD (required)
  "description": "...",
  "status": "draft",
  "version": "0.1",
  "execution_model": "sequential",
  "agents": ["agt_001"],
  "connections": [...],
  "metadata": {
    "canvas_layout": {
      "width": 5000,
      "height": 5000
    }
  }
}
```

**Naming Strategy:**
- Explicit: User provides workflow_name
- Auto-generated: From first agent name + " Workflow"
- Fallback: "Workflow YYYYMMDD_HHMM"

#### 6. **Execution Model Inference**

Mapper automatically infers execution model from canvas structure:

```python
Sequential: Linear node flow
Parallel: Map nodes or multiple branches from single node
Hierarchical: Subagent delegation pattern
Hybrid: Mix of above patterns
```

#### 7. **Validation Rules**

Mapper validates:
- ✅ Start node exists and is unique
- ✅ All node types are supported
- ✅ Connections reference valid nodes
- ✅ UI layout data is present or auto-generated

### New API Endpoints

#### **POST** `/workflows/canvas/to-backend`
Convert frontend canvas to backend workflow format.

**Request:**
```json
{
  "canvas_nodes": [
    {
      "id": 123,
      "type": "Start",
      "name": "Start",
      "x": 100,
      "y": 100,
      "icon": "▶️",
      "color": "#10b981",
      "config": {
        "inputVariables": [
          {"name": "query", "type": "string", "required": true}
        ]
      }
    },
    {
      "id": 456,
      "type": "Agent",
      "name": "Researcher",
      "x": 300,
      "y": 100,
      "icon": "🔶",
      "color": "#f59e0b",
      "config": {
        "model": {
          "modelName": "gpt-4o-mini"
        },
        "tools": []
      }
    }
  ],
  "connections": [
    {"id": 1, "from": 123, "to": 456}
  ],
  "workflow_name": "Research Pipeline"
}
```

**Response:**
```json
{
  "workflow": {
    "workflow_id": "wf_generated",
    "name": "Research Pipeline",
    "status": "draft",
    "version": "0.1",
    "execution_model": "sequential",
    "agents": ["agt_123", "agt_456"],
    "connections": [
      {"id": 1, "from": "agt_123", "to": "agt_456"}
    ]
  },
  "agents": {
    "agt_123": {
      "agent_id": "agt_123",
      "name": "Start",
      "role": "Workflow entry point",
      "metadata": {
        "node_type": "Start",
        "ui_layout": {"x": 100, "y": 100, "icon": "▶️", "color": "#10b981"}
      }
    },
    "agt_456": {
      "agent_id": "agt_456",
      "name": "Researcher",
      "role": "Autonomous AI agent",
      "llm": {"model": "gpt-4o-mini"},
      "tools": [],
      "metadata": {
        "node_type": "Agent",
        "ui_layout": {"x": 300, "y": 100, "icon": "🔶", "color": "#f59e0b"}
      }
    }
  }
}
```

#### **POST** `/workflows/backend/to-canvas`
Convert backend workflow to frontend canvas format.

**Request:**
```json
{
  "workflow": {
    "workflow_id": "wf_123",
    "name": "Customer Support Workflow",
    "agents": ["agt_001", "agt_002"],
    "connections": [{"from": "agt_001", "to": "agt_002"}]
  },
  "agents": {
    "agt_001": {
      "name": "Ticket Analyzer",
      "metadata": {
        "node_type": "Agent",
        "ui_layout": {"x": 200, "y": 150, "icon": "🔶"}
      }
    }
  }
}
```

**Response:**
```json
{
  "canvas_nodes": [
    {
      "id": 1,
      "type": "Agent",
      "name": "Ticket Analyzer",
      "x": 200,
      "y": 150,
      "icon": "🔶",
      "color": "#f59e0b",
      "backend_id": "agt_001",
      "status": "idle"
    }
  ],
  "connections": [
    {"id": 1, "from": 1, "to": 2}
  ],
  "workflow_name": "Customer Support Workflow"
}
```

#### **POST** `/workflows/canvas/save`
Save canvas workflow directly (converts + validates + saves).

**Request:**
```json
{
  "canvas_nodes": [...],
  "connections": [...],
  "workflow_name": "My Workflow",
  "save_as": "draft"
}
```

**Response:**
```json
{
  "success": true,
  "workflow_id": "wf_generated",
  "workflow_name": "My Workflow",
  "state": "draft",
  "path": "/path/to/storage/workflows/draft/wf_generated.draft.json"
}
```

**Error Response (Validation Failed):**
```json
{
  "success": false,
  "errors": [
    "Workflow must have a Start node",
    "Agent 'agt_003' expects 'data' but no producer found"
  ],
  "warnings": [
    "Tool 'legacy_tool' is deprecated"
  ]
}
```

### Agent Templates System

#### **Static Templates File**
**File:** `apps/storage/agent_templates.json`

Contains 9 pre-defined agent templates:
- Research Analyst
- Customer Support Agent
- Data Analyst
- Content Writer
- Code Reviewer
- Project Manager
- Sales Assistant
- HR Coordinator
- Financial Analyst

Each template includes:
- name, icon, description, role
- prompt, tools, variables
- settings (temperature, max_token, max_iteration)

#### **GET** `/agents/templates/all`
Get all agent templates (static + created agents).

**Response:**
```json
{
  "templates": [
    {
      "name": "Research Analyst",
      "icon": "🔬",
      "description": "Conducts comprehensive research",
      "role": "Research and Analysis",
      "prompt": "You are a Research Analyst...",
      "tools": ["Web Search", "Web Fetch"],
      "variables": [...],
      "settings": {
        "temperature": 0.3,
        "max_token": 4000,
        "max_iteration": 5
      }
    }
  ],
  "created": [
    {
      "name": "Custom Agent",
      "icon": "🤖",
      "description": "User-created agent",
      "role": "Custom Role",
      "agent_id": "agt_001",
      "source": "created"
    }
  ],
  "total_templates": 9,
  "total_created": 5,
  "total": 14
}
```

#### **GET** `/agents/templates/static`
Get only static agent templates.

**Response:**
```json
{
  "templates": [...],
  "count": 9
}
```

### Storage Structure Updated

```
echoAI/
├── apps/
│   ├── storage/
│   │   ├── agents/
│   │   │   ├── ai_agents.json          ← Master list
│   │   │   ├── agt_001.json            ← Individual agents
│   │   │   └── ...
│   │   └── agent_templates.json        ← Static templates (NEW)
│   └── workflow/
│       ├── storage/
│       │   └── workflows/
│       │       ├── draft/
│       │       ├── temp/
│       │       ├── final/
│       │       └── archive/
│       └── visualization/
│           └── node_mapper.py          ← Mapper service (NEW)
```

### Integration with Frontend

#### Loading Canvas from Backend Workflow
```javascript
// Fetch workflow
const workflow = await fetch('/workflows/wf_123/temp').then(r => r.json());
const agents = {}; // Fetch agents separately or embedded

// Convert to canvas
const response = await fetch('/workflows/backend/to-canvas', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({workflow, agents})
});

const {canvas_nodes, connections, workflow_name} = await response.json();

// Load into canvas
activeNodes.value = canvas_nodes;
connectionsState.value = connections;
```

#### Saving Canvas to Backend
```javascript
// Convert canvas to backend format
const response = await fetch('/workflows/canvas/to-backend', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    canvas_nodes: activeNodes.value,
    connections: connectionsState.value,
    workflow_name: workflowName.value
  })
});

const {workflow, agents} = await response.json();

// Now validate and save
await fetch('/workflows/validate/draft', {
  method: 'POST',
  body: JSON.stringify({workflow, agents})
});
```

#### Loading Agent Templates
```javascript
const response = await fetch('/agents/templates/all');
const {templates, created} = await response.json();

// Display templates in Agents tab
agentTemplates.value = [...templates, ...created];
```

### Files Created/Modified

**Created:**
1. `apps/workflow/visualization/node_mapper.py` - Mapper service
2. `apps/storage/agent_templates.json` - Static templates

**Modified:**
1. `apps/workflow/routes.py` - Added 3 mapper endpoints
2. `apps/workflow/container.py` - Registered node_mapper service
3. `apps/agent/routes.py` - Added 2 template endpoints
4. `echolib/schemas/workflow_schema.json` - Already had name field

**Total:** 6 files

### Testing the Mapper

#### Test Frontend → Backend Conversion
```bash
curl -X POST http://localhost:8000/workflows/canvas/to-backend \
  -H "Content-Type: application/json" \
  -d '{
    "canvas_nodes": [
      {
        "id": 1,
        "type": "Start",
        "name": "Start",
        "x": 100,
        "y": 100,
        "config": {
          "inputVariables": [{"name": "query", "type": "string"}]
        }
      }
    ],
    "connections": [],
    "workflow_name": "Test Workflow"
  }'
```

#### Test Backend → Frontend Conversion
```bash
curl -X POST http://localhost:8000/workflows/backend/to-canvas \
  -H "Content-Type": application/json" \
  -d '{
    "workflow": {
      "workflow_id": "wf_test",
      "name": "Test",
      "agents": ["agt_001"],
      "connections": []
    },
    "agents": {
      "agt_001": {
        "name": "Agent 1",
        "metadata": {
          "node_type": "Agent",
          "ui_layout": {"x": 200, "y": 200}
        }
      }
    }
  }'
```

#### Test Agent Templates
```bash
curl http://localhost:8000/agents/templates/all
```

### Benefits

1. **Clean Separation:**
   - Frontend focuses on visual canvas
   - Backend focuses on workflow logic
   - Mapper bridges the gap

2. **All Node Types Supported:**
   - 16 different node types
   - Each with specific handling
   - Extensible for future types

3. **Connection Preservation:**
   - Arrow rendering works after load
   - Connection IDs preserved
   - Conditional connections supported

4. **Auto-Layout Fallback:**
   - Missing positions auto-generated
   - Horizontal flow with vertical stagger
   - No workflow breaks on missing UI data

5. **Agent Templates:**
   - 9 pre-built templates
   - Combines with user-created agents
   - Easy to extend with more templates

### ✅ Implementation Complete

**Mapper Layer:**
- ✅ Bidirectional conversion (Frontend ↔ Backend)
- ✅ All 16 node types handled
- ✅ Connection preservation for arrows
- ✅ Workflow name support (explicit + auto)
- ✅ Start node validation
- ✅ Auto-layout fallback
- ✅ UI layout in metadata

**Agent Templates:**
- ✅ Static templates JSON file
- ✅ 9 pre-defined templates
- ✅ Template + created agents endpoint
- ✅ Static-only endpoint

**API Endpoints:**
- ✅ POST /workflows/canvas/to-backend
- ✅ POST /workflows/backend/to-canvas
- ✅ POST /workflows/canvas/save (convert + validate + save)
- ✅ GET /agents/templates/all
- ✅ GET /agents/templates/static

**Ready for Frontend Integration!** 🚀
