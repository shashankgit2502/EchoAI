# Workflow Orchestrator

A workflow orchestration system.


workflow-orchestrator/
│
├── app/
│   ├── main.py                         # FastAPI entrypoint
│
│   ├── api/                            # API layer (HTTP only)
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── validate.py             # /validate/* endpoints
│   │   │   ├── workflow.py             # create/edit/save/import/export workflows
│   │   │   ├── agent.py                # agent CRUD & validation
│   │   │   ├── runtime.py              # chat/test execution
│   │   │   ├── visualize.py            # graph data for UI (nodes + edges)
│   │   │   ├── telemetry.py            # runtime metrics & traces API
│   │   │   └── health.py               # system health checks
│
│   ├── core/                           # Core business logic (NO FastAPI)
│   │   ├── __init__.py
│   │   ├── config.py                   # env, settings
│   │   ├── constants.py                # enums, limits, workflow states
│   │   ├── logging.py                  # logging config
│   │   └── telemetry.py                # OpenTelemetry bootstrap (global)
│
│   ├── schemas/                        # JSON schemas & Pydantic models
│   │   ├── __init__.py
│   │   ├── workflow_schema.json
│   │   ├── agent_schema.json
│   │   ├── tool_schema.json
│   │   ├── graph_schema.json           # nodes/edges schema for visualization
│   │   └── api_models.py               # request/response models
│
│   ├── validator/                      # 🔑 Compiler layer
│   │   ├── __init__.py
│   │   ├── validator.py                # main validate_workflow()
│   │   ├── sync_rules.py               # sync validation rules
│   │   ├── async_rules.py              # async checks (MCP, LLM)
│   │   ├── retry.py                    # retry + timeout helpers
│   │   └── errors.py                   # validator error types
│
│   ├── workflow/                       # Workflow design & lifecycle
│   │   ├── __init__.py
│   │   ├── designer.py                 # LLM workflow designer
│   │   ├── compiler.py                 # Workflow JSON → LangGraph
│   │   ├── graph_builder.py            # Workflow JSON → graph (nodes/edges)
│   │   ├── versioning.py               # draft/final/version logic
│   │   └── state.py                    # workflow state schema helpers
│
│   ├── agents/                         # Agent system
│   │   ├── __init__.py
│   │   ├── registry.py                 # load/store agent JSON
│   │   ├── factory.py                  # instantiate agent at runtime
│   │   ├── permissions.py              # agent permission rules
│   │   └── templates/                  # default agent templates
│
│   ├── tools/                          # MCP integration
│   │   ├── __init__.py
│   │   ├── mcp_client.py               # MCP client wrapper
│   │   ├── registry.py                 # tool registry/cache
│   │   └── health.py                   # MCP health checks
│
│   ├── runtime/                        # Execution layer
│   │   ├── __init__.py
│   │   ├── executor.py                 # LangGraph execution
│   │   ├── hitl.py                     # Human-in-the-loop interrupts
│   │   ├── checkpoints.py              # state persistence
│   │   ├── guards.py                   # cost, timeout, step limits
│   │   └── telemetry.py                # OTel spans for workflow/agent/tool
│
│   ├── visualization/                  # 🔹 Design-time graph support
│   │   ├── __init__.py
│   │   ├── graph_mapper.py             # workflow → UI graph mapping
│   │   ├── graph_editor.py             # apply UI edits → workflow JSON
│   │   └── layout.py                   # auto-layout helpers (DAG, hierarchy)
│
│   ├── storage/                        # Persistence
│   │   ├── __init__.py
│   │   ├── filesystem.py               # JSON file storage
│   │   ├── workflows/
│   │   │   ├── draft/
│   │   │   ├── temp/
│   │   │   ├── final/
│   │   │   └── archive/
│   │   └── agents/
│   │       └── *.json
│
│   ├── services/                       # Cross-cutting services
│   │   ├── __init__.py
│   │   ├── prompt_generator.py         # meta-prompt logic
│   │   ├── llm_provider.py             # LLM abstraction
│   │   └── hashing.py                  # validation hash logic
│
│   └── utils/
│       ├── __init__.py
│       ├── json_utils.py
│       ├── time.py
│       └── ids.py
│
├── tests/
│   ├── unit/
│   │   ├── test_validator.py
│   │   ├── test_agent_schema.py
│   │   ├── test_workflow_schema.py
│   │   ├── test_graph_builder.py
│   │   └── test_telemetry.py
│   ├── integration/
│   │   ├── test_validate_api.py
│   │   ├── test_visualization_api.py
│   │   └── test_runtime_execution.py
│   └── fixtures/
│       ├── agents/
│       └── workflows/
│
├── scripts/
│   ├── init_mcp_registry.py
│   ├── migrate_workflows.py
│   ├── cleanup_temp.py
│   └── export_telemetry.py
│
├── .env
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md



My Understanding of Your Project

  After thoroughly reviewing your CLAUDE.md file, here's my comprehensive understanding:

  Core Concept

  You're building a Dynamic Workflow-Centric Multi-Agent System Builder where users describe requirements in natural language, and the system autonomously designs, validates, and executes multi-agent workflows.

  Key Architecture Principles

  From User's Perspective:
  - Single chatbot interface
  - Natural language input only
  - No manual selection of agents, tools, or workflow types
  - System figures everything out dynamically

  From System's Perspective:
  The system must:
  1. Analyze user prompts
  2. Decide optimal workflow type (sequential/parallel/hierarchical/hybrid)
  3. Determine number and roles of agents
  4. Select appropriate tools for each agent
  5. Draft complete workflow with validation
  6. Enable human-in-the-loop editing
  7. Support testing before finalization
  8. Save versioned, reusable workflows

  Critical Architecture Stack

  USER CHAT
     ↓
  PROMPT ANALYZER
     ↓
  WORKFLOW DESIGNER (LLM)
     ↓
  DRAFT AGENTS + DRAFT WORKFLOW (JSON)
     ↓
  AUTO VALIDATOR
     ↓
  HUMAN-IN-THE-LOOP EDITOR
     ↓
  RE-VALIDATE
     ↓
  SAVE → TEMP JSON
     ↓
  CHAT / TEST WORKFLOW
     ↓
  EDIT (optional) → back to VALIDATE
     ↓
  FINAL SAVE (VERSIONED JSON)

  Core Components

  1. Workflow System:
  - Workflows are data (JSON), not code
  - Support 4 execution models: sequential, parallel, hierarchical, hybrid
  - Lifecycle: DRAFT → VALIDATED → TESTING (TEMP) → FINAL
  - FINAL workflows are immutable, versioned (clone to edit)

  2. Agent System:
  - Agents are configurable JSON components
  - Each agent has its own LLM, tools, input/output schema
  - MCP-first tooling (all tools via Model Context Protocol)
  - Agents stored as JSON, reusable across workflows

  3. Validator (Compiler):
  - Runs before and after human-in-the-loop
  - Sync validation (schema, topology, I/O contracts)
  - Async validation (MCP servers, LLM availability)
  - With retries, timeouts, and bounded execution

  4. Runtime Orchestrator:
  - LangGraph-based execution
  - Supports HITL interrupts
  - Cost/timeout/step guards
  - OpenTelemetry instrumentation for observability

  5. Visualization Layer:
  - Workflow → graph (nodes/edges) conversion
  - UI editing → workflow JSON sync
  - Auto-layout for different topologies

  Key Technologies

  - Backend: FastAPI + Python
  - GenAI: LangChain v1, LangGraph v1, Azure AI Agents SDK
  - LLMs: Azure OpenAI, Ollama, multi-provider support
  - Tools: MCP (Model Context Protocol) servers
  - UI: Streamlit
  - Observability: OpenTelemetry → Grafana/Jaeger

  Project Structure (13 Phases)

  You've defined a clear coding order:
  1. Bootstrap (FastAPI setup)
  2. Schemas (workflow, agent, tool, graph)
  3. Storage (filesystem with versioning)
  4. Validator (compiler layer)
  5. Tool System (MCP registry)
  6. Agent System
  7. Workflow Designer (LLM-based)
  8. Visualization
  9. Runtime Execution
  10. Telemetry
  11. API Routes
  12. Tests & Scripts
  13. Tools (MCP) - TO BE DONE LAST ON YOUR COMMAND

  Critical Design Rules

  ✅ Non-Negotiables:
  - Workflows are data, not behavior
  - Agents are configurable components
  - LLM proposes, validator decides, human approves
  - FINAL workflows are immutable (clone to modify)
  - All tools are MCP-based
  - Validation is mandatory before execution
  - TEMP workflows are sandboxes for testing
  - No static graphs or predefined agent counts

  ✅ Lifecycle Management:
  - DRAFT = editable
  - TEMP = executable for testing (mutable)
  - FINAL = executable for reuse (immutable, versioned)
  - Archive old versions when creating new ones

  ✅ Multi-LLM Support:
  - Default LLM configurable
  - Per-agent LLM selection
  - User can change LLMs in DRAFT/TEMP
  - FINAL workflows lock LLM configuration

  What Makes This Unique

  This is NOT a hierarchical agent system. This is a workflow design system where:
  - Hierarchy, parallelism, sequential execution are outcomes, not constraints
  - The LLM designs workflow topology, not just agent chains
  - Everything is decided at runtime based on user intent
  - Human-in-the-loop is a hard gate, not optional
  - Export/import is first-class (reusable workflows)

  Implementation Approach

  You've emphasized:
  - Always use latest documentation (web search + MCP for LangChain/LangGraph)
  - NO patchwork solutions - always root cause analysis
  - Modular, well-structured code
  - SKIP MCP tool implementation until your command

  ---
  My Understanding in One Sentence:

  You're building an intelligent workflow compiler and runtime that takes natural language → dynamically generates multi-agent workflows → validates them → lets humans refine them → executes them safely → and makes them reusable, all without hardcoding any agent logic or workflow structures.

 
Clarified Architecture Flow:

  USER CHAT (external system)
     ↓
  [YOUR API] META-PROMPT GENERATOR (Analyzer LLM)
     │
     ├─ analyze_request() → Extract domain, entities, operations
     ├─ _calculate_complexity() → Score 1-10
     ├─ _suggest_patterns() → Recommend communication patterns
     └─ generate_meta_prompt() → Create structured prompt with JSON schema
     ↓
  [YOUR API] WORKFLOW DESIGNER (Designer LLM)
     │
     └─ Takes meta-prompt → Returns complete agent system JSON
     ↓
  DRAFT AGENTS + DRAFT WORKFLOW (JSON)
     {
       "system_name": "...",
       "agents": [...],
       "tools": [...],
       "workflows": [...]
     }
     ↓
  [YOUR CORE] AUTO VALIDATOR
     │
     ├─ Sync: Schema, topology, I/O contracts
     └─ Async: Tool availability, LLM availability (bounded retries)
     ↓
  [EXTERNAL UI] HUMAN-IN-THE-LOOP EDITOR
     │
     └─ User modifies agents/tools/workflows/execution mode
     ↓
  [YOUR CORE] RE-VALIDATE
     │
     └─ Same validation logic after human edits
     ↓
  [YOUR STORAGE] SAVE → TEMP JSON
     │
     └─ storage/workflows/temp/{workflow_id}.temp.json
     ↓
  [YOUR RUNTIME] CHAT / TEST WORKFLOW
     │
     ├─ Compile JSON → LangGraph StateGraph
     ├─ Execute with checkpointing
     └─ Support HITL interrupts
     ↓
  [LOOP] EDIT (optional) → back to VALIDATE
     │
     └─ If changes needed, loop back to validator
     ↓
  [YOUR STORAGE] FINAL SAVE (VERSIONED JSON)
     │
     └─ storage/workflows/final/{workflow_id}.v{version}.json

  ---
  Component Mapping:

  | Flow Step             | Implementation                                     |
  |-----------------------|----------------------------------------------------|
  | Meta-Prompt Generator | app/services/meta_prompt_generator.py              |
  | Workflow Designer     | app/workflow/designer.py                           |
  | Auto Validator        | app/validator/validator.py (sync + async)          |
  | Storage (Temp)        | app/storage/filesystem.py → temp/                  |
  | Runtime Executor      | app/runtime/executor.py + app/workflow/compiler.py |
  | Storage (Final)       | app/storage/filesystem.py → final/                 |

  ---
  What I Will Build (In Order):

  Phase 1: Core Design Engine

  1. ✅ Schemas (agent_system, agent, tool, workflow)
  2. ✅ MetaPromptGenerator (with your reference code)
  3. ✅ WorkflowDesigner (LLM-powered)
  4. ✅ API models for requests/responses

  Phase 2: Validation & Storage

  5. ✅ Validator (updated for richer JSON)
  6. ✅ Storage (draft/temp/final with versioning)

  Phase 3: Runtime Execution

  7. ✅ Workflow Compiler (JSON → LangGraph)
  8. ✅ Runtime Executor (with HITL support)

  Phase 4: API Layer

  9. ✅ API routes (design, validate, save, execute)



  =============================================================================================
  =============================================================================================
  =============================================================================================

  ## API for future microservice 
  Current State: Service Boundary Violations Found

  ❌ Problems Identified

  1. Direct Core Imports in API Layer
  # main.py lines 23-28 - VIOLATION
  from app.services.meta_prompt_generator import MetaPromptGenerator
  from app.workflow.designer import WorkflowDesigner
  from app.validator.validator import AgentSystemValidator
  from app.storage.filesystem import WorkflowStorage
  from app.runtime.executor import WorkflowExecutor
  Issue: API directly calls core modules, not through service abstraction

  2. Monolithic main.py (640 lines)
  - All routes in one file
  - Router stub files (workflow.py, validate.py, etc.) are empty/unused
  - No component isolation

  3. No Internal Service Layer
  - Missing: app/services/<component>_service.py abstraction
  - Components communicate via direct function calls
  - Not microservice-ready

  4. No Internal API Structure
  - Missing: /api/internal/validator/*
  - Missing: /api/internal/workflow/*
  - Missing: /api/internal/runtime/*

  ---
  What Must Be Done

  Phase A: Create Service Layer ⚡ (NEW)

  Add service wrappers for each component:

  app/services/
  ├── validator_service.py      # Wraps app.validator
  ├── workflow_service.py        # Wraps app.workflow
  ├── agent_service.py           # Wraps app.agents
  ├── runtime_service.py         # Wraps app.runtime
  ├── storage_service.py         # Wraps app.storage
  ├── visualization_service.py   # Wraps app.visualization
  └── telemetry_service.py       # Wraps app.runtime.telemetry

  Each service:
  - ✅ Has request/response DTOs (Pydantic models)
  - ✅ Async + idempotent methods
  - ✅ No FastAPI imports (pure business logic)
  - ✅ Ready for network transport

  ---
  Phase B: Create Internal APIs 🔌 (NEW)

  Add internal API routes for component-to-component communication:

  app/api/internal/
  ├── validator.py       # /api/internal/validator/*
  ├── workflow.py        # /api/internal/workflow/*
  ├── agent.py           # /api/internal/agent/*
  ├── runtime.py         # /api/internal/runtime/*
  ├── storage.py         # /api/internal/storage/*
  └── visualize.py       # /api/internal/visualize/*

  Purpose: Enable component decoupling (future microservice extraction)

  ---
  Phase C: Refactor API Routes 🔧

  Populate stub files to call services (not core directly):

  app/api/routes/
  ├── workflow.py    → calls workflow_service
  ├── validate.py    → calls validator_service
  ├── runtime.py     → calls runtime_service
  ├── agent.py       → calls agent_service
  ├── visualize.py   → calls visualization_service
  └── telemetry.py   → calls telemetry_service

  ---
  Phase D: Refactor main.py 🧹

  - Remove direct core imports
  - Import only routers
  - Use dependency injection for services
  - Reduce to ~100 lines

  ---
  Implementation Order

  1️⃣ Create Service DTOs (api_models.py extensions)
  2️⃣ Implement Service Layer (Phase A)
  3️⃣ Implement Internal APIs (Phase B)
  4️⃣ Refactor External API Routes (Phase C)
  5️⃣ Refactor main.py (Phase D)
  6️⃣ Test end-to-end with service boundaries

  ---
  Final Verdict
  ┌──────────────────────────────────────┬───────────────────────────────────────────────────────────┐
  │               Question               │                          Answer                           │
  ├──────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Is your API-first reasoning correct? │ ✅ YES - Correct approach for microservices               │
  ├──────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Is it currently implemented?         │ ⚠️ PARTIALLY - API exists but violates service boundaries │
  ├──────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Can it support microservices?        │ ❌ NO - Needs service layer + internal APIs first         │
  ├──────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Is it fixable?                       │ ✅ YES - Clear refactoring path (Phases A-D)              │
  └──────────────────────────────────────┴───────────────────────────────────────────────────────────┘
  ---
  🎯 Next Steps

  Shall I proceed with implementation in this order:
  1. Service Layer (Phase A) - Wrap core modules
  2. Internal APIs (Phase B) - Enable component decoupling
  3. Refactor Routes (Phase C & D) - Clean API structure



error: 
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
langchain-anthropic 0.3.22 requires langchain-core<1.0.0,>=0.3.78, but you have langchain-core 1.2.7 which is incompatible.
langchain-mcp 0.2.1 requires langchain-core~=0.3.37, but you have langchain-core 1.2.7 which is incompatible.
langchain-openai 0.3.35 requires langchain-core<1.0.0,>=0.3.78, but you have langchain-core 1.2.7 which is incompatible.
Successfully installed langchain-1.2.3 langchain-core-1.2.7 langgraph-1.0.5 langgraph-prebuilt-1.0.5 langgraph-sdk-0.3.2


==========================================================================================
## How to use tool ?
What’s in place

  - Core search models, interfaces, providers, and service:
      - workflow-orchestrator/app/core/web_search/models.py
      - workflow-orchestrator/app/core/web_search/interfaces.py
      - workflow-orchestrator/app/core/web_search/providers/bing.py
      - workflow-orchestrator/app/core/web_search/providers/duckduckgo.py
      - workflow-orchestrator/app/core/web_search/providers/google.py
      - workflow-orchestrator/app/core/web_search/service.py
  - MCP tool + server:
      - workflow-orchestrator/app/mcp/tools/web_search_tool.py
      - workflow-orchestrator/app/mcp/server.py
  - Package init files:
      - workflow-orchestrator/app/core/web_search/__init__.py                                           
      - workflow-orchestrator/app/core/web_search/providers/__init__.py                                 
      - workflow-orchestrator/app/mcp/__init__.py                                                       
      - workflow-orchestrator/app/mcp/tools/__init__.py                                                 
  - Dependencies:                                                                                       
      - workflow-orchestrator/requirements.txt now includes httpx and mcp                               
                                                                                                        
  Notes                                                                                                 
                                                                                                        
  - Default allowed providers are whitelisted to { "bing" } as per your spec.                           
  - Google and DuckDuckGo providers are registered but blocked by policy until allowed.                 
  - The code uses app.* absolute imports to match your project layout.                                  
                                                                                                        
  Next steps (optional)                                                                                 
                                                                                                        
  1. Run the MCP server: python -m app.mcp.server                                                       
  2. Set BING_API_KEY before running.

 What’s in place

  - Core calculator + math modules:
      - workflow-orchestrator/app/core/calculator/models.py
      - workflow-orchestrator/app/core/calculator/service.py
      - workflow-orchestrator/app/core/math/arithmetic.py
      - workflow-orchestrator/app/core/math/statistics.py
      - workflow-orchestrator/app/core/math/linear_algebra.py
      - workflow-orchestrator/app/core/calculator/__init__.py
      - workflow-orchestrator/app/core/math/__init__.py
  - MCP tool + server registration:
      - workflow-orchestrator/app/tools/mcp_servers/agent_tools/tools/calculator_tool.py
      - workflow-orchestrator/app/tools/mcp_servers/agent_tools/server.py

  Logic checks I enforced (safe + deterministic)

  - Empty lists throw a clear error.                                                                                                             
  - Vector add/subtract/dot require same length.                                                                                                 
  - Matrix add requires same dimensions; multiply requires aligned dimensions.                                                                   
  - Precision/rounding applies only to scalar results (same as your spec).                                                                       
                                                                                                                                                 
  Important note for “agent invocation”                                                                                                          
  The calculator MCP tool is implemented, but agents won’t call it automatically until the MCP tool registry/binding (Phase 3 tools system) is   
  wired into agent execution. Right now the runtime uses placeholder tools, so even a “mean 1..10” request won’t hit MCP unless you connect the  
  tool registry and bind it in the agent factory.                                                                                                
                                                                                                                                                 
  If you want “mean 1..10” to invoke the calculator tool end‑to‑end, tell me and I’ll:                                                           
                                                                                                                                                 
  1. Add the calculator tool to tools/registry.py                                                                                                
  2. Bind it in agents/factory.py
  3. Update validator rules to allow it                                                                                                          
  4. Add tool schema entry for MCP                                                                                                               
                                                                                                                                                 
  If you want me to validate the MCP server works right now, I can add a quick local call example or a small script. 