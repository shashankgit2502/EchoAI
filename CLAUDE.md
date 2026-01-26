# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EchoAI is an LLM-driven workflow orchestration platform. Users describe requirements in natural language, and the system automatically determines how many agents are needed, what each does, what tools they need, and the execution topology (sequential, parallel, hierarchical, or hybrid). The system uses LangGraph for workflow execution and optionally CrewAI for agent collaboration.

**Core principle**: Workflow is data, not behavior. Agents are configurable components. LLM proposes, system validates, human approves.

## Tech Stack

- **Framework**: FastAPI + Uvicorn (ASGI)
- **Orchestration**: LangGraph (StateGraph-based workflow execution)
- **LLM SDK**: LangChain Core + LangChain OpenAI
- **Agent Framework**: CrewAI (optional, for intra-agent collaboration)
- **LLM Providers**: OpenRouter (default for dev), Ollama, Azure OpenAI, OpenAI direct, Anthropic
- **Validation**: Pydantic v2
- **Telemetry**: OpenTelemetry
- **Testing**: pytest + pytest-asyncio

## Build & Run Commands

```bash
# Install dependencies (from echoAI/ directory)
pip install -r requirements.txt

# Start the gateway server (all services included)
cd echoAI
uvicorn apps.gateway.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_workflow_integration.py -v

# Run with coverage
pytest tests/ --cov=apps --cov-report=html
```

## Architecture

### Entry Point & Routing

The gateway (`apps/gateway/main.py`) is the single entry point. All services register as FastAPI routers:

- `/workflows` - Workflow orchestration (design, validate, execute, HITL)
- `/agents` - Agent CRUD, design, permissions
- `/chat` - Chat/conversation testing interface
- `/llm` - LLM provider management
- `/tool` - Tool registry (MCP-based)
- `/rag` - RAG/document retrieval
- `/session` - Session management
- `/connector` - External integrations

### Workflow Lifecycle

```
User Prompt → Workflow Designer (LLM) → Draft → Validator → HITL Editor
→ Re-Validate → Temp Save → Chat/Test → Final Save (versioned, immutable)
```

Storage states:
- `draft/` - Editable design phase
- `temp/` - Executable testing phase (agents remain editable here)
- `final/` - Immutable, versioned production state
- `archive/` - Old versions for rollback

Editing a final workflow requires: clone → modify → re-validate → save as new version.

### Key Modules (under `echoAI/apps/`)

| Module | Purpose |
|--------|---------|
| `workflow/designer/designer.py` | LLM-powered workflow design from prompts |
| `workflow/designer/compiler.py` | Workflow JSON → LangGraph StateGraph |
| `workflow/runtime/executor.py` | LangGraph execution engine |
| `workflow/runtime/hitl.py` | Human-in-the-loop checkpoint manager |
| `workflow/routes.py` | All workflow API endpoints (~1000 lines) |
| `workflow/visualization/node_mapper.py` | Canvas ↔ backend bidirectional conversion |
| `agent/designer/agent_designer.py` | LLM-based agent design |
| `agent/factory/factory.py` | Runtime agent instantiation |

### Shared Library (`echoAI/echolib/`)

- `config.py` - LLM provider config (loads from `.env`)
- `di.py` - Dependency injection container
- `types.py` - Pydantic models
- `services.py` - Shared service implementations

### Two-Phase Validation

1. **Sync phase** (fast, deterministic): Schema validation, agent existence, I/O contract checks, topology rules, hierarchy rules, HITL rules
2. **Async phase** (bounded retries + timeouts): MCP server health, LLM availability, credential verification

Sync runs first. Async only runs if sync passes.

### Execution Models

The compiler supports these topologies (inferred by the LLM designer):
- **Sequential** - Linear agent chain
- **Parallel** - Concurrent agents with merge
- **Hierarchical** - Master agent delegates to sub-agents
- **Hybrid** - Mixed topologies (partial support)

## LLM Configuration

Configure via `.env` in `echoAI/` directory. Set exactly one provider flag to `true`:

- `USE_OPENROUTER=true` (recommended for development, has free tier)
- `USE_OLLAMA=true` (on-premise)
- `USE_AZURE=true` (Azure OpenAI)
- `USE_OPENAI=true` (direct OpenAI)

Model registry is in `echoAI/llm_provider.json`. Each agent can use a different LLM.

## Development Guidelines

- Always search the web for latest LangChain v1 / LangGraph v1 documentation rather than relying on training data
- Perform root cause analysis before implementing fixes; no patchwork solutions
- Write modular API code with clear separation of concerns
- Workflows and agents are JSON data, not code. Never hardcode execution logic
- Any edit to a validated workflow resets its validation status
- HITL is a hard gate - human approval is required before execution
- Telemetry (OpenTelemetry) must be non-blocking and never instrumented inside agent logic
- Tools are MCP-based: agents reference tools by ID only, runtime binds dynamically

## Canvas Node Types (16 total)

Start, End, Agent, Sub-agent, Prompt, Conditional, Loop, Map, Self-Review, HITL, API, MCP, Code, Template, Failsafe, Merge

## Key API Patterns

```bash
# Design workflow from prompt
POST /workflows/design/prompt

# Validate (sync-only for drafts, sync+async for finals)
POST /workflows/validate/draft
POST /workflows/validate/final

# Temp lifecycle
POST /workflows/temp/save
GET  /workflows/{id}/temp
DELETE /workflows/{id}/temp

# Final lifecycle (versioned)
POST /workflows/final/save
GET  /workflows/{id}/final/{version}
GET  /workflows/{id}/versions
POST /workflows/clone

# Execution
POST /workflows/execute
POST /workflows/chat/start
POST /workflows/chat/send

# HITL
POST /workflows/hitl/approve
POST /workflows/hitl/reject
POST /workflows/hitl/modify
```
