# Workflow Orchestrator - API Endpoints

## Architecture

The API is organized into two layers:

1. **External APIs** - User-facing endpoints (public)
2. **Internal APIs** - Component-to-component communication (microservice-ready)

---

## External APIs (User-Facing)

### Health & Status

#### `GET /`
Root endpoint with API information
- Returns service status and available endpoints

#### `GET /health`
Detailed health check
- Returns component health status

#### `GET /ping`
Simple ping for load balancers
- Returns `{"status": "ok"}`

---

### Validation

#### `POST /validate`
Validate agent system design
- **Body**: `AgentSystemDesign`
- **Response**: `ValidationResponse`
- Checks: agent uniqueness, tool references, workflow topology, LLM configs

#### `POST /validate/quick`
Quick validation (sync only)
- **Body**: `AgentSystemDesign`
- **Response**: `{"valid": boolean}`
- Faster validation for draft mode

---

### Workflow Management

#### `POST /workflow/design`
Design workflow from natural language
- **Body**: `UserRequest` (natural language description)
- **Response**: `DesignWorkflowResponse` (agent system + analysis)
- Complete pipeline: analyze → meta-prompt → LLM design

#### `POST /workflow/modify`
Modify existing workflow (HITL)
- **Body**: `ModifyWorkflowRequest`
- **Response**: `AgentSystemDesign`
- Human-in-the-loop workflow editing

#### `POST /workflow/save/draft`
Save as DRAFT (editable)
- **Body**: `SaveWorkflowRequest`
- **Response**: `SaveWorkflowResponse`

#### `POST /workflow/save/temp`
Save as TEMP (validated, for testing)
- **Body**: `SaveWorkflowRequest`
- **Response**: `SaveWorkflowResponse`
- Requires validation to pass

#### `POST /workflow/save/final`
Save as FINAL (immutable, versioned)
- **Body**: `SaveWorkflowRequest`
- **Response**: `SaveWorkflowResponse`
- Requires validation to pass

#### `POST /workflow/load`
Load workflow from storage
- **Body**: `LoadWorkflowRequest`
- **Response**: `AgentSystemDesign`

#### `POST /workflow/clone`
Clone FINAL → DRAFT for editing
- **Body**: `CloneWorkflowRequest`
- **Response**: `AgentSystemDesign`

#### `GET /workflow/versions/{workflow_id}`
List all versions of a workflow
- **Response**: `List[string]`

#### `POST /workflow/version/bump`
Bump workflow version
- **Body**: `VersionWorkflowRequest`
- **Response**: `VersionWorkflowResponse`

#### `DELETE /workflow/delete/draft/{workflow_id}`
Delete DRAFT workflow
- **Response**: `{"success": true, "message": "..."}`

#### `DELETE /workflow/delete/temp/{workflow_id}`
Delete TEMP workflow
- **Response**: `{"success": true, "message": "..."}`

#### `POST /workflow/archive/{workflow_id}?version={version}`
Archive FINAL version
- **Response**: `{"success": true, "message": "..."}`

#### `GET /workflow/list?state={draft|temp|final}`
List all workflows (optional state filter)
- **Response**: `ListWorkflowsResponse`

---

### Agent Management

#### `GET /agent/list`
List all registered agents
- **Response**: `ListAgentsResponse`

#### `GET /agent/{agent_id}`
Get agent by ID
- **Response**: `AgentDefinition`

#### `DELETE /agent/{agent_id}`
Delete agent from registry
- **Response**: `{"success": true, "agent_id": "..."}`

---

### Runtime Execution

#### `POST /runtime/execute`
Execute a workflow
- **Body**: `ExecuteWorkflowRequest`
- **Response**: `ExecutionStatus`
- Modes: `test` (TEMP) or `final` (FINAL)

#### `POST /runtime/execute/stream`
Execute with streaming updates (SSE)
- **Body**: `ExecuteWorkflowRequest`
- **Response**: Server-Sent Events stream
- Real-time execution progress

#### `POST /runtime/resume`
Resume paused execution (HITL)
- **Body**: `{workflow_id, thread_id, human_decision}`
- **Response**: `ExecutionStatus`

#### `GET /runtime/status/{run_id}`
Get execution status by run ID
- **Response**: `ExecutionStatus`

#### `DELETE /runtime/cancel/{run_id}`
Cancel running execution
- **Response**: `{"success": true, "run_id": "..."}`

#### `GET /runtime/active`
List all active executions
- **Response**: `List[ExecutionStatus]`

#### `POST /runtime/batch`
Execute multiple workflows in parallel
- **Body**: `List[ExecuteWorkflowRequest]`
- **Response**: `List[ExecutionStatus]`

#### `GET /runtime/history?workflow_id={id}&limit={n}`
Get execution history
- **Response**: `List[ExecutionStatus]`

---

### Visualization

#### `POST /visualize/graph`
Generate workflow graph
- **Body**: `WorkflowGraphRequest`
- **Response**: `WorkflowGraphResponse` (nodes + edges)

#### `POST /visualize/apply-edits`
Apply UI graph edits back to workflow
- **Body**: `ApplyGraphEditRequest`
- **Response**: `ApplyGraphEditResponse`

---

### Telemetry & Metrics

#### `POST /telemetry/metrics`
Query execution metrics
- **Body**: `TelemetryQuery`
- **Response**: `TelemetryResponse`
- Filters: workflow_id, run_id, agent_id, time range

#### `GET /telemetry/workflow/{workflow_id}/history?limit={n}`
Get execution history for workflow
- **Response**: `List[dict]`

#### `GET /telemetry/cost/{run_id}`
Get cost breakdown (INR)
- **Response**: Cost breakdown by agent/tool

---

## Internal APIs (Component-to-Component)

### Validator

- `POST /api/internal/validator/agent-system`
- `POST /api/internal/validator/agent`
- `POST /api/internal/validator/workflow`

### Workflow

- `POST /api/internal/workflow/compile`
- `POST /api/internal/workflow/modify`
- `POST /api/internal/workflow/version/bump`

### Agent

- `POST /api/internal/agent/create`
- `POST /api/internal/agent/validate-permissions`
- `GET /api/internal/agent/list`
- `GET /api/internal/agent/{agent_id}`

### Runtime

- `POST /api/internal/runtime/execute`
- `POST /api/internal/runtime/resume`
- `GET /api/internal/runtime/metrics/{run_id}`
- `GET /api/internal/runtime/checkpoint/{thread_id}`

### Storage

- `POST /api/internal/storage/save/draft`
- `POST /api/internal/storage/save/temp`
- `POST /api/internal/storage/save/final`
- `POST /api/internal/storage/load`
- `POST /api/internal/storage/clone`
- `POST /api/internal/storage/delete`
- `GET /api/internal/storage/versions/{workflow_id}`

### Visualization

- `POST /api/internal/visualize/graph`
- `POST /api/internal/visualize/apply-edits`

---

## API Documentation

Interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## Architecture Notes

### Service Boundaries
All external APIs call service layer (no direct core imports):
```
External API → Service Layer → Core Logic
```

### Internal APIs
Used for component-to-component communication:
```
Component A → Internal API → Service Layer → Component B
```

### Microservice Readiness
Today:
```
Runtime → internal API → Validator (same process)
```

Tomorrow (zero business logic change):
```
Runtime → HTTP/gRPC → Validator Service (different process)
```

---

## LangGraph v1 Integration

- Uses latest LangGraph v1 patterns
- StateGraph compilation
- Async execution (`ainvoke`, `astream`)
- Built-in HITL with interrupts
- Durable execution with checkpointing

---

## Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 404 | Not Found |
| 422 | Validation Failed |
| 500 | Internal Server Error |
| 501 | Not Implemented (TODO markers) |
