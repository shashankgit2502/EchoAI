# EchoAI Workflow Orchestrator - API Documentation

Base URL: `http://localhost:8000`

---

## Table of Contents

1. [Workflow APIs](#workflow-apis)
   - [Design](#workflow-design)
   - [Validation](#workflow-validation)
   - [Storage](#workflow-storage)
   - [Execution](#workflow-execution)
   - [Visualization](#workflow-visualization)
2. [Agent APIs](#agent-apis)
   - [Registry](#agent-registry)
   - [Factory](#agent-factory)
   - [Permissions](#agent-permissions)
3. [Legacy APIs](#legacy-apis-backward-compatible)

---

## Workflow APIs

### Workflow Design

#### **POST** `/workflows/design/prompt`
Design workflow from natural language prompt.

**Request:**
```json
{
  "prompt": "Create a workflow to analyze sales data and generate insights",
  "default_llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.2
  }
}
```

**Response:**
```json
{
  "workflow": {
    "workflow_id": "wf_abc123",
    "name": "Workflow from prompt",
    "description": "Create a workflow to analyze sales data...",
    "status": "draft",
    "version": "0.1",
    "execution_model": "sequential",
    "agents": ["agt_001", "agt_002"],
    "connections": [
      {
        "from": "agt_001",
        "to": "agt_002"
      }
    ],
    "state_schema": {},
    "human_in_loop": {
      "enabled": false,
      "review_points": []
    },
    "metadata": {
      "created_by": "designer",
      "created_at": "2026-01-15T10:00:00",
      "tags": ["auto-generated"]
    }
  },
  "agents": {
    "agt_001": {
      "agent_id": "agt_001",
      "name": "Analyzer",
      "role": "Data Analysis",
      "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.2
      },
      "tools": [],
      "input_schema": ["input_data"],
      "output_schema": ["analysis_result"]
    },
    "agt_002": {
      "agent_id": "agt_002",
      "name": "Synthesizer",
      "role": "Result Synthesis",
      "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.2
      },
      "tools": [],
      "input_schema": ["analysis_result"],
      "output_schema": ["final_output"]
    }
  }
}
```

---

### Workflow Validation

#### **POST** `/workflows/validate/draft`
Validate draft workflow (sync only, before HITL).

**Request:**
```json
{
  "workflow": {
    "workflow_id": "wf_abc123",
    "name": "Test Workflow",
    "status": "draft",
    "version": "0.1",
    "execution_model": "sequential",
    "agents": ["agt_001", "agt_002"],
    "connections": [
      {
        "from": "agt_001",
        "to": "agt_002"
      }
    ],
    "state_schema": {},
    "metadata": {}
  },
  "agents": {
    "agt_001": { /* agent definition */ },
    "agt_002": { /* agent definition */ }
  }
}
```

**Response:**
```json
{
  "valid": true,
  "errors": [],
  "warnings": []
}
```

**Error Response:**
```json
{
  "valid": false,
  "errors": [
    "Agent 'agt_003' not found",
    "State key 'missing_data' written by multiple agents"
  ],
  "warnings": [
    "Tool 'legacy_db_reader' is deprecated"
  ]
}
```

---

#### **POST** `/workflows/validate/final`
Validate workflow after HITL (full async validation).

**Request:** Same as `/validate/draft`

**Response:**
```json
{
  "valid": true,
  "errors": [],
  "warnings": []
}
```

**Note:** If validation succeeds, workflow status is automatically updated to "validated".

---

### Workflow Storage

#### **POST** `/workflows/temp/save`
Save workflow as temp for testing.

**Request:**
```json
{
  "workflow_id": "wf_abc123",
  "name": "Test Workflow",
  "status": "validated",
  "version": "0.1",
  "execution_model": "sequential",
  "agents": ["agt_001", "agt_002"],
  "connections": [...],
  "state_schema": {},
  "metadata": {}
}
```

**Response:**
```json
{
  "workflow_id": "wf_abc123",
  "path": "/path/to/storage/workflows/temp/wf_abc123.temp.json",
  "state": "temp"
}
```

**Error (400):**
```json
{
  "detail": "Workflow must be validated before saving as temp"
}
```

---

#### **GET** `/workflows/{workflow_id}/temp`
Load temp workflow.

**URL Parameters:**
- `workflow_id` (string, required): Workflow identifier

**Response:**
```json
{
  "workflow_id": "wf_abc123",
  "name": "Test Workflow",
  "status": "testing",
  "version": "0.1",
  "execution_model": "sequential",
  "agents": ["agt_001", "agt_002"],
  "connections": [...],
  "metadata": {
    "is_temp": true
  }
}
```

**Error (404):**
```json
{
  "detail": "Temp workflow not found"
}
```

---

#### **DELETE** `/workflows/{workflow_id}/temp`
Delete temp workflow.

**URL Parameters:**
- `workflow_id` (string, required): Workflow identifier

**Response:**
```json
{
  "message": "Temp workflow deleted",
  "workflow_id": "wf_abc123"
}
```

---

#### **POST** `/workflows/final/save`
Save workflow as final (versioned, immutable).

**Request:**
```json
{
  "workflow": {
    "workflow_id": "wf_abc123",
    "name": "Production Workflow",
    "status": "validated",
    "version": "1.0",
    "execution_model": "hierarchical",
    "agents": ["agt_001", "agt_002"],
    "connections": [...],
    "metadata": {}
  }
}
```

**Response:**
```json
{
  "workflow_id": "wf_abc123",
  "version": "1.0",
  "path": "/path/to/storage/workflows/final/wf_abc123.v1.0.json"
}
```

---

#### **GET** `/workflows/{workflow_id}/final/{version}`
Load specific final version.

**URL Parameters:**
- `workflow_id` (string, required): Workflow identifier
- `version` (string, required): Version number (e.g., "1.0")

**Response:**
```json
{
  "workflow_id": "wf_abc123",
  "name": "Production Workflow",
  "status": "final",
  "version": "1.0",
  "execution_model": "hierarchical",
  "agents": [...],
  "metadata": {
    "immutable": true
  }
}
```

**Error (404):**
```json
{
  "detail": "Workflow version not found"
}
```

---

#### **GET** `/workflows/{workflow_id}/versions`
List all final versions of a workflow.

**URL Parameters:**
- `workflow_id` (string, required): Workflow identifier

**Response:**
```json
{
  "workflow_id": "wf_abc123",
  "versions": ["1.0", "1.1", "2.0"]
}
```

**Error (404):**
```json
{
  "detail": "No versions found"
}
```

---

#### **POST** `/workflows/clone`
Clone final workflow to draft for editing.

**Request:**
```json
{
  "workflow_id": "wf_abc123",
  "from_version": "1.0"
}
```

**Response:**
```json
{
  "message": "Workflow cloned to draft",
  "workflow_id": "wf_abc123",
  "base_version": "1.0"
}
```

**Error (404):**
```json
{
  "detail": "Final workflow not found"
}
```

---

### Workflow Execution

#### **POST** `/workflows/execute`
Execute workflow (test or final mode).

**Request:**
```json
{
  "workflow_id": "wf_abc123",
  "mode": "test",
  "version": null,
  "input_payload": {
    "input_data": "Sample sales data",
    "parameters": {
      "date_range": "2024-01-01 to 2024-12-31"
    }
  }
}
```

**Request (Final Mode):**
```json
{
  "workflow_id": "wf_abc123",
  "mode": "final",
  "version": "1.0",
  "input_payload": {
    "input_data": "Production data"
  }
}
```

**Response:**
```json
{
  "run_id": "run_xyz789",
  "workflow_id": "wf_abc123",
  "status": "completed",
  "execution_mode": "test",
  "output": {
    "message": "Workflow executed successfully (placeholder)",
    "input_received": {
      "input_data": "Sample sales data"
    }
  }
}
```

**Modes:**
- `"test"`: Executes temp workflow (version not required)
- `"final"`: Executes final workflow (version required)

---

### Workflow Visualization

#### **GET** `/workflows/{workflow_id}/graph`
Get graph representation of workflow.

**URL Parameters:**
- `workflow_id` (string, required): Workflow identifier

**Query Parameters:**
- `state` (string, optional, default: "temp"): Workflow state ("temp" or "final")

**Response:**
```json
{
  "nodes": [
    {
      "id": "agt_001",
      "label": "Analyzer",
      "type": "agent",
      "metadata": {
        "role": "Data Analysis",
        "llm": {
          "provider": "openai",
          "model": "gpt-4o-mini"
        },
        "tools": []
      }
    },
    {
      "id": "agt_002",
      "label": "Synthesizer",
      "type": "agent",
      "metadata": {
        "role": "Result Synthesis",
        "llm": {
          "provider": "openai",
          "model": "gpt-4o-mini"
        }
      }
    }
  ],
  "edges": [
    {
      "source": "agt_001",
      "target": "agt_002",
      "condition": null
    }
  ]
}
```

**Error (404):**
```json
{
  "detail": "Workflow not found"
}
```

---

## Agent APIs

### Agent Registry

#### **POST** `/agents/register`
Register a new agent in the registry.

**Request:**
```json
{
  "agent_id": "custom_agt_001",
  "name": "Custom Analyzer",
  "role": "Data Analysis",
  "description": "Analyzes complex datasets",
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "max_tokens": 2000
  },
  "tools": ["tool_001", "tool_002"],
  "input_schema": ["input_data", "parameters"],
  "output_schema": ["analysis_result", "metrics"],
  "constraints": {
    "max_steps": 5,
    "timeout_seconds": 30,
    "budget_tokens": 5000
  },
  "permissions": {
    "can_call_agents": false,
    "allowed_agents": []
  }
}
```

**Response:**
```json
{
  "agent_id": "custom_agt_001",
  "path": "/path/to/storage/agents/custom_agt_001.json"
}
```

---

#### **GET** `/agents/{agent_id}`
Get agent by ID from registry.

**URL Parameters:**
- `agent_id` (string, required): Agent identifier

**Response:**
```json
{
  "agent_id": "custom_agt_001",
  "name": "Custom Analyzer",
  "role": "Data Analysis",
  "description": "Analyzes complex datasets",
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.2
  },
  "tools": ["tool_001", "tool_002"],
  "input_schema": ["input_data", "parameters"],
  "output_schema": ["analysis_result"],
  "metadata": {
    "registered_at": "2026-01-15T10:00:00"
  }
}
```

**Error (404):**
```json
{
  "detail": "Agent not found"
}
```

---

#### **GET** `/agents/registry/list`
List all agents in registry.

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "agt_001",
      "name": "Analyzer",
      "role": "Data Analysis"
    },
    {
      "agent_id": "agt_002",
      "name": "Synthesizer",
      "role": "Result Synthesis"
    }
  ],
  "count": 2
}
```

---

#### **PUT** `/agents/{agent_id}`
Update an existing agent.

**URL Parameters:**
- `agent_id` (string, required): Agent identifier

**Request:**
```json
{
  "name": "Updated Analyzer",
  "llm": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-5",
    "temperature": 0.3
  },
  "tools": ["tool_001", "tool_003"]
}
```

**Response:** (Returns full updated agent)
```json
{
  "agent_id": "custom_agt_001",
  "name": "Updated Analyzer",
  "role": "Data Analysis",
  "llm": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-5",
    "temperature": 0.3
  },
  "tools": ["tool_001", "tool_003"],
  "metadata": {
    "registered_at": "2026-01-15T10:00:00",
    "updated_at": "2026-01-15T11:30:00"
  }
}
```

**Error (404):**
```json
{
  "detail": "Agent 'custom_agt_001' not found"
}
```

---

#### **DELETE** `/agents/{agent_id}`
Delete an agent from registry.

**URL Parameters:**
- `agent_id` (string, required): Agent identifier

**Response:**
```json
{
  "message": "Agent deleted",
  "agent_id": "custom_agt_001"
}
```

**Error (404):**
```json
{
  "detail": "Agent 'custom_agt_001' not found"
}
```

---

#### **GET** `/agents/role/{role}`
Get agents by role.

**URL Parameters:**
- `role` (string, required): Agent role

**Response:**
```json
{
  "role": "Data Analysis",
  "agents": [
    {
      "agent_id": "agt_001",
      "name": "Analyzer",
      "role": "Data Analysis"
    },
    {
      "agent_id": "custom_agt_001",
      "name": "Custom Analyzer",
      "role": "Data Analysis"
    }
  ],
  "count": 2
}
```

---

### Agent Factory

#### **POST** `/agents/instantiate/{agent_id}`
Create runtime agent instance from definition.

**URL Parameters:**
- `agent_id` (string, required): Agent identifier

**Query Parameters:**
- `bind_tools` (boolean, optional, default: true): Bind tools to agent

**Response:**
```json
{
  "agent_id": "agt_001",
  "name": "Analyzer",
  "role": "Data Analysis",
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "ready": true
  },
  "tools": [
    {
      "tool_id": "tool_001",
      "name": "Database Reader",
      "bound": true
    }
  ],
  "constraints": {
    "max_steps": 5,
    "timeout_seconds": 30
  },
  "runtime_ready": true
}
```

**Error (404):**
```json
{
  "detail": "Agent not found"
}
```

---

#### **POST** `/agents/instantiate/batch`
Create multiple runtime agent instances.

**Request:**
```json
{
  "agent_ids": ["agt_001", "agt_002", "agt_003"]
}
```

**Response:**
```json
{
  "instances": {
    "agt_001": {
      "agent_id": "agt_001",
      "name": "Analyzer",
      "runtime_ready": true
    },
    "agt_002": {
      "agent_id": "agt_002",
      "name": "Synthesizer",
      "runtime_ready": true
    },
    "agt_003": {
      "agent_id": "agt_003",
      "name": "Reporter",
      "runtime_ready": true
    }
  },
  "count": 3
}
```

---

### Agent Permissions

#### **POST** `/agents/permissions/check`
Check if caller can communicate with target agent.

**Request:**
```json
{
  "caller_id": "agt_001",
  "target_id": "agt_002",
  "workflow": {
    "workflow_id": "wf_abc123",
    "execution_model": "hierarchical",
    "agents": ["agt_001", "agt_002", "agt_003"],
    "hierarchy": {
      "master_agent": "agt_001",
      "delegation_order": ["agt_002", "agt_003"]
    }
  },
  "agents": {
    "agt_001": { /* agent definition */ },
    "agt_002": { /* agent definition */ }
  }
}
```

**Response:**
```json
{
  "caller_id": "agt_001",
  "target_id": "agt_002",
  "allowed": true
}
```

**Permission Rules:**
- **Hierarchical**: Master can call sub-agents, sub-agents can only call master
- **Sequential**: Agent can only call next in sequence
- **Parallel**: All agents can communicate freely

---

#### **POST** `/agents/permissions/validate`
Validate all permissions in a workflow.

**Request:**
```json
{
  "workflow": {
    "workflow_id": "wf_abc123",
    "execution_model": "hierarchical",
    "agents": ["agt_001", "agt_002"],
    "hierarchy": {
      "master_agent": "agt_001",
      "delegation_order": ["agt_002"]
    }
  },
  "agents": {
    "agt_001": {
      "agent_id": "agt_001",
      "permissions": {
        "can_call_agents": true
      }
    },
    "agt_002": {
      "agent_id": "agt_002",
      "permissions": {
        "can_call_agents": false
      }
    }
  }
}
```

**Response:**
```json
{
  "valid": true,
  "errors": []
}
```

**Error Response:**
```json
{
  "valid": false,
  "errors": [
    "Sub-agent 'agt_002' cannot have can_call_agents in hierarchical workflow"
  ]
}
```

---

#### **GET** `/agents/permissions/targets/{agent_id}`
Get list of agents that the given agent can call.

**URL Parameters:**
- `agent_id` (string, required): Agent identifier

**Request Body:**
```json
{
  "workflow": {
    "workflow_id": "wf_abc123",
    "execution_model": "hierarchical",
    "agents": ["agt_001", "agt_002", "agt_003"],
    "hierarchy": {
      "master_agent": "agt_001",
      "delegation_order": ["agt_002", "agt_003"]
    }
  },
  "agents": {
    "agt_001": { /* master agent */ },
    "agt_002": { /* sub-agent */ },
    "agt_003": { /* sub-agent */ }
  }
}
```

**Response:**
```json
{
  "agent_id": "agt_001",
  "allowed_targets": ["agt_002", "agt_003"]
}
```

**Response (for sub-agent):**
```json
{
  "agent_id": "agt_002",
  "allowed_targets": ["agt_001"]
}
```

---

## Legacy APIs (Backward Compatible)

These existing APIs continue to work unchanged:

### Workflows

- **POST** `/workflows/create/prompt` - Simple workflow creation
- **POST** `/workflows/create/canvas` - Canvas-based workflow creation
- **POST** `/workflows/validate` - Simple workflow validation

### Agents

- **POST** `/agents/create/prompt` - Simple agent creation from prompt
- **POST** `/agents/create/card` - Canvas-based agent creation
- **POST** `/agents/validate` - Simple agent validation
- **GET** `/agents/list` - List all agents (simple)

---

## Error Response Format

All API errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**
- `200` - Success
- `400` - Bad Request (validation failed, invalid input)
- `404` - Not Found (resource doesn't exist)
- `422` - Unprocessable Entity (validation error)
- `500` - Internal Server Error

---

## Workflow Lifecycle

```
1. Design       → POST /workflows/design/prompt
2. Validate     → POST /workflows/validate/draft
3. HITL Edit    → (Frontend editing)
4. Re-validate  → POST /workflows/validate/final
5. Save Temp    → POST /workflows/temp/save
6. Test Execute → POST /workflows/execute (mode: "test")
7. Save Final   → POST /workflows/final/save
8. Production   → POST /workflows/execute (mode: "final", version: "1.0")
```

---

## Agent Lifecycle

```
1. Register     → POST /agents/register
2. Update       → PUT /agents/{agent_id}
3. Instantiate  → POST /agents/instantiate/{agent_id}
4. Use in Workflow
5. Delete       → DELETE /agents/{agent_id}
```

---

## Execution Models

### Sequential
- Agents execute in linear order
- Each agent passes output to next
- Connections define the sequence

### Parallel
- Multiple agents execute simultaneously
- Results merged at the end
- No direct agent-to-agent communication

### Hierarchical
- Master agent coordinates sub-agents
- Master delegates tasks
- Sub-agents report back to master only
- Sub-agents cannot communicate with each other

### Hybrid
- Combination of above models
- Complex workflows with multiple patterns

---

## Authentication

**Current Status:** Not implemented in this version.

**Future:** All endpoints will require JWT token in Authorization header:
```
Authorization: Bearer <token>
```

---

## Rate Limiting

**Current Status:** Not implemented.

**Future:** Rate limits will be applied per API key/user.

---

## Swagger UI

Interactive API documentation available at:
**http://localhost:8000/docs**

- Test all endpoints
- See request/response schemas
- Try out API calls directly from browser

---

## Examples

See the main documentation for complete frontend integration examples using:
- JavaScript Fetch API
- Axios
- React/Vue integration patterns

---

## Support

For issues or questions:
1. Check Swagger UI at `/docs`
2. Review error messages in response
3. Check backend logs for detailed errors

---

**Version:** 1.0
**Last Updated:** January 15, 2026
**Base URL:** http://localhost:8000
