# Agent Design API - Prompt-Based Agent Creation

Base URL: `http://localhost:8000`

---

## Overview

Create intelligent agents from natural language descriptions without writing code. The system uses LLM (OpenAI/Ollama) to analyze your prompt and generate a complete, executable agent definition.

---

## Endpoint

### **POST** `/agents/design/prompt`

Design and register an agent from natural language prompt.
---
## Request

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "prompt": "Create an agent that analyzes customer feedback and extracts sentiment",
  "default_llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.2
  }
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | Natural language description of what the agent should do |
| `default_llm` | object | No | LLM configuration for the agent |
| `default_llm.provider` | string | No | LLM provider: "openai", "anthropic", "azure" (default: "openai") |
| `default_llm.model` | string | No | Model name (default: "gpt-4o-mini") |
| `default_llm.temperature` | float | No | Temperature 0.0-1.0 (default: 0.2) |
| `default_llm.max_tokens` | integer | No | Maximum tokens (default: 1000) |

---

## Response

**Status:** `200 OK`

**Body:**
```json
{
  "agent": {
    "agent_id": "agt_abc123",
    "name": "Customer Feedback Analyzer",
    "role": "Sentiment Analysis",
    "description": "Analyzes customer feedback to extract sentiment and key themes",
    "llm": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "temperature": 0.2,
      "max_tokens": 1000
    },
    "tools": [],
    "input_schema": ["customer_feedback"],
    "output_schema": ["sentiment_analysis", "key_themes"],
    "constraints": {
      "max_steps": 5,
      "timeout_seconds": 30,
      "budget_tokens": 5000
    },
    "permissions": {
      "can_call_agents": false,
      "allowed_agents": []
    },
    "metadata": {
      "created_by": "designer_llm",
      "created_at": "2026-01-15T10:00:00",
      "tags": ["auto-generated", "llm-designed"]
    }
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `agent.agent_id` | string | Unique identifier for the agent |
| `agent.name` | string | Human-readable agent name |
| `agent.role` | string | Agent's primary role/responsibility |
| `agent.description` | string | Detailed description of agent capabilities |
| `agent.llm` | object | LLM configuration for agent execution |
| `agent.tools` | array | List of tool IDs the agent can use |
| `agent.input_schema` | array | Expected input keys |
| `agent.output_schema` | array | Output keys the agent produces |
| `agent.constraints` | object | Execution constraints (steps, timeout, tokens) |
| `agent.permissions` | object | Agent-to-agent communication permissions |
| `agent.metadata` | object | Creation timestamp, tags, etc. |

---

## Error Responses

**400 Bad Request:**
```json
{
  "detail": "Prompt cannot be empty"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "LLM design failed: OPENAI_API_KEY not set"
}
```

---

## Examples

### Example 1: Data Analysis Agent

**Request:**
```bash
curl -X POST http://localhost:8000/agents/design/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create an agent that analyzes sales data and generates insights",
    "default_llm": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "temperature": 0.2
    }
  }'
```

**Response:**
```json
{
  "agent": {
    "agent_id": "agt_xyz789",
    "name": "Sales Data Analyzer",
    "role": "Data Analysis",
    "description": "Analyzes sales data to generate actionable insights",
    "llm": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "temperature": 0.2
    },
    "input_schema": ["sales_data"],
    "output_schema": ["insights", "recommendations"]
  }
}
```

---

### Example 2: Content Generator Agent

**Request:**
```bash
curl -X POST http://localhost:8000/agents/design/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create an agent that generates marketing copy from product descriptions"
  }'
```

**Response:**
```json
{
  "agent": {
    "agent_id": "agt_content_001",
    "name": "Marketing Copy Generator",
    "role": "Content Generation",
    "description": "Transforms product descriptions into engaging marketing copy",
    "llm": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "temperature": 0.7
    },
    "input_schema": ["product_description", "target_audience"],
    "output_schema": ["marketing_copy", "key_features"]
  }
}
```

---

### Example 3: Using Ollama (Local LLM)

**Request:**
```bash
curl -X POST http://localhost:8000/agents/design/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create an agent that summarizes legal documents",
    "default_llm": {
      "provider": "openai",
      "model": "mistral-nemo:12b-instruct-2407-fp16",
      "temperature": 0.1
    }
  }'
```

**Note:** When `USE_OLLAMA=true` in `.env`, the system automatically uses your local Ollama endpoint.

---

## Use Cases

### 1. Standalone Agents
Create agents that work independently without workflows:
```
"Create an agent that translates text from English to Spanish"
```

### 2. Specialized Processors
Design agents for specific data processing tasks:
```
"Create an agent that extracts key entities from medical records"
```

### 3. Rapid Prototyping
Quickly test agent capabilities:
```
"Create an agent that classifies customer support tickets by urgency"
```

### 4. Template Generation
Generate base templates for later customization:
```
"Create an agent that performs financial analysis"
```

---

## What Happens After Creation

The agent is automatically:
1. ✅ **Registered** in the agent registry
2. ✅ **Ready** to be used in workflows
3. ✅ **Editable** via `PUT /agents/{agent_id}`
4. ✅ **Instantiable** via `POST /agents/instantiate/{agent_id}`
5. ✅ **Executable** when added to workflows

---

## Next Steps

After creating an agent:

### Use in Workflow
```bash
POST /workflows/design/prompt
{
  "prompt": "Create workflow using agent agt_abc123"
}
```

### Update Agent
```bash
PUT /agents/agt_abc123
{
  "llm": {
    "model": "gpt-4",
    "temperature": 0.3
  }
}
```

### Instantiate for Testing
```bash
POST /agents/instantiate/agt_abc123
```

### Delete Agent
```bash
DELETE /agents/agt_abc123
```

---

## LLM Configuration

### Environment Variables

**Using Ollama (Local):**
```bash
USE_OLLAMA=true
OLLAMA_BASE_URL=http://10.188.100.131:8004/v1
OLLAMA_MODEL=mistral-nemo:12b-instruct-2407-fp16
```

**Using OpenAI (Cloud):**
```bash
USE_OLLAMA=false
OPENAI_API_KEY=sk-your-key-here
```

**Using Anthropic (Cloud):**
```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

---

## Integration Example

### JavaScript/Fetch
```javascript
async function createAgent(prompt) {
  const response = await fetch('http://localhost:8000/agents/design/prompt', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt: prompt,
      default_llm: {
        provider: 'openai',
        model: 'gpt-4o-mini',
        temperature: 0.2
      }
    })
  });

  const data = await response.json();
  console.log('Agent created:', data.agent.agent_id);
  return data.agent;
}

// Usage
const agent = await createAgent(
  'Create an agent that analyzes product reviews'
);
```

### Python/Requests
```python
import requests

def create_agent(prompt):
    response = requests.post(
        'http://localhost:8000/agents/design/prompt',
        json={
            'prompt': prompt,
            'default_llm': {
                'provider': 'openai',
                'model': 'gpt-4o-mini',
                'temperature': 0.2
            }
        }
    )
    data = response.json()
    print(f"Agent created: {data['agent']['agent_id']}")
    return data['agent']

# Usage
agent = create_agent('Create an agent that extracts dates from text')
```

---

## Comparison with Manual Registration

### Prompt-Based (This API)
```bash
POST /agents/design/prompt
{
  "prompt": "Create a sentiment analyzer"
}
```
✅ Fast
✅ No schema design needed
✅ Auto-generates fields
✅ LLM-powered

### Manual Registration
```bash
POST /agents/register
{
  "agent_id": "...",
  "name": "...",
  "role": "...",
  "llm": {...},
  "input_schema": [...],
  "output_schema": [...]
}
```
✅ Full control
✅ Precise configuration
✅ No LLM needed
✅ Deterministic

**Choose:**
- **Prompt-based** for rapid prototyping
- **Manual** for production-grade agents

---

## Version Information

**API Version:** 1.0
**Last Updated:** January 15, 2026
**Endpoint:** `/agents/design/prompt`
**Method:** POST
**Authentication:** None (future: JWT required)

---

## Support

For issues:
1. Check response error messages
2. Verify LLM configuration (`.env` file)
3. Ensure Ollama/OpenAI is accessible
4. Review backend logs for detailed errors

---

**Related Documentation:**
- [EchoAI-API.md](./EchoAI-API.md) - Full API reference
- [Agent Registry API](./EchoAI-API.md#agent-registry) - Manual agent management
- [Workflow Design API](./EchoAI-API.md#workflow-design) - Create workflows from prompts
