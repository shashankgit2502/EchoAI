# LLM Provider Architecture

## Overview

The LLM Provider system is a **unified abstraction layer** that enables the workflow orchestrator to work with multiple LLM providers (Anthropic, OpenAI, OpenRouter, On-Premise) through a single, consistent interface.

**Core Principle**: LLM selection is **configuration, not logic**.

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  (Workflow Designer, Agent Factory, Runtime Executor)        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   LLM PROVIDER LAYER                         │
│            (app/services/llm_provider.py)                    │
│                                                              │
│  ┌─────────────────────────────────────────────────┐       │
│  │           LLMProvider (Singleton)               │       │
│  │  - Unified invoke interface                     │       │
│  │  - Model availability checking                  │       │
│  │  - Cost tracking & estimation                   │       │
│  │  - Token counting                               │       │
│  │  - Streaming support                            │       │
│  └──────────────────────┬──────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────┐       │
│  │           ModelCatalog                          │       │
│  │  Loads: schemas/llm_models.json                 │       │
│  │  - Model metadata (context, cost, capabilities) │       │
│  │  - Provider metadata (API keys, rate limits)    │       │
│  └──────────────────────┬──────────────────────────┘       │
└─────────────────────────┼────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  PROVIDER ADAPTERS                           │
│       (LangChain Integration Layer)                          │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Anthropic   │  │   OpenAI     │  │ OpenRouter   │     │
│  │ ChatAnthropic│  │  ChatOpenAI  │  │  ChatOpenAI  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────┐                                           │
│  │   On-Prem    │                                           │
│  │   (Ollama)   │                                           │
│  │  ChatOpenAI  │                                           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    ACTUAL LLM APIs                           │
│  Anthropic API | OpenAI API | OpenRouter | Ollama Server    │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Model Catalog (`schemas/llm_models.json`)

**Central registry of all available LLM models**

```json
{
  "models": [
    {
      "id": "claude-sonnet-4-5-20250929",
      "name": "Claude Sonnet 4.5",
      "provider": "anthropic",
      "tier": "standard",
      "context_window": 200000,
      "max_output_tokens": 16384,
      "supports_tools": true,
      "supports_vision": true,
      "cost_per_million_input_tokens": 3.0,
      "cost_per_million_output_tokens": 15.0,
      "recommended_for": ["workflow_design", "agent_coordination"]
    }
  ],
  "providers": [
    {
      "id": "anthropic",
      "name": "Anthropic",
      "api_base_url": "https://api.anthropic.com/v1",
      "requires_api_key": true,
      "api_key_env_var": "ANTHROPIC_API_KEY"
    }
  ]
}
```

**Features:**
- Model metadata (context window, cost, capabilities)
- Provider configuration (API endpoints, auth requirements)
- Tier-based organization (premium, standard, fast)
- Extensible: Add new models by editing JSON

---

### 2. LLMProvider Class (`app/services/llm_provider.py`)

**Unified interface for all LLM operations**

#### Key Methods:

##### `ainvoke()` - Invoke LLM Asynchronously
```python
response = await llm_provider.ainvoke(
    model="claude-sonnet-4-5-20250929",
    messages=[HumanMessage(content="Analyze this data")],
    temperature=0.7,
    max_tokens=4000
)
```

##### `astream()` - Stream LLM Response
```python
async for chunk in llm_provider.astream(
    model="gpt-4",
    messages=[HumanMessage(content="Generate report")],
    temperature=0.5
):
    print(chunk, end="")
```

##### `check_availability()` - Verify Model Availability
```python
is_available = await llm_provider.check_availability(
    "claude-sonnet-4-5-20250929"
)
```

##### `estimate_cost()` - Calculate Token Cost
```python
cost = llm_provider.estimate_cost(
    model="claude-sonnet-4-5-20250929",
    input_tokens=1000,
    output_tokens=500
)
# Returns: 0.0105 (in USD)
```

##### `list_available_models()` - Get Available Models
```python
models = llm_provider.list_available_models(
    provider="anthropic",
    tier="standard"
)
```

#### Internal Architecture:

**Model Caching:**
```python
# Models are cached with key: "model_id:temperature:max_tokens"
_model_cache = {
    "claude-sonnet-4-5-20250929:0.7:4000": <ChatAnthropic instance>,
    "gpt-4:0.5:2000": <ChatOpenAI instance>
}
```

**Provider-Specific Creation:**
```python
def _create_model(model_id, temperature, max_tokens):
    model_meta = catalog.get_model(model_id)

    if model_meta.provider == "anthropic":
        return ChatAnthropic(model=model_id, ...)

    elif model_meta.provider == "openai":
        return ChatOpenAI(model=model_id, ...)

    elif model_meta.provider == "openrouter":
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model=model_id,
            ...
        )

    elif model_meta.provider == "onprem":
        return ChatOpenAI(
            base_url="http://10.188.100.131:8004/v1",
            model=model_id,
            ...
        )
```

---

## Usage Throughout the System

### 1. Agent Factory (`app/agents/factory.py`)

**Agents are created with LLM configuration from their definition**

```python
class AgentFactory:
    def __init__(self):
        self.llm_provider = get_llm_provider()

    async def create_agent(self, agent_def: AgentDefinition):
        # Get LLM model ID from agent definition
        model_id = agent_def.llm_config.model

        # Verify model availability
        _ = self.llm_provider._get_or_create_model(
            model_id=model_id,
            temperature=agent_def.llm_config.temperature,
            max_tokens=agent_def.llm_config.max_tokens
        )

        # Create agent with LangChain v1
        agent = create_agent(
            model=model_id,  # ← Just the model ID string
            tools=filtered_tools,
            system_prompt=system_prompt
        )
```

**Flow:**
```
Agent JSON Definition
    ↓
{
  "agent_id": "security_analyst",
  "llm": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-5-20250929",
    "temperature": 0.2
  }
}
    ↓
Agent Factory reads config
    ↓
LLM Provider verifies model exists
    ↓
LangChain create_agent() receives model string
    ↓
Runtime: LangChain automatically uses LLM Provider
```

---

### 2. Workflow Designer (`app/workflow/designer.py`)

**Designer uses LLM to generate workflow drafts**

```python
class WorkflowDesigner:
    def __init__(self):
        # Designer uses its own LLM (can be different from agents)
        self.designer_llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model="allenai/olmo-3.1-32b-think:free"
        )

    async def design_from_meta_prompt(self, meta_prompt: str):
        # LLM generates agent system design
        response = await self.designer_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=meta_prompt)
        ])

        # Parse JSON response into AgentSystemDesign
        return self._parse_design(response.content)
```

**Flow:**
```
User Request
    ↓
Meta-Prompt Generator
    ↓
Workflow Designer LLM
    ↓
Draft Workflow JSON
    +
Draft Agent JSON (with LLM configs)
    ↓
Validator checks agents & tools
    ↓
TEMP workflow saved
```

---

### 3. Runtime Execution (Indirect Usage)

**Runtime doesn't call LLM Provider directly—agents do**

```python
# Runtime executor compiles workflow to LangGraph
class WorkflowExecutor:
    async def execute(self, request: ExecuteWorkflowRequest):
        # Load workflow & agents
        agent_system = self._load_agent_system(request)

        # Compile to LangGraph
        compiled_workflow = self.compiler.compile(agent_system)

        # Execute
        final_state = await compiled_workflow.ainvoke(initial_state)
```

**Agent execution internally:**
```python
# When agent runs, LangChain calls the configured LLM
RuntimeAgent.ainvoke(input_data)
    ↓
agent.ainvoke({"messages": [...]})
    ↓
LangChain internally routes to correct provider
    ↓
LLM Provider returns response
```

---

## Configuration Flow

### How LLM Selection Works at Each Stage

#### 1. **System Default** (`core/config.py`)
```python
DEFAULT_LLM = {
    "provider": "anthropic",
    "model": "claude-sonnet-4-5-20250929",
    "temperature": 0.7
}
```

#### 2. **Workflow Designer Override**
```python
# Designer uses free OpenRouter model for cost efficiency
designer_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="allenai/olmo-3.1-32b-think:free"
)
```

#### 3. **Agent-Level Override** (Most Common)
```json
{
  "agent_id": "data_analyst",
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.2,
    "max_tokens": 2000
  }
}
```

#### 4. **TEMP Workflow Override** (Testing)
```json
{
  "workflow_id": "sales_analysis",
  "status": "testing",
  "agents": {
    "analyst": {
      "llm": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929"
      }
    }
  }
}
```

---

## Multi-Provider Support

### Current Providers

| Provider | API Type | Use Case | Cost |
|----------|----------|----------|------|
| **Anthropic** | Direct | Production agents, complex reasoning | Paid |
| **OpenAI** | Direct | Production agents, general tasks | Paid |
| **OpenRouter** | Proxy | Free/cheap models, experimentation | Free/Paid |
| **On-Prem (Ollama)** | Local | Privacy-sensitive, offline, dev | Free |

### Adding New Provider

**1. Update `llm_models.json`:**
```json
{
  "providers": [
    {
      "id": "gemini",
      "name": "Google Gemini",
      "api_base_url": "https://generativelanguage.googleapis.com/v1",
      "requires_api_key": true,
      "api_key_env_var": "GEMINI_API_KEY"
    }
  ],
  "models": [
    {
      "id": "gemini-1.5-pro",
      "provider": "gemini",
      "tier": "standard",
      ...
    }
  ]
}
```

**2. Update `llm_provider.py`:**
```python
def _get_api_key(self, provider: str):
    if provider == "gemini":
        return self.settings.GEMINI_API_KEY
    # ...

def _create_model(self, model_id, ...):
    if model_meta.provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=api_key,
            ...
        )
```

**3. Done!** All agents can now use Gemini models.

---

## Cost Tracking & Optimization

### Cost Estimation
```python
# Before running agent
cost = llm_provider.estimate_cost(
    model="claude-sonnet-4-5-20250929",
    input_tokens=5000,
    output_tokens=2000
)
# Returns: 0.045 USD

# Decide if cost is acceptable
if cost > MAX_COST_PER_AGENT:
    # Use cheaper model
    agent_def.llm_config.model = "gpt-3.5-turbo"
```

### Tier-Based Selection
```python
# For simple validation tasks → use fast tier
fast_models = llm_provider.list_available_models(tier="fast")

# For complex reasoning → use premium tier
premium_models = llm_provider.list_available_models(tier="premium")
```

---

## Availability Checking

### Before Workflow Execution
```python
async def validate_runtime_feasibility(workflow, agents):
    for agent_id in workflow["agents"]:
        agent = agents[agent_id]
        model = agent["llm"]["model"]

        is_available = await llm_provider.check_availability(model)

        if not is_available:
            errors.append(f"Model {model} not available for agent {agent_id}")
```

**What it checks:**
- Model exists in catalog ✓
- API key is configured ✓
- API is reachable ✓
- Makes minimal test call (10 tokens) ✓

---

## Singleton Pattern

**Why Singleton?**
- Reuse model cache across requests
- Avoid reloading catalog repeatedly
- Share rate limit tracking

**Usage:**
```python
# Everywhere in codebase
from app.services.llm_provider import get_llm_provider

llm_provider = get_llm_provider()  # Always same instance
```

---

## Real Execution Example

### Complete Flow: User Request → Agent Execution

```python
# 1. User submits request
user_request = UserRequest(
    request="Analyze Q4 sales and create improvement strategy",
    preferred_llm="claude-sonnet-4-5-20250929"
)

# 2. Workflow Designer uses LLM to generate draft
designer = WorkflowDesigner()
agent_system, analysis, meta_prompt = await designer.design_from_user_request(
    user_request
)

# Designer outputs:
agent_system = {
    "agents": [
        {
            "id": "data_analyst",
            "llm": {
                "model": "claude-sonnet-4-5-20250929",
                "temperature": 0.2
            },
            "tools": ["read_sales_db"]
        },
        {
            "id": "strategy_planner",
            "llm": {
                "model": "gpt-4",
                "temperature": 0.7
            },
            "tools": ["knowledge_base"]
        }
    ]
}

# 3. Validator checks all models are available
validator = WorkflowValidator()
result = await validator.validate_runtime(agent_system)

# 4. Agent Factory creates runtime agents
factory = AgentFactory()
runtime_agents = await factory.create_agents_batch(
    agent_ids=["data_analyst", "strategy_planner"],
    tools=all_tools
)

# 5. Runtime executes workflow
executor = WorkflowExecutor()
final_result = await executor.execute(
    ExecuteWorkflowRequest(
        workflow_id="sales_analysis",
        execution_mode="test"
    )
)

# During execution:
#   - data_analyst agent uses Claude Sonnet 4.5
#   - strategy_planner agent uses GPT-4
#   - Each agent calls LLM Provider internally
#   - Costs are tracked per agent
```

---

## Key Benefits

### ✅ Flexibility
- Add new providers without touching agent code
- Switch models per agent without refactoring
- Test with cheap models, deploy with premium models

### ✅ Cost Control
- Real-time cost estimation
- Tier-based model selection
- Track spending per agent/workflow

### ✅ Reliability
- Availability checking before execution
- Automatic fallback to cached models
- Graceful error handling

### ✅ Scalability
- Model caching prevents redundant initialization
- Singleton pattern reduces memory usage
- Rate limit awareness (future enhancement)

### ✅ Maintainability
- Single source of truth (llm_models.json)
- Clear separation: config vs logic
- Easy to add new models/providers

---

## Important Rules (from CLAUDE.md)

### 1. **LLM selection is configuration, not logic**
```python
# ✅ CORRECT
agent_def.llm_config.model = "claude-sonnet-4-5-20250929"

# ❌ WRONG
if task_type == "analysis":
    use_claude()
elif task_type == "generation":
    use_gpt4()
```

### 2. **LLM choice lives in agent JSON**
```python
# ✅ Agent definition controls its LLM
{
  "agent_id": "analyst",
  "llm": {
    "model": "claude-sonnet-4-5-20250929"
  }
}

# ❌ Not hardcoded in factory
def create_agent():
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")  # WRONG
```

### 3. **Runtime does not care about providers**
```python
# Runtime just executes workflow
# LLM Provider handles all provider-specific logic
executor.execute(workflow)  # Works with any provider
```

---

## Troubleshooting

### "Model not available"
**Check:**
1. Model exists in `llm_models.json`
2. API key is set in environment
3. API endpoint is reachable
4. API key has permissions

### "Import error: langchain.agents"
**Solution:**
```bash
pip install -U langchain>=1.0.0 langgraph>=1.0.0 langchain-core>=1.0.0
```

### "Cost too high"
**Options:**
1. Use tier="fast" models
2. Use OpenRouter free models
3. Use on-premise Ollama
4. Reduce max_tokens

---

## Provider Features & Model Recommendations

### Provider Supported Features

**What are "supported_features"?**

Each provider has different capabilities. The `supported_features` array tells the system what a provider can do.

```json
{
  "id": "anthropic",
  "supported_features": ["tools", "vision", "streaming", "system_prompts"]
}
```

#### Feature Types Explained

| Feature | Description | When Needed | Example Use Case |
|---------|-------------|-------------|------------------|
| **tools** | Function/tool calling support | Agents need to call external tools | Agent needs to query database or call APIs |
| **vision** | Image understanding | Agents process screenshots/images | Security analyst reviewing SIEM dashboard screenshots |
| **streaming** | Token-by-token response | Real-time UI updates | Chat interfaces, live agent output |
| **system_prompts** | Native system message support | Better prompt control | Claude's system prompt vs OpenAI's approach |

#### How Features Are Used

**In Validator (`validator/async_rules.py`):**
```python
async def validate_agent_capabilities(agent_def, llm_provider):
    model_meta = llm_provider.get_model_metadata(agent_def.llm_config.model)
    provider_meta = llm_provider.catalog.get_provider(model_meta.provider)

    # Check if agent needs tools
    if agent_def.tools and not model_meta.supports_tools:
        errors.append(
            f"Agent {agent_def.id} requires tools but "
            f"model {model_meta.id} doesn't support tools"
        )

    # Check if agent needs vision
    if agent_def.requires_vision and not model_meta.supports_vision:
        errors.append(
            f"Agent {agent_def.id} requires vision but "
            f"model {model_meta.id} doesn't support vision"
        )
```

**In Runtime (Automatic):**
```python
# LangChain automatically uses streaming if supported
if model_meta.supports_streaming:
    async for chunk in agent.astream(input_data):
        yield chunk
else:
    response = await agent.ainvoke(input_data)
    yield response
```

#### Customizing Provider Features

**Add a new feature:**

```json
{
  "id": "anthropic",
  "supported_features": [
    "tools",
    "vision",
    "streaming",
    "system_prompts",
    "json_mode",          // ← NEW: Structured JSON output
    "caching"             // ← NEW: Prompt caching
  ]
}
```

**Then update validator logic:**
```python
# In validator/sync_rules.py
if workflow.requires_structured_output:
    for agent_id in workflow["agents"]:
        model = agents[agent_id]["llm"]["model"]
        provider = get_provider_for_model(model)

        if "json_mode" not in provider.supported_features:
            errors.append(f"Agent {agent_id} requires json_mode")
```

---

### Model Recommendations (`recommended_for`)

**What is "recommended_for"?**

The `recommended_for` array tells the system which tasks a model is optimized for. This helps the Workflow Designer automatically select the best model for each agent.

```json
{
  "id": "claude-sonnet-4-5-20250929",
  "recommended_for": [
    "general_purpose",
    "workflow_design",
    "agent_coordination",
    "balanced_tasks"
  ]
}
```

#### Standard Recommendation Categories

| Category | Meaning | Best Used For |
|----------|---------|---------------|
| **general_purpose** | All-around capability | Any task without specific requirements |
| **workflow_design** | Complex planning & reasoning | Meta-level workflow generation |
| **agent_coordination** | Managing multi-agent systems | Master agents in hierarchical workflows |
| **reasoning** | Deep analytical thinking | Complex problem-solving, strategy |
| **fast_tasks** | Quick, simple operations | Validation, formatting, simple queries |
| **data_analysis** | SQL, statistics, data processing | Database querying, data transformation |
| **security** | Threat detection, analysis | SOC agents, alert triage |
| **code_generation** | Writing/reviewing code | DevOps agents, automation |
| **conversational** | Natural chat interactions | User-facing chatbots |
| **cost_free** | No API cost | Development, testing, experimentation |
| **onprem_deployment** | Runs locally | Privacy-sensitive, offline scenarios |
| **privacy_sensitive** | Data stays local | GDPR, HIPAA, classified data |

#### How Recommendations Are Used

**1. In Workflow Designer (Automatic Model Selection):**

```python
class WorkflowDesigner:
    def select_model_for_agent(self, agent_role: str, agent_task: str):
        # Workflow designer suggests best model based on task
        if agent_role == "master_orchestrator":
            # Need strong reasoning + coordination
            models = llm_provider.catalog.list_all_models()
            candidates = [
                m for m in models
                if "agent_coordination" in m.recommended_for
                and "workflow_design" in m.recommended_for
            ]
            return candidates[0].id  # claude-sonnet-4-5-20250929

        elif agent_role == "data_analyst":
            # Need data processing
            candidates = [
                m for m in models
                if "data_analysis" in m.recommended_for
            ]
            return candidates[0].id

        elif agent_role == "simple_validator":
            # Need fast, cheap model
            candidates = [
                m for m in models
                if "fast_tasks" in m.recommended_for
            ]
            return candidates[0].id
```

**2. In UI (Model Suggestions):**

```python
# Frontend: When user edits agent LLM
GET /api/models/recommended?agent_role=security_analyst

# Response:
[
  {
    "id": "claude-sonnet-4-5-20250929",
    "name": "Claude Sonnet 4.5",
    "reason": "Recommended for: general_purpose, agent_coordination",
    "cost_estimate": "moderate"
  },
  {
    "id": "gpt-4",
    "name": "GPT-4",
    "reason": "Recommended for: general_purpose, reasoning",
    "cost_estimate": "high"
  }
]
```

**3. In Cost Optimizer:**

```python
# Auto-downgrade to cheaper model if task is simple
def optimize_agent_costs(agent_def):
    if agent_def.task_complexity == "simple":
        # Find cheapest model with "fast_tasks"
        models = llm_provider.list_available_models(tier="fast")
        cheapest = min(
            [m for m in models if "fast_tasks" in m.recommended_for],
            key=lambda m: m.cost_per_million_input_tokens
        )
        agent_def.llm_config.model = cheapest.id
```

---

### Customizing Model Recommendations

#### Adding Custom Categories

**Scenario:** You're building SOC agents and want "siem_analysis" category.

**Step 1: Define in `llm_models.json`**
```json
{
  "models": [
    {
      "id": "claude-sonnet-4-5-20250929",
      "recommended_for": [
        "general_purpose",
        "siem_analysis",        // ← NEW
        "threat_detection",     // ← NEW
        "alert_triage"          // ← NEW
      ]
    },
    {
      "id": "mistral-nemo:12b-instruct-2407-fp16",
      "recommended_for": [
        "fast_tasks",
        "alert_triage"          // ← Fast model for simple triage
      ]
    }
  ]
}
```

**Step 2: Update Workflow Designer**
```python
class WorkflowDesigner:
    def select_model_for_agent(self, agent_role: str):
        if agent_role == "siem_analyst":
            # Prioritize models with siem_analysis
            candidates = [
                m for m in self.llm_provider.catalog.list_all_models()
                if "siem_analysis" in m.recommended_for
            ]
            return candidates[0].id if candidates else DEFAULT_MODEL

        elif agent_role == "alert_triager":
            # Use fast model with alert_triage
            candidates = [
                m for m in self.llm_provider.catalog.list_all_models()
                if "alert_triage" in m.recommended_for
                and m.tier == "fast"  # Must be fast
            ]
            return candidates[0].id
```

**Step 3: Document Category**
```json
// In llm_models.json
{
  "recommendation_categories": {
    "siem_analysis": {
      "description": "Security Information and Event Management analysis",
      "required_capabilities": ["reasoning", "pattern_recognition"],
      "typical_agents": ["siem_analyst", "threat_hunter"]
    },
    "alert_triage": {
      "description": "Fast initial alert classification",
      "required_capabilities": ["classification", "speed"],
      "typical_agents": ["alert_classifier", "priority_scorer"]
    }
  }
}
```

#### Changing Model Recommendations Based on Your Use Case

**Example 1: You prioritize cost over capability**

```json
{
  "id": "allenai/olmo-3.1-32b-think:free",
  "recommended_for": [
    "general_purpose",      // ← Add this (originally just "fast_tasks")
    "workflow_design",      // ← Add this if you're on a budget
    "fast_tasks",
    "cost_free"
  ]
}
```

**Impact:**
- Workflow Designer will now suggest this free model first
- Agents will default to OLMo instead of Claude/GPT

**Example 2: You need on-premise for everything**

```json
{
  "id": "mistral-nemo:12b-instruct-2407-fp16",
  "recommended_for": [
    "general_purpose",        // ← Force all agents to consider this
    "workflow_design",        // ← Even for complex tasks
    "agent_coordination",     // ← Even for orchestration
    "onprem_deployment",
    "privacy_sensitive"
  ]
}
```

**Impact:**
- System prefers on-premise model by default
- Only falls back to cloud if on-prem doesn't support required features

**Example 3: Domain-specific tuning**

```json
{
  "models": [
    {
      "id": "custom-medical-gpt",
      "provider": "openai",
      "recommended_for": [
        "medical_diagnosis",     // ← Custom category
        "hipaa_compliant",       // ← Custom category
        "clinical_notes"         // ← Custom category
      ]
    }
  ]
}
```

---

### Practical Configuration Examples

#### Configuration 1: Budget-Conscious Setup

**Goal:** Minimize costs while maintaining functionality

```json
{
  "models": [
    {
      "id": "allenai/olmo-3.1-32b-think:free",
      "tier": "fast",
      "recommended_for": [
        "general_purpose",        // ← Use for most tasks
        "workflow_design",        // ← Even complex tasks
        "agent_coordination",
        "cost_free"
      ]
    },
    {
      "id": "claude-sonnet-4-5-20250929",
      "tier": "standard",
      "recommended_for": [
        "critical_decisions",     // ← Only for critical tasks
        "final_validation"        // ← Use sparingly
      ]
    }
  ]
}
```

**Workflow Designer Strategy:**
```python
def select_model(agent_role):
    if agent_role in ["final_validator", "critical_decision_maker"]:
        return "claude-sonnet-4-5-20250929"  # Premium model
    else:
        return "allenai/olmo-3.1-32b-think:free"  # Free model
```

---

#### Configuration 2: Privacy-First (On-Premise Only)

**Goal:** All data stays local, no cloud APIs

```json
{
  "models": [
    {
      "id": "mistral-nemo:12b-instruct-2407-fp16",
      "provider": "onprem",
      "recommended_for": [
        "general_purpose",
        "workflow_design",
        "agent_coordination",
        "data_analysis",
        "privacy_sensitive",
        "hipaa_compliant"
      ]
    }
  ]
}
```

**Validator Enforcement:**
```python
# In validator/sync_rules.py
def validate_privacy_requirements(workflow, agents):
    if workflow.metadata.get("requires_privacy"):
        for agent_id in workflow["agents"]:
            model = agents[agent_id]["llm"]["model"]
            model_meta = catalog.get_model(model)

            if "privacy_sensitive" not in model_meta.recommended_for:
                errors.append(
                    f"Workflow requires privacy but agent {agent_id} "
                    f"uses non-private model {model}"
                )
```

---

#### Configuration 3: Hybrid (Fast Local + Smart Cloud)

**Goal:** Use on-premise for simple tasks, cloud for complex ones

```json
{
  "models": [
    {
      "id": "mistral-nemo:12b-instruct-2407-fp16",
      "provider": "onprem",
      "tier": "fast",
      "recommended_for": [
        "fast_tasks",
        "data_preprocessing",
        "simple_classification"
      ]
    },
    {
      "id": "claude-sonnet-4-5-20250929",
      "provider": "anthropic",
      "tier": "standard",
      "recommended_for": [
        "complex_reasoning",
        "workflow_design",
        "agent_coordination"
      ]
    }
  ]
}
```

**Automatic Selection:**
```python
def select_model(agent_task_complexity):
    if agent_task_complexity <= 3:  # Simple (0-3)
        return "mistral-nemo:12b-instruct-2407-fp16"  # On-prem
    else:  # Complex (4-10)
        return "claude-sonnet-4-5-20250929"  # Cloud
```

---

### How to Change Recommendations (Step-by-Step)

**Scenario:** You want GPT-4 to be used for workflow design instead of Claude

**Before:**
```json
{
  "id": "claude-sonnet-4-5-20250929",
  "recommended_for": ["workflow_design", "agent_coordination"]
},
{
  "id": "gpt-4",
  "recommended_for": ["reasoning"]
}
```

**After:**
```json
{
  "id": "claude-sonnet-4-5-20250929",
  "recommended_for": ["agent_coordination"]  // ← Removed workflow_design
},
{
  "id": "gpt-4",
  "recommended_for": ["reasoning", "workflow_design"]  // ← Added
}
```

**Impact:**
- `WorkflowDesigner.__init__()` will now prefer GPT-4
- All agents with "workflow_design" task will get GPT-4
- No code changes needed—pure configuration

---

### API Endpoints for Model Selection

**GET /api/models/recommended**
```python
@router.get("/models/recommended")
def get_recommended_models(
    agent_role: str,
    required_features: List[str] = [],
    max_cost_per_million: float = None
):
    """
    Get recommended models for an agent

    Query params:
        agent_role: e.g., "security_analyst", "data_processor"
        required_features: e.g., ["tools", "vision"]
        max_cost_per_million: Budget constraint
    """
    models = llm_provider.list_available_models()

    # Filter by role recommendations
    role_keywords = ROLE_TO_KEYWORDS.get(agent_role, ["general_purpose"])
    candidates = [
        m for m in models
        if any(kw in m.recommended_for for kw in role_keywords)
    ]

    # Filter by required features
    if required_features:
        candidates = [
            m for m in candidates
            if all(
                getattr(m, f"supports_{feat}", False)
                for feat in required_features
            )
        ]

    # Filter by cost
    if max_cost_per_million:
        candidates = [
            m for m in candidates
            if m.cost_per_million_input_tokens <= max_cost_per_million
        ]

    # Sort by tier (premium → standard → fast)
    tier_priority = {"premium": 0, "standard": 1, "fast": 2}
    candidates.sort(key=lambda m: tier_priority.get(m.tier, 3))

    return candidates
```

---

### Quick Reference: Feature Flags

**Model Capabilities (Boolean Flags):**
```json
{
  "supports_tools": true,        // Can call functions/tools
  "supports_vision": true,       // Can process images
  "supports_streaming": true     // Can stream responses
}
```

**How to Add New Capability:**
1. Add boolean field to model in `llm_models.json`
2. Add check in validator if needed
3. Use in agent factory if runtime behavior changes

**Example: Adding "supports_json_mode"**
```json
{
  "id": "gpt-4",
  "supports_tools": true,
  "supports_vision": false,
  "supports_streaming": true,
  "supports_json_mode": true    // ← NEW
}
```

```python
# In agent factory
if agent_def.requires_structured_output:
    model_meta = llm_provider.get_model_metadata(agent_def.llm_config.model)
    if not model_meta.supports_json_mode:
        raise ValueError(f"Model doesn't support JSON mode")
```

---

## Summary

**The LLM Provider is the abstraction layer that makes multi-LLM, multi-provider agent systems possible.**

| Component | Responsibility |
|-----------|----------------|
| **llm_models.json** | Model catalog & provider registry |
| **LLMProvider** | Unified interface & model lifecycle |
| **Agent Factory** | Reads agent LLM config, verifies availability |
| **Workflow Designer** | Uses LLM to generate agent configurations |
| **Runtime** | Executes agents (LLM calls happen internally) |

**Key Insight:** Agents don't know or care which provider they use—they just execute with whatever LLM was configured in their definition.

---

**Architecture Status:** ✅ Production-ready, scalable, extensible



## where you need to add LLM

  You need to edit these files:

    1. .env file

    Location: C:\Users\Shashank Singh\Desktop\Phase 2 - ECHO\workflow-orchestrator\.env

    Add your OpenRouter API key:
    OPENROUTER_API_KEY=your_key_here
    LOG_LEVEL=INFO

    2. app/core/config.py

    Location: C:\Users\Shashank Singh\Desktop\Phase 2 - ECHO\workflow-orchestrator\app\core\config.py     

    Add OpenRouter configuration (around line 20):
    OPENROUTER_API_KEY: Optional[str] = None

    3. app/services/llm_provider.py (most important)

    Location: C:\Users\Shashank Singh\Desktop\Phase 2 - ECHO\workflow-orchestrator\app\services\llm_provider.py

    This is where the LLM client is created. You need to configure it to use OpenRouter's API endpoint.   


==========================================================================================================
## IMPORTANT

 Current State

  Your codebase currently uses ChatOpenAI (standard OpenAI) in multiple places for LLM operations.

  Your Requirement

  You want to add AzureChatOpenAI as a commented alternative alongside the existing ChatOpenAI implementations, so that when deploying to Azure cloud, you can:
  1. Comment out the ChatOpenAI import and initialization
  2. Uncomment the AzureChatOpenAI import and initialization

  Files Where Changes Are Needed

  I've identified 5 files where ChatOpenAI is actively used and needs the Azure alternative added:

  1. app/services/llm_provider.py (4 locations)

  - Line 12: Import statement
  - Line 313: OpenAI provider instantiation
  - Line 324: OpenRouter provider instantiation
  - Line 336: On-prem provider instantiation
  - Line 343: Default fallback instantiation

  2. app/workflow/designer.py (2 locations)

  - Line 9: Import statement
  - Line 52: Designer LLM initialization (OpenRouter)
  - Line 61: Fallback LLM initialization

  3. app/workflow/compiler.py (Multiple locations)

  - Line 7: Import statement
  - Lines 309, 313, 317, 324, 333, 343: Agent LLM initialization for different providers

  4. app/services/meta_prompt_generator.py (2 locations)

  - Line 8: Import statement
  - Line 30: Analyzer LLM initialization

  5. app/core/file_reader/summarizer.py (2 locations)

  - Line 3: Import statement
  - Line 25: Summarizer LLM initialization

  Pattern to Follow

  For each location, add the Azure alternative as a commented block above the current ChatOpenAI code:

  # For Azure deployment - uncomment this block
  # from langchain_openai import AzureChatOpenAI
  # self.llm = AzureChatOpenAI(
  #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
  #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
  #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
  #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
  #     temperature=0,
  #     streaming=stream,
  #     callbacks=[self.collector] if stream else None,
  # )

  # For local/standard OpenAI - comment this when deploying to Azure
  from langchain_openai import ChatOpenAI
  self.llm = ChatOpenAI(...)

  Notes I Found

  - Two files (csv_agent.py and csv_summarizer.py) already use AzureChatOpenAI, so they don't need changes
  - The pattern will need slight adjustments for each context (e.g., some don't have callbacks, some use different parameter names)

  ---
  Ready for implementation? Let me know when you want me to proceed with the code changes!
