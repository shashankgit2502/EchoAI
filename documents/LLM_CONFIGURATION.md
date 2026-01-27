# LLM Configuration Guide

## Where to Change Default LLM Provider

### 1. **Environment Variables (.env file)** - RECOMMENDED
```bash
# Change these to switch providers globally
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-key-here
OPENROUTER_MODEL=mistralai/devstral-2512:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

**To switch to OpenAI**:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o-mini
```

**To switch to Ollama**:
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=mistral-nemo
```

---

### 2. **Code Files** (if you need hardcoded defaults)

#### **A. CrewAI Adapter** (for workflow agent execution)
**File**: `echoAI/apps/workflow/crewai_adapter.py`
**Line**: ~389

```python
# DEFAULT: Use OpenRouter (change this line to modify default provider)
provider = llm_config.get("provider", os.getenv("LLM_PROVIDER", "openrouter"))
model = llm_config.get("model", os.getenv("OPENROUTER_MODEL", "mistralai/devstral-2512:free"))
```

**Change to OpenAI**:
```python
provider = llm_config.get("provider", "openai")
model = llm_config.get("model", "gpt-4o-mini")
```

---

#### **B. Workflow Designer** (for workflow generation LLM)
**File**: `echoAI/apps/workflow/designer/designer.py`
**Line**: ~166

```python
# DEFAULT LLM for agents - CHANGE HERE to modify default
default_llm = {
    "provider": os.getenv("LLM_PROVIDER", "openrouter"),
    "model": os.getenv("OPENROUTER_MODEL", "mistralai/devstral-2512:free"),
    "temperature": 0.3
}
```

**Change to OpenAI**:
```python
default_llm = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.3
}
```

---

#### **C. Agent Designer** (for agent creation LLM)
**File**: `echoAI/apps/agent/designer/agent_designer.py`
**Line**: ~85

Already reads from .env, but you can add fallback:
```python
if os.getenv("USE_OPENROUTER", "true").lower() == "true":
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    # ...
```

---

## Provider-Specific Configuration

### **OpenRouter**
```python
{
    "provider": "openrouter",
    "model": "mistralai/devstral-2512:free",  # Free tier
    # Other options:
    # "model": "google/gemini-2.0-flash-thinking-exp:free"
    # "model": "anthropic/claude-3-haiku"
    "temperature": 0.3
}
```

### **OpenAI**
```python
{
    "provider": "openai",
    "model": "gpt-4o-mini",
    # Other options: "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"
    "temperature": 0.3
}
```

### **Ollama (Local)**
```python
{
    "provider": "ollama",
    "model": "mistral-nemo",
    # Other options: "llama3", "codellama", "mixtral"
    "temperature": 0.3
}
```

### **Azure OpenAI**
```python
{
    "provider": "azure",
    "model": "gpt-4",  # Your deployment name
    "temperature": 0.3
}
```

---

## Per-Agent LLM Configuration

You can override the default for specific agents in the workflow JSON:

```json
{
  "agent_id": "agt_123",
  "name": "Code Analyzer",
  "llm": {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.1
  }
}
```

This allows mixing providers in the same workflow (e.g., cheap model for simple tasks, powerful for complex).

---

## Quick Fix for Current Issue

**Your OpenRouter API key is invalid.** Two options:

### Option 1: Get New OpenRouter Key
1. Go to https://openrouter.ai/keys
2. Sign up with new email
3. Create API key
4. Update `.env`:
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-YOUR-NEW-KEY
   ```
5. Restart server

### Option 2: Switch to Ollama (if available)
1. Update `.env`:
   ```bash
   LLM_PROVIDER=ollama
   USE_OLLAMA=true
   OLLAMA_BASE_URL=http://10.188.100.131:8004/v1
   OLLAMA_MODEL=mistral-nemo:12b-instruct-2407-fp16
   ```
2. Restart server

---

## Testing

After changing configuration:

```bash
cd echoAI
python test_api_key.py  # Test OpenRouter key
# or
python -c "from apps.workflow.crewai_adapter import CrewAIAdapter; print('OK')"  # Test imports
```

Then restart server:
```bash
uvicorn apps.gateway.main:app --reload
```

---

## Summary

**Priority Order for Changes**:
1. ✅ **Update .env file** (easiest, recommended)
2. Restart server
3. Only modify code files if you need hardcoded defaults

**Files to change if not using .env**:
- `echoAI/apps/workflow/crewai_adapter.py` (line ~389)
- `echoAI/apps/workflow/designer/designer.py` (line ~166)
- `echoAI/apps/agent/designer/agent_designer.py` (line ~85)


Latest: 

Two places to change:                                                                                        
  ┌──────────────────────────┬──────────────────────────────────────────────────────────────────────────────┐  
  │           File           │                                What to change                                │    ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤  
  │ echoAI/llm_provider.json │ Set "is_default": true on desired model (line 10 currently), false on others │    ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ echoAI/.env              │ Set USE_OPENROUTER=false and USE_<YOUR_PROVIDER>=true                        │  
  └──────────────────────────┴──────────────────────────────────────────────────────────────────────────────┘  
  If no .env exists, the fallback defaults are hardcoded in:
  ┌──────────────────────────┬──────────────────────────────────────────┐
  │           File           │                  Lines                   │
  ├──────────────────────────┼──────────────────────────────────────────┤
  │ echoAI/echolib/config.py │ Line 30: use_openrouter defaults to true │
  ├──────────────────────────┼──────────────────────────────────────────┤
  │ echoAI/echolib/config.py │ Line 71: fallback returns 'openrouter'   │
  └──────────────────────────┴──────────────────────────────────────────┘
  Simplest approach: Create/edit .env in echoAI/ folder:
  USE_OPENROUTER=false
  USE_OLLAMA=true   # or USE_AZURE=true or USE_OPENAI=true

  Then in llm_provider.json, flip the is_default flag. That's it

===========

## llm_provider.json 
    is loaded in:

  echoAI/apps/agent/designer/agent_designer.py → lines 236-241

  self._llm_providers = self._load_llm_providers()

  def _load_llm_providers(self) -> Dict[str, Any]:
      """Load LLM provider configurations from llm_provider.json."""

    Purpose: Registry of available LLM models for agent assignment. 
    - Lists all configured models (OpenRouter, Ollama, OpenAI, etc.)
    - Each model has: id, name, provider, base_url, api_key, model_name                                          
    - is_default: true marks which model agents use when none specified
    - Loaded by agent_designer.py to let users pick LLM per agent


Done. Added comments at:

  - Lines 29-31: Commented Ollama implementation to swap with current OpenRouter defaults
  - Line 74: Comment showing fallback change for Ollama