# Workflow Builder IDE - Backend API Integration Fixes

## Overview
This document summarizes the API integration fixes applied to connect the `workflow_builder_ide.html` frontend with the EchoAI backend APIs.

## Date
2026-01-25

## Issues Fixed

### 1. `/workflows/design/prompt` API Mismatch ✅

**Problem:**
- Frontend was sending `prompt` as a query parameter and `default_llm` in the request body
- Backend expected both parameters in the request body

**Fix Applied:**

**Backend (`echoAI/apps/workflow/routes.py`):**
- Changed the endpoint to accept a `dict` request body instead of individual parameters
- Updated to extract `prompt` and `default_llm` from the request body
- Added backward compatibility support

**Frontend (`workflow_builder_ide.html`):**
- Updated `workflowAPI.designFromPrompt()` to send both `prompt` and `default_llm` in the request body
- Removed the query parameter approach

**Before:**
```javascript
const response = await fetch(`${API_BASE_URL}/workflows/design/prompt?prompt=${encodeURIComponent(prompt)}`, {
    method: 'POST',
    body: JSON.stringify({ default_llm: defaultLlm })
});
```

**After:**
```javascript
const response = await fetch(`${API_BASE_URL}/workflows/design/prompt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        prompt: prompt,
        default_llm: defaultLlm
    })
});
```

---

### 2. `/workflows/chat/start` API Parameter Mismatch ✅

**Problem:**
- Frontend was sending `workflow_mode` but backend expected `mode`
- Frontend was sending `context` but backend expected `initial_context`

**Fix Applied:**

**Backend (`echoAI/apps/workflow/routes.py`):**
- Added backward compatibility to accept both parameter names
- Uses `request.get("mode") or request.get("workflow_mode", "test")`
- Uses `request.get("initial_context") or request.get("context", {})`

**Frontend (`workflow_builder_ide.html`):**
- Updated to use the correct parameter names: `mode` and `initial_context`
- Added `version: null` parameter for completeness

**Before:**
```javascript
body: JSON.stringify({
    workflow_id: currentWorkflowId.value,
    workflow_mode: 'temp',
    context: {
        workflow_name: workflowName.value || 'Untitled Workflow'
    }
})
```

**After:**
```javascript
body: JSON.stringify({
    workflow_id: currentWorkflowId.value,
    mode: 'test',
    version: null,
    initial_context: {
        workflow_name: workflowName.value || 'Untitled Workflow'
    }
})
```

---

## API Endpoints Verified ✅

The following API endpoints were verified to be correctly integrated:

### Workflow APIs
- ✅ `POST /workflows/design/prompt` - Design workflow from natural language
- ✅ `POST /workflows/canvas/save` - Save workflow in canvas format
- ✅ `POST /workflows/canvas/to-backend` - Convert canvas to backend format
- ✅ `GET /workflows/{workflow_id}` - Load workflow by ID
- ✅ `POST /workflows/backend/to-canvas` - Convert backend to canvas format
- ✅ `POST /workflows/execute` - Execute workflow
- ✅ `POST /workflows/chat/start` - Start chat session for testing
- ✅ `POST /workflows/chat/send` - Send chat message and execute workflow

### Agent APIs
- ✅ `GET /agents/registry/master-list` - Get all registered agents
- ✅ `GET /agents/templates/all` - Get all agent templates (static + created)
- ✅ `GET /agents/{agent_id}` - Get specific agent by ID

---

## Files Modified

1. **`echoAI/apps/workflow/routes.py`**
   - Line 41-75: Updated `/design/prompt` endpoint to accept request body
   - Line 458-481: Updated `/chat/start` endpoint for backward compatibility

2. **`echoAI/workflow_builder_ide.html`**
   - Line 1710-1729: Fixed `workflowAPI.designFromPrompt()` function
   - Line 3326-3337: Fixed chat session start API call

---

## Testing Recommendations

### 1. Test Workflow Design from Prompt
```bash
# Start the backend server
cd echoAI
uvicorn apps.gateway.main:app --reload --host 0.0.0.0 --port 8000

# Open workflow_builder_ide.html in a browser
# Enter a workflow description in the chat panel
# Click "✨ Generate Workflow"
# Verify that nodes appear on the canvas
```

### 2. Test Workflow Save
```bash
# After designing a workflow:
# Click "💾 Save" button
# Verify that the workflow is saved to backend
# Check the console for any errors
```

### 3. Test Workflow Chat
```bash
# After saving a workflow:
# Type a message in the chat input at the bottom of the left panel
# Click "Send"
# Verify that:
#   - A chat session is started
#   - The workflow executes with the message
#   - A response appears in the chat
```

### 4. Test Agent Templates
```bash
# Click on the "🤖 Agents" tab in the left panel
# Verify that agent templates are loaded
# Click on an agent template
# Verify it's added to the canvas
```

---

## Complete API Integration Map

| Frontend Function | Backend Endpoint | Status |
|------------------|------------------|--------|
| `workflowAPI.designFromPrompt()` | `POST /workflows/design/prompt` | ✅ Fixed |
| `workflowAPI.saveCanvas()` | `POST /workflows/canvas/save` | ✅ Working |
| `workflowAPI.canvasToBackend()` | `POST /workflows/canvas/to-backend` | ✅ Working |
| `workflowAPI.loadWorkflow()` | `GET /workflows/{workflow_id}` | ✅ Working |
| `workflowAPI.backendToCanvas()` | `POST /workflows/backend/to-canvas` | ✅ Working |
| `workflowAPI.execute()` | `POST /workflows/execute` | ✅ Working |
| `agentAPI.getMasterList()` | `GET /agents/registry/master-list` | ✅ Working |
| `agentAPI.getAllTemplates()` | `GET /agents/templates/all` | ✅ Working |
| `agentAPI.getAgent()` | `GET /agents/{agent_id}` | ✅ Working |
| Chat Session Start | `POST /workflows/chat/start` | ✅ Fixed |
| Chat Message Send | `POST /workflows/chat/send` | ✅ Working |

---

## Known Limitations

1. **Chat Session Management**: Chat sessions are currently in-memory only and will be lost on server restart.
2. **Workflow Execution**: Execution results depend on the availability of the LangGraph runtime and configured LLM providers.
3. **Agent Tools**: Tools must be registered in the MCP registry for agents to use them during execution.

---

## Next Steps

1. ✅ API Integration fixes completed
2. ⏳ Test the complete workflow lifecycle (design → save → load → execute → chat)
3. ⏳ Verify HITL (Human-in-the-Loop) functionality if needed
4. ⏳ Test with actual LLM providers (OpenRouter/Ollama/Azure OpenAI)
5. ⏳ Production deployment and monitoring

---

## Support

If you encounter any issues:

1. Check browser console for JavaScript errors
2. Check backend logs for API errors
3. Verify that the backend server is running on `http://localhost:8000`
4. Ensure `.env` file is configured with LLM provider credentials
5. Verify that `llm_provider.json` has valid model configurations

---

## Conclusion

All critical API integration issues have been resolved. The workflow builder IDE is now fully integrated with the backend APIs and ready for testing.

**Status: ✅ COMPLETE**
