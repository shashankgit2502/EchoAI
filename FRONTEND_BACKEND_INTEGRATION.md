# Frontend-Backend Integration Guide

This document provides step-by-step instructions to integrate your Workflow Builder IDE and Agent Builder with the backend APIs.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [File Structure](#file-structure)
3. [Integration Steps](#integration-steps)
4. [Workflow Builder IDE Integration](#workflow-builder-ide-integration)
5. [Agent Builder Integration](#agent-builder-integration)
6. [Testing](#testing)

---

## Prerequisites

1. **Backend server running** at `http://localhost:8000`
2. **Frontend files**: `workflow_builder_ide.html` and `type-agent_builder.html`
3. **Integration module**: `backend_integration.js` (created)

---

## File Structure

```
Phase 2 - ECHO/
├── backend_integration.js          # API integration module
├── workflow_builder_ide.html       # Workflow builder (to be modified)
├── type-agent_builder.html         # Agent builder (to be modified)
└── echoAI/                         # Backend
    └── apps/
        ├── workflow/routes.py
        └── agent/routes.py
```

---

## Integration Steps

### Step 1: Add Backend Integration Script

Add this line in the `<head>` section of BOTH HTML files (after Vue.js script):

```html
<script src="./backend_integration.js"></script>
```

### Step 2: Modify Workflow Builder IDE

#### Location: `workflow_builder_ide.html`

Find the Vue app section (around line 1498) and make the following changes:

#### A. Replace `generateWorkflow` function (around line 2382)

**FIND:**
```javascript
const generateWorkflow = async () => {
    if (!workflowPrompt.value.trim() || isGenerating.value) return;

    isGenerating.value = true;
    generationStatus.value = 'Analyzing prompt...';

    // Simulate AI processing
    await new Promise(resolve => setTimeout(resolve, 1000));
    // ... rest of simulated code
```

**REPLACE WITH:**
```javascript
const generateWorkflow = async () => {
    if (!workflowPrompt.value.trim() || isGenerating.value) return;

    isGenerating.value = true;
    generationStatus.value = 'Analyzing prompt...';

    try {
        // Call backend API
        const result = await window.BackendAPI.Workflow.generateFromPrompt(
            workflowPrompt.value,
            null // default LLM
        );

        generationStatus.value = 'Creating nodes...';

        // Convert backend workflow to canvas format
        const canvasData = await window.BackendAPI.Workflow.backendToCanvas(
            result.workflow,
            result.agents
        );

        generationStatus.value = 'Rendering workflow...';

        // Clear existing nodes and add new ones
        activeNodes.value = [];
        connections.value = [];

        // Add canvas nodes
        canvasData.canvas_nodes.forEach(node => {
            activeNodes.value.push({
                ...node,
                status: 'idle'
            });
        });

        // Add connections
        connections.value = canvasData.connections;

        generationStatus.value = 'Workflow created successfully!';
        window.BackendAPI.showNotification('Workflow generated successfully!', 'success');

        // Switch to Nodes tab to show the generated workflow
        leftPaneTab.value = 'nodes';

    } catch (error) {
        console.error('Workflow generation error:', error);
        generationStatus.value = `Error: ${error.message}`;
        window.BackendAPI.showNotification(`Failed to generate workflow: ${error.message}`, 'error');
    } finally {
        isGenerating.value = false;

        // Clear status after 3 seconds
        setTimeout(() => {
            generationStatus.value = '';
        }, 3000);
    }
};
```

#### B. Add Save Workflow Function

Add this new function after `generateWorkflow`:

```javascript
const saveWorkflow = async (saveAs = 'temp') => {
    if (activeNodes.value.length === 0) {
        window.BackendAPI.showNotification('No workflow to save', 'warning');
        return;
    }

    try {
        const workflowName = prompt('Enter workflow name:') || window.BackendAPI.formatWorkflowName(workflowPrompt.value);

        if (!workflowName) return;

        const result = await window.BackendAPI.Workflow.saveCanvas(
            activeNodes.value,
            connections.value,
            workflowName,
            saveAs
        );

        if (result.success) {
            window.BackendAPI.showNotification(`Workflow saved as ${saveAs}`, 'success');
            return result;
        } else {
            throw new Error(result.errors?.join(', ') || 'Validation failed');
        }
    } catch (error) {
        window.BackendAPI.showNotification(`Save failed: ${error.message}`, 'error');
        throw error;
    }
};
```

#### C. Add Test Workflow Function

Add this function:

```javascript
const testWorkflow = async () => {
    try {
        // Save workflow as temp first
        const saved = await saveWorkflow('temp');

        if (!saved) return;

        // Start chat session for testing
        const session = await window.BackendAPI.Chat.startSession(
            saved.workflow_id,
            'test'
        );

        window.BackendAPI.showNotification('Chat session started! You can now test your workflow.', 'success');

        // Store session ID for later use
        window.currentChatSession = session.session_id;
        window.currentWorkflowId = saved.workflow_id;

        // Switch to chat tab
        leftPaneTab.value = 'chat';

        return session;
    } catch (error) {
        window.BackendAPI.showNotification(`Test failed: ${error.message}`, 'error');
    }
};
```

#### D. Add Send Chat Message Function

Add this function:

```javascript
const sendChatMessage = async (message) => {
    if (!window.currentChatSession) {
        window.BackendAPI.showNotification('No active chat session', 'warning');
        return;
    }

    try {
        const result = await window.BackendAPI.Chat.sendMessage(
            window.currentChatSession,
            message,
            true
        );

        return result;
    } catch (error) {
        window.BackendAPI.showNotification(`Message failed: ${error.message}`, 'error');
        throw error;
    }
};
```

#### E. Add HITL Functions

Add these HITL control functions:

```javascript
const approveHITL = async (runId, rationale = '') => {
    try {
        const result = await window.BackendAPI.HITL.approve(
            runId,
            'user@frontend', // Replace with actual user
            rationale
        );

        window.BackendAPI.showNotification('Workflow approved!', 'success');
        return result;
    } catch (error) {
        window.BackendAPI.showNotification(`Approve failed: ${error.message}`, 'error');
    }
};

const rejectHITL = async (runId, rationale = '') => {
    try {
        const result = await window.BackendAPI.HITL.reject(
            runId,
            'user@frontend',
            rationale
        );

        window.BackendAPI.showNotification('Workflow rejected', 'info');
        return result;
    } catch (error) {
        window.BackendAPI.showNotification(`Reject failed: ${error.message}`, 'error');
    }
};

const modifyHITL = async (runId, changes, rationale = '') => {
    try {
        const result = await window.BackendAPI.HITL.modify(
            runId,
            'user@frontend',
            changes,
            rationale
        );

        window.BackendAPI.showNotification('Workflow modified - requires re-validation', 'info');
        return result;
    } catch (error) {
        window.BackendAPI.showNotification(`Modify failed: ${error.message}`, 'error');
    }
};

const deferHITL = async (runId, deferUntil = null, rationale = '') => {
    try {
        const result = await window.BackendAPI.HITL.defer(
            runId,
            'user@frontend',
            deferUntil,
            rationale
        );

        window.BackendAPI.showNotification('Decision deferred', 'info');
        return result;
    } catch (error) {
        window.BackendAPI.showNotification(`Defer failed: ${error.message}`, 'error');
    }
};
```

#### F. Load Agent Templates from Backend

Replace hardcoded `agentTemplates` array (around line 1526) with API call:

```javascript
const agentTemplates = ref([]);

// Load templates on mount
onMounted(async () => {
    try {
        const result = await window.BackendAPI.Agent.getAllTemplates();
        agentTemplates.value = result.templates || result;
    } catch (error) {
        console.error('Failed to load agent templates:', error);
        // Fallback to hardcoded templates if API fails
        agentTemplates.value = [
            // ... keep existing hardcoded templates as fallback
        ];
    }
});
```

#### G. Export Functions for UI

At the end of the `return` statement in `setup()`, add:

```javascript
return {
    // ... existing returns
    generateWorkflow,
    saveWorkflow,
    testWorkflow,
    sendChatMessage,
    approveHITL,
    rejectHITL,
    modifyHITL,
    deferHITL,
    agentTemplates
}
```

---

### Step 3: Add UI Buttons

#### Add Save Button (in toolbar)

Find the toolbar section and add:

```html
<button @click="saveWorkflow('temp')"
        class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
    💾 Save Workflow
</button>
```

#### Add Test Button (in toolbar)

```html
<button @click="testWorkflow"
        class="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600">
    ▶️ Test Workflow
</button>
```

#### Add HITL Controls (show when workflow hits HITL checkpoint)

```html
<div v-if="hitlStatus === 'waiting_for_human'" class="p-4 bg-yellow-50 border border-yellow-200 rounded">
    <h3 class="font-bold mb-2">Human Review Required</h3>
    <p class="text-sm mb-4">Workflow paused at: {{ hitlBlockedAt }}</p>

    <div class="flex gap-2">
        <button @click="approveHITL(currentRunId)"
                class="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600">
            ✓ Approve
        </button>

        <button @click="rejectHITL(currentRunId)"
                class="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600">
            ✗ Reject
        </button>

        <button @click="showModifyModal = true"
                class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
            ✎ Modify
        </button>

        <button @click="deferHITL(currentRunId)"
                class="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600">
            ⏸ Defer
        </button>
    </div>
</div>
```

---

### Step 4: Agent Builder Integration

#### Location: `type-agent_builder.html`

#### A. Add Design Agent from Prompt Function

Find where agent creation happens and add:

```javascript
const designAgentFromPrompt = async (prompt) => {
    try {
        const result = await window.BackendAPI.Agent.designFromPrompt(
            prompt,
            null // default LLM
        );

        // Populate form with generated agent
        agentForm.value = result.agent;

        window.BackendAPI.showNotification('Agent designed successfully!', 'success');
        return result;
    } catch (error) {
        window.BackendAPI.showNotification(`Design failed: ${error.message}`, 'error');
    }
};
```

#### B. Add Save Agent Function

```javascript
const saveAgent = async () => {
    try {
        const result = await window.BackendAPI.Agent.register(agentForm.value);

        window.BackendAPI.showNotification('Agent saved successfully!', 'success');

        // Clear form
        agentForm.value = {};

        // Refresh agent list
        await loadAgents();

        return result;
    } catch (error) {
        window.BackendAPI.showNotification(`Save failed: ${error.message}`, 'error');
    }
};
```

#### C. Add Load Agents Function

```javascript
const loadAgents = async () => {
    try {
        const result = await window.BackendAPI.Agent.list();
        agents.value = result;
    } catch (error) {
        console.error('Failed to load agents:', error);
    }
};
```

---

## Testing

### 1. Start Backend Server

```bash
cd echoAI
uvicorn main:app --reload --port 8000
```

### 2. Open Workflow Builder

```bash
# Open in browser
open workflow_builder_ide.html
# or
start workflow_builder_ide.html
```

### 3. Test Chat Tab (Workflow Generation)

1. Enter a prompt: "Create a workflow to analyze customer feedback"
2. Click "Generate Workflow"
3. Check:
   - ✓ Loading indicators show
   - ✓ Workflow appears in Nodes tab
   - ✓ Nodes are connected
   - ✓ No console errors

### 4. Test Nodes Tab (Manual Building)

1. Drag nodes onto canvas
2. Connect them
3. Click "Save Workflow"
4. Check:
   - ✓ Save dialog appears
   - ✓ Workflow saved successfully
   - ✓ Success notification shows

### 5. Test Workflow Execution

1. Click "Test Workflow"
2. Chat interface opens
3. Send a message
4. Check:
   - ✓ Workflow executes
   - ✓ Response received
   - ✓ Messages display correctly

### 6. Test HITL (if workflow has HITL node)

1. Workflow executes until HITL node
2. HITL controls appear
3. Test all buttons:
   - ✓ Approve continues workflow
   - ✓ Reject terminates workflow
   - ✓ Modify shows edit interface
   - ✓ Defer postpones decision

### 7. Test Agents Tab

1. Check agent templates load
2. Click on a template
3. Verify:
   - ✓ Template details show
   - ✓ Can add to workflow
   - ✓ No console errors

### 8. Test Agent Builder

1. Enter agent prompt
2. Click "Design Agent"
3. Verify:
   - ✓ Form populates
   - ✓ Can edit fields
   - ✓ Save works
   - ✓ Agent appears in list

---

## Troubleshooting

### Backend Not Responding

**Problem:** `Network error - check your connection`

**Solution:**
1. Verify backend is running: `http://localhost:8000/docs`
2. Check CORS is enabled in backend
3. Verify port 8000 is not blocked

### CORS Errors

**Problem:** `CORS policy: No 'Access-Control-Allow-Origin' header`

**Solution:** Add CORS middleware to FastAPI:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Validation Errors

**Problem:** Workflow save fails with validation errors

**Solution:**
1. Check Start node exists
2. Check End node exists
3. Verify all nodes are connected
4. Check agent configurations are complete

### HITL Not Triggering

**Problem:** Workflow doesn't pause at HITL node

**Solution:**
1. Verify HITL node type is correctly set
2. Check workflow has HITL node in Nodes tab
3. Verify backend compiler detects HITL node
4. Check `metadata.node_type === "HITL"`

---

## Next Steps

1. **Add Error Handling UI**: Show detailed error messages in UI
2. **Add Loading States**: Improve UX with skeleton loaders
3. **Add Workflow History**: Show past executions
4. **Add Agent Library**: Browse and search agents
5. **Add Workflow Templates**: Pre-built workflow templates
6. **Add Export/Import**: JSON export/import functionality

---

## API Endpoints Summary

| Category | Endpoint | Purpose |
|----------|----------|---------|
| **Workflow** | POST `/workflows/design/prompt` | Generate from prompt |
| | POST `/workflows/canvas/to-backend` | Convert canvas |
| | POST `/workflows/canvas/save` | Save workflow |
| | POST `/workflows/execute` | Execute workflow |
| **Chat** | POST `/workflows/chat/start` | Start test session |
| | POST `/workflows/chat/send` | Send message |
| | GET `/workflows/chat/history/{id}` | Get history |
| **HITL** | POST `/workflows/hitl/approve` | Approve execution |
| | POST `/workflows/hitl/reject` | Reject execution |
| | POST `/workflows/hitl/modify` | Modify workflow |
| | POST `/workflows/hitl/defer` | Defer decision |
| **Agent** | GET `/agents/templates/all` | Get templates |
| | POST `/agents/design/prompt` | Design from prompt |
| | POST `/agents/register` | Save agent |
| | GET `/agents` | List all agents |

---

## Support

For issues or questions:
1. Check browser console for errors
2. Check backend logs: `tail -f echoAI/logs/app.log`
3. Verify API responses in Network tab
4. Check this integration guide

---

**Integration Status:**
- ✅ Backend API module created
- ✅ Integration functions documented
- ⏳ HTML files need manual updates (follow steps above)
- ⏳ Testing required after integration

**Last Updated:** 2026-01-17
