# Chat Feature Addition Plan - Workflow Chat + Draft Loader

## Executive Summary

This document outlines the implementation plan for adding two major features to EchoAI:
1. **Draft Workflow Chat Execution** - Enable chat-based interaction with draft workflows
2. **Enhanced Load Button** - Display list of draft workflows for easy selection

---

## Frontend Analysis (workflow_builder_ide.html)

### Button Implementations

#### 1. Save Button (Line 340)
```html
<button @click="saveWorkflow('draft')" class="...">💾 Save</button>
```

**Function: `saveWorkflow(saveAs = 'draft')` (Lines 3303-3345)**
- Validates canvas has nodes
- Calls `workflowAPI.saveCanvas()` with `save_as: 'draft'`
- On success: Sets `workflowSaved = true`, `workflowState = 'saved'`, updates `currentWorkflowId`
- Storage: Saves to `/apps/workflow/storage/workflows/draft/` folder

#### 2. Run Workflow Button (Lines 347-355)
```html
<button @click="runWorkflow" :disabled="isExecuting" class="...">▶ Run Workflow</button>
```

**Function: `runWorkflow()` (Lines 3373-3400)**
- Forces save as `'temp'` before execution (line 3383, 3389)
- Opens input modal for user payload
- Calls `executeWithInput()` which calls `workflowAPI.execute(workflowId, 'test', null, inputPayload)`
- Storage: Saves to `/apps/workflow/storage/workflows/temp/` before execution

#### 3. Load Button (Lines 334-336)
```html
<button @click="loadWorkflowDialog" class="...">📂 Load</button>
```

**Function: `loadWorkflowDialog()` (Lines 3347-3371)**
- Uses `prompt('Enter workflow ID to load')` - manual ID entry only
- Calls `workflowAPI.loadWorkflow(workflowId)` → `GET /workflows/{id}`
- Converts to canvas format via `workflowAPI.backendToCanvas()`
- Populates canvas with loaded workflow

### Chat Implementation

#### State Variables (Lines 2051-2054)
```javascript
const chatMessages = ref([]);       // Message history
const chatInput = ref('');          // Current input
const chatSessionId = ref(null);    // Session ID for chat API
const isSendingMessage = ref(false); // Loading state
```

#### Chat Panel (Lines 424-493)
- Located in left pane under "Chat" tab
- Shows message history (`chatMessages`)
- Input textarea with Send button
- **Current behavior**: Forces save as `temp` before chat (lines 3484, 3494)

#### sendChatMessage() Function (Lines 3472-3566)
```javascript
const sendChatMessage = async () => {
    // CURRENT: Forces temp save before chat
    if (!workflowSaved.value || !currentWorkflowId.value) {
        await saveWorkflow('temp');  // Line 3484
    } else {
        await saveWorkflow('temp');  // Line 3494
    }

    // Start chat session with mode='temp'
    if (!chatSessionId.value) {
        const startResponse = await fetch(`${API_BASE_URL}/workflows/chat/start`, {
            body: JSON.stringify({
                workflow_id: currentWorkflowId.value,
                workflow_mode: 'temp',  // HARDCODED to 'temp'
                context: {...}
            })
        });
    }

    // Send message
    await fetch(`${API_BASE_URL}/workflows/chat/send`, {...});
};
```

### Key State Variables (Lines 2047-2062)
```javascript
const workflowSaved = ref(false);      // Whether workflow is saved
const currentWorkflowId = ref(null);    // Current workflow ID
const workflowModified = ref(false);    // Whether modified since save
const workflowState = ref('empty');     // 'empty' | 'unsaved' | 'saved'
```

### workflowAPI Object (Lines 1729-1860)
```javascript
const workflowAPI = {
    saveCanvas(nodes, connections, name, saveAs, model),  // POST /workflows/canvas/save
    loadWorkflow(workflowId),                              // GET /workflows/{id}
    backendToCanvas(workflow, agents),                     // POST /workflows/backend/to-canvas
    execute(workflowId, mode, version, inputPayload),      // POST /workflows/execute
    // ... other methods
};
```

---

## Gap Analysis

### What Needs to Change

| Component | Current Behavior | Required Change |
|-----------|-----------------|-----------------|
| **sendChatMessage()** | Always saves as `temp` | Use `draft` mode if already saved as draft |
| **Chat session start** | `workflow_mode: 'temp'` hardcoded | Use `draft` if workflow state is draft |
| **loadWorkflowDialog()** | Manual `prompt()` only | Add modal with draft list option |
| **workflowAPI** | No `listDrafts()` method | Add `listDrafts()` API call |

### Backend Gaps
| Component | Gap | Solution |
|-----------|-----|----------|
| Chat Session | No "draft" mode support | Add "draft" to `workflow_mode` enum |
| Routes | No `/draft/list` endpoint | Add new endpoint |
| Executor | Chat with draft not explicit | Ensure executor loads from draft state |

---

## Detailed Implementation Plan

### Backend Changes (via backend-python-dev agent)

#### 1. Storage Layer - `filesystem.py`
Add method to list all draft workflows:
```python
def list_draft_workflows(self) -> List[Dict]:
    """Return list of all workflows in draft folder with metadata."""
    draft_dir = self._get_storage_path("draft")
    workflows = []
    for file in draft_dir.glob("*.draft.json"):
        workflow = self._load_json(file)
        workflows.append({
            "workflow_id": workflow.get("workflow_id"),
            "name": workflow.get("name"),
            "execution_model": workflow.get("execution_model"),
            "agent_count": len(workflow.get("agents", [])),
            "created_at": workflow.get("metadata", {}).get("created_at")
        })
    return sorted(workflows, key=lambda x: x.get("created_at", ""), reverse=True)
```

#### 2. Chat Session - `chat_session.py`
Update type hint to allow "draft" mode:
```python
# From:
workflow_mode: Literal["test", "final"]
# To:
workflow_mode: Literal["draft", "test", "final"]
```

#### 3. Routes - `routes.py`
Add new endpoint and update chat routes:
```python
@router.get('/draft/list')
async def list_draft_workflows():
    """List all workflows in draft folder."""
    storage = container.resolve('workflow.storage')
    workflows = storage.list_draft_workflows()
    return {"workflows": workflows, "total": len(workflows)}

# Update /workflows/chat/start to accept "draft" mode
# Update /workflows/chat/send to handle draft execution
```

#### 4. Executor - `executor.py`
Ensure draft mode execution:
```python
def execute_workflow(workflow_id, execution_mode, ...):
    # execution_mode can be "draft", "test", or "final"
    if execution_mode == "draft":
        workflow = storage.load_workflow(workflow_id, state="draft")
    elif execution_mode == "test":
        workflow = storage.load_workflow(workflow_id, state="temp")
    else:
        workflow = storage.load_workflow(workflow_id, state="final")
    # ... rest of execution
```

### Frontend Changes

#### 1. Add Draft List API Method
```javascript
// Add to workflowAPI object
async listDrafts() {
    const response = await fetch(`${API_BASE_URL}/workflows/draft/list`);
    if (!response.ok) throw new Error('Failed to load draft list');
    return await response.json();
}
```

#### 2. Add State for Load Modal
```javascript
const showLoadModal = ref(false);
const draftWorkflows = ref([]);
const isLoadingDrafts = ref(false);
const manualWorkflowId = ref('');
```

#### 3. Update loadWorkflowDialog()
```javascript
const loadWorkflowDialog = async () => {
    showLoadModal.value = true;
    isLoadingDrafts.value = true;
    try {
        const result = await workflowAPI.listDrafts();
        draftWorkflows.value = result.workflows || [];
    } catch (error) {
        console.error('Failed to load drafts:', error);
    } finally {
        isLoadingDrafts.value = false;
    }
};

const loadSelectedDraft = async (workflowId) => {
    showLoadModal.value = false;
    // Reuse existing load logic
    try {
        const result = await workflowAPI.loadWorkflow(workflowId);
        // ... same as current loadWorkflowDialog logic
    } catch (error) {
        alert(`Failed to load workflow: ${error.message}`);
    }
};

const loadManualId = async () => {
    if (!manualWorkflowId.value.trim()) return;
    await loadSelectedDraft(manualWorkflowId.value.trim());
    manualWorkflowId.value = '';
};
```

#### 4. Add Load Modal HTML
```html
<!-- Load Workflow Modal -->
<div v-if="showLoadModal" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
    <div class="bg-white rounded-xl shadow-2xl w-[500px] max-h-[600px] flex flex-col">
        <div class="p-4 border-b flex justify-between items-center">
            <h3 class="text-sm font-bold text-slate-700">Load Workflow</h3>
            <button @click="showLoadModal = false" class="text-slate-400 hover:text-slate-600">×</button>
        </div>

        <!-- Manual ID Entry -->
        <div class="p-4 border-b">
            <div class="text-xs font-semibold text-slate-600 mb-2">Load by ID</div>
            <div class="flex gap-2">
                <input v-model="manualWorkflowId" placeholder="Enter workflow ID..." class="flex-1 px-3 py-2 text-xs border rounded-lg">
                <button @click="loadManualId" class="px-4 py-2 text-xs font-semibold text-white bg-indigo-600 rounded-lg">Load</button>
            </div>
        </div>

        <!-- Draft List -->
        <div class="flex-1 overflow-y-auto p-4">
            <div class="text-xs font-semibold text-slate-600 mb-2">Saved Drafts</div>
            <div v-if="isLoadingDrafts" class="text-center py-4">Loading...</div>
            <div v-else-if="draftWorkflows.length === 0" class="text-center py-4 text-slate-400 text-xs">No drafts found</div>
            <div v-else class="space-y-2">
                <div v-for="wf in draftWorkflows" :key="wf.workflow_id"
                     @click="loadSelectedDraft(wf.workflow_id)"
                     class="p-3 border rounded-lg hover:border-indigo-300 hover:bg-indigo-50 cursor-pointer">
                    <div class="text-xs font-semibold text-slate-700">{{ wf.name || 'Untitled' }}</div>
                    <div class="text-[10px] text-slate-400">{{ wf.workflow_id }} • {{ wf.agent_count }} agents • {{ wf.execution_model }}</div>
                </div>
            </div>
        </div>
    </div>
</div>
```

#### 5. Update sendChatMessage() for Draft Support
```javascript
const sendChatMessage = async () => {
    const message = chatInput.value.trim();
    if (!message || isSendingMessage.value) return;

    if (activeNodes.value.length === 0) {
        alert('No workflow on canvas. Please create a workflow first.');
        return;
    }

    // NEW: Check if workflow is saved as draft (don't force temp save)
    let chatMode = 'draft';  // Default to draft

    if (!workflowSaved.value || !currentWorkflowId.value) {
        const shouldSave = confirm('Workflow must be saved before chatting. Save now?');
        if (shouldSave) {
            await saveWorkflow('draft');  // Save as draft (NOT temp)
            if (!currentWorkflowId.value) {
                alert('Failed to save workflow. Please try again.');
                return;
            }
        } else {
            return;
        }
    }
    // NOTE: Do NOT re-save as temp - keep as draft

    // ... rest of chat logic with workflow_mode: chatMode
    if (!chatSessionId.value) {
        const startResponse = await fetch(`${API_BASE_URL}/workflows/chat/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow_id: currentWorkflowId.value,
                workflow_mode: chatMode,  // Use 'draft' instead of hardcoded 'temp'
                context: { workflow_name: workflowName.value }
            })
        });
        // ...
    }
};
```

---

## Non-Breaking Guarantees

| Existing Feature | Protection |
|------------------|------------|
| Save button | Unchanged - still saves to draft |
| Run Workflow | Unchanged - still saves to temp then executes |
| Manual ID load | Still available in modal |
| Execute API | No changes to signature |
| Temp/Final workflows | Chat still works with test/final modes |

---

## Implementation Order

1. **Backend Phase 1** - Storage layer `list_draft_workflows()`
2. **Backend Phase 2** - Update chat session to support "draft" mode
3. **Backend Phase 3** - Add `/draft/list` endpoint
4. **Backend Phase 4** - Ensure executor handles draft mode in chat context
5. **Frontend Phase 1** - Add load modal and draft list API
6. **Frontend Phase 2** - Update sendChatMessage for draft support
7. **Testing** - Full regression + new feature validation

---

## Files to Modify

### Backend
| File | Changes |
|------|---------|
| `apps/workflow/storage/filesystem.py` | Add `list_draft_workflows()` |
| `apps/workflow/runtime/chat_session.py` | Add "draft" to mode enum |
| `apps/workflow/runtime/executor.py` | Handle draft mode |
| `apps/workflow/routes.py` | Add `/draft/list`, update chat routes |

### Frontend
| File | Changes |
|------|---------|
| `echoAI/workflow_builder_ide.html` | Add modal, update chat logic, add API method |

---

## Next Steps

**Awaiting user approval to proceed with implementation.**

1. First: Backend implementation via backend-python-dev agent
2. Then: Frontend implementation directly
3. Finally: Integration testing
