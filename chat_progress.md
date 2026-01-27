# Chat Feature Progress Tracker

## Project: Workflow Chat + Draft Loader Feature Addition

**Started:** 2026-01-27
**Status:** Planning Complete - Awaiting Approval

---

## Phase Overview

| Phase | Description | Status |
|-------|-------------|--------|
| 1. Analysis & Planning | Understand codebase, design solution | ✅ Complete |
| 2. Backend Implementation | API endpoints, executor updates | ✅ Complete |
| 3. Frontend Implementation | Load modal, chat wiring | ✅ Complete |
| 4. Integration Testing | End-to-end validation | ⏳ Pending |
| 5. Final Verification | Regression testing | ⏳ Pending |

---

## Detailed Task Breakdown

### Phase 1: Analysis & Planning ✅

- [x] Explore codebase structure
- [x] Understand current draft save mechanism
- [x] Analyze existing chat infrastructure
- [x] Review executor workflow loading
- [x] Map frontend Load button implementation
- [x] **Deep-dive frontend analysis** (workflow_builder_ide.html)
  - [x] Save button: `saveWorkflow('draft')` → Line 340
  - [x] Run Workflow: `runWorkflow()` → Lines 3373-3400 (forces temp save)
  - [x] Load button: `loadWorkflowDialog()` → Lines 3347-3371 (prompt-based)
  - [x] Chat: `sendChatMessage()` → Lines 3472-3566 (hardcoded to temp mode)
  - [x] State variables: Lines 2047-2062
  - [x] workflowAPI object: Lines 1729-1860
- [x] Document gap analysis
- [x] Create implementation plan (`chat_plan.md`)
- [x] Create progress tracker (`chat_progress.md`)

### Phase 2: Backend Implementation ✅

#### 2.1 Storage Layer
- [x] Add `list_draft_workflows()` method to `filesystem.py`
  - Read all files from `draft/` directory
  - Extract workflow metadata (id, name, execution_model, agent_count, created_at)
  - Return sorted list (newest first)

#### 2.2 Chat Session Updates
- [x] Update `chat_session.py`
  - Updated comment to include "draft" mode: `# "draft" | "test" | "final"`
  - Session creation works with draft mode

#### 2.3 Executor Updates
- [x] Update `executor.py`
  - Added `execution_mode="draft"` handling (lines 56-63)
  - Updated `load_for_execution()` to handle draft mode (lines 184-218)
  - Draft workflows load from state="draft"

#### 2.4 API Routes
- [x] Add `GET /workflows/draft/list` endpoint (lines 171-180)
  - Returns `{"workflows": [...], "total": N}`
- [x] `/workflows/chat/start` already accepts any mode string
- [x] `/workflows/chat/send` will use the mode from session

#### 2.5 Backend Testing
- [ ] Test draft list endpoint returns correct data
- [ ] Test chat start with draft mode
- [ ] Test chat send executes draft workflow
- [ ] Verify existing endpoints still work

### Phase 3: Frontend Implementation ✅

#### 3.1 Load Button Enhancement
- [x] Create modal component for workflow selection
  - Draft list view with workflow cards (scrollable)
  - Manual ID entry field with Enter key support
  - Close/Cancel functionality (click outside or X button)
- [x] Fetch draft list from backend on modal open (`workflowAPI.listDrafts()`)
- [x] Handle workflow selection (`loadSelectedDraft()` function)
- [x] Handle manual ID entry (`loadManualId()` function)

#### 3.2 Chat Interface Wiring
- [x] Update chat session creation to use "draft" mode
  - Changed `workflow_mode: 'temp'` → `workflow_mode: 'draft'`
  - Changed `saveWorkflow('temp')` → `saveWorkflow('draft')`
  - Removed re-save to temp on every chat message
- [x] Wire sendChatMessage to work with current draft
- [x] Reset chat session when loading new workflow

#### 3.3 Frontend Changes Summary
- Added `listDrafts()` method to workflowAPI (lines ~1858-1870)
- Added state variables: `showLoadModal`, `draftWorkflows`, `isLoadingDrafts`, `manualWorkflowId`
- Updated `loadWorkflowDialog()` to show modal + fetch drafts
- Added `loadWorkflowById()` helper function
- Added `loadSelectedDraft()` and `loadManualId()` functions
- Updated `sendChatMessage()` to use draft mode
- Added Load Workflow Modal HTML (after Execution Modal)
- Exported new state/functions in Vue return statement

### Phase 4: Integration Testing ⏳

- [ ] Full flow: Save draft → Chat with it
- [ ] Full flow: Load button → Select from list → Edit
- [ ] Full flow: Execute → Chat (verify temp mode works)
- [ ] Verify no regression in Save button
- [ ] Verify no regression in Execute button
- [ ] Verify no regression in existing Load behavior

---

## Bug Fix Applied (2026-01-27)

### Issue
Chat was failing with "Workflow not found" error because:
1. Each `saveWorkflow()` call generated a NEW workflow_id
2. Run Workflow saved to `temp/` with one ID
3. Chat (draft mode) looked in `draft/` with same ID - file not there

### Root Cause
`node_mapper.map_frontend_to_backend()` always called `new_id("wf_")` generating fresh IDs

### Fix Applied

**1. node_mapper.py** - Accept existing workflow_id parameter:
```python
def map_frontend_to_backend(..., workflow_id: Optional[str] = None):
    if not workflow_id:
        workflow_id = new_id("wf_")  # Only generate if not provided
```

**2. routes.py** - Pass workflow_id to mapper:
```python
workflow_id=request.get("workflow_id")
```

**3. workflow_builder_ide.html** - Frontend passes existing ID:
```javascript
saveCanvas(..., currentWorkflowId.value)
```

**4. executor.py** - Fallback to temp if draft not found:
```python
try:
    workflow = self.storage.load_workflow(workflow_id, state="draft")
except FileNotFoundError:
    # Fallback: try temp folder for backwards compatibility
    workflow = self.storage.load_workflow(workflow_id, state="temp")
```

### Files Modified
- `apps/workflow/visualization/node_mapper.py`
- `apps/workflow/routes.py`
- `apps/workflow/runtime/executor.py`
- `echoAI/workflow_builder_ide.html`

### Phase 5: Final Verification ⏳

- [ ] Code review
- [ ] Documentation update if needed
- [ ] Final testing sign-off

---

## Implementation Complete - Testing Required

**Date:** 2026-01-27

### Files Modified

**Backend (4 files):**
1. `apps/workflow/storage/filesystem.py` - Added `list_draft_workflows()` method
2. `apps/workflow/runtime/chat_session.py` - Updated mode comment to include "draft"
3. `apps/workflow/runtime/executor.py` - Added draft execution mode handling
4. `apps/workflow/routes.py` - Added `GET /workflows/draft/list` endpoint

**Frontend (1 file):**
1. `echoAI/workflow_builder_ide.html` - Load modal, draft list API, chat draft mode

### How to Test

1. **Start the server:**
   ```bash
   cd echoAI
   uvicorn apps.gateway.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Test Draft List API:**
   ```bash
   curl http://localhost:8000/workflows/draft/list
   ```

3. **Test Load Button:**
   - Open http://localhost:8000/workflow_builder_ide.html
   - Click "📂 Load" button
   - Should see modal with draft list + manual ID entry

4. **Test Draft Chat:**
   - Create a workflow on canvas
   - Click "💾 Save" (saves as draft)
   - Type in chat input and send
   - Should work without moving to temp/

5. **Test Run Workflow (regression):**
   - Click "▶ Run Workflow"
   - Should still save to temp and execute normally

---

## Implementation Notes

### Key Files to Modify

**Backend:**
1. `echoAI/apps/workflow/storage/filesystem.py` - Add list_draft_workflows()
2. `echoAI/apps/workflow/runtime/chat_session.py` - Add "draft" mode
3. `echoAI/apps/workflow/runtime/executor.py` - Add draft execution
4. `echoAI/apps/workflow/routes.py` - Add endpoint, update chat routes

**Frontend:**
1. `echoAI/workflow_builder_ide.html` - Load modal, chat wiring

### Critical Constraints

- ❌ Do NOT modify existing Save behavior
- ❌ Do NOT modify existing Execute behavior
- ❌ Do NOT auto-move drafts to temp
- ✅ Reuse existing execution logic
- ✅ Keep manual ID entry option
- ✅ Make changes backward-compatible

---

## Blockers & Issues

| Issue | Status | Resolution |
|-------|--------|------------|
| None identified | - | - |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-27 | Initial planning complete | Claude |
| 2026-01-27 | Created chat_plan.md | Claude |
| 2026-01-27 | Created chat_progress.md | Claude |

---

## Next Action

**Awaiting user approval to proceed with backend implementation using backend-python-dev agent.**
