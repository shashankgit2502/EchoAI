# EchoAI Agent System Bug Fixes Plan

## Overview
Three issues need to be addressed in the agent creation/update flow to improve decision-making and output structure.

---

## Issue 1: REUSE BEFORE CREATE (Semantic Similarity Check)

### Problem
When a user requests agent creation, the system creates a new agent even if a semantically similar agent already exists in the registry.

### Root Cause
- `AgentService.createFromPrompt()` in `echolib/services.py` checks templates but not existing agents in the registry
- Routes in `apps/agent/routes.py` directly create agents without similarity checking

### Solution
1. Add semantic similarity check in `AgentService` before creation
2. Query `AgentRegistry` for existing agents and compare intent/keywords
3. Return `AGENT_EXISTS` action if similar agent found (threshold-based match)
4. Only create new agent if no semantic match exists

### Files to Modify
- `echoAI/echolib/services.py` - Add `_check_existing_agents()` method
- `echoAI/apps/agent/routes.py` - Update `/create/prompt` and `/design/prompt` endpoints

### Response Format (when similar agent exists)
```json
{
  "action": "AGENT_EXISTS",
  "agent_id": "<existing_agent_id>",
  "agent_name": "<existing_agent_name>",
  "similarity_score": 0.XX,
  "message": "A similar agent already exists. You can configure or modify it."
}
```

---

## Issue 2: UPDATE VS RENAME ERROR

### Problem
When modifying an existing agent (changing tools, goals, behavior), the system incorrectly renames the agent using the user instruction as the new name.

### Root Cause
- `AgentDesigner._design_with_llm()` in `designer/agent_designer.py` generates a new name from prompt
- No detection of "update intent" vs "create intent"
- LLM prompt does not preserve existing agent name during updates

### Solution
1. Add intent detection: is user creating new agent or modifying existing one?
2. Modify `AgentDesigner` to accept optional `existing_agent` parameter
3. When existing agent provided, preserve name and only update specified fields
4. Update routes to detect modification keywords and pass existing agent context

### Files to Modify
- `echoAI/apps/agent/designer/agent_designer.py` - Add update mode with name preservation
- `echoAI/echolib/services.py` - Add intent detection for update vs create
- `echoAI/apps/agent/routes.py` - Add logic to detect update requests

### Response Format (for updates)
```json
{
  "action": "UPDATE_AGENT",
  "agent_id": "<existing_agent_id>",
  "agent_name": "<unchanged_name>",
  "updated_fields": {
    "tools": ["..."],
    "description": "..."
  }
}
```

---

## Issue 3: TOOL AUTO-SELECTION

### Problem
Tools are not auto-selected based on agent purpose. System either assigns empty list or blindly uses template tools.

### Root Cause
- `AgentDesigner.design_from_prompt()` accepts tools as parameter but doesn't infer them
- No logic to map agent purpose/domain to appropriate tools
- Tool registry not queried during agent design

### Solution
1. Add tool inference logic in `AgentDesigner` based on analyzed intent
2. Query tool registry for available tools
3. Apply tool selection rules:
   - Min: 0, Max: 2 tools
   - web_search: allowed for any agent (optional)
   - file_reader: for document/file processing agents
   - code_executor, code_reviewer: for coding agents
   - Empty list if no clear need

### Files to Modify
- `echoAI/apps/agent/designer/agent_designer.py` - Add `_select_tools()` method
- `echoAI/apps/agent/container.py` - Inject tool registry into designer

### Tool Mapping Rules
| Agent Domain/Keywords | Allowed Tools |
|----------------------|---------------|
| research, analyze, search, explore | web_search |
| file, document, pdf, read, parse | file_reader |
| code, programming, develop, debug | code_executor, code_reviewer |
| general/unclear | [] |

---

## Implementation Order
1. Issue 2 (UPDATE VS RENAME) - Foundation for proper action selection
2. Issue 1 (REUSE BEFORE CREATE) - Depends on proper intent detection
3. Issue 3 (TOOL AUTO-SELECTION) - Independent, can be done last

---

## Constraints (CRITICAL)
- Preserve all existing backend & frontend logic
- Preserve existing response schemas
- Only adjust: action choice, update vs create decision, tool selection list
- Do not introduce new fields beyond those specified
- Do not rename agents unless explicitly asked

---

## Issue 4: TEMPLATE CHECK ON WRONG ENDPOINT (2026-01-27)

### Problem
When user creates "code reviewer" agent, system creates "CodeCraft Pro" instead of finding the existing "Code Reviewer" template.

### Root Cause Analysis
- Frontend calls `/agents/design/prompt` endpoint, NOT `/agents/create/prompt`
- Template checking code (`_check_existing_agents()`) was added to `/create/prompt` only
- `/design/prompt` endpoint calls `designer.design_from_prompt()` directly, bypassing ALL similarity/template checking

### Evidence
```
Frontend: designAgentFromPrompt() → POST /agents/design/prompt
Backend: /design/prompt → designer.design_from_prompt() → NO template check
```

### Solution
Add template/similarity checking to `/design/prompt` endpoint in `apps/agent/routes.py`:
1. Before calling `designer.design_from_prompt()`, call `service._check_existing_agents()`
2. If match found, return `AGENT_EXISTS` response
3. Only proceed with design if no match

### Files to Modify
- `echoAI/apps/agent/routes.py` - Add similarity check to `/design/prompt` endpoint

---

## Issue 5: TOOLS NOT PERSISTING TO CONFIGURE TAB (2026-01-27)

### Problem
User adds tools via chat ("add code executor"), chat confirms tools added, but Configure tab shows no tools.

### Root Cause Analysis
- Chat updates `selectedTools` state correctly via `setSelectedTools(newTools)`
- But Configure tab may re-read from original backend response or different state
- State update is lost when switching tabs

### Solution
Ensure the tools added via chat persist to Configure tab:
1. Verify Configure tab reads from `selectedTools` state
2. If needed, ensure state is preserved across tab switches

### Files to Modify
- `echoAI/type-agent_builder.html` - Verify/fix state management for tools

---

## Issue 6: CHAT Q&A LOGIC FLAW (2026-01-27)

### Problem
User says "yes, save it" to finalize, but system treats it as refinement and appends text to goal.

### Root Cause Analysis
Question asked: *"Would you like to make any other changes, OR are we ready to finalize?"*

The `noChangePhrases` list only contains "no"-type responses:
```javascript
['no', 'nope', 'none', 'done', 'finish', 'finalize', ...]
```

"yes, save it" means "yes, finalize" but:
- "yes" is NOT in the list
- "save" is NOT in the list
- Falls through to else branch → appends to goal

### Solution
Add affirmative finalization phrases to `noChangePhrases`:
```javascript
'yes', 'yep', 'yeah', 'sure', 'save', 'save it', 'yes save', 'yes save it',
'finalize it', 'yes finalize', 'go ahead', 'lets go', "let's go"
```

### Files to Modify
- `echoAI/type-agent_builder.html` - Update `noChangePhrases` in refine step

---

## Issue 7: TEMPLATE TOOLS USE WRONG IDS (2026-01-27)

### Problem
When agent is created from template, tools don't appear in Configure tab.

### Root Cause Analysis
- `agent_templates.json` uses human-readable tool names: `["Code Analysis", "Security Scan", "GitHub"]`
- Frontend `availableTools` uses IDs: `["code_executor", "code_reviewer", "web_search"]`
- When `AGENT_EXISTS` returns template data, tool names don't match frontend IDs
- `availableTools.find(t => t.id === toolId)` returns `null` → tools don't render

### Solution
In `_check_existing_agents()`, when returning `AGENT_EXISTS` for a template match:
1. Run `_auto_select_tools()` on template's name + role + description
2. Replace template's `tools` field with auto-selected tool IDs
3. Return modified template with correct tool IDs

### Files to Modify
- `echoAI/echolib/services.py` - Add tool auto-selection before returning template match

---

## Issue 8: TEMPLATE MATCHING FAILS FOR ANY TEMPLATE (2026-01-27)

### Problem
Template matching only worked for specific cases. User asked for "financial analyst" but got "FinGenius Pro" instead of the "Financial Analyst" template.

### Root Cause Analysis
- Jaccard similarity uses EXACT word matching
- "analyst" ≠ "analysis" (different strings)
- "financial" matched but "analyst" vs "analysis" didn't
- Threshold was too high (0.5) for natural language variations

### Solution
Implemented SMART template matching with multiple strategies:
1. **Direct name match**: If template name appears verbatim in prompt → 0.95 score
2. **Name word matching**: If all words from template name (normalized) appear in prompt → 0.85 score
3. **Role matching**: If role words match → 0.5-0.8 score
4. **Normalized Jaccard**: Fallback with word normalization (analyst→analy, analysis→analy)

Added `_normalize_word()` method that maps word families:
- analyst/analysis/analyze → "analy"
- financial/finance → "financ"
- etc.

### Files Modified
- `echoAI/echolib/services.py` - Added `_normalize_word()`, `_score_template_match()`, rewrote `_check_existing_agents()`
- `echoAI/apps/agent/routes.py` - Pass `user_prompt` to `_check_existing_agents()`

---

## Issue 9: NAME CONFIRMATION LITERAL INTERPRETATION (2026-01-27)

### Problem
User says "keep this name" and system sets agent name TO "keep this name" literally.

### Root Cause Analysis
- Frontend used simple phrase matching: `confirmationPhrases.includes(input)`
- "keep this name" was not in the list
- "keep it" was in list but "keep this name".includes("keep it") = FALSE

### Solution
Implemented SMART name confirmation with 3 strategies:
1. **Direct phrases**: Standard confirmations ("yes", "ok", "keep it", etc.)
2. **Pattern matching**: Regex patterns for natural variations
   - `/\bkeep\b.*\b(this|that|the|it|name)\b/i` matches "keep this name", "keep the name", etc.
   - `/\bno\b.*\bchange/i` matches "no change needed"
3. **New name indicators**: Detect when user is providing a NEW name
   - "call it X", "name it Y", "rename to Z", "how about X"
   - Extract the actual name from these patterns

### Files Modified
- `echoAI/type-agent_builder.html` - Rewrote name step logic with regex patterns

---

## Issue 10: TOOLS NOT AUTO-SELECTED FOR FINANCIAL AGENTS (2026-01-27)

### Problem
Financial analyst agent didn't get web_search tool auto-selected.

### Root Cause Analysis
- TOOL_SELECTION_RULES didn't include financial/finance keywords for web_search
- Financial analysts need web search for market research

### Solution
Added financial keywords to web_search rule:
```python
"web_search": [
    ...,
    "financial", "finance", "stock", "market", "report", "data",
    "information", "news", "trends", "analyst", "advisor"
]
```

Also added `calculator` tool rule for financial calculations.

### Files Modified
- `echoAI/echolib/services.py` - Updated TOOL_SELECTION_RULES

---

## Issue 11: LLM-BASED INTENT CLASSIFICATION (2026-01-27)

### Problem
Frontend uses pattern matching for intent detection which requires constant updates for new phrases. User wants "real chatbot" behavior where ANY natural language is understood.

### Root Cause Analysis
- Frontend used `confirmationPhrases.some(phrase => input.includes(phrase))`
- Every new vocabulary requires code update
- Not scalable, not how real chatbots work

### Solution
Implemented **backend LLM-based intent classification**:

1. **New Endpoint**: `POST /agents/classify-intent`
   - Accepts: context, suggested_value, user_message
   - Returns: intent (CONFIRMATION/MODIFICATION/REJECTION/CLARIFICATION), confidence, reasoning, extracted_value

2. **Backend Method**: `AgentService.classify_user_intent()`
   - Uses LLM to understand natural language intent
   - System prompt instructs LLM on context types and intent types
   - Fallback heuristics if LLM unavailable

3. **Frontend Integration**:
   - Name step calls `/agents/classify-intent` instead of pattern matching
   - Refine step calls `/agents/classify-intent` instead of pattern matching
   - Handles all 4 intent types appropriately

### Intent Types
| Intent | Meaning | Action |
|--------|---------|--------|
| CONFIRMATION | User approves/accepts | Keep suggested value, advance flow |
| MODIFICATION | User wants to change | Extract new value, update state |
| REJECTION | User declines | Stay in current step, ask again |
| CLARIFICATION | User asks question | Provide explanation, stay in step |

### Files Modified
- `echoAI/echolib/services.py` - Added `classify_user_intent()` and `_fallback_intent_classification()`
- `echoAI/apps/agent/routes.py` - Added `/agents/classify-intent` endpoint
- `echoAI/type-agent_builder.html` - Updated name and refine steps to use backend intent classification

---

## Issue 12: FALSE POSITIVE SIMILARITY MATCHING (2026-01-27)

### Problem
User asked to create "Booking of tickets" agent, but system matched "Agent 3 - Hybrid Integrator" at 85% similarity.

### Root Cause Analysis
The `_score_template_match()` function had a flaw in Strategy 2 (Name Word Matching):

1. Template name "Agent 3" was filtered to `["agent"]` (the "3" was removed because `len("3") <= 2`)
2. User prompt contained "agent" (as in "creating an **agent**")
3. Single-word match ratio = 1/1 = 100%
4. This returned 0.85 (85%) score

**The fundamental problem**: In an agent-building system, the word "agent" appears in nearly EVERY user prompt. Using it for similarity matching guarantees false positives.

### Solution
Implemented multi-layered protection against false positives:

1. **AGENT_STOP_WORDS**: Added a set of generic words to exclude from matching:
   - `agent, agents, assistant, assistants, helper, helpers, bot, bots, ai`
   - `system, systems, tool, tools`
   - `create, creating, build, building, make, making`
   - `help, helping, want, need, please, would, could`
   - `the, a, an, for, me, my, that, will, do, does`

2. **GENERIC_AGENT_NAME_PATTERN**: Regex `^agent\s*\d+$` to skip placeholder names like "Agent 1", "Agent 2", "Agent 3"

3. **Modified `_score_template_match()`**:
   - PRE-CHECK: Return 0.0 immediately for generic agent names
   - Strategy 1: Verify direct name match has at least 1 significant (non-stop) word
   - Strategy 2: Filter stop words from name_words AND prompt_words
   - Strategy 2: Require at least 2 significant words for 0.85 score
   - Strategy 2: Single-word matches get max 0.4 score (needs role/description confirmation)
   - Strategy 3 & 4: Also filter stop words for consistency

### Files Modified
- `echoAI/echolib/services.py` - Added `AGENT_STOP_WORDS`, `GENERIC_AGENT_NAME_PATTERN`, updated `_score_template_match()` with stop word filtering

---

## Issue 13: TOOL ID MISMATCH - TOOLS NOT RENDERING (2026-01-27)

### Problem
User adds tools via chat ("add web search") or tools are auto-selected for agents (e.g., "Create a travel agent"), but Configure tab shows no tools selected.

### Root Cause Analysis
**Critical ID mismatch between frontend and backend:**

| Component | Uses ID | Backend Actual ID |
|-----------|---------|-------------------|
| Frontend `availableTools` | `web_search` | `tool_web_search` |
| Frontend `toolMappings` | `web_search` | `tool_web_search` |
| Backend `TOOL_SELECTION_RULES` | `web_search` | `tool_web_search` |
| Backend tool files | - | `tool_web_search` |

**Flow breakdown:**
1. User says "add web search" or creates "travel agent"
2. Chat/Backend adds `"web_search"` to tools array
3. Configure tab calls `getAvailableTools().find(t => t.id === "web_search")`
4. `getAvailableTools()` returns backend tools with IDs like `"tool_web_search"`
5. `.find()` returns `undefined` → tool doesn't render!

**Additional Issue:** Missing keywords for travel/booking agents in `TOOL_SELECTION_RULES`

### Solution

**Backend (`echoAI/echolib/services.py`):**
1. Update `TOOL_SELECTION_RULES` keys to use actual backend tool IDs:
   - `web_search` → `tool_web_search`
   - `file_reader` → `tool_file_reader`
   - `code_executor` → `tool_code_generator`
   - `code_reviewer` → `tool_code_reviewer`
   - `calculator` → `tool_calculator`

2. Add travel/booking keywords to `tool_web_search`:
   - `travel`, `trip`, `vacation`, `holiday`, `booking`, `flight`, `hotel`
   - `reservation`, `destination`, `tour`, `itinerary`, `ticket`, `tickets`

**Frontend (`echoAI/type-agent_builder.html`):**
1. Update `availableTools` IDs to match backend format
2. Update `toolMappings` values to use backend IDs
3. Add `normalizeToolId()` function for safety/backwards compatibility
4. Apply normalization when:
   - Loading agent from backend response
   - Adding tools via chat
   - Loading from AGENT_EXISTS response

### Files to Modify
- `echoAI/echolib/services.py` - Fix `TOOL_SELECTION_RULES` keys, add travel keywords
- `echoAI/type-agent_builder.html` - Fix IDs, add normalization layer
