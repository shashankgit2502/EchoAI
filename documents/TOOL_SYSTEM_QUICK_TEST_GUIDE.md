# EchoAI Tool System - Quick Test Guide

**For**: Human testers who need to quickly validate the system
**Full Specification**: See `TOOL_SYSTEM_MANUAL_TEST_SPECIFICATION.md`

---

## Quick Start

### 1. Start the Server
```bash
cd echoAI
uvicorn apps.gateway.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Run Discovery
```bash
curl -X POST http://localhost:8000/tools/discover
```
Expected: `{"status": "success", "discovered_count": 5, ...}`

### 3. List Tools
```bash
curl http://localhost:8000/tools/list
```
Expected: Array of 5 tools (calculator, web_search, file_reader, code_generator, code_reviewer)

### 4. Test Calculator
```bash
curl -X POST http://localhost:8000/tools/invoke/id/tool_calculator \
  -H "Content-Type: application/json" \
  -d '{"operation": "add", "values": [10, 20, 30]}'
```
Expected: `{"success": true, "output": {"result": 60}, ...}`

---

## Critical Test Paths

### Foundation Path (Must Pass First)
1. **A1**: Health Check → `GET /tools/health` → status: "healthy"
2. **A2**: Discovery → `POST /tools/discover` → discovered_count: 5
3. **A3**: List Tools → `GET /tools/list` → 5 tools returned
4. **B1**: Invoke Tool → `POST /tools/invoke/id/tool_calculator` → success: true

**If any fail, STOP and debug before continuing.**

### Integration Path (Verify Agent-Tool Binding)
1. **C1**: Get Agent Tools → `GET /tools/agent/{agent_id}`
2. **C2**: Create Agent with Tool
3. **D1**: Single Agent Workflow Execution (E2E)

### E2E Validation Path (Deferred Tasks)
1. **D1**: Single agent with tool → Verify tool invocation in workflow
2. **D2**: Three agents sequential → Verify tool output propagation
3. **D3**: Parallel workflow → Verify concurrent tool execution
4. **D4**: Hierarchical workflow → Verify manager-worker tool usage

---

## Quick Test Commands (curl)

### Health Check
```bash
curl http://localhost:8000/tools/health
```

### Discover Tools
```bash
curl -X POST http://localhost:8000/tools/discover
```

### List All Tools
```bash
curl http://localhost:8000/tools/list
```

### Get Specific Tool
```bash
curl http://localhost:8000/tools/tool_calculator
```

### Invoke Calculator (Add)
```bash
curl -X POST http://localhost:8000/tools/invoke/id/tool_calculator \
  -H "Content-Type: application/json" \
  -d '{"operation": "add", "values": [10, 20]}'
```

### Invoke Calculator (Multiply)
```bash
curl -X POST http://localhost:8000/tools/invoke/id/tool_calculator \
  -H "Content-Type: application/json" \
  -d '{"operation": "multiply", "values": [6, 7]}'
```

### Validate Input (Valid)
```bash
curl -X POST http://localhost:8000/tools/validate/tool_calculator \
  -H "Content-Type: application/json" \
  -d '{"operation": "add", "values": [1, 2]}'
```

### Validate Input (Invalid - Missing Field)
```bash
curl -X POST http://localhost:8000/tools/validate/tool_calculator \
  -H "Content-Type: application/json" \
  -d '{"values": [1, 2]}'
```

### Search Tools
```bash
curl "http://localhost:8000/tools/search?q=calculator"
```

### List by Type
```bash
curl http://localhost:8000/tools/list/type/local
```

### Clear Cache
```bash
curl -X POST http://localhost:8000/tools/cache/clear
```

### Register Test Tool
```bash
curl -X POST http://localhost:8000/tools/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Tool",
    "description": "A test tool",
    "tool_type": "local",
    "input_schema": {"type": "object", "properties": {"test": {"type": "string"}}},
    "output_schema": {"type": "object"},
    "execution_config": {"module": "builtins", "class": "dict", "method": "get"}
  }'
```

### Delete Tool
```bash
curl -X DELETE http://localhost:8000/tools/tool_test_tool
```

---

## Expected Response Patterns

### Success Response
```json
{
  "name": "Calculator",
  "tool_id": "tool_calculator",
  "output": { ... },
  "success": true,
  "error": null,
  "metadata": { ... }
}
```

### Failure Response
```json
{
  "name": "Calculator",
  "tool_id": "tool_calculator",
  "output": {},
  "success": false,
  "error": "Input validation failed at 'operation': ...",
  "metadata": {
    "error_type": "validation_error"
  }
}
```

### Discovery Response
```json
{
  "status": "success",
  "discovered_count": 5,
  "tools": [
    "tool_calculator",
    "tool_web_search",
    "tool_file_reader",
    "tool_code_generator",
    "tool_code_reviewer"
  ]
}
```

---

## Quick Validation Checklist

### Phase 5 Tasks 5.1 & 5.2 (Unit + Integration)

**Unit Tests (14 tests)**:
- [ ] A1: Health check returns healthy
- [ ] A2: Discovery finds 5 tools
- [ ] A3: List returns all tools
- [ ] A4: Get tool by ID returns full definition
- [ ] A5: Manual registration works
- [ ] A6: Delete removes tool
- [ ] B1: Calculator executes correctly
- [ ] B2: Valid input passes validation
- [ ] B3: Invalid input fails validation
- [ ] B4: Invoke by name works
- [ ] B5: Instance caching reduces execution time
- [ ] B6: Cache clearing works
- [ ] C3: Search finds tools
- [ ] C4: List by type filters correctly

**Integration Tests (8 tests)**:
- [ ] C1: Get agent tools returns correct data
- [ ] C2: Create agent with tools succeeds
- [ ] E1: Timeout handling works
- [ ] E2: Invalid input handled gracefully
- [ ] E3: Missing tool handled gracefully
- [ ] E4: Missing config detected
- [ ] E5: Tool failure in workflow handled
- [ ] E6: Concurrent invocations work

**E2E Tests (4 tests - includes deferred 3.3 & 3.4)**:
- [ ] D1: Single agent workflow with tool (Task 3.3)
- [ ] D2: Three-agent sequential with propagation (Task 3.4)
- [ ] D3: Parallel workflow with tools
- [ ] D4: Hierarchical workflow with tools

---

## Troubleshooting

### Discovery Returns 0 Tools
```bash
# Check folder exists
ls echoAI/AgentTools/

# Check manifests exist
ls echoAI/AgentTools/*/tool_manifest.json

# Check manifest syntax
cat echoAI/AgentTools/calculator/tool_manifest.json | python -m json.tool
```

### Tool Invocation Fails
```bash
# Check tool exists
curl http://localhost:8000/tools/tool_calculator

# Check import works
cd echoAI
python -c "from AgentTools.calculator.service import CalculatorService; print('OK')"

# Check __init__.py files
ls AgentTools/__init__.py
ls AgentTools/calculator/__init__.py
```

### Server Errors
```bash
# Check server logs in terminal
# Look for errors from apps.tool.executor or apps.tool.registry

# Restart server with debug logging
uvicorn apps.gateway.main:app --reload --log-level debug
```

---

## Test Result Template

Copy this for reporting:

```
DATE: ___________
TESTER: ___________

FOUNDATION TESTS (A1-A6, B1-B6):
- Tests Passed: ___ / 12
- Tests Failed: ___
- Critical Issues: ___

INTEGRATION TESTS (C1-C4, E1-E6):
- Tests Passed: ___ / 10
- Tests Failed: ___
- Critical Issues: ___

E2E TESTS (D1-D4):
- Tests Passed: ___ / 4
- Tests Failed: ___
- Critical Issues: ___

DEFERRED TASKS VERIFICATION:
- Task 3.3 (D1): [ ] PASS [ ] FAIL
- Task 3.4 (D2): [ ] PASS [ ] FAIL

OVERALL STATUS: [ ] READY FOR PRODUCTION [ ] NEEDS FIXES

NOTES:
_______________________________________________
```

---

## Next Steps After Testing

1. Mark all tests in `TOOL_SYSTEM_MANUAL_TEST_SPECIFICATION.md`
2. Fill defects table with any issues found
3. Update `tools_progress.md` Phase 5 status
4. Report results to development team
5. If all pass, proceed to automated test creation

---

**Full Details**: See `TOOL_SYSTEM_MANUAL_TEST_SPECIFICATION.md`
**Questions**: Contact EchoAI development team
