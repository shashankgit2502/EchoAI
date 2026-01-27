# EchoAI Tool System - Manual Test Specification

**Version**: 1.0
**Created**: 2026-01-26
**Purpose**: Comprehensive manual testing guide for Phase 5 validation
**Target Audience**: Human testers unfamiliar with the codebase

---

## Table of Contents

1. [Test Environment Setup](#test-environment-setup)
2. [Section A: Tool Discovery & Registration](#section-a-tool-discovery--registration)
3. [Section B: Direct Tool Invocation](#section-b-direct-tool-invocation)
4. [Section C: Agent-Tool Binding](#section-c-agent-tool-binding)
5. [Section D: Workflow Execution with Tools](#section-d-workflow-execution-with-tools)
6. [Section E: Error Handling & Edge Cases](#section-e-error-handling--edge-cases)
7. [Test Results Tracking](#test-results-tracking)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## Test Environment Setup

### Prerequisites

1. **Environment Confirmation**: Verify you are in a sandbox/test environment
   - Check: No production data should be accessible
   - Check: System is running locally on `localhost:8000`

2. **Service Status**: Ensure backend server is running
   ```bash
   cd echoAI
   uvicorn apps.gateway.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Tools Available**: Verify AgentTools folder exists
   ```bash
   ls echoAI/AgentTools/
   # Should show: calculator, web_search, file_reader, code_generator, code_reviewer
   ```

4. **HTTP Client**: Install a REST client (Postman, Insomnia, or curl)
   - Base URL: `http://localhost:8000`
   - All requests use `Content-Type: application/json`

5. **Test Data Directory**: Create a test data directory
   ```bash
   mkdir -p test_data
   ```

### Environment Assumptions

- Server is running on `http://localhost:8000`
- Tool storage directory: `echoAI/apps/storage/tools/`
- AgentTools directory: `echoAI/AgentTools/`
- Agent storage directory: `echoAI/apps/storage/agents/`

---

## Section A: Tool Discovery & Registration (Foundation)

### Test A1: Health Check - Tool System Status

**Test ID**: A1
**Test Type**: Unit
**Purpose**: Verify tool system components are initialized and healthy

#### Test Steps

1. **HTTP Request**
   ```
   GET http://localhost:8000/tools/health
   ```

2. **Expected Status Code**: `200 OK`

3. **Expected Response Structure**
   ```json
   {
     "status": "healthy",
     "registry": {
       "tool_count": <number>,
       "discovery_dirs": <number>
     },
     "executor": {
       "cached_instances": <number>,
       "default_timeout": 60
     }
   }
   ```

4. **Validation Checklist**
   - [ ] Response status is "healthy"
   - [ ] registry.tool_count >= 0
   - [ ] registry.discovery_dirs >= 1
   - [ ] executor.default_timeout == 60
   - [ ] No error messages in response

5. **Common Errors**
   - `{"status": "unhealthy", "error": "..."}` → DI container initialization failed
   - 404 Not Found → Route not registered, check gateway.main.py includes tool routes
   - Connection refused → Server not running

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test A2: Tool Discovery from AgentTools Folder

**Test ID**: A2
**Test Type**: Integration
**Purpose**: Verify system can discover and register tools from AgentTools directory

#### Test Steps

1. **HTTP Request**
   ```
   POST http://localhost:8000/tools/discover
   Content-Type: application/json

   (No body required)
   ```

2. **Expected Status Code**: `201 Created` or `200 OK`

3. **Expected Response Structure**
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

4. **Validation Checklist**
   - [ ] discovered_count == 5 (or actual number of tools with manifests)
   - [ ] tools array contains expected tool_ids
   - [ ] Each tool_id starts with "tool_"
   - [ ] No duplicate tool_ids in list
   - [ ] Status is "success"

5. **Common Errors**
   - `discovered_count: 0` → Check tool_manifest.json files exist in AgentTools subfolders
   - `"error": "Tool discovery failed"` → Check discovery_dirs path in container.py
   - Invalid JSON in manifest → Check manifest file syntax
   - ImportError in logs → Check AgentTools __init__.py files exist

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test A3: List All Registered Tools

**Test ID**: A3
**Test Type**: Unit
**Purpose**: Verify registry returns all registered tools with correct metadata

#### Test Steps

1. **Prerequisite**: Run Test A2 (discovery) first

2. **HTTP Request**
   ```
   GET http://localhost:8000/tools/list
   ```

3. **Expected Status Code**: `200 OK`

4. **Expected Response Structure**
   ```json
   [
     {
       "tool_id": "tool_calculator",
       "name": "Calculator",
       "description": "Performs mathematical calculations...",
       "tool_type": "local",
       "status": "active",
       "version": "1.0",
       "tags": ["math", "calculation"]
     },
     ...
   ]
   ```

5. **Validation Checklist**
   - [ ] Response is an array
   - [ ] Array length matches discovered_count from A2
   - [ ] Each tool has required fields: tool_id, name, description, tool_type, status
   - [ ] tool_type values are valid: "local", "mcp", "api", or "crewai"
   - [ ] status values are valid: "active", "deprecated", or "disabled"
   - [ ] All tool_ids are unique

6. **Common Errors**
   - Empty array → Discovery didn't run or tools not saved to storage
   - Missing fields → ToolDef model incomplete or serialization issue
   - 500 Internal Server Error → Check storage directory permissions

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test A4: Get Tool by ID

**Test ID**: A4
**Test Type**: Unit
**Purpose**: Verify individual tool retrieval returns complete definition

#### Test Steps

1. **HTTP Request**
   ```
   GET http://localhost:8000/tools/tool_calculator
   ```

2. **Expected Status Code**: `200 OK`

3. **Expected Response Structure**
   ```json
   {
     "tool_id": "tool_calculator",
     "name": "Calculator",
     "description": "Performs mathematical calculations including arithmetic...",
     "tool_type": "local",
     "input_schema": {
       "type": "object",
       "properties": {
         "operation": {"type": "string"},
         "values": {"type": "array"}
       },
       "required": ["operation", "values"]
     },
     "output_schema": {
       "type": "object",
       "properties": {
         "operation": {"type": "string"},
         "result": {}
       }
     },
     "execution_config": {
       "module": "AgentTools.calculator.service",
       "class": "CalculatorService",
       "method": "calculate"
     },
     "version": "1.0",
     "tags": ["math", "calculation"],
     "status": "active",
     "metadata": {}
   }
   ```

4. **Validation Checklist**
   - [ ] All ToolDef fields present (tool_id through metadata)
   - [ ] input_schema has "type", "properties", "required" fields
   - [ ] output_schema has "type", "properties" fields
   - [ ] execution_config has "module", "class", "method" for LOCAL tools
   - [ ] Response matches tool_manifest.json content

5. **Common Errors**
   - 404 Not Found → tool_id doesn't exist or wrong spelling
   - Missing execution_config → Manifest incomplete
   - Invalid schema structure → JSON Schema validation failed

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test A5: Manual Tool Registration

**Test ID**: A5
**Test Type**: Unit
**Purpose**: Verify tools can be registered via API without discovery

#### Test Steps

1. **HTTP Request**
   ```
   POST http://localhost:8000/tools/register
   Content-Type: application/json

   {
     "name": "Test Mock Tool",
     "description": "A simple mock tool for testing",
     "tool_type": "local",
     "input_schema": {
       "type": "object",
       "properties": {
         "message": {"type": "string"}
       },
       "required": ["message"]
     },
     "output_schema": {
       "type": "object",
       "properties": {
         "echo": {"type": "string"}
       }
     },
     "execution_config": {
       "module": "builtins",
       "class": "dict",
       "method": "get"
     },
     "version": "1.0",
     "tags": ["test", "mock"],
     "status": "active"
   }
   ```

2. **Expected Status Code**: `200 OK` or `201 Created`

3. **Expected Response Structure**
   ```json
   {
     "tool_id": "tool_test_mock_tool",
     "status": "registered",
     "message": "Tool 'Test Mock Tool' registered successfully"
   }
   ```

4. **Validation Checklist**
   - [ ] Response includes auto-generated tool_id
   - [ ] Status is "registered"
   - [ ] tool_id follows pattern: tool_{name_lowercase_underscored}
   - [ ] Subsequent GET /tools/tool_test_mock_tool returns the tool

5. **Common Errors**
   - 400 Bad Request → Invalid ToolDef structure, check required fields
   - "Tool must have a name" → Missing name field
   - "status must be one of {active, deprecated, disabled}" → Invalid status value

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test A6: Delete Tool

**Test ID**: A6
**Test Type**: Unit
**Purpose**: Verify tools can be removed from registry

#### Test Steps

1. **Prerequisite**: Register a test tool (use A5)

2. **HTTP Request**
   ```
   DELETE http://localhost:8000/tools/tool_test_mock_tool
   ```

3. **Expected Status Code**: `200 OK`

4. **Expected Response Structure**
   ```json
   {
     "status": "deleted",
     "tool_id": "tool_test_mock_tool",
     "message": "Tool 'tool_test_mock_tool' deleted successfully"
   }
   ```

5. **Validation Checklist**
   - [ ] Status is "deleted"
   - [ ] tool_id matches request
   - [ ] Subsequent GET /tools/tool_test_mock_tool returns 404
   - [ ] Tool no longer appears in /tools/list

6. **Common Errors**
   - 404 Not Found → Tool doesn't exist (already deleted or wrong ID)
   - File permission error in logs → Check storage directory permissions

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

## Section B: Direct Tool Invocation (Executor Validation)

### Test B1: Invoke Calculator Tool (LOCAL type)

**Test ID**: B1
**Test Type**: Integration
**Purpose**: Verify local tool execution with dynamic module loading

#### Test Steps

1. **Prerequisite**: Tool discovery completed (Test A2)

2. **HTTP Request**
   ```
   POST http://localhost:8000/tools/invoke/id/tool_calculator
   Content-Type: application/json

   {
     "operation": "add",
     "values": [10, 20, 30]
   }
   ```

3. **Expected Status Code**: `200 OK`

4. **Expected Response Structure**
   ```json
   {
     "name": "Calculator",
     "tool_id": "tool_calculator",
     "output": {
       "operation": "add",
       "result": 60
     },
     "success": true,
     "error": null,
     "metadata": {
       "execution_time": <float>,
       "tool_type": "local",
       "tool_version": "1.0",
       "context": {}
     }
   }
   ```

5. **Validation Checklist**
   - [ ] success == true
   - [ ] error == null
   - [ ] output.result == 60 (correct calculation)
   - [ ] metadata.execution_time > 0
   - [ ] metadata.tool_type == "local"

6. **Test Variations**
   - Subtract: `{"operation": "subtract", "values": [100, 25]}` → result: 75
   - Multiply: `{"operation": "multiply", "values": [5, 7]}` → result: 35
   - Divide: `{"operation": "divide", "values": [100, 4]}` → result: 25.0

7. **Common Errors**
   - success: false, error: "Module not found" → Check AgentTools.calculator.service exists
   - success: false, error: "Input validation failed" → Check input against input_schema
   - success: false, error: "Method not found" → Check CalculatorService.calculate exists
   - Timeout error → Check default_timeout setting, tool might be hanging

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test B2: Input Validation - Valid Input

**Test ID**: B2
**Test Type**: Unit
**Purpose**: Verify input validation accepts valid data

#### Test Steps

1. **HTTP Request**
   ```
   POST http://localhost:8000/tools/validate/tool_calculator
   Content-Type: application/json

   {
     "operation": "add",
     "values": [1, 2, 3]
   }
   ```

2. **Expected Status Code**: `200 OK`

3. **Expected Response Structure**
   ```json
   {
     "valid": true,
     "tool_id": "tool_calculator",
     "message": "Input validation passed"
   }
   ```

4. **Validation Checklist**
   - [ ] valid == true
   - [ ] Message indicates success
   - [ ] No error field present

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test B3: Input Validation - Invalid Input

**Test ID**: B3
**Test Type**: Unit
**Purpose**: Verify input validation rejects invalid data

#### Test Steps

1. **HTTP Request** (missing required field)
   ```
   POST http://localhost:8000/tools/validate/tool_calculator
   Content-Type: application/json

   {
     "values": [1, 2, 3]
   }
   ```

2. **Expected Status Code**: `200 OK`

3. **Expected Response Structure**
   ```json
   {
     "valid": false,
     "tool_id": "tool_calculator",
     "error": "Input validation failed at 'root': 'operation' is a required property"
   }
   ```

4. **Validation Checklist**
   - [ ] valid == false
   - [ ] error field present with descriptive message
   - [ ] Error indicates missing "operation" field

5. **Test Variations**
   - Wrong type: `{"operation": 123, "values": [1,2]}` → error about type mismatch
   - Invalid enum: `{"operation": "invalid_op", "values": [1,2]}` → error about enum validation

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test B4: Invoke Tool by Name (Backward Compatibility)

**Test ID**: B4
**Test Type**: Unit
**Purpose**: Verify name-based tool invocation for backward compatibility

#### Test Steps

1. **HTTP Request**
   ```
   POST http://localhost:8000/tools/invoke/Calculator
   Content-Type: application/json

   {
     "operation": "multiply",
     "values": [6, 7]
   }
   ```

2. **Expected Status Code**: `200 OK`

3. **Expected Response Structure**
   ```json
   {
     "name": "Calculator",
     "tool_id": "tool_calculator",
     "output": {
       "operation": "multiply",
       "result": 42
     },
     "success": true,
     "error": null,
     "metadata": {...}
   }
   ```

4. **Validation Checklist**
   - [ ] Tool resolved by name "Calculator"
   - [ ] Result is correct (42)
   - [ ] success == true

5. **Common Errors**
   - 404 Not Found → Name lookup failed, check registry.get_by_name implementation

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test B5: Executor Instance Caching

**Test ID**: B5
**Test Type**: Unit
**Purpose**: Verify local tool instances are cached for performance

#### Test Steps

1. **First Invocation**
   ```
   POST http://localhost:8000/tools/invoke/id/tool_calculator
   Content-Type: application/json

   {"operation": "add", "values": [1, 1]}
   ```
   Note the execution_time from metadata (e.g., 0.045s)

2. **Second Invocation** (immediately after)
   ```
   POST http://localhost:8000/tools/invoke/id/tool_calculator
   Content-Type: application/json

   {"operation": "add", "values": [2, 2]}
   ```
   Note the execution_time from metadata (e.g., 0.012s)

3. **Check Cache Status**
   ```
   GET http://localhost:8000/tools/health
   ```
   Verify `executor.cached_instances >= 1`

4. **Validation Checklist**
   - [ ] Second invocation is faster than first (cached instance reused)
   - [ ] cached_instances count increased after first use
   - [ ] Both invocations return correct results

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test B6: Clear Instance Cache

**Test ID**: B6
**Test Type**: Unit
**Purpose**: Verify cache can be cleared for development/testing

#### Test Steps

1. **Prerequisite**: Run B5 to populate cache

2. **HTTP Request**
   ```
   POST http://localhost:8000/tools/cache/clear
   ```

3. **Expected Status Code**: `200 OK`

4. **Expected Response Structure**
   ```json
   {
     "status": "success",
     "cleared_instances": 1
   }
   ```

5. **Validation Checklist**
   - [ ] cleared_instances > 0
   - [ ] Subsequent health check shows cached_instances == 0

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

## Section C: Agent-Tool Binding (Integration)

### Test C1: Get Tools for Agent

**Test ID**: C1
**Test Type**: Integration
**Purpose**: Verify agent-tool relationship can be queried

#### Test Steps

1. **Find an Existing Agent** (list all agents)
   ```
   GET http://localhost:8000/agents/list
   ```
   Pick an agent_id from the response (e.g., "agt_xxx")

2. **Check Agent's Tools**
   ```
   GET http://localhost:8000/tools/agent/{agent_id}
   ```
   Replace {agent_id} with actual ID from step 1

3. **Expected Status Code**: `200 OK`

4. **Expected Response Structure**
   ```json
   {
     "agent_id": "agt_xxx",
     "agent_name": "Code Executor Agent",
     "tool_count": 2,
     "tools": [
       {
         "tool_id": "tool_calculator",
         "name": "Calculator",
         "description": "...",
         ...
       }
     ]
   }
   ```

5. **Validation Checklist**
   - [ ] agent_id matches request
   - [ ] tool_count matches length of tools array
   - [ ] Each tool has complete ToolDef structure
   - [ ] Only "active" tools are included

6. **Common Errors**
   - 404 Not Found → Agent doesn't exist, check agent_id
   - Empty tools array → Agent has no tools assigned

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test C2: Create Agent with Tool Assignment

**Test ID**: C2
**Test Type**: Integration
**Purpose**: Verify agents can be created with tools via API

#### Test Steps

1. **HTTP Request**
   ```
   POST http://localhost:8000/agents/register
   Content-Type: application/json

   {
     "name": "Test Calculator Agent",
     "role": "Mathematical Assistant",
     "description": "An agent that performs calculations",
     "tools": ["tool_calculator"],
     "llm": {
       "provider": "openrouter",
       "model": "anthropic/claude-3-5-sonnet",
       "temperature": 0.7
     },
     "input_schema": {
       "type": "object",
       "properties": {
         "calculation_request": {"type": "string"}
       }
     },
     "output_schema": {
       "type": "object",
       "properties": {
         "result": {"type": "string"}
       }
     }
   }
   ```

2. **Expected Status Code**: `201 Created` or `200 OK`

3. **Expected Response Structure**
   ```json
   {
     "agent_id": "agt_xxx",
     "status": "registered",
     "message": "Agent registered successfully"
   }
   ```

4. **Validation Checklist**
   - [ ] agent_id is returned
   - [ ] Status is "registered"
   - [ ] GET /tools/agent/{agent_id} returns tool_calculator

5. **Common Errors**
   - Tool not found → tool_id doesn't exist in registry
   - Invalid agent schema → Check required fields

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test C3: Search Tools by Query

**Test ID**: C3
**Test Type**: Unit
**Purpose**: Verify tool search functionality for agent tool selection

#### Test Steps

1. **HTTP Request**
   ```
   GET http://localhost:8000/tools/search?q=calculator
   ```

2. **Expected Status Code**: `200 OK`

3. **Expected Response Structure**
   ```json
   [
     {
       "tool_id": "tool_calculator",
       "name": "Calculator",
       "description": "Performs mathematical calculations...",
       "tool_type": "local",
       "status": "active"
     }
   ]
   ```

4. **Validation Checklist**
   - [ ] Results include tools matching "calculator" in name or description
   - [ ] Search is case-insensitive
   - [ ] Only active tools are returned

5. **Test Variations**
   - Search for "code": Should return code_generator and code_reviewer
   - Search for "web": Should return web_search
   - Search for "nonexistent": Should return empty array

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test C4: List Tools by Type

**Test ID**: C4
**Test Type**: Unit
**Purpose**: Verify tools can be filtered by execution type

#### Test Steps

1. **HTTP Request**
   ```
   GET http://localhost:8000/tools/list/type/local
   ```

2. **Expected Status Code**: `200 OK`

3. **Expected Response Structure**
   ```json
   [
     {
       "tool_id": "tool_calculator",
       "name": "Calculator",
       "description": "...",
       "tool_type": "local",
       "status": "active"
     },
     ...
   ]
   ```

4. **Validation Checklist**
   - [ ] All returned tools have tool_type == "local"
   - [ ] Count matches expected number of LOCAL tools
   - [ ] No MCP or API tools included

5. **Test Variations**
   - type=mcp → Should return MCP tools (if any registered)
   - type=api → Should return API tools (if any registered)
   - type=invalid → Should return 400 Bad Request with error

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

## Section D: Workflow Execution with Tools (E2E)

### Test D1: Single Agent Sequential Workflow with Tool

**Test ID**: D1
**Test Type**: E2E
**Purpose**: Verify tool invocation during workflow execution (Deferred Task 3.3)

#### Test Steps

1. **Create Test Workflow** (via IDE or API)
   - Workflow: Single agent (Calculator Agent) with tool_calculator assigned
   - Agent task: "Calculate the sum of 45 and 55"

2. **Execute Workflow**
   ```
   POST http://localhost:8000/workflows/execute
   Content-Type: application/json

   {
     "workflow_id": "<workflow_id>",
     "input": {
       "user_input": "Calculate the sum of 45 and 55"
     }
   }
   ```

3. **Expected Status Code**: `200 OK`

4. **Expected Response Structure**
   ```json
   {
     "status": "completed",
     "workflow_id": "<workflow_id>",
     "output": {
       "crew_result": "The sum of 45 and 55 is 100. [Tool: Calculator used]",
       "messages": [...]
     },
     "execution_time": <float>
   }
   ```

5. **Validation Checklist**
   - [ ] Status is "completed"
   - [ ] crew_result contains correct answer (100)
   - [ ] crew_result or logs indicate tool was invoked
   - [ ] No error in output
   - [ ] execution_time is reasonable

6. **Verify in Logs**
   - [ ] Check server logs for "Invoking tool 'Calculator'"
   - [ ] Check for "Tool 'Calculator' executed successfully"
   - [ ] Check for CrewAI tool binding messages

7. **Common Errors**
   - Tool not invoked → Check CrewAI adapter tool binding
   - LLM-only response without calculation → Tool wrapper might not be working
   - Timeout → Check tool execution and workflow execution timeouts

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test D2: Three-Agent Sequential Workflow with Tool Propagation

**Test ID**: D2
**Test Type**: E2E
**Purpose**: Verify tool output propagates between agents (Deferred Task 3.4)

#### Test Steps

1. **Create Test Workflow** (via IDE or API)
   - Agent 1: Code Generator (tool: code_generator)
     - Task: "Write a Python function to calculate factorial"
   - Agent 2: Code Reviewer (tool: code_reviewer)
     - Task: "Review the code from previous agent"
   - Agent 3: Code Tester (no tool)
     - Task: "Summarize the code quality assessment"

2. **Execute Workflow**
   ```
   POST http://localhost:8000/workflows/execute
   Content-Type: application/json

   {
     "workflow_id": "<workflow_id>",
     "input": {
       "user_input": "Write a factorial function, review it, and summarize"
     }
   }
   ```

3. **Expected Status Code**: `200 OK`

4. **Expected Response Structure**
   ```json
   {
     "status": "completed",
     "workflow_id": "<workflow_id>",
     "output": {
       "crew_result": "Summary: Factorial function written, reviewed with score X/100, suggestions: ...",
       "original_user_input": "Write a factorial function, review it, and summarize",
       "messages": [
         {"agent": "Agent 1", "output": "def factorial(n): ..."},
         {"agent": "Agent 2", "output": "Code review: Score 85/100..."},
         {"agent": "Agent 3", "output": "Summary: ..."}
       ]
     }
   }
   ```

5. **Validation Checklist**
   - [ ] Agent 1 output contains generated code
   - [ ] Agent 2 output contains review with tool-enhanced analysis
   - [ ] Agent 3 output references content from both Agent 1 and Agent 2
   - [ ] original_user_input is preserved
   - [ ] crew_result contains final output from Agent 3
   - [ ] All three agents executed in order

6. **Verify Tool Output Propagation**
   - [ ] Agent 2 received Agent 1's tool-enriched output (code + generator analysis)
   - [ ] Agent 3 received Agent 2's tool-enriched output (code + review results)
   - [ ] Each agent's context included `crew_result` from previous agent

7. **Common Errors**
   - Agent 2 doesn't receive Agent 1 output → Check StateGraph state propagation
   - Tool output missing → Check CrewAI result parsing and state update
   - Agents out of order → Check workflow compilation and edge creation

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test D3: Parallel Workflow with Multiple Tools

**Test ID**: D3
**Test Type**: E2E
**Purpose**: Verify multiple tools execute in parallel workflow topology

#### Test Steps

1. **Create Test Workflow**
   - Parallel agents:
     - Agent A: Calculator (tool: calculator) - "Calculate 10 + 20"
     - Agent B: Web Search (tool: web_search) - "Search for 'Python tutorial'"
   - Merge agent: Summarizer (no tool) - "Combine results"

2. **Execute Workflow**
   ```
   POST http://localhost:8000/workflows/execute
   Content-Type: application/json

   {
     "workflow_id": "<workflow_id>",
     "input": {
       "user_input": "Calculate 10+20 and search for Python tutorials"
     }
   }
   ```

3. **Expected Status Code**: `200 OK`

4. **Validation Checklist**
   - [ ] Both Agent A and Agent B executed
   - [ ] Both tools were invoked (check logs)
   - [ ] Merge agent received outputs from both agents
   - [ ] Final output includes both calculation result and search results

5. **Verify Parallel Execution**
   - [ ] Check execution_time is reasonable (not 2x sequential time)
   - [ ] Check logs show parallel agent execution

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test D4: Hierarchical Workflow with Tools

**Test ID**: D4
**Test Type**: E2E
**Purpose**: Verify tools work in hierarchical (manager-worker) topology

#### Test Steps

1. **Create Test Workflow**
   - Manager agent (no tool): Coordinates sub-agents
   - Worker 1: Calculator (tool: calculator)
   - Worker 2: File Reader (tool: file_reader)

2. **Execute Workflow**
   ```
   POST http://localhost:8000/workflows/execute
   Content-Type: application/json

   {
     "workflow_id": "<workflow_id>",
     "input": {
       "user_input": "Calculate 5*5 and read data.txt, then summarize"
     }
   }
   ```

3. **Expected Status Code**: `200 OK`

4. **Validation Checklist**
   - [ ] Manager delegated tasks to workers
   - [ ] Worker tools were invoked
   - [ ] Manager received worker outputs
   - [ ] Final output includes synthesized information

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

## Section E: Error Handling & Edge Cases

### Test E1: Tool Execution Timeout

**Test ID**: E1
**Test Type**: Unit
**Purpose**: Verify timeout handling for long-running tools

#### Test Steps

1. **Create a Mock Slow Tool** (optional - for testing only)
   - Register a tool with execution that sleeps for 70s (exceeds default 60s timeout)

2. **Invoke the Slow Tool**
   ```
   POST http://localhost:8000/tools/invoke/id/tool_slow_mock
   Content-Type: application/json

   {"delay": 70}
   ```

3. **Expected Status Code**: `200 OK` (request succeeds, but result indicates timeout)

4. **Expected Response Structure**
   ```json
   {
     "name": "Slow Mock Tool",
     "tool_id": "tool_slow_mock",
     "output": {},
     "success": false,
     "error": "Execution timed out after 60s",
     "metadata": {
       "execution_time": 60.x,
       "error_type": "timeout_error"
     }
   }
   ```

5. **Validation Checklist**
   - [ ] success == false
   - [ ] error mentions timeout
   - [ ] metadata.error_type == "timeout_error"
   - [ ] Server didn't crash or hang

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test E2: Tool Execution Failure (Invalid Input)

**Test ID**: E2
**Test Type**: Unit
**Purpose**: Verify graceful handling of invalid tool input

#### Test Steps

1. **HTTP Request** (invalid operation)
   ```
   POST http://localhost:8000/tools/invoke/id/tool_calculator
   Content-Type: application/json

   {
     "operation": "invalid_operation",
     "values": [1, 2]
   }
   ```

2. **Expected Status Code**: `400 Bad Request`

3. **Expected Response Structure**
   ```json
   {
     "detail": "Input validation failed at 'operation': ... (enum constraint)"
   }
   ```

4. **Validation Checklist**
   - [ ] Error is descriptive
   - [ ] Status code is 400 (client error)
   - [ ] Workflow doesn't crash if this happens during execution

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test E3: Tool Not Found in Agent Execution

**Test ID**: E3
**Test Type**: Integration
**Purpose**: Verify agent handles missing tool gracefully

#### Test Steps

1. **Create Agent with Non-Existent Tool**
   ```
   POST http://localhost:8000/agents/register
   Content-Type: application/json

   {
     "name": "Test Agent",
     "role": "Tester",
     "description": "Test agent",
     "tools": ["tool_nonexistent"],
     "llm": {...}
   }
   ```

2. **Execute Workflow with This Agent**

3. **Validation Checklist**
   - [ ] Agent creation succeeds (tools are not validated at registration time)
   - [ ] During workflow execution, agent runs without the tool (LLM-only mode)
   - [ ] OR: Workflow fails gracefully with clear error message
   - [ ] Server logs warning about missing tool

4. **Common Errors**
   - Server crash → Need error handling in tool binding
   - Silent failure → Need logging of tool binding failures

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test E4: Tool with Missing Execution Config

**Test ID**: E4
**Test Type**: Unit
**Purpose**: Verify validation of tool execution config

#### Test Steps

1. **Attempt to Register Invalid Tool**
   ```
   POST http://localhost:8000/tools/register
   Content-Type: application/json

   {
     "name": "Invalid Tool",
     "description": "A tool with no execution config",
     "tool_type": "local",
     "execution_config": {}
   }
   ```

2. **Expected Behavior**
   - Registration succeeds (validation is at execution time, not registration)

3. **Attempt to Invoke Invalid Tool**
   ```
   POST http://localhost:8000/tools/invoke/id/tool_invalid_tool
   Content-Type: application/json

   {"test": "data"}
   ```

4. **Expected Status Code**: `400 Bad Request` or `200 OK` with success: false

5. **Validation Checklist**
   - [ ] Error message indicates missing execution config fields
   - [ ] Error is clear: "Tool 'Invalid Tool' execution_config missing required fields"

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test E5: Workflow Execution with Tool Failure

**Test ID**: E5
**Test Type**: E2E
**Purpose**: Verify workflow handles tool failure gracefully

#### Test Steps

1. **Create Workflow with Agent Using Calculator**

2. **Execute with Input That Causes Tool Error** (e.g., divide by zero)
   ```
   POST http://localhost:8000/workflows/execute
   Content-Type: application/json

   {
     "workflow_id": "<workflow_id>",
     "input": {
       "user_input": "Calculate 10 divided by 0"
     }
   }
   ```

3. **Expected Behavior**
   - Workflow doesn't crash
   - Agent receives tool error and can respond with fallback (LLM reasoning)
   - OR: Workflow fails with clear error message

4. **Validation Checklist**
   - [ ] Workflow execution completes or fails gracefully
   - [ ] Error is logged clearly
   - [ ] Agent can explain the error to the user

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

### Test E6: Concurrent Tool Invocations

**Test ID**: E6
**Test Type**: Integration
**Purpose**: Verify system handles concurrent tool invocations

#### Test Steps

1. **Send Multiple Concurrent Requests** (use parallel curl or tool like Apache Bench)
   ```bash
   # Terminal 1
   curl -X POST http://localhost:8000/tools/invoke/id/tool_calculator \
     -H "Content-Type: application/json" \
     -d '{"operation": "add", "values": [1, 1]}'

   # Terminal 2 (simultaneously)
   curl -X POST http://localhost:8000/tools/invoke/id/tool_calculator \
     -H "Content-Type: application/json" \
     -d '{"operation": "multiply", "values": [5, 5]}'

   # Terminal 3 (simultaneously)
   curl -X POST http://localhost:8000/tools/invoke/id/tool_web_search \
     -H "Content-Type: application/json" \
     -d '{"query": "test"}'
   ```

2. **Validation Checklist**
   - [ ] All requests complete successfully
   - [ ] Results are correct (no race conditions)
   - [ ] No server errors or crashes
   - [ ] Cached instances are thread-safe

**Result**: [ ] PASS [ ] FAIL [ ] BLOCKED
**Notes**: ____________________________________________

---

## Test Results Tracking

### Summary Table

| Section | Test ID | Test Name | Status | Notes |
|---------|---------|-----------|--------|-------|
| A | A1 | Health Check | [ ] | |
| A | A2 | Tool Discovery | [ ] | |
| A | A3 | List Tools | [ ] | |
| A | A4 | Get Tool by ID | [ ] | |
| A | A5 | Manual Registration | [ ] | |
| A | A6 | Delete Tool | [ ] | |
| B | B1 | Invoke Calculator | [ ] | |
| B | B2 | Validate Input (Valid) | [ ] | |
| B | B3 | Validate Input (Invalid) | [ ] | |
| B | B4 | Invoke by Name | [ ] | |
| B | B5 | Instance Caching | [ ] | |
| B | B6 | Clear Cache | [ ] | |
| C | C1 | Get Agent Tools | [ ] | |
| C | C2 | Create Agent with Tools | [ ] | |
| C | C3 | Search Tools | [ ] | |
| C | C4 | List by Type | [ ] | |
| D | D1 | Single Agent E2E | [ ] | |
| D | D2 | Sequential 3-Agent E2E | [ ] | |
| D | D3 | Parallel Workflow | [ ] | |
| D | D4 | Hierarchical Workflow | [ ] | |
| E | E1 | Timeout Handling | [ ] | |
| E | E2 | Invalid Input | [ ] | |
| E | E3 | Missing Tool | [ ] | |
| E | E4 | Missing Config | [ ] | |
| E | E5 | Tool Failure in Workflow | [ ] | |
| E | E6 | Concurrent Invocations | [ ] | |

### Test Execution Log

| Date | Tester | Tests Run | Pass | Fail | Blocked | Notes |
|------|--------|-----------|------|------|---------|-------|
| | | | | | | |
| | | | | | | |

### Defects Found

| Defect ID | Test ID | Severity | Description | Status | Notes |
|-----------|---------|----------|-------------|--------|-------|
| | | | | | |
| | | | | | |

---

## Troubleshooting Guide

### Common Issues

#### 1. Server Not Running
**Symptom**: Connection refused errors
**Solution**:
```bash
cd echoAI
uvicorn apps.gateway.main:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Tool Discovery Returns 0 Tools
**Symptom**: `discovered_count: 0`
**Check**:
- AgentTools folder exists: `ls echoAI/AgentTools/`
- Manifests exist: `ls echoAI/AgentTools/*/tool_manifest.json`
- Manifest syntax is valid JSON
- Container.py has correct discovery_dirs path

#### 3. Tool Invocation Fails with Module Not Found
**Symptom**: `"error": "Cannot import module 'AgentTools.calculator.service'"`
**Solution**:
- Verify `__init__.py` files exist in AgentTools and subfolders
- Check Python path includes echoAI directory
- Try manual import: `python -c "from AgentTools.calculator.service import CalculatorService"`

#### 4. Input Validation Always Fails
**Symptom**: All inputs rejected as invalid
**Check**:
- jsonschema package installed: `pip install jsonschema`
- Input matches schema exactly (including types)
- Required fields are present

#### 5. Workflow Doesn't Use Tools
**Symptom**: Agent runs but never invokes tools
**Check**:
- Agent has tools assigned: GET /tools/agent/{agent_id}
- CrewAI adapter binds tools (check logs for "Binding X tools to agent")
- LLM is configured correctly
- Agent task mentions tool capability

#### 6. Timeout on Tool Invocation
**Symptom**: Tool times out after 60s
**Solutions**:
- Increase timeout in container.py: `ToolExecutor(registry=_registry, default_timeout=120)`
- Check if tool implementation has infinite loop
- Verify external dependencies (for MCP/API tools) are reachable

### Log Locations

- Server logs: Console output from uvicorn
- Tool execution logs: Look for logger messages from `apps.tool.executor`
- Workflow execution logs: Look for logger messages from `apps.workflow.runtime.executor`

### Environment Reset

If tests are inconsistent, reset the environment:

```bash
# Stop server
# Delete tool storage
rm -rf echoAI/apps/storage/tools/*

# Restart server
cd echoAI
uvicorn apps.gateway.main:app --reload --host 0.0.0.0 --port 8000

# Re-run discovery
curl -X POST http://localhost:8000/tools/discover
```

---

## Appendix: Sample Test Data

### Sample Calculator Input
```json
{
  "operation": "add",
  "values": [10, 20, 30]
}
```

### Sample Agent Registration
```json
{
  "name": "Math Agent",
  "role": "Mathematical Assistant",
  "description": "Performs calculations",
  "tools": ["tool_calculator"],
  "llm": {
    "provider": "openrouter",
    "model": "anthropic/claude-3-5-sonnet",
    "temperature": 0.7
  }
}
```

### Sample Tool Registration
```json
{
  "name": "Test Echo Tool",
  "description": "Echoes input for testing",
  "tool_type": "local",
  "input_schema": {
    "type": "object",
    "properties": {
      "message": {"type": "string"}
    },
    "required": ["message"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "echo": {"type": "string"}
    }
  },
  "execution_config": {
    "module": "builtins",
    "class": "dict",
    "method": "get"
  }
}
```

---

**End of Manual Test Specification**

**Next Steps**:
1. Execute tests in order (A → B → C → D → E)
2. Mark each test as PASS/FAIL/BLOCKED
3. Document any defects found
4. Report results to development team

**Questions?** Contact the EchoAI development team.
