"""
Meta-Prompt Generator Service
Analyzes user requests and generates structured meta-prompts for the agent system designer
"""
import json
from typing import Dict, List
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
# For Azure deployment - uncomment the line below
# from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from app.schemas.api_models import UserRequest, DomainAnalysis, MetaPromptResponse
from app.core.config import get_settings


class MetaPromptGenerator:
    """
    Analyzes user requests and generates structured meta-prompts
    for the agent system designer LLM

    This is Step 1 of the two-step LLM process:
    1. MetaPromptGenerator (Analyzer) - Understands intent, extracts structure
    2. AgentSystemDesigner (Designer) - Designs complete agent system
    """

    def __init__(self):
        settings = get_settings()

        # Use OpenRouter for analysis (low temperature for consistency)
        # For Azure deployment - uncomment this block
        # self.analyzer_llm = AzureChatOpenAI(
        #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        #     temperature=0.3,
        #     max_tokens=4000
        # )

        # For local/OpenRouter - comment this when deploying to Azure
        self.analyzer_llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=settings.OPENROUTER_API_KEY or "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",
            model=settings.DEFAULT_LLM_MODEL,
            temperature=0.3,
            max_tokens=4000
        )

    async def analyze_request(self, user_request: UserRequest) -> DomainAnalysis:
        """
        Step 1: Analyze the user's natural language request
        Extract domain, entities, operations, and constraints
        """

        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert system analyst specializing in multi-agent workflow design.

Your task: Analyze the user's request and extract structured information that will be used to design a multi-agent system.

Extract the following information:

1. **Primary domain**: The main area this system operates in (e.g., inventory, sales, customer_service, finance, healthcare, logistics, etc.)

2. **Key entities**: The main objects/concepts involved (e.g., products, orders, customers, invoices, tickets, etc.)

3. **Required operations**: The actions that need to be performed (e.g., track, analyze, report, alert, predict, validate, process, etc.)

4. **Data sources needed**: Where data comes from (e.g., database, api, file, user_input, sensor, webhook, etc.)

5. **Output requirements**: What the system produces (e.g., dashboard, report, alerts, api_response, file, email, etc.)

6. **Temporal requirements**: When things happen (e.g., real-time, daily, weekly, on-demand, event-driven, etc.)

7. **Integration points**: External systems to connect with (e.g., email, slack, webhook, third_party_api, etc.)

Based on the analysis, also determine:

8. **Suggested patterns**: Recommend communication patterns that fit the use case:
   - "event-driven" - for monitoring and alerting scenarios
   - "sequential" - for step-by-step processing pipelines
   - "parallel" - for independent simultaneous tasks
   - "hierarchical" - for master-worker coordination
   - "conditional" - for decision-based routing

9. **Complexity score**: Rate the system complexity from 1-10 based on:
   - Number of entities (+2 points each)
   - Number of operations (+1.5 points each)
   - Number of data sources (+1 point each)
   - Integration requirements (+2 points for external integrations)

Output ONLY a valid JSON object with these exact fields:
{{
  "original_request": "the original user request",
  "domain": "primary domain",
  "entities": ["entity1", "entity2"],
  "operations": ["operation1", "operation2"],
  "data_sources": ["source1", "source2"],
  "output_requirements": ["output1", "output2"],
  "temporal_requirements": "timing description",
  "integration_points": ["integration1", "integration2"],
  "suggested_patterns": ["pattern1", "pattern2"],
  "complexity_score": 5
}}

Do not include any explanation or markdown formatting. Only output the JSON object."""),
            ("user", "{request}")
        ])

        response = await self.analyzer_llm.ainvoke(
            analysis_prompt.format_messages(request=user_request.request)
        )

        # Parse LLM response to structured format
        try:
            analysis_dict = json.loads(response.content)

            # Create DomainAnalysis model
            analysis = DomainAnalysis(**analysis_dict)

            return analysis

        except json.JSONDecodeError as e:
            # Fallback: create basic analysis
            return self._create_fallback_analysis(user_request.request)

    def _create_fallback_analysis(self, request: str) -> DomainAnalysis:
        """Create a basic analysis if LLM response parsing fails"""
        return DomainAnalysis(
            original_request=request,
            domain="general",
            entities=["data", "task"],
            operations=["process", "analyze"],
            data_sources=["user_input"],
            output_requirements=["result"],
            temporal_requirements="on-demand",
            integration_points=[],
            suggested_patterns=["sequential"],
            complexity_score=3
        )

    def _suggest_patterns(self, analysis: DomainAnalysis) -> List[str]:
        """
        Suggest communication patterns based on analysis
        (This is a backup heuristic in case LLM doesn't provide good suggestions)
        """
        patterns = []

        operations = [op.lower() for op in analysis.operations]

        # Event-driven patterns
        if any(op in operations for op in ["track", "monitor", "alert", "detect"]):
            patterns.append("event-driven")

        # Sequential patterns
        if any(op in operations for op in ["analyze", "process", "transform", "report"]):
            patterns.append("sequential")

        # Parallel patterns
        if len(analysis.entities) > 3 or any(op in operations for op in ["distribute", "parallel", "concurrent"]):
            patterns.append("parallel")

        # Hierarchical patterns
        if any(op in operations for op in ["coordinate", "orchestrate", "manage", "delegate"]):
            patterns.append("hierarchical")

        return patterns if patterns else ["sequential"]

    def _calculate_complexity(self, analysis: DomainAnalysis) -> int:
        """
        Calculate system complexity score (1-10)
        (This is a backup heuristic)
        """
        score = 0
        score += len(analysis.entities) * 2
        score += len(analysis.operations) * 1.5
        score += len(analysis.data_sources) * 1
        score += len(analysis.integration_points) * 2

        return min(int(score), 10)

    async def generate_meta_prompt(self, analysis: DomainAnalysis) -> str:
        """
        Step 2: Generate a structured meta-prompt for the designer LLM
        This prompt will instruct the designer to create a complete agent system
        """

        meta_prompt = f"""# Agent System Design Request

## User Requirement
{analysis.original_request}

## Domain Analysis
- **Primary Domain**: {analysis.domain}
- **Key Entities**: {', '.join(analysis.entities)}
- **Required Operations**: {', '.join(analysis.operations)}
- **Data Sources**: {', '.join(analysis.data_sources)}
- **Output Requirements**: {', '.join(analysis.output_requirements)}
- **Temporal Requirements**: {analysis.temporal_requirements}
- **Integration Points**: {', '.join(analysis.integration_points) if analysis.integration_points else 'None'}

## System Constraints
- **Complexity Level**: {analysis.complexity_score}/10
- **Suggested Patterns**: {', '.join(analysis.suggested_patterns)}

## Design Task
Design a multi-agent system that fulfills the above requirements.

### Agent Specifications Required
For each agent, provide:

1. **id**: Unique identifier (lowercase, underscore-separated, e.g., "inventory_tracker")
2. **name**: Human-readable name (e.g., "Inventory Tracker")
3. **role**: Concise role description (e.g., "Monitor stock levels")
4. **responsibilities**: List of 3-5 specific tasks this agent handles
5. **system_prompt**: Detailed instructions for the agent's LLM (minimum 50 characters)
   - Include context about the agent's role
   - Specify expected behavior
   - Define interaction patterns
6. **tools**: List of tool names this agent requires (e.g., ["database_query", "email_sender"])
7. **llm_config**: LLM configuration
   - **model**: Choose appropriate model (e.g., "gpt-4", "claude-sonnet-4-5-20250929", "gpt-3.5-turbo")
   - **temperature**: 0.0-2.0 (lower for deterministic tasks, higher for creative tasks)
   - **max_tokens**: 100-128000 (based on task complexity)
   - **top_p**: 0.0-1.0 (default 1.0)
8. **input_schema**: Expected input structure (optional but recommended)
9. **output_schema**: Expected output structure (optional but recommended)
10. **is_master**: Set to true for master/coordinator agents (default false)

### Tool Specifications Required
For each tool referenced by agents, provide:

1. **name**: Unique tool identifier (e.g., "database_query")
2. **type**: Tool category - one of: "database", "api", "calculator", "email", "search", "file", "custom"
3. **description**: What this tool does
4. **config**: Tool-specific configuration (e.g., connection details, endpoints)
5. **auth_required**: Whether authentication is needed (true/false)
6. **parameters**: Parameter schema for the tool (optional)

### Workflow Specifications Required
For each workflow, provide:

1. **name**: Unique workflow identifier (e.g., "stock_monitoring_workflow")
2. **description**: What this workflow accomplishes
3. **trigger**: How this workflow starts - one of: "manual", "scheduled", "event", "webhook"
4. **steps**: Ordered list of workflow steps, where each step has:
   - **agent_id**: Which agent executes this step
   - **action**: What action to perform
   - **inputs**: Input data for this step (optional)
   - **condition**: Conditional execution logic (optional, e.g., "if stock < threshold")
   - **parallel_with**: List of step IDs to execute in parallel (optional)
5. **communication_pattern**: How steps execute - one of: "sequential", "parallel", "conditional", "graph"

### Communication Architecture
- Define how agents communicate (message passing, shared state, etc.)
- Specify any required synchronization points
- Define error handling strategy

### Success Criteria
- Define how to measure if the system meets requirements
- Specify monitoring metrics

## Output Format
Provide the complete design as a valid JSON object matching this schema:

```json
{{
  "system_name": "string (unique name for this system)",
  "description": "string (what this system does)",
  "domain": "{analysis.domain}",
  "agents": [
    {{
      "id": "string (lowercase_underscore)",
      "name": "string (human-readable)",
      "role": "string (concise role)",
      "responsibilities": ["string", "string", ...],
      "system_prompt": "string (detailed LLM instructions, min 50 chars)",
      "tools": ["tool_name1", "tool_name2"],
      "llm_config": {{
        "model": "string (model identifier)",
        "temperature": number (0.0-2.0),
        "max_tokens": number (100-128000),
        "top_p": number (0.0-1.0)
      }},
      "input_schema": {{}} (optional),
      "output_schema": {{}} (optional),
      "is_master": boolean (default false)
    }}
  ],
  "tools": [
    {{
      "name": "string (unique tool name)",
      "type": "database|api|calculator|email|search|file|custom",
      "description": "string (what this tool does)",
      "config": {{}} (tool-specific config),
      "auth_required": boolean,
      "parameters": {{}} (optional)
    }}
  ],
  "workflows": [
    {{
      "name": "string (unique workflow name)",
      "description": "string (what this accomplishes)",
      "trigger": "manual|scheduled|event|webhook",
      "steps": [
        {{
          "agent_id": "string (agent id)",
          "action": "string (action to perform)",
          "inputs": {{}} (optional),
          "condition": "string (optional conditional logic)",
          "parallel_with": ["step_id"] (optional)
        }}
      ],
      "communication_pattern": "sequential|parallel|conditional|graph"
    }}
  ],
  "communication_pattern": "sequential|parallel|hierarchical|conditional|graph"
}}
```

## Design Principles

1. **Modularity**: Each agent should have a clear, focused responsibility
2. **Scalability**: Design should handle increased load gracefully
3. **Observability**: Include logging and monitoring points
4. **Error Handling**: Define fallback strategies
5. **Cost Efficiency**: Choose appropriate models for each task complexity
   - Use cheaper models (gpt-3.5-turbo) for simple tasks
   - Use powerful models (gpt-4, claude-sonnet-4-5) for complex reasoning
6. **User Experience**: Consider response times and feedback mechanisms

## Important Notes

- Ensure all agent IDs are unique and follow the pattern: lowercase_with_underscores
- Ensure all tool names referenced in agents' "tools" arrays are defined in the "tools" section
- Ensure all agent_ids referenced in workflow steps exist in the "agents" section
- Set is_master=true for exactly one agent in hierarchical patterns
- For parallel workflows, use "parallel_with" in steps to indicate concurrent execution
- For conditional workflows, use "condition" in steps to specify when they should run

Generate the complete agent system design now. Output ONLY the JSON object, no markdown formatting or explanation.
"""

        return meta_prompt

    async def generate(self, user_request: UserRequest) -> MetaPromptResponse:
        """
        Complete meta-prompt generation pipeline

        1. Analyze user request → structured domain analysis
        2. Generate meta-prompt from analysis

        Returns:
            MetaPromptResponse containing analysis and meta-prompt
        """
        # Step 1: Analyze request
        analysis = await self.analyze_request(user_request)

        # Step 2: Generate meta-prompt
        meta_prompt = await self.generate_meta_prompt(analysis)

        return MetaPromptResponse(
            analysis=analysis,
            meta_prompt=meta_prompt
        )
