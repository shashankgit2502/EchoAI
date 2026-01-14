"""
Workflow Designer Service
Takes meta-prompt and generates complete agent system design using LLM
"""
import json
import os  # For Azure deployment environment variables
import re
from typing import Optional
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
# For Azure deployment - uncomment the line below
# from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from app.schemas.api_models import (
    AgentSystemDesign,
    DomainAnalysis,
    UserRequest,
    MetaPromptResponse
)
from app.services.meta_prompt_generator import MetaPromptGenerator
from app.services.llm_provider import get_llm_provider
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class WorkflowDesigner:
    """
    Agent System Designer - Step 2 of the two-step LLM process

    Takes a meta-prompt from MetaPromptGenerator and uses an LLM to design
    a complete multi-agent system with:
    - Agent specifications (roles, tools, LLM configs, system prompts)
    - Tool definitions
    - Workflow steps
    - Communication patterns
    """

    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self._llm_provider = get_llm_provider()

        # Use Claude Sonnet 4.5 for design (balance of power and cost)
        # Can be configurable to use GPT-4, Gemini, etc.
        # self.designer_llm = ChatAnthropic(
        #     model="claude-sonnet-4-5-20250929",
        #     temperature=0.7,  # Higher temp for creative design
        #     max_tokens=16000  # Need lots of tokens for complex JSON output
        # )

        # For Azure deployment - uncomment this block
        # self.designer_llm = AzureChatOpenAI(
        #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        #     temperature=0.7,
        #     max_tokens=16000
        # )

        # For local/OpenRouter - comment this when deploying to Azure
        self.designer_llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=settings.OPENROUTER_API_KEY or "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",
            model=settings.DEFAULT_LLM_MODEL
        )

        # Fallback to OpenAI if needed (disabled if no OpenAI key)
        self.fallback_llm = None
        if settings.OPENAI_API_KEY:
            # For Azure deployment - uncomment this block
            # self.fallback_llm = AzureChatOpenAI(
            #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            #     temperature=0.7,
            #     max_tokens=16000
            # )

            # For local/standard OpenAI - comment this when deploying to Azure
            self.fallback_llm = ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model="gpt-4",
                temperature=0.7,
                max_tokens=16000
            )

        # Initialize meta-prompt generator
        self.meta_prompt_gen = MetaPromptGenerator()

    async def design_from_user_request(
        self,
        user_request: UserRequest
    ) -> tuple[AgentSystemDesign, DomainAnalysis, str]:
        """
        Complete pipeline: User request → Agent system design

        1. Generate meta-prompt (via MetaPromptGenerator)
        2. Send to designer LLM
        3. Parse and validate response

        Returns:
            Tuple of (agent_system, domain_analysis, meta_prompt_used)
        """
        logger.info(f"Starting workflow design for request: {user_request.request[:100]}...")

        # Step 1: Generate meta-prompt
        meta_response = await self.meta_prompt_gen.generate(user_request)

        logger.info(f"Domain analysis complete: {meta_response.analysis.domain}, "
                   f"complexity: {meta_response.analysis.complexity_score}/10")

        # Step 2: Design agent system
        agent_system = await self.design_from_meta_prompt(
            meta_response.meta_prompt,
            meta_response.analysis
        )

        return agent_system, meta_response.analysis, meta_response.meta_prompt

    async def design_from_meta_prompt(
        self,
        meta_prompt: str,
        analysis: Optional[DomainAnalysis] = None
    ) -> AgentSystemDesign:
        """
        Design agent system from a meta-prompt

        Args:
            meta_prompt: Structured prompt for the designer LLM
            analysis: Optional domain analysis for enrichment

        Returns:
            Complete AgentSystemDesign
        """
        logger.info("Sending meta-prompt to designer LLM...")

        # Create the design prompt
        design_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert multi-agent system architect.

Your task: Design a complete, production-ready multi-agent system based on the specifications provided.

CRITICAL RULES:
1. Output ONLY valid JSON - no markdown, no explanations, no ```json blocks
2. Follow the exact schema provided in the meta-prompt
3. Ensure all agent IDs are unique and lowercase_with_underscores
4. Ensure all tools referenced by agents are defined in the tools section
5. Ensure all agent_ids in workflow steps exist in the agents section
6. Create realistic, detailed system_prompts for each agent (minimum 50 characters)
7. Use only the configured default LLM model unless the user explicitly requests another:
   - {default_model}
8. Set appropriate temperatures:
   - 0.0-0.3 for deterministic tasks (data processing, validation)
   - 0.5-0.8 for balanced tasks (analysis, reporting)
   - 0.9-1.5 for creative tasks (content generation)

Design a complete, coherent system that solves the user's problem effectively."""),
            ("user", "{meta_prompt}")
        ])

        try:
            # Invoke designer LLM
            response = await self.designer_llm.ainvoke(
                design_prompt.format_messages(
                    meta_prompt=meta_prompt,
                    default_model=self.settings.DEFAULT_LLM_MODEL
                )
            )

            # Extract JSON from response
            agent_system_json = self._extract_json(response.content)

            logger.info(f"Designer LLM response received, parsing JSON...")

            # Fix common LLM mistakes before validation
            agent_system_json = self._fix_agent_permissions(agent_system_json)

            # Parse to Pydantic model
            agent_system = AgentSystemDesign(**agent_system_json)

            # Normalize models to available/default to avoid unusable providers
            self._normalize_agent_models(agent_system)

            # Enrich with metadata if analysis provided
            if analysis:
                agent_system.metadata = {
                    "complexity_score": analysis.complexity_score,
                    "suggested_patterns": analysis.suggested_patterns,
                    "domain_analysis": analysis.dict()
                }

            logger.info(f"Agent system designed: {agent_system.system_name}, "
                       f"{len(agent_system.agents)} agents, "
                       f"{len(agent_system.workflows)} workflows")

            return agent_system

        except Exception as e:
            logger.error(f"Designer LLM failed: {e}")

            # Try fallback LLM if available
            if self.fallback_llm:
                try:
                    logger.info("Attempting fallback with GPT-4...")
                    response = await self.fallback_llm.ainvoke(
                        design_prompt.format_messages(
                            meta_prompt=meta_prompt,
                            default_model=self.settings.DEFAULT_LLM_MODEL
                        )
                    )

                    agent_system_json = self._extract_json(response.content)
                    agent_system_json = self._fix_agent_permissions(agent_system_json)
                    agent_system = AgentSystemDesign(**agent_system_json)

                    self._normalize_agent_models(agent_system)

                    if analysis:
                        agent_system.metadata = {
                            "complexity_score": analysis.complexity_score,
                            "suggested_patterns": analysis.suggested_patterns,
                            "domain_analysis": analysis.dict(),
                            "used_fallback_llm": True
                        }

                    return agent_system

                except Exception as fallback_error:
                    logger.error(f"Fallback LLM also failed: {fallback_error}")
            else:
                logger.warning("No fallback LLM configured (missing OPENAI_API_KEY)")

            # Last resort: create minimal valid system
            return self._create_minimal_system(analysis)

    def _extract_json(self, content: str) -> dict:
        """
        Extract JSON from LLM response, handling various formats

        Handles:
        - Plain JSON
        - JSON in ```json``` code blocks
        - JSON with explanatory text before/after
        """
        # Remove markdown code blocks if present
        content = re.sub(r'^```json\s*\n', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n```\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^```\s*\n', '', content, flags=re.MULTILINE)

        # Try to find JSON object
        # Look for outermost { }
        start = content.find('{')
        end = content.rfind('}') + 1

        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")

        json_str = content[start:end]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Content: {json_str[:500]}...")
            raise ValueError(f"Invalid JSON in response: {e}")

    def _create_minimal_system(
        self,
        analysis: Optional[DomainAnalysis] = None
    ) -> AgentSystemDesign:
        """
        Create a minimal valid agent system as fallback

        This is used when both primary and fallback LLMs fail
        """
        logger.warning("Creating minimal fallback agent system")

        domain = analysis.domain if analysis else "general"
        entities = analysis.entities if analysis else ["data"]
        operations = analysis.operations if analysis else ["process"]

        return AgentSystemDesign(
            system_name=f"{domain.title()} System",
            description=f"Minimal agent system for {domain} domain",
            domain=domain,
            agents=[
                {
                    "id": "primary_agent",
                    "name": "Primary Agent",
                    "role": f"{operations[0].title()} Specialist",
                    "responsibilities": [
                        f"Handle {entity} {operations[0]}"
                        for entity in entities[:3]
                    ],
                    "system_prompt": f"You are a specialist agent responsible for {operations[0]} operations in the {domain} domain. Process requests carefully and provide accurate results.",
                    "tools": [],
                    "llm_config": {
                        "model": self.settings.DEFAULT_LLM_MODEL,
                        "temperature": self.settings.DEFAULT_LLM_TEMPERATURE,
                        "max_tokens": self.settings.DEFAULT_LLM_MAX_TOKENS,
                        "top_p": 1.0
                    },
                    "is_master": False
                }
            ],
            tools=[],
            workflows=[
                {
                    "name": "main_workflow",
                    "description": f"Primary workflow for {domain} operations",
                    "trigger": "manual",
                    "steps": [
                        {
                            "agent_id": "primary_agent",
                            "action": operations[0] if operations else "process",
                            "inputs": {}
                        }
                    ],
                    "communication_pattern": "sequential"
                }
            ],
            communication_pattern="sequential",
            metadata={
                "is_fallback": True,
                "reason": "LLM design failed, using minimal system"
            }
        )

    def _fix_agent_permissions(self, system_json: dict) -> dict:
        """
        Fix common LLM mistakes in agent permissions

        Common issues:
        1. Setting can_call_agents to boolean instead of list
        2. Missing permissions field
        3. Inconsistent hierarchical setup
        4. Missing is_master for hierarchical workflows
        """
        agents = system_json.get("agents", [])
        communication_pattern = system_json.get("communication_pattern", "sequential")

        # Get all agent IDs
        all_agent_ids = [agent.get("id") for agent in agents]

        # Find master agents
        master_agents = [agent for agent in agents if agent.get("is_master", False)]

        # CRITICAL FIX: For hierarchical workflows, ensure at least one master agent exists
        if communication_pattern == "hierarchical" and len(master_agents) == 0 and len(agents) > 0:
            # Try to find the coordinator/orchestrator agent by common naming patterns
            coordinator_agent = None
            coordinator_keywords = ["coordinator", "orchestrator", "master", "manager", "supervisor", "planner"]

            for agent in agents:
                agent_id = agent.get("id", "").lower()
                agent_role = agent.get("role", "").lower()
                agent_name = agent.get("name", "").lower()

                # Check if any keyword matches
                for keyword in coordinator_keywords:
                    if keyword in agent_id or keyword in agent_role or keyword in agent_name:
                        coordinator_agent = agent
                        break
                if coordinator_agent:
                    break

            # If no coordinator found by name, use the first agent
            if coordinator_agent is None:
                coordinator_agent = agents[0]

            # Set is_master for the identified coordinator
            coordinator_agent["is_master"] = True
            master_agents = [coordinator_agent]
            logger.info(f"Auto-set is_master=true for agent '{coordinator_agent.get('id')}' in hierarchical workflow")

        for agent in agents:
            # Ensure permissions field exists
            if "permissions" not in agent:
                agent["permissions"] = {}

            permissions = agent["permissions"]

            # Fix can_call_agents if it's a boolean
            if "can_call_agents" in permissions:
                call_agents_value = permissions["can_call_agents"]

                # If it's True (boolean), convert to list of all other agents
                if isinstance(call_agents_value, bool):
                    if call_agents_value:
                        # If True, allow calling all other agents
                        permissions["can_call_agents"] = [
                            aid for aid in all_agent_ids if aid != agent.get("id")
                        ]
                        logger.info(f"Fixed can_call_agents for {agent.get('id')}: converted True to agent list")
                    else:
                        # If False, set to empty list
                        permissions["can_call_agents"] = []
                        logger.info(f"Fixed can_call_agents for {agent.get('id')}: converted False to []")

            # For hierarchical workflows, ensure master agents can call all sub-agents
            if communication_pattern == "hierarchical" and agent.get("is_master", False):
                if "can_call_agents" not in permissions or not permissions["can_call_agents"]:
                    # Master agent should be able to call all other agents
                    permissions["can_call_agents"] = [
                        aid for aid in all_agent_ids if aid != agent.get("id")
                    ]
                    logger.info(f"Set can_call_agents for master agent {agent.get('id')}")

                # Ensure master can delegate
                if "can_delegate" not in permissions:
                    permissions["can_delegate"] = True

            # For non-master agents in hierarchical workflows
            if communication_pattern == "hierarchical" and not agent.get("is_master", False):
                # Sub-agents shouldn't delegate or call other agents directly
                if "can_delegate" not in permissions:
                    permissions["can_delegate"] = False

                if "can_call_agents" not in permissions:
                    permissions["can_call_agents"] = []

            # Set default max_tool_calls if missing
            if "max_tool_calls" not in permissions:
                permissions["max_tool_calls"] = 50 if agent.get("is_master", False) else 10

        return system_json

    def _normalize_agent_models(self, agent_system: AgentSystemDesign) -> None:
        """
        Ensure agents use an available model; fall back to default if not.
        """
        available_ids = {m.id for m in self._llm_provider.list_available_models()}
        available_prefixed = {f"openrouter/{m.id}" for m in self._llm_provider.list_available_models()}

        for agent in agent_system.agents:
            model_id = agent.llm_config.model
            if model_id in available_ids or model_id in available_prefixed:
                continue
            agent.llm_config.model = self.settings.DEFAULT_LLM_MODEL

    async def modify_agent_system(
        self,
        current_system: AgentSystemDesign,
        modification_request: str
    ) -> AgentSystemDesign:
        """
        Modify an existing agent system based on user feedback

        This allows HITL (Human-in-the-Loop) modifications.
        Uses targeted modification approach - returns only changes, not full system.
        """
        logger.info(f"Modifying agent system: {current_system.system_name}")
        logger.info(f"Modification request: {modification_request}")

        # Use targeted modification prompt that returns ONLY the changes
        modification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are modifying an existing multi-agent system based on user feedback.

CRITICAL RULES:
1. Return ONLY a JSON object containing the specific changes to apply
2. Do NOT regenerate the entire system - only specify what needs to change
3. Preserve all existing fields and values unless explicitly asked to change them
4. Use the default model: {default_model}

OUTPUT FORMAT - Return a JSON object with these optional fields:
{{
  "changes": {{
    "system_name": "new name (only if changing)",
    "description": "new description (only if changing)",
    "communication_pattern": "new pattern (only if changing)",
    "agent_modifications": [
      {{
        "agent_id": "id of agent to modify",
        "action": "modify" | "add" | "remove",
        "changes": {{
          "name": "new name",
          "role": "new role",
          "system_prompt": "new prompt",
          "tools": ["updated", "tools"],
          "llm_config": {{ "model": "new model", "temperature": 0.7 }}
        }}
      }}
    ],
    "workflow_modifications": [
      {{
        "workflow_name": "name of workflow to modify",
        "action": "modify" | "add" | "remove",
        "changes": {{
          "description": "new description",
          "steps": [...]
        }}
      }}
    ]
  }},
  "reasoning": "Brief explanation of what was changed and why"
}}

For simple changes like updating an agent's prompt or tools, only include those specific fields.

Current system summary:
- Name: {system_name}
- Agents: {agent_list}
- Pattern: {communication_pattern}

User modification request:
{modification_request}

Return ONLY the JSON object with targeted changes."""),
            ("user", "Apply the requested modifications. Return ONLY the changes needed, not the full system.")
        ])

        try:
            # Build agent list summary
            agent_list = ", ".join([f"{a.id} ({a.role})" for a in current_system.agents])

            response = await self.designer_llm.ainvoke(
                modification_prompt.format_messages(
                    system_name=current_system.system_name,
                    agent_list=agent_list,
                    communication_pattern=current_system.communication_pattern,
                    modification_request=modification_request,
                    default_model=self.settings.DEFAULT_LLM_MODEL
                )
            )

            changes_json = self._extract_json(response.content)
            logger.info(f"Received changes: {changes_json}")

            # Apply targeted changes to the current system
            modified_system = self._apply_targeted_changes(current_system, changes_json)

            # Fix common LLM mistakes before validation
            modified_dict = modified_system.model_dump()
            modified_dict = self._fix_agent_permissions(modified_dict)
            modified_system = AgentSystemDesign(**modified_dict)

            self._normalize_agent_models(modified_system)

            logger.info(f"System modified successfully: {modified_system.system_name}")

            return modified_system

        except Exception as e:
            logger.error(f"Targeted modification failed: {e}", exc_info=True)
            # Fallback to full regeneration approach if targeted fails
            logger.info("Falling back to full regeneration approach...")
            return await self._modify_full_regeneration(current_system, modification_request)

    async def _modify_full_regeneration(
        self,
        current_system: AgentSystemDesign,
        modification_request: str
    ) -> AgentSystemDesign:
        """
        Fallback: Full regeneration approach for complex modifications
        """
        logger.info("Using full regeneration approach for modification")

        modification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are modifying an existing multi-agent system based on user feedback.

CRITICAL RULES:
1. Preserve the existing system structure unless explicitly asked to change it
2. Maintain all agent IDs unless renaming is requested
3. Output ONLY valid JSON following the same schema
4. Ensure backward compatibility where possible
5. Use the default model unless the user explicitly requests another: {default_model}

Current system:
```json
{current_system_json}
```

User modification request:
{modification_request}

Provide the COMPLETE modified system as JSON."""),
            ("user", "Apply the requested modifications and return the complete updated system.")
        ])

        try:
            response = await self.designer_llm.ainvoke(
                modification_prompt.format_messages(
                    current_system_json=current_system.model_dump_json(indent=2),
                    modification_request=modification_request,
                    default_model=self.settings.DEFAULT_LLM_MODEL
                )
            )

            modified_json = self._extract_json(response.content)
            modified_json = self._fix_agent_permissions(modified_json)
            modified_system = AgentSystemDesign(**modified_json)
            self._normalize_agent_models(modified_system)

            logger.info(f"System modified (full regeneration): {modified_system.system_name}")
            return modified_system

        except Exception as e:
            logger.error(f"Full regeneration modification failed: {e}")
            return current_system

    def _apply_targeted_changes(
        self,
        current_system: AgentSystemDesign,
        changes_json: dict
    ) -> AgentSystemDesign:
        """
        Apply targeted changes to the current system without regenerating everything
        """
        import copy

        # Deep copy current system to avoid mutation
        system_dict = current_system.model_dump()

        changes = changes_json.get("changes", changes_json)

        # Apply system-level changes
        if "system_name" in changes:
            system_dict["system_name"] = changes["system_name"]

        if "description" in changes:
            system_dict["description"] = changes["description"]

        if "communication_pattern" in changes:
            system_dict["communication_pattern"] = changes["communication_pattern"]

        # Apply agent modifications
        agent_mods = changes.get("agent_modifications", [])
        for mod in agent_mods:
            agent_id = mod.get("agent_id")
            action = mod.get("action", "modify")
            agent_changes = mod.get("changes", {})

            if action == "modify":
                # Find and update existing agent
                for i, agent in enumerate(system_dict["agents"]):
                    if agent["id"] == agent_id:
                        # Apply only specified changes
                        for key, value in agent_changes.items():
                            if key == "llm_config" and isinstance(value, dict):
                                # Merge llm_config
                                if "llm_config" not in agent:
                                    agent["llm_config"] = {}
                                agent["llm_config"].update(value)
                            else:
                                agent[key] = value
                        break

            elif action == "add":
                # Build role for new agent
                role = agent_changes.get("role", "Agent")

                # Generate default responsibilities if not provided or empty
                responsibilities = agent_changes.get("responsibilities", [])
                if not responsibilities:
                    # Generate meaningful default responsibility from role
                    responsibilities = [f"Handle {role.lower()} tasks for the workflow"]

                # Generate unique ID for new agent
                # Priority: 1) name-based ID, 2) provided agent_id (if unique), 3) auto-generate
                existing_ids = {a["id"] for a in system_dict["agents"]}

                new_id = None
                if "name" in agent_changes:
                    # Convert name to valid ID format: lowercase, replace spaces with underscores
                    name_based_id = agent_changes["name"].lower().replace(" ", "_").replace("-", "_")
                    # Remove any non-alphanumeric characters except underscores
                    name_based_id = "".join(c for c in name_based_id if c.isalnum() or c == "_")
                    if name_based_id and name_based_id not in existing_ids:
                        new_id = name_based_id

                if not new_id and agent_id not in existing_ids:
                    new_id = agent_id

                if not new_id:
                    # Auto-generate unique ID
                    base_id = agent_changes.get("name", "agent").lower().replace(" ", "_")[:20]
                    counter = 1
                    new_id = f"{base_id}_{counter}"
                    while new_id in existing_ids:
                        counter += 1
                        new_id = f"{base_id}_{counter}"

                # Add new agent with all required fields
                new_agent = {
                    "id": new_id,
                    "name": agent_changes.get("name", new_id.replace("_", " ").title()),
                    "role": role,
                    "responsibilities": responsibilities,
                    "system_prompt": agent_changes.get("system_prompt", f"You are the {new_id} agent. Your role is to {role.lower()}."),
                    "tools": agent_changes.get("tools", []),
                    "llm_config": agent_changes.get("llm_config", {
                        "model": self.settings.DEFAULT_LLM_MODEL,
                        "temperature": self.settings.DEFAULT_LLM_TEMPERATURE,
                        "max_tokens": self.settings.DEFAULT_LLM_MAX_TOKENS,
                        "top_p": 1.0
                    }),
                    "is_master": agent_changes.get("is_master", False)
                }

                # Ensure llm_config has all required fields
                if "top_p" not in new_agent["llm_config"]:
                    new_agent["llm_config"]["top_p"] = 1.0
                if "max_tokens" not in new_agent["llm_config"]:
                    new_agent["llm_config"]["max_tokens"] = self.settings.DEFAULT_LLM_MAX_TOKENS

                system_dict["agents"].append(new_agent)

            elif action == "remove":
                # Remove agent
                system_dict["agents"] = [
                    a for a in system_dict["agents"] if a["id"] != agent_id
                ]

        # Apply workflow modifications
        workflow_mods = changes.get("workflow_modifications", [])
        for mod in workflow_mods:
            workflow_name = mod.get("workflow_name")
            action = mod.get("action", "modify")
            workflow_changes = mod.get("changes", {})

            if action == "modify":
                for i, workflow in enumerate(system_dict["workflows"]):
                    if workflow["name"] == workflow_name:
                        for key, value in workflow_changes.items():
                            # Transform workflow steps if provided with wrong format
                            if key == "steps" and isinstance(value, list):
                                transformed_steps = self._transform_workflow_steps(value)
                                workflow[key] = transformed_steps
                            else:
                                workflow[key] = value
                        break

        # CRITICAL: Sync workflow communication_pattern with system pattern
        # When system pattern changes, workflows should match
        if "communication_pattern" in changes:
            new_pattern = changes["communication_pattern"]
            for workflow in system_dict["workflows"]:
                workflow["communication_pattern"] = new_pattern
            logger.info(f"Synced workflow communication_pattern to: {new_pattern}")

        # CRITICAL: Validate and regenerate workflow steps if agents changed
        # Get current agent IDs after all modifications
        current_agent_ids = {agent["id"] for agent in system_dict["agents"]}

        for workflow in system_dict["workflows"]:
            # Check if any step references a non-existent agent
            steps_valid = all(
                step.get("agent_id") in current_agent_ids
                for step in workflow.get("steps", [])
            )

            if not steps_valid:
                logger.info(f"Regenerating workflow steps for '{workflow['name']}' - agent references invalid")
                # Regenerate steps based on current agents and communication pattern
                workflow["steps"] = self._generate_workflow_steps(
                    agents=system_dict["agents"],
                    communication_pattern=workflow.get("communication_pattern", "sequential")
                )

        return AgentSystemDesign(**system_dict)

    def _generate_workflow_steps(self, agents: list, communication_pattern: str) -> list:
        """
        Generate workflow steps based on available agents and communication pattern.

        For parallel: All agents run concurrently
        For sequential: Agents run one after another
        For hierarchical: Master agent delegates to workers
        """
        steps = []
        agent_ids = [agent["id"] for agent in agents]

        if communication_pattern == "parallel":
            # Parallel: All agents execute concurrently
            # Each agent runs with parallel_with pointing to all other agents
            for i, agent in enumerate(agents):
                other_agents = [a["id"] for a in agents if a["id"] != agent["id"]]
                steps.append({
                    "agent_id": agent["id"],
                    "action": agent.get("role", "process").lower().replace(" ", "_"),
                    "inputs": {},
                    "parallel_with": other_agents
                })
            logger.info(f"Generated {len(steps)} parallel workflow steps")

        elif communication_pattern == "hierarchical":
            # Hierarchical: Master first, then workers, then master aggregates
            master_agents = [a for a in agents if a.get("is_master", False)]
            worker_agents = [a for a in agents if not a.get("is_master", False)]

            # If no master designated, use first agent as coordinator
            if not master_agents and agents:
                master_agents = [agents[0]]
                worker_agents = agents[1:]

            # Master initiates
            if master_agents:
                steps.append({
                    "agent_id": master_agents[0]["id"],
                    "action": "coordinate",
                    "inputs": {}
                })

            # Workers execute (can be parallel)
            worker_ids = [w["id"] for w in worker_agents]
            for worker in worker_agents:
                other_workers = [w["id"] for w in worker_agents if w["id"] != worker["id"]]
                steps.append({
                    "agent_id": worker["id"],
                    "action": worker.get("role", "process").lower().replace(" ", "_"),
                    "inputs": {},
                    "parallel_with": other_workers
                })

            # Master aggregates results
            if master_agents:
                steps.append({
                    "agent_id": master_agents[0]["id"],
                    "action": "aggregate",
                    "inputs": {}
                })
            logger.info(f"Generated {len(steps)} hierarchical workflow steps")

        else:
            # Sequential (default): One after another
            for agent in agents:
                steps.append({
                    "agent_id": agent["id"],
                    "action": agent.get("role", "process").lower().replace(" ", "_"),
                    "inputs": {},
                    "parallel_with": []
                })
            logger.info(f"Generated {len(steps)} sequential workflow steps")

        return steps

    def _transform_workflow_steps(self, steps: list) -> list:
        """
        Transform workflow steps from various LLM formats to the expected schema format.

        Expected format (WorkflowStep schema):
            {"agent_id": "...", "action": "...", "inputs": {}, ...}

        Common LLM formats to handle:
            {"agent": "...", "next": "..."}
            {"agent_id": "...", "next_agent": "..."}
        """
        transformed = []

        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                continue

            # Build the transformed step
            new_step = {}

            # Handle agent_id (might be "agent" or "agent_id")
            if "agent_id" in step:
                new_step["agent_id"] = step["agent_id"]
            elif "agent" in step:
                new_step["agent_id"] = step["agent"]
            else:
                # Skip malformed step
                logger.warning(f"Skipping step {i}: no agent_id or agent field found")
                continue

            # Handle action (required field)
            if "action" in step:
                new_step["action"] = step["action"]
            else:
                # Generate action from agent_id if not provided
                agent_id = new_step["agent_id"]
                # Common pattern: "code_validator" -> "validate"
                action_word = agent_id.split("_")[-1] if "_" in agent_id else "process"
                new_step["action"] = action_word

            # Handle inputs (optional, default to empty dict)
            new_step["inputs"] = step.get("inputs", {})

            # Handle condition (optional)
            if "condition" in step:
                new_step["condition"] = step["condition"]

            # Handle parallel_with (optional)
            if "parallel_with" in step:
                new_step["parallel_with"] = step["parallel_with"]
            else:
                new_step["parallel_with"] = []

            transformed.append(new_step)

        return transformed
