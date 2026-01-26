"""
LLM-based workflow designer.
Analyzes user prompts and generates workflow + agent definitions using real LLM.

LLM Provider Configuration:
---------------------------
This module supports multiple LLM providers. Configure via .env file:
- OPTION 1: Ollama (On-Premise) - Set USE_OLLAMA=true
- OPTION 2: OpenRouter (Current) - Set USE_OPENROUTER=true
- OPTION 3: Azure OpenAI - Set USE_AZURE=true
- OPTION 4: OpenAI Direct - Set USE_OPENAI=true

See .env file for detailed configuration options.
"""
import json
import os
from typing import Dict, Any, Tuple, List, Optional
from echolib.utils import new_id
from datetime import datetime


class WorkflowDesigner:
    """
    Workflow designer service.
    Uses LLM to generate workflows from natural language prompts.
    """

    def __init__(self, llm_service=None, api_key: Optional[str] = None, agent_registry=None):
        """
        Initialize designer.

        Args:
            llm_service: LLM service for prompt analysis
            api_key: OpenAI API key (optional, reads from env if not provided)
            agent_registry: Agent registry for saving agents
        """
        self.llm_service = llm_service
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._openai_client = None
        self.agent_registry = agent_registry

    def _get_openai_client(self):
        """
        Get LLM client using centralized LLM Manager.

        All LLM configuration is now in llm_manager.py
        To change provider/model, edit llm_manager.py
        """
        if self._openai_client is None:
            try:
                from llm_manager import LLMManager

                # Get LLM from centralized manager
                # Uses default configuration from llm_manager.py
                # Temperature and max_tokens can be overridden if needed
                self._openai_client = LLMManager.get_llm(
                    temperature=0.3,
                    max_tokens=4000
                )

            except Exception as e:
                raise RuntimeError(f"Failed to get LLM from LLMManager: {e}")

        return self._openai_client

    def design_from_prompt(
        self,
        user_prompt: str,
        default_llm: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Design workflow and agents from user prompt using real LLM.

        Args:
            user_prompt: Natural language description of desired workflow
            default_llm: Default LLM configuration for agents

        Returns:
            Tuple of (workflow_definition, agent_definitions)
        """
        if default_llm is None:
            # DEFAULT LLM for agents - delegates to LLMManager
            # Agents will use defaults from llm_manager.py unless overridden
            default_llm = {
                # Leave empty to use LLMManager defaults
                # Or specify: "provider": "openai", "model": "gpt-4", etc.
                "temperature": 0.3
            }

        # Always try LLM first (OpenRouter is available)
        try:
            return self._design_with_llm(user_prompt, default_llm)
        except Exception as e:
            print(f"LLM design failed, falling back to heuristics: {e}")
            return self._design_with_heuristics(user_prompt, default_llm)

    def _design_with_llm(
        self,
        user_prompt: str,
        default_llm: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Design workflow using real LLM analysis."""

        # Enhanced system prompt with better pattern detection
        system_prompt = """You are an expert workflow architect. Analyze the user's request and design an optimal multi-agent workflow.

## WORKFLOW TYPES AND WHEN TO USE THEM

1. **Sequential**: Linear chain where tasks must happen in specific order
   - Example: "Generate code, then review it, then deploy"
   - Pattern: Output of A feeds into B feeds into C
   - Use when: Dependencies exist between stages

2. **Parallel**: Independent tasks that can run simultaneously
   - Example: "Analyze code for bugs, security issues, and performance problems"
   - Pattern: Multiple agents process same/different inputs concurrently
   - Use when: Tasks are independent and can benefit from concurrency

3. **Hierarchical**: Manager coordinates specialist workers
   - Example: "Project manager assigns tasks to frontend, backend, and DevOps teams"
   - Pattern: One manager delegates to multiple workers
   - Use when: Central coordination and delegation is needed

4. **Hybrid**: Mixed patterns combining parallel and sequential
   - Example: "Three agents analyze different aspects in parallel, then synthesizer combines results, then reviewer validates"
   - Pattern: Parallel stages → merge → sequential stages
   - Use when: Some stages benefit from parallelism, others need sequential processing

## DECISION TREE

Ask yourself these questions in order:
1. Is there ONE agent coordinating/managing others? → **Hierarchical**
2. Are there distinct stages where some run in parallel, then merge into sequential? → **Hybrid**
3. Can ALL tasks run simultaneously with no dependencies? → **Parallel**
4. Must tasks happen in a specific order? → **Sequential**

## RESPONSE FORMAT

Return JSON with this structure:

For Sequential/Parallel/Hierarchical:
{
  "execution_model": "sequential|parallel|hierarchical",
  "workflow_name": "Brief name",
  "reasoning": "1-2 sentences why you chose this model",
  "agents": [
    {
      "name": "Agent name",
      "role": "Clear role",
      "goal": "What this agent aims to achieve",
      "description": "What this agent does",
      "input_schema": ["input_keys"],
      "output_schema": ["output_keys"]
    }
  ]
}

For Hybrid workflows, ALSO include topology:
{
  "execution_model": "hybrid",
  "workflow_name": "Brief name",
  "reasoning": "Why hybrid is needed",
  "agents": [...],
  "topology": {
    "parallel_groups": [
      {
        "agents": [0, 1, 2],  // indices into agents array
        "merge_strategy": "combine"  // "combine", "vote", or "prioritize"
      }
    ],
    "sequential_chains": [
      {
        "agents": [3, 4]  // indices into agents array (after merge)
      }
    ]
  }
}

For Hierarchical workflows, ALSO include hierarchy:
{
  "execution_model": "hierarchical",
  "hierarchy": {
    "master_agent_index": 0,  // index in agents array
    "sub_agent_indices": [1, 2, 3],  // worker indices
    "delegation_strategy": "dynamic"  // "dynamic", "all", or "sequential"
  }
}

## EXAMPLES

User: "Generate Python code for an API endpoint"
→ Sequential (design → implement → test)

User: "Check code for security, performance, and maintainability issues"
→ Parallel (3 independent analyses)

User: "A tech lead coordinates frontend, backend, and DevOps work"
→ Hierarchical (1 manager + 3 specialists)

User: "Extract data from 3 sources in parallel, then transform, then load to database"
→ Hybrid (3 parallel extractors → transformer → loader)

## RULES

1. Design 2-5 agents (more complex tasks may need more)
2. Each agent needs clear role, goal, and I/O schema
3. For sequential: ensure output keys match next agent's input keys
4. For parallel: agents should have similar input but different focus areas
5. For hierarchical: manager's goal should mention coordination/delegation
6. For hybrid: be explicit about which agents run in parallel vs sequential
7. Always include "reasoning" to explain your choice
8. Be practical and concise

Now analyze the user's request:"""

        llm = self._get_openai_client()

        # Combine system and user prompts for ChatOpenAI
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}\n\nProvide your response as a valid JSON object."

        # Invoke LLM
        response = llm.invoke(full_prompt)

        # Parse LLM response
        content = response.content if hasattr(response, 'content') else str(response)

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        llm_output = json.loads(content)

        # Build workflow from LLM design
        return self._build_workflow_from_llm_response(
            llm_output, user_prompt, default_llm
        )

    def _build_workflow_from_llm_response(
        self,
        llm_output: Dict[str, Any],
        user_prompt: str,
        default_llm: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Build workflow structure from LLM response."""

        workflow_id = new_id("wf_")
        timestamp = datetime.utcnow().isoformat()

        execution_model = llm_output.get("execution_model", "sequential")
        workflow_name = llm_output.get("workflow_name", "Workflow from prompt")

        # Create agents from LLM design
        agents = []
        for agent_spec in llm_output.get("agents", []):
            agent_id = new_id("agt_")
            agents.append({
                "agent_id": agent_id,
                "name": agent_spec.get("name", "Agent"),
                "role": agent_spec.get("role", "Processing"),
                "goal": agent_spec.get("goal") or f"Complete task: {agent_spec.get('name', 'processing')}",
                "description": agent_spec.get("description", ""),
                "llm": default_llm.copy(),
                "tools": [],
                "input_schema": agent_spec.get("input_schema", ["input"]),
                "output_schema": agent_spec.get("output_schema", ["output"]),
                "constraints": {
                    "max_steps": 10,
                    "timeout_seconds": 60
                },
                "permissions": {
                    "can_call_agents": execution_model == "hierarchical" and len(agents) == 0
                },
                "metadata": {
                    "created_at": timestamp
                }
            })

        # Generate connections (for non-hybrid workflows)
        connections = self._generate_connections(agents, execution_model)

        # Build topology for hybrid workflows
        topology = None
        if execution_model == "hybrid":
            llm_topology = llm_output.get("topology", {})
            parallel_groups_indices = llm_topology.get("parallel_groups", [])
            sequential_chains_indices = llm_topology.get("sequential_chains", [])

            # Convert agent indices to agent IDs
            topology = {
                "parallel_groups": [],
                "sequential_chains": []
            }

            for group in parallel_groups_indices:
                agent_indices = group.get("agents", [])
                topology["parallel_groups"].append({
                    "agents": [agents[i]["agent_id"] for i in agent_indices if i < len(agents)],
                    "merge_strategy": group.get("merge_strategy", "combine")
                })

            for chain in sequential_chains_indices:
                agent_indices = chain.get("agents", [])
                topology["sequential_chains"].append({
                    "agents": [agents[i]["agent_id"] for i in agent_indices if i < len(agents)]
                })

        # Build hierarchy for hierarchical workflows
        hierarchy = None
        if execution_model == "hierarchical":
            llm_hierarchy = llm_output.get("hierarchy", {})
            master_index = llm_hierarchy.get("master_agent_index", 0)
            sub_indices = llm_hierarchy.get("sub_agent_indices", list(range(1, len(agents))))
            delegation_strategy = llm_hierarchy.get("delegation_strategy", "dynamic")

            if len(agents) > 0:
                hierarchy = {
                    "master_agent": agents[master_index]["agent_id"] if master_index < len(agents) else agents[0]["agent_id"],
                    "delegation_order": [agents[i]["agent_id"] for i in sub_indices if i < len(agents)],
                    "delegation_strategy": delegation_strategy
                }

        # Build workflow
        workflow = {
            "workflow_id": workflow_id,
            "name": workflow_name,
            "description": user_prompt[:200],
            "status": "draft",
            "version": "0.1",
            "execution_model": execution_model,
            "agents": [agent["agent_id"] for agent in agents],
            "connections": connections,
            "hierarchy": hierarchy,
            "topology": topology,  # For hybrid workflows
            "state_schema": {},
            "human_in_loop": {
                "enabled": False,
                "review_points": []
            },
            "metadata": {
                "created_by": "designer_llm",
                "created_at": timestamp,
                "reasoning": llm_output.get("reasoning", ""),  # LLM's reasoning for workflow type
                "tags": ["auto-generated", "llm-designed"]
            }
        }

        # Convert agents list to dict
        agent_dict = {agent["agent_id"]: agent for agent in agents}

        # Save agents to registry if available
        if self.agent_registry:
            for agent in agents:
                try:
                    self.agent_registry.register_agent(agent)
                except Exception as e:
                    print(f"Warning: Failed to register agent {agent.get('agent_id')}: {e}")

        return workflow, agent_dict

    def _design_with_heuristics(
        self,
        user_prompt: str,
        default_llm: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Fallback: Design workflow using simple heuristics."""

        workflow_id = new_id("wf_")
        timestamp = datetime.utcnow().isoformat()

        # Determine execution model
        execution_model = self._infer_execution_model(user_prompt)

        # Generate agents
        agents = self._generate_agents_heuristic(user_prompt, default_llm)

        # Generate connections
        connections = self._generate_connections(agents, execution_model)

        # Build hierarchy if needed
        hierarchy = None
        if execution_model == "hierarchical" and len(agents) > 0:
            hierarchy = {
                "master_agent": agents[0]["agent_id"],
                "delegation_order": [a["agent_id"] for a in agents[1:]]
            }

        # Build workflow
        workflow = {
            "workflow_id": workflow_id,
            "name": "Workflow from prompt",
            "description": user_prompt[:200],
            "status": "draft",
            "version": "0.1",
            "execution_model": execution_model,
            "agents": [agent["agent_id"] for agent in agents],
            "connections": connections,
            "hierarchy": hierarchy,
            "state_schema": {},
            "human_in_loop": {
                "enabled": False,
                "review_points": []
            },
            "metadata": {
                "created_by": "designer_heuristic",
                "created_at": timestamp,
                "tags": ["auto-generated"]
            }
        }

        agent_dict = {agent["agent_id"]: agent for agent in agents}

        # Save agents to registry if available
        if self.agent_registry:
            for agent in agents:
                try:
                    self.agent_registry.register_agent(agent)
                except Exception as e:
                    print(f"Warning: Failed to register agent {agent.get('agent_id')}: {e}")

        return workflow, agent_dict

    def _infer_execution_model(self, prompt: str) -> str:
        """Infer execution model from prompt using keywords."""
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["coordinate", "orchestrate", "manage", "master"]):
            return "hierarchical"
        elif any(word in prompt_lower for word in ["parallel", "simultaneously", "at once", "concurrent"]):
            return "parallel"
        elif any(word in prompt_lower for word in ["step", "then", "after", "sequence", "pipeline"]):
            return "sequential"
        else:
            return "sequential"

    def _generate_agents_heuristic(
        self,
        prompt: str,
        default_llm: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate basic agents using heuristics."""
        timestamp = datetime.utcnow().isoformat()

        agents = [
            {
                "agent_id": new_id("agt_"),
                "name": "Analyzer",
                "role": "Data Analysis",
                "description": "Analyzes input data",
                "llm": default_llm.copy(),
                "tools": [],
                "input_schema": ["input_data"],
                "output_schema": ["analysis_result"],
                "constraints": {
                    "max_steps": 5,
                    "timeout_seconds": 30
                },
                "permissions": {
                    "can_call_agents": False
                },
                "metadata": {
                    "created_at": timestamp
                }
            },
            {
                "agent_id": new_id("agt_"),
                "name": "Synthesizer",
                "role": "Result Synthesis",
                "description": "Synthesizes analysis into final output",
                "llm": default_llm.copy(),
                "tools": [],
                "input_schema": ["analysis_result"],
                "output_schema": ["final_output"],
                "constraints": {
                    "max_steps": 3,
                    "timeout_seconds": 20
                },
                "permissions": {
                    "can_call_agents": False
                },
                "metadata": {
                    "created_at": timestamp
                }
            }
        ]

        return agents

    def _generate_connections(
        self,
        agents: List[Dict[str, Any]],
        execution_model: str
    ) -> List[Dict[str, str]]:
        """Generate workflow connections based on execution model."""
        if execution_model == "sequential":
            connections = []
            for i in range(len(agents) - 1):
                connections.append({
                    "from": agents[i]["agent_id"],
                    "to": agents[i + 1]["agent_id"]
                })
            return connections

        elif execution_model == "parallel":
            return []

        elif execution_model == "hierarchical":
            connections = []
            if len(agents) > 1:
                master_id = agents[0]["agent_id"]
                for agent in agents[1:]:
                    connections.append({
                        "from": master_id,
                        "to": agent["agent_id"]
                    })
            return connections

        else:
            # Default: sequential
            connections = []
            for i in range(len(agents) - 1):
                connections.append({
                    "from": agents[i]["agent_id"],
                    "to": agents[i + 1]["agent_id"]
                })
            return connections
