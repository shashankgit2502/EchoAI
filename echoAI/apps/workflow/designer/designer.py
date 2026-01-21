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
        Get LLM client based on environment configuration.

        Provider Priority (based on .env settings):
        1. USE_AZURE=true -> Azure OpenAI
        2. USE_OPENROUTER=true -> OpenRouter (current default)
        3. USE_OLLAMA=true -> Ollama (on-premise)
        4. USE_OPENAI=true -> OpenAI Direct
        """
        if self._openai_client is None:
            try:
                from langchain_openai import ChatOpenAI
                # For Azure deployment - uncomment the line below
                # from langchain_openai import AzureChatOpenAI

                # =================================================================
                # OPTION 1: AZURE OPENAI (For Azure Deployment)
                # =================================================================
                # Uncomment this block when deploying to Azure
                # -----------------------------------------------------------------
                # if os.getenv("USE_AZURE", "false").lower() == "true":
                #     self._openai_client = AzureChatOpenAI(
                #         azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                #         api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                #         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                #         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                #         temperature=0.3,
                #         max_tokens=4000
                #     )
                #     return self._openai_client

                # =================================================================
                # OPTION 2: OPENROUTER (Current - For Development)
                # =================================================================
                # Using OpenRouter with free tier model - CURRENTLY ACTIVE
                # -----------------------------------------------------------------
                if os.getenv("USE_OPENROUTER", "true").lower() == "true":
                    openrouter_key = os.getenv(
                        "OPENROUTER_API_KEY",
                        "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"
                    )
                    openrouter_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                    openrouter_model = os.getenv("OPENROUTER_MODEL", "mistralai/devstral-2512:free")
                    self._openai_client = ChatOpenAI(
                        base_url=openrouter_url,
                        api_key=openrouter_key,
                        model=openrouter_model,
                        temperature=0.3,
                        max_tokens=4000
                    )
                    return self._openai_client

                # =================================================================
                # OPTION 3: OLLAMA (On-Premise/Local)
                # =================================================================
                # Uncomment this block for local Ollama deployment
                # -----------------------------------------------------------------
                # if os.getenv("USE_OLLAMA", "false").lower() == "true":
                #     ollama_url = os.getenv("OLLAMA_BASE_URL", "http://10.188.100.131:8004/v1")
                #     ollama_model = os.getenv("OLLAMA_MODEL", "mistral-nemo:12b-instruct-2407-fp16")
                #     self._openai_client = ChatOpenAI(
                #         base_url=ollama_url,
                #         api_key="ollama",
                #         model=ollama_model,
                #         temperature=0.3,
                #         max_tokens=4000
                #     )
                #     return self._openai_client

                # =================================================================
                # OPTION 4: OPENAI DIRECT
                # =================================================================
                # Uncomment this block for direct OpenAI API
                # -----------------------------------------------------------------
                # if os.getenv("USE_OPENAI", "false").lower() == "true":
                #     self._openai_client = ChatOpenAI(
                #         api_key=os.getenv("OPENAI_API_KEY", self.api_key),
                #         model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                #         temperature=0.3,
                #         max_tokens=4000
                #     )
                #     return self._openai_client

                # =================================================================
                # FALLBACK: OpenRouter (always available)
                # =================================================================
                self._openai_client = ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key="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",
                    model="mistralai/devstral-2512:free",
                    temperature=0.3,
                    max_tokens=4000
                )

            except ImportError:
                raise ImportError(
                    "langchain-openai not installed. Run: pip install langchain-openai"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize LLM client: {e}")
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
            default_llm = {
                "provider": "openrouter",
                "model": "mistralai/devstral-2512:free",
                "temperature": 0.2
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

        # Create system prompt for workflow design
        system_prompt = """You are a workflow design assistant. Analyze the user's request and design a multi-agent workflow.

Return a JSON response with this exact structure:
{
  "execution_model": "sequential|parallel|hierarchical|hybrid",
  "workflow_name": "Brief workflow name",
  "agents": [
    {
      "name": "Agent name",
      "role": "Agent role/responsibility",
      "description": "What this agent does",
      "input_schema": ["list", "of", "input", "keys"],
      "output_schema": ["list", "of", "output", "keys"]
    }
  ]
}

Rules:
1. execution_model:
   - "sequential": Tasks done one after another
   - "parallel": Tasks done simultaneously
   - "hierarchical": One master agent coordinates sub-agents
   - "hybrid": Mix of above

2. Design 2-5 agents based on complexity
3. Each agent should have clear role and I/O schema
4. Ensure output of one agent matches input of next (for sequential)
5. Be concise and practical"""

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
                    "can_call_agents": execution_model == "hierarchical" and agents == []
                },
                "metadata": {
                    "created_at": timestamp
                }
            })

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
            "name": workflow_name,
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
                "created_by": "designer_llm",
                "created_at": timestamp,
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
