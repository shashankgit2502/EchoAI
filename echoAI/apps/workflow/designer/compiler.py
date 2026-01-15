"""
Workflow compiler.
Compiles workflow JSON definitions into executable LangGraph.
"""
from typing import Dict, Any, TypedDict, List, Annotated
import operator
import os


class WorkflowCompiler:
    """
    Compiles workflow JSON to executable LangGraph.
    """

    def __init__(self):
        """Initialize compiler."""
        self._compiled_cache = {}

    def compile_to_langgraph(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> Any:
        """
        Compile workflow JSON to executable LangGraph.

        Args:
            workflow: Workflow definition
            agent_registry: Agent definitions

        Returns:
            Compiled LangGraph instance (runnable)
        """
        try:
            from langgraph.graph import StateGraph, END
            from langgraph.checkpoint.memory import MemorySaver
        except ImportError:
            raise ImportError(
                "LangGraph not installed. Run: pip install langgraph langchain-core"
            )

        execution_model = workflow.get("execution_model", "sequential")

        # Create state schema
        WorkflowState = self._create_state_class(workflow, agent_registry)

        # Build graph based on execution model
        if execution_model == "sequential":
            return self._compile_sequential(workflow, agent_registry, WorkflowState)
        elif execution_model == "parallel":
            return self._compile_parallel(workflow, agent_registry, WorkflowState)
        elif execution_model == "hierarchical":
            return self._compile_hierarchical(workflow, agent_registry, WorkflowState)
        else:
            # Default to sequential
            return self._compile_sequential(workflow, agent_registry, WorkflowState)

    def _create_state_class(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> type:
        """
        Create TypedDict state class for workflow.

        Args:
            workflow: Workflow definition
            agent_registry: Agent definitions

        Returns:
            TypedDict class for state
        """
        # Collect all state keys from agents
        state_keys = set()

        for agent_id in workflow.get("agents", []):
            agent = agent_registry.get(agent_id, {})
            state_keys.update(agent.get("input_schema", []))
            state_keys.update(agent.get("output_schema", []))

        # Add workflow-level state keys
        state_keys.update(workflow.get("state_schema", {}).keys())

        # Create TypedDict dynamically
        fields = {
            key: Annotated[Any, operator.add] for key in state_keys
        }

        # Add messages field for agent communication
        fields["messages"] = Annotated[List[Dict[str, Any]], operator.add]

        WorkflowState = TypedDict("WorkflowState", fields)
        return WorkflowState

    def _compile_sequential(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
        WorkflowState: type
    ) -> Any:
        """Compile sequential workflow."""
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        graph = StateGraph(WorkflowState)

        agents = workflow.get("agents", [])
        connections = workflow.get("connections", [])

        # Add agent nodes
        for agent_id in agents:
            agent = agent_registry.get(agent_id, {})
            node_func = self._create_agent_node(agent_id, agent)
            graph.add_node(agent_id, node_func)

        # Add edges based on connections
        for i, connection in enumerate(connections):
            from_agent = connection.get("from")
            to_agent = connection.get("to")

            if i == 0:
                # First connection - set entry point
                graph.set_entry_point(from_agent)

            graph.add_edge(from_agent, to_agent)

        # Set finish point
        if agents:
            graph.add_edge(agents[-1], END)

        # Compile with memory
        memory = MemorySaver()
        compiled = graph.compile(checkpointer=memory)

        return compiled

    def _compile_parallel(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
        WorkflowState: type
    ) -> Any:
        """Compile parallel workflow."""
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        graph = StateGraph(WorkflowState)

        agents = workflow.get("agents", [])

        # Create coordinator node
        def coordinator(state: Dict[str, Any]) -> Dict[str, Any]:
            """Distribute work to parallel agents."""
            return state

        # Create aggregator node
        def aggregator(state: Dict[str, Any]) -> Dict[str, Any]:
            """Aggregate results from parallel agents."""
            return state

        graph.add_node("coordinator", coordinator)
        graph.add_node("aggregator", aggregator)

        # Add parallel agent nodes
        for agent_id in agents:
            agent = agent_registry.get(agent_id, {})
            node_func = self._create_agent_node(agent_id, agent)
            graph.add_node(agent_id, node_func)

            # Connect coordinator to each agent
            graph.add_edge("coordinator", agent_id)
            # Connect each agent to aggregator
            graph.add_edge(agent_id, "aggregator")

        # Set entry and exit
        graph.set_entry_point("coordinator")
        graph.add_edge("aggregator", END)

        memory = MemorySaver()
        compiled = graph.compile(checkpointer=memory)

        return compiled

    def _compile_hierarchical(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
        WorkflowState: type
    ) -> Any:
        """Compile hierarchical workflow."""
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        graph = StateGraph(WorkflowState)

        hierarchy = workflow.get("hierarchy", {})
        master_agent_id = hierarchy.get("master_agent")
        sub_agents = hierarchy.get("delegation_order", [])

        # Add master agent node
        if master_agent_id:
            master_agent = agent_registry.get(master_agent_id, {})
            master_func = self._create_agent_node(master_agent_id, master_agent)
            graph.add_node(master_agent_id, master_func)

            # Set master as entry point
            graph.set_entry_point(master_agent_id)

        # Add sub-agent nodes
        for agent_id in sub_agents:
            agent = agent_registry.get(agent_id, {})
            node_func = self._create_agent_node(agent_id, agent)
            graph.add_node(agent_id, node_func)

            # Connect master to sub-agent and back
            if master_agent_id:
                graph.add_edge(master_agent_id, agent_id)
                graph.add_edge(agent_id, master_agent_id)

        # Master agent decides when to end
        if master_agent_id and sub_agents:
            # Add conditional edge from master
            graph.add_edge(master_agent_id, END)

        memory = MemorySaver()
        compiled = graph.compile(checkpointer=memory)

        return compiled

    def _create_agent_node(
        self,
        agent_id: str,
        agent_config: Dict[str, Any]
    ):
        """
        Create agent node function with REAL LLM execution.

        Args:
            agent_id: Agent identifier
            agent_config: Agent configuration

        Returns:
            Callable node function
        """
        def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Agent node execution function with real LLM calls.

            Args:
                state: Current workflow state

            Returns:
                Updated state
            """
            # Get agent configuration
            agent_name = agent_config.get("name", agent_id)
            agent_role = agent_config.get("role", "Processing")
            agent_description = agent_config.get("description", "")
            input_schema = agent_config.get("input_schema", [])
            output_schema = agent_config.get("output_schema", [])
            llm_config = agent_config.get("llm", {})

            # Extract inputs from state
            inputs = {key: state.get(key) for key in input_schema if key in state}

            # Build prompt for LLM
            prompt = f"""You are {agent_name}, a specialized agent with the following role:
{agent_role}

{agent_description}

Your task is to process the following inputs and generate outputs:

Inputs:
{inputs}

Please provide your response in a clear, structured format. Focus on your specific role and responsibilities."""

            # Execute real LLM call
            try:
                llm_response = self._execute_llm_call(llm_config, prompt)

                # Create outputs based on LLM response
                outputs = {}
                for key in output_schema:
                    outputs[key] = llm_response

            except Exception as e:
                # Fallback if LLM call fails
                outputs = {
                    key: f"Error in {agent_name}: {str(e)}" for key in output_schema
                }

            # Add message to history
            message = {
                "agent": agent_id,
                "role": agent_role,
                "inputs": inputs,
                "outputs": outputs
            }

            messages = state.get("messages", [])
            messages.append(message)

            # Update state
            updated_state = {**state, **outputs, "messages": messages}

            return updated_state

        return agent_node

    def _execute_llm_call(self, llm_config: Dict[str, Any], prompt: str) -> str:
        """
        Execute actual LLM call based on provider.

        Args:
            llm_config: LLM configuration (provider, model, temperature)
            prompt: Prompt to send to LLM

        Returns:
            LLM response text
        """
        provider = llm_config.get("provider", "openai")
        model = llm_config.get("model", "gpt-4o-mini")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_tokens", 1000)

        if provider == "openai":
            return self._call_openai(model, prompt, temperature, max_tokens)
        elif provider == "anthropic":
            return self._call_anthropic(model, prompt, temperature, max_tokens)
        elif provider == "azure":
            return self._call_azure(model, prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _call_openai(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call OpenAI API or Ollama using ChatOpenAI."""
        try:
            from langchain_openai import ChatOpenAI

            # Check if using Ollama
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://10.188.100.131:8004/v1")
            use_ollama = os.getenv("USE_OLLAMA", "true").lower() == "true"

            if use_ollama:
                # Use Ollama endpoint
                llm = ChatOpenAI(
                    base_url=ollama_url,
                    api_key="ollama",
                    model=os.getenv("OLLAMA_MODEL", "mistral-nemo:12b-instruct-2407-fp16"),
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # Use OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                llm = ChatOpenAI(
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            response = llm.invoke(prompt)
            return response.content

        except ImportError:
            raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

    def _call_anthropic(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call Anthropic API."""
        try:
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text

        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}")

    def _call_azure(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call Azure OpenAI API."""
        try:
            from openai import AzureOpenAI

            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

            if not api_key or not endpoint:
                raise ValueError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")

            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint
            )

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API call failed: {e}")

    def _determine_graph_type(self, execution_model: str) -> str:
        """
        Determine LangGraph graph type from execution model.

        Args:
            execution_model: Execution model

        Returns:
            LangGraph graph type
        """
        mapping = {
            "sequential": "StateGraph",
            "parallel": "StateGraph",
            "hierarchical": "StateGraph",
            "hybrid": "StateGraph"
        }
        return mapping.get(execution_model, "StateGraph")
