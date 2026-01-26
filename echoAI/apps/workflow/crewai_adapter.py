"""
CrewAI Adapter Module

CRITICAL ARCHITECTURAL RULES:
============================
1. LangGraph OWNS: Workflow topology, execution order, branching, merging, state
2. CrewAI is ONLY invoked INSIDE LangGraph node functions
3. CrewAI HANDLES: Agent collaboration, delegation, parallelism WITHIN nodes
4. CrewAI NEVER: Controls graph traversal or state transitions
5. State flow: LangGraph state → CrewAI → LangGraph state

This adapter creates LangGraph node functions that execute CrewAI crews.
The node functions are called BY LangGraph, not the other way around.
"""

from typing import Dict, Any, List, Callable, Optional
import os
import asyncio
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CrewAIAdapter:
    """
    Adapter for integrating CrewAI with LangGraph workflows.

    This class creates LangGraph-compatible node functions that execute
    CrewAI crews for agent collaboration within nodes.

    IMPORTANT: This adapter does NOT create graph structures. It creates
    node functions that LangGraph calls as part of ITS graph execution.
    """

    def __init__(self):
        """Initialize the CrewAI adapter."""
        # LLM caching is now handled by LLMManager
        pass

    # ========================================================================
    # TOOL BINDING METHODS
    # ========================================================================

    def _get_tool_executor(self):
        """
        Get ToolExecutor from DI container.

        Returns:
            ToolExecutor instance for invoking tools
        """
        from echolib.di import container
        return container.resolve('tool.executor')

    def _get_tool_registry(self):
        """
        Get ToolRegistry from DI container.

        Returns:
            ToolRegistry instance for looking up tool definitions
        """
        from echolib.di import container
        return container.resolve('tool.registry')

    def _create_crewai_tool_wrapper(self, tool_def, executor):
        """
        Create a CrewAI-compatible tool wrapper from a ToolDef.

        This method creates a dynamic CrewAI BaseTool subclass that wraps
        our ToolDef and uses the ToolExecutor to invoke it. CrewAI requires
        synchronous tool methods, so we handle the async executor here.

        Args:
            tool_def: ToolDef instance with tool metadata and execution config
            executor: ToolExecutor instance for tool execution

        Returns:
            CrewAI BaseTool instance that wraps the tool
        """
        from crewai.tools import BaseTool

        # Capture tool_def and executor in closure for the dynamic class
        captured_tool_def = tool_def
        captured_executor = executor

        class DynamicCrewAITool(BaseTool):
            """
            Dynamically created CrewAI tool wrapper.

            This class wraps an EchoAI ToolDef and executes it via ToolExecutor.
            """
            name: str = captured_tool_def.name
            description: str = captured_tool_def.description

            def _run(self, **kwargs) -> str:
                """
                Execute the tool synchronously.

                CrewAI expects sync methods, so we handle the async executor here
                by running it in an event loop.

                Args:
                    **kwargs: Tool input parameters

                Returns:
                    JSON string of tool output or error
                """
                try:
                    # ToolExecutor.invoke is async, so we need to run it
                    # Try to get running event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, need to handle carefully
                        # Create a new thread to run the async code
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            future = pool.submit(
                                asyncio.run,
                                captured_executor.invoke(captured_tool_def.tool_id, kwargs)
                            )
                            result = future.result(timeout=60)
                    except RuntimeError:
                        # No running loop, create new one
                        result = asyncio.run(
                            captured_executor.invoke(captured_tool_def.tool_id, kwargs)
                        )

                    # Check if execution was successful
                    if result.success:
                        # Serialize output to string for CrewAI
                        return json.dumps(result.output)
                    else:
                        # Return error message as JSON
                        return json.dumps({
                            "error": result.error,
                            "tool_id": result.tool_id
                        })

                except Exception as e:
                    # Handle any execution errors gracefully
                    logger.error(f"Tool execution failed for {captured_tool_def.name}: {e}")
                    return json.dumps({
                        "error": f"Tool execution failed: {str(e)}",
                        "tool_id": captured_tool_def.tool_id
                    })

        # Instantiate and return the tool
        return DynamicCrewAITool()

    def _bind_tools_to_agent(self, agent_config: Dict[str, Any]) -> List[Any]:
        """
        Bind tools to an agent based on its configuration.

        This method retrieves tool definitions from the registry and creates
        CrewAI-compatible tool wrappers for each one.

        Args:
            agent_config: Agent configuration dictionary with 'tools' key

        Returns:
            List of CrewAI tool instances ready for agent use
        """
        tool_ids = agent_config.get("tools", [])
        crewai_tools = []

        if not tool_ids:
            logger.debug(f"Agent has no tools configured")
            return crewai_tools

        try:
            tool_registry = self._get_tool_registry()
            tool_executor = self._get_tool_executor()
        except KeyError as e:
            logger.warning(f"Tool system not available: {e}. Agent will run without tools.")
            return crewai_tools

        for tool_id in tool_ids:
            try:
                tool_def = tool_registry.get(tool_id)
                if tool_def:
                    # Create CrewAI-compatible tool wrapper
                    crewai_tool = self._create_crewai_tool_wrapper(tool_def, tool_executor)
                    crewai_tools.append(crewai_tool)
                    logger.info(f"Bound tool '{tool_def.name}' (id={tool_id}) to agent")
                else:
                    # Try finding by name as fallback (for frontend compatibility)
                    tool_def = tool_registry.get_by_name(tool_id)
                    if tool_def:
                        crewai_tool = self._create_crewai_tool_wrapper(tool_def, tool_executor)
                        crewai_tools.append(crewai_tool)
                        logger.info(f"Bound tool '{tool_def.name}' (by name lookup) to agent")
                    else:
                        logger.warning(f"Tool '{tool_id}' not found in registry, skipping")
            except Exception as e:
                # Log but don't fail - agent can still work without this tool
                logger.warning(f"Failed to bind tool '{tool_id}': {e}")

        logger.info(f"Successfully bound {len(crewai_tools)}/{len(tool_ids)} tools to agent")
        return crewai_tools

    # ========================================================================
    # HIERARCHICAL WORKFLOWS: Manager + Workers with Dynamic Delegation
    # ========================================================================

    def create_hierarchical_crew_node(
        self,
        master_agent_config: Dict[str, Any],
        sub_agent_configs: List[Dict[str, Any]],
        delegation_strategy: str = "dynamic"
    ) -> Callable:
        """
        Create a LangGraph node function that uses CrewAI for hierarchical coordination.

        ARCHITECTURE (FIXED):
        - LangGraph calls this node as part of its graph execution
        - Inside this node, CrewAI Manager delegates to workers
        - Manager receives ORIGINAL user input to make delegation decisions
        - CrewAI returns results to LangGraph state
        - LangGraph decides what node to execute next

        Args:
            master_agent_config: Manager agent configuration
            sub_agent_configs: Worker agent configurations
            delegation_strategy: "dynamic" (manager decides) or "all" (invoke all)

        Returns:
            Callable node function compatible with LangGraph StateGraph
        """
        def hierarchical_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            LangGraph node that executes CrewAI hierarchical crew.

            This function is CALLED BY LangGraph during graph execution.
            CrewAI handles agent collaboration WITHIN this function.
            """
            try:
                from crewai import Crew, Agent, Task, Process

                logger.info(f"Hierarchical Crew node executing with {len(sub_agent_configs)} workers")

                # FIXED: Extract ORIGINAL user input (preserved through workflow)
                original_input = (
                    state.get("original_user_input")
                    or state.get("user_input")
                    or state.get("task_description")
                    or state.get("message")
                    or ""
                )

                # Get any previous context
                previous_context = state.get("crew_result", "")

                logger.info(f"Hierarchical execution for user request: {original_input[:100]}...")

                # Build comprehensive task description for manager
                task_parts = [f"USER REQUEST: {original_input}"]

                manager_prompt = master_agent_config.get("prompt", "")
                if manager_prompt:
                    task_parts.append(f"MANAGER INSTRUCTIONS: {manager_prompt}")

                if previous_context:
                    task_parts.append(f"PREVIOUS CONTEXT: {previous_context}")

                # List available workers for manager
                worker_descriptions = []
                for idx, wc in enumerate(sub_agent_configs):
                    worker_descriptions.append(
                        f"- {wc.get('name', f'Worker{idx+1}')}: {wc.get('role', 'Worker')} - {wc.get('goal', 'Complete assigned tasks')}"
                    )
                if worker_descriptions:
                    task_parts.append(f"AVAILABLE WORKERS:\n" + "\n".join(worker_descriptions))

                task_parts.append("Coordinate the workers to accomplish the user's request. Delegate tasks appropriately.")

                task_description = "\n\n".join(task_parts)
                expected_output = state.get("expected_output", "Completed results from hierarchical coordination addressing the user's request")

                # Bind tools to manager agent
                manager_tools = self._bind_tools_to_agent(master_agent_config)
                if manager_tools:
                    logger.info(f"Manager agent will use {len(manager_tools)} tool(s)")

                # Create CrewAI Manager Agent (can delegate)
                manager = Agent(
                    role=master_agent_config.get("role", "Manager"),
                    goal=master_agent_config.get("goal") or f"Coordinate workers to accomplish: {original_input[:200]}",
                    backstory=master_agent_config.get("description", "Experienced manager coordinating specialized workers"),
                    tools=manager_tools,  # Pass bound tools to manager
                    allow_delegation=True,  # CRITICAL: Manager can delegate
                    llm=self._get_llm_for_agent(master_agent_config),
                    verbose=True
                )

                # Create CrewAI Worker Agents (cannot delegate)
                workers = []
                for worker_config in sub_agent_configs:
                    # Bind tools to each worker agent
                    worker_tools = self._bind_tools_to_agent(worker_config)
                    if worker_tools:
                        logger.info(f"Worker '{worker_config.get('name', 'Worker')}' will use {len(worker_tools)} tool(s)")

                    worker_goal = worker_config.get("goal") or f"Complete specialized task: {worker_config.get('name', 'work')}"
                    worker = Agent(
                        role=worker_config.get("role", "Worker"),
                        goal=worker_goal,
                        backstory=worker_config.get("description", "Specialized worker agent"),
                        tools=worker_tools,  # Pass bound tools to worker
                        allow_delegation=False,  # CRITICAL: Workers don't delegate (prevents loops)
                        llm=self._get_llm_for_agent(worker_config),
                        verbose=True
                    )
                    workers.append(worker)

                # Create main task for manager
                main_task = Task(
                    description=task_description,
                    expected_output=expected_output,
                    agent=manager  # Assigned to manager
                )

                # Create CrewAI Crew with HIERARCHICAL process
                crew = Crew(
                    agents=[manager] + workers,  # Manager first, then workers
                    tasks=[main_task],
                    process=Process.hierarchical,  # CRITICAL: Hierarchical delegation
                    manager_llm=self._get_llm_for_agent(master_agent_config),
                    verbose=True
                )

                # Execute crew (CrewAI handles delegation internally)
                logger.info(f"Executing hierarchical Crew with manager + {len(workers)} workers...")
                result = crew.kickoff()

                # Extract result (CrewAI returns CrewOutput object)
                output_text = result.raw if hasattr(result, 'raw') else str(result)

                # Return only new data - LangGraph merges via state schema
                logger.info("Hierarchical Crew execution completed")
                return {
                    "hierarchical_output": output_text,
                    "crew_result": output_text,
                    "original_user_input": original_input,  # Preserve for downstream
                    "messages": [{
                        "node": "hierarchical_crew",
                        "manager": master_agent_config.get("name", "Manager"),
                        "workers": [w.get("name", "Worker") for w in sub_agent_configs],
                        "original_request": original_input[:200] if original_input else "",
                        "output_preview": output_text[:500] if output_text else "",
                        "timestamp": datetime.utcnow().isoformat()
                    }]
                }

            except Exception as e:
                logger.error(f"CrewAI hierarchical execution failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Fail fast - no silent errors
                raise RuntimeError(f"CrewAI hierarchical node failed: {e}")

        return hierarchical_node

    # ========================================================================
    # PARALLEL WORKFLOWS: Multiple Agents Executing Concurrently
    # ========================================================================

    def create_parallel_crew_node(
        self,
        agent_configs: List[Dict[str, Any]],
        aggregation_strategy: str = "combine"
    ) -> Callable:
        """
        Create a LangGraph node function for parallel agent execution.

        ARCHITECTURE (FIXED):
        - LangGraph calls this node during graph execution
        - Inside this node, CrewAI executes ALL agents in a single Crew
        - Each agent gets a task with the ORIGINAL user input + their specific focus
        - Results are aggregated and returned to LangGraph state
        - True parallel context: all agents can see each other's work

        Args:
            agent_configs: List of agent configurations to run in parallel
            aggregation_strategy: How to merge results ("combine", "vote", "prioritize")

        Returns:
            Callable node function compatible with LangGraph StateGraph
        """
        def parallel_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            LangGraph node that executes CrewAI agents in parallel.

            This function is CALLED BY LangGraph during graph execution.
            CrewAI handles parallel agent execution WITHIN this function.
            """
            try:
                from crewai import Crew, Agent, Task, Process

                logger.info(f"Parallel Crew node executing with {len(agent_configs)} agents")

                # FIXED: Extract ORIGINAL user input (preserved from coordinator)
                original_input = (
                    state.get("original_user_input")
                    or state.get("user_input")
                    or state.get("message")
                    or state.get("task_description")
                    or ""
                )

                # Also get any previous context
                previous_context = state.get("crew_result", "")

                logger.info(f"Parallel execution for user request: {original_input[:100]}...")

                # Create CrewAI agents (all work on the same user request)
                agents = []
                tasks = []
                individual_results = []

                for idx, agent_config in enumerate(agent_configs):
                    agent_name = agent_config.get("name", f"Agent{idx+1}")
                    agent_role = agent_config.get("role", f"Parallel Agent {idx+1}")
                    agent_goal = agent_config.get("goal", "")
                    agent_prompt = agent_config.get("prompt", "")

                    # Bind tools to parallel agent
                    agent_tools = self._bind_tools_to_agent(agent_config)
                    if agent_tools:
                        logger.info(f"Parallel agent '{agent_name}' will use {len(agent_tools)} tool(s)")

                    # Create agent
                    agent = Agent(
                        role=agent_role,
                        goal=agent_goal or f"Process: {agent_name}",
                        backstory=agent_config.get("description", f"Specialized {agent_role} agent"),
                        tools=agent_tools,  # Pass bound tools to agent
                        allow_delegation=False,  # Parallel agents work independently
                        llm=self._get_llm_for_agent(agent_config),
                        verbose=True
                    )
                    agents.append(agent)

                    # FIXED: Build comprehensive task description
                    task_parts = [f"USER REQUEST: {original_input}"]

                    if agent_prompt:
                        task_parts.append(f"YOUR SPECIFIC INSTRUCTIONS: {agent_prompt}")

                    if previous_context:
                        task_parts.append(f"PREVIOUS CONTEXT: {previous_context}")

                    task_parts.append(f"YOUR ROLE: {agent_role}")
                    task_parts.append(f"Focus on your specific expertise and provide a complete response.")

                    task_description = "\n\n".join(task_parts)

                    # Create task for this agent
                    task = Task(
                        description=task_description,
                        expected_output=f"Complete response from {agent_name} addressing the user request",
                        agent=agent
                    )
                    tasks.append(task)

                # Create crew with sequential process
                # Note: CrewAI's sequential here means tasks execute in order,
                # but each agent works on its own task independently
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    process=Process.sequential,
                    verbose=True
                )

                # Execute crew
                logger.info(f"Executing parallel Crew with {len(agents)} agents...")
                result = crew.kickoff()

                # Collect individual results from each task
                for task in crew.tasks:
                    if hasattr(task, 'output'):
                        output = task.output.raw if hasattr(task.output, 'raw') else str(task.output)
                        individual_results.append(output)

                # Aggregate results based on strategy
                if aggregation_strategy == "combine":
                    aggregated = self._combine_results(individual_results)
                elif aggregation_strategy == "vote":
                    aggregated = self._vote_on_results(individual_results)
                elif aggregation_strategy == "prioritize":
                    aggregated = individual_results[0] if individual_results else ""
                else:
                    aggregated = "\n\n".join(individual_results)

                # Return only new data - LangGraph merges via state schema
                logger.info(f"Parallel Crew execution completed: {len(individual_results)} results aggregated")
                return {
                    "parallel_output": aggregated,
                    "individual_outputs": individual_results,
                    "crew_result": aggregated,
                    "messages": [{
                        "node": "parallel_crew",
                        "agents": [a.get("name", f"Agent{i+1}") for i, a in enumerate(agent_configs)],
                        "original_request": original_input[:200] if original_input else "",
                        "result_count": len(individual_results),
                        "aggregation_strategy": aggregation_strategy,
                        "timestamp": datetime.utcnow().isoformat()
                    }]
                }

            except Exception as e:
                logger.error(f"CrewAI parallel execution failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Fail fast
                raise RuntimeError(f"CrewAI parallel node failed: {e}")

        return parallel_node

    # ========================================================================
    # SEQUENTIAL WORKFLOWS: Single Agent with CrewAI
    # ========================================================================

    def create_sequential_agent_node(
        self,
        agent_config: Dict[str, Any]
    ) -> Callable:
        """
        Create a LangGraph node function for a single agent using CrewAI.

        ARCHITECTURE (FIXED):
        - LangGraph calls this node as part of sequential execution
        - Inside this node, CrewAI executes a single agent
        - Agent receives BOTH original user input AND previous agent output
        - Results returned to LangGraph state
        - LangGraph decides next node in sequence

        Args:
            agent_config: Configuration for the agent

        Returns:
            Callable node function compatible with LangGraph StateGraph
        """
        def sequential_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            LangGraph node that executes a single CrewAI agent.

            This function is CALLED BY LangGraph during graph execution.
            """
            try:
                from crewai import Crew, Agent, Task, Process

                agent_name = agent_config.get("name", "Agent")
                agent_role = agent_config.get("role", "Processor")
                agent_goal = agent_config.get("goal", "")
                agent_description = agent_config.get("description", "")

                logger.info(f"Sequential node executing agent: {agent_name}")

                # --- FIXED: Extract ORIGINAL user input (preserved through workflow) ---
                original_input = (
                    state.get("original_user_input")
                    or state.get("user_input")
                    or state.get("message")
                    or state.get("task_description")
                    or state.get("question")
                    or state.get("input")
                    or ""
                )

                # Current user input (may differ from original in chat scenarios)
                current_input = (
                    state.get("user_input")
                    or state.get("message")
                    or original_input
                )

                logger.info(f"Agent {agent_name} processing request: {original_input[:100]}...")

                # Extract inputs from agent's input_schema
                input_schema = agent_config.get("input_schema", [])
                inputs = {key: state.get(key) for key in input_schema if state.get(key) is not None}

                # Get previous agent's output as context
                previous_output = state.get("crew_result", "")

                # Get parallel results if this agent is after a parallel section
                parallel_output = state.get("parallel_output", "")

                # --- Build comprehensive task description ---
                task_parts = []

                # ALWAYS include original user request first
                if original_input:
                    task_parts.append(f"ORIGINAL USER REQUEST: {original_input}")

                # Add agent's configured prompt/instructions
                agent_prompt = agent_config.get("prompt", "")
                if agent_prompt:
                    task_parts.append(f"YOUR INSTRUCTIONS: {agent_prompt}")

                # Add context from previous agents in the workflow
                if previous_output:
                    task_parts.append(f"PREVIOUS AGENT OUTPUT:\n{previous_output}")

                # Add parallel results if available (for post-merge agents)
                if parallel_output and parallel_output != previous_output:
                    task_parts.append(f"PARALLEL EXECUTION RESULTS:\n{parallel_output}")

                # Add extracted inputs if any
                if inputs:
                    task_parts.append(f"INPUT DATA: {inputs}")

                # Add role context
                task_parts.append(f"YOUR ROLE: {agent_role}")
                task_parts.append(f"YOUR GOAL: {agent_goal or 'Complete your assigned task based on the user request'}")

                # Build final task description
                if task_parts:
                    task_description = "\n\n".join(task_parts)
                else:
                    # Fallback: use agent role/goal as the task
                    task_description = (
                        f"You are a {agent_role}. "
                        f"{agent_goal or agent_description or 'Complete your assigned task.'}"
                    )

                # --- Bind tools to agent ---
                # Get tools assigned to this agent and create CrewAI wrappers
                crewai_tools = self._bind_tools_to_agent(agent_config)
                if crewai_tools:
                    logger.info(f"Agent '{agent_name}' will use {len(crewai_tools)} tool(s)")

                # --- Create CrewAI agent ---
                agent = Agent(
                    role=agent_role,
                    goal=agent_goal or f"Complete task: {agent_name}",
                    backstory=agent_description or f"Specialized {agent_role} agent",
                    tools=crewai_tools,  # Pass bound tools to agent
                    allow_delegation=False,
                    llm=self._get_llm_for_agent(agent_config),
                    verbose=True
                )

                # Create task
                task = Task(
                    description=task_description,
                    expected_output=f"Complete response from {agent_name} addressing the user's request",
                    agent=agent
                )

                # Create crew with single agent
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True
                )

                # Execute
                logger.info(f"Executing sequential agent: {agent_name}...")
                result = crew.kickoff()
                output_text = result.raw if hasattr(result, 'raw') else str(result)

                # --- Map output to state ---
                output_schema = agent_config.get("output_schema", [])
                outputs = {}
                for key in output_schema:
                    outputs[key] = output_text

                # Always store result for next agent to pick up
                outputs["crew_result"] = output_text

                # Preserve original user input for downstream agents
                outputs["original_user_input"] = original_input

                # If this is the last agent (exit point), store as "result"
                if "result" in output_schema or agent_role in ("Workflow exit point", "Output"):
                    outputs["result"] = output_text

                # Return only new data - LangGraph merges via state schema
                logger.info(f"Sequential agent {agent_name} completed")
                return {
                    **outputs,
                    "messages": [{
                        "agent": agent_config.get("agent_id", "unknown"),
                        "name": agent_name,
                        "role": agent_role,
                        "tools_bound": len(crewai_tools),
                        "tool_names": [t.name for t in crewai_tools] if crewai_tools else [],
                        "original_request": original_input[:200] if original_input else "",
                        "had_previous_context": bool(previous_output),
                        "had_parallel_context": bool(parallel_output),
                        "output_preview": output_text[:500] if output_text else "",
                        "timestamp": datetime.utcnow().isoformat()
                    }]
                }

            except Exception as e:
                logger.error(f"CrewAI sequential agent execution failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise RuntimeError(f"CrewAI sequential agent node failed: {e}")

        return sequential_agent_node

    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================

    def _get_llm_for_agent(self, agent_config: Dict[str, Any]):
        """
        Get LLM instance for agent using centralized LLM Manager.

        Supports per-agent LLM configuration with different providers.
        All LLM configuration is now centralized in llm_manager.py

        This method simply extracts agent preferences and delegates to LLMManager.
        """
        from llm_manager import LLMManager

        llm_config = agent_config.get("llm", {})

        # Extract agent's LLM preferences (or None to use defaults)
        provider = llm_config.get("provider")  # None = use LLMManager default
        model = llm_config.get("model")        # None = use LLMManager default
        temperature = llm_config.get("temperature")
        max_tokens = llm_config.get("max_tokens")

        # OVERRIDE FIX: Ignore provider/model from agent config if they conflict
        # Always use LLMManager defaults to avoid API key mismatches
        # Remove this override once all agent configs are cleaned up
        logger.info(f"Agent requested: provider={provider}, model={model}")
        logger.info(f"Using LLMManager defaults instead (configured in llm_manager.py)")
        provider = None  # Force use of LLMManager default
        model = None     # Force use of LLMManager default

        # Delegate to centralized LLM Manager (CrewAI-specific method)
        # CrewAI requires its own LLM class, not LangChain's ChatOpenAI
        try:
            llm = LLMManager.get_crewai_llm(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return llm

        except Exception as e:
            logger.error(f"Failed to get LLM for agent: {e}")
            logger.error("Check llm_manager.py configuration")
            raise

    def _combine_results(self, results: List[str]) -> str:
        """Combine multiple results by concatenating."""
        if not results:
            return ""
        return "\n\n---\n\n".join(results)

    def _vote_on_results(self, results: List[str]) -> str:
        """Select result that appears most frequently (simple voting)."""
        if not results:
            return ""

        # Count occurrences
        from collections import Counter
        counter = Counter(results)
        most_common = counter.most_common(1)[0][0]
        return most_common

    # ========================================================================
    # VALIDATION HELPERS
    # ========================================================================

    @staticmethod
    def validate_no_orchestration_in_crewai(crew_config: Dict[str, Any]) -> bool:
        """
        Validate that CrewAI configuration doesn't contain orchestration logic.

        This prevents architectural violations where CrewAI tries to control
        graph-level decisions.
        """
        # Check for forbidden patterns
        forbidden_patterns = [
            "next_node",
            "graph.add_edge",
            "workflow_control",
            "decide_next",
            "routing_logic"
        ]

        config_str = str(crew_config).lower()
        for pattern in forbidden_patterns:
            if pattern in config_str:
                raise ValueError(
                    f"CrewAI configuration contains orchestration logic: '{pattern}'. "
                    f"CrewAI must only handle agent collaboration, not workflow control."
                )

        return True


# ============================================================================
# UTILITY FUNCTIONS FOR LANGGRAPH INTEGRATION
# ============================================================================

def create_crewai_merge_node(
    parallel_agent_configs: List[Dict[str, Any]],
    merge_strategy: str = "combine"
) -> Callable:
    """
    Create a merge node that aggregates results from parallel CrewAI agents.

    This is used in hybrid workflows where parallel branches need to merge
    before continuing sequentially.

    FIXED: Properly preserves original_user_input and crew_result for downstream agents.

    Args:
        parallel_agent_configs: Configs of agents that ran in parallel
        merge_strategy: How to merge ("combine", "vote", "llm_synthesis")

    Returns:
        Callable merge node function for LangGraph
    """
    def merge_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from parallel execution."""
        try:
            logger.info(f"Merge node processing results from {len(parallel_agent_configs)} parallel agents")

            # Preserve original user input
            original_input = state.get("original_user_input", "")

            # Collect outputs from parallel agents
            parallel_outputs = []

            # First check for individual_outputs from parallel Crew execution
            individual_outputs = state.get("individual_outputs", [])
            if individual_outputs:
                parallel_outputs.extend(individual_outputs)

            # Also check parallel_output (aggregated by parallel Crew)
            parallel_output = state.get("parallel_output", "")

            # Check crew_result
            crew_result = state.get("crew_result", "")

            # Fallback: check output_schema keys from each agent config
            if not parallel_outputs:
                for agent_config in parallel_agent_configs:
                    agent_id = agent_config.get("agent_id")
                    output_schema = agent_config.get("output_schema", [])
                    for key in output_schema:
                        if key in state and state[key]:
                            parallel_outputs.append(state[key])

            # Merge based on strategy
            if merge_strategy == "combine":
                merged = "\n\n---\n\n".join(str(o) for o in parallel_outputs if o)
            elif merge_strategy == "vote":
                from collections import Counter
                counter = Counter(parallel_outputs)
                merged = counter.most_common(1)[0][0] if counter else ""
            elif merge_strategy == "prioritize":
                merged = parallel_outputs[0] if parallel_outputs else ""
            else:
                merged = "\n\n".join(str(o) for o in parallel_outputs if o)

            # Use parallel_output if no individual outputs
            if not merged and parallel_output:
                merged = parallel_output

            # Use crew_result as fallback
            if not merged and crew_result:
                merged = crew_result

            logger.info(f"Merge node aggregated {len(parallel_outputs)} outputs using {merge_strategy} strategy")

            # Return merged state - preserve original_user_input for downstream
            return {
                "merged_output": merged,
                "crew_result": merged,  # For downstream sequential agents
                "parallel_outputs": parallel_outputs,
                "original_user_input": original_input,  # CRITICAL: Preserve for downstream
                "messages": [{
                    "node": "merge",
                    "action": "merged_parallel_results",
                    "strategy": merge_strategy,
                    "input_count": len(parallel_outputs),
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }

        except Exception as e:
            logger.error(f"Merge node failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to merge parallel results: {e}")

    return merge_node
