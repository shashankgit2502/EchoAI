
from echolib.di import container
from echolib.adapters import OTelLogger, KeyVaultClient
from echolib.services import TemplateRepository, LangGraphBuilder, AgentService

# Import new orchestrator services
from .registry.registry import AgentRegistry
from .factory.factory import AgentFactory
from .permissions.permissions import AgentPermissions
from .designer.agent_designer import AgentDesigner

# New orchestrator services (initialized first so AgentService can use them)
_registry = AgentRegistry()
_factory = AgentFactory(tool_registry={})  # TODO: Add real tool registry
_permissions = AgentPermissions()
_designer = AgentDesigner()

# Existing services enhanced with registry and designer injection
_tpl = TemplateRepository()
_graph = LangGraphBuilder()
_agentsvc = AgentService(
    tpl_repo=_tpl,
    graph_builder=_graph,
    registry=_registry,
    designer=_designer
)

# Register existing services
container.register('agent.service', lambda: _agentsvc)
container.register('cred.store', lambda: KeyVaultClient())
container.register('logger', lambda: OTelLogger())

# Register new orchestrator services
container.register('agent.registry', lambda: _registry)
container.register('agent.factory', lambda: _factory)
container.register('agent.permissions', lambda: _permissions)
container.register('agent.designer', lambda: _designer)
