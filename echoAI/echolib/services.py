
from typing import List, Callable
from pathlib import Path
from .interfaces import ILogger, IEventBus, ICredentialStore
from .types import *
from .utils import new_id

from typing import Dict, Any, Optional, List
import asyncio
from echolib.adaptors.Get_MCP.http_script import HTTPMCPConnector
from echolib.adaptors.Get_MCP.sse import SSEMCPConnector
from echolib.adaptors.Get_MCP.stdio import STDIOMCPConnector
from echolib.adaptors.Get_MCP.storage import get_storage
from echolib.adaptors.Get_MCP.validator import validate_and_normalize, ValidationError
from echolib.adaptors.Get_API.connectors.factory import ConnectorFactory
from echolib.adaptors.Get_API.models import ConnectorConfig


class DocumentStore:
    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
    def put(self, doc: Document) -> None:
        self._docs[doc.id] = doc
    def get(self, id: str) -> Document:
        return self._docs[id]
    def search(self, query: str) -> List[Document]:
        q = query.lower()
        return [d for d in self._docs.values() if q in d.title.lower() or q in d.content.lower()]

class StateStore:
    def __init__(self) -> None:
        self._s: dict[str, dict] = {}
    def put(self, key: str, value: dict) -> None:
        self._s[key] = value
    def get(self, key: str) -> dict:
        return self._s.get(key, {})
    def del_(self, key: str) -> None:
        self._s.pop(key, None)

class ToolService:
    def __init__(self, cred_store: ICredentialStore | None = None):
        self._tools: dict[str, ToolDef] = {}
        self._cred = cred_store
    def registerTool(self, tool: ToolDef) -> ToolRef:
        self._tools[tool.name] = tool
        return ToolRef(name=tool.name)
    def listTools(self) -> List[ToolRef]:
        return [ToolRef(name=n) for n in self._tools.keys()]
    def invokeTool(self, name: str, args: dict) -> ToolResult:
        if name not in self._tools:
            raise ValueError('tool not found')
        return ToolResult(name=name, output={'echo': args})

class RAGService:
    def __init__(self, store: DocumentStore):
        self.store = store
    def indexDocs(self, docs: List[Document]) -> IndexSummary:
        for d in docs:
            self.store.put(d)
        return IndexSummary(count=len(docs))
    def queryIndex(self, query: str, filters: dict) -> ContextBundle:
        return ContextBundle(documents=self.store.search(query))
    def vectorize(self, text: str) -> List[float]:
        return [float(len(text))]

class LLMService:
    def __init__(self, toolsvc: ToolService):
        self.toolsvc = toolsvc
    def generate(self, prompt: str, tools: List[ToolRef]) -> LLMOutput:
        return LLMOutput(text=f"LLM says: {prompt}")
    def stream(self, prompt: str, onToken: Callable[[str], None]) -> None:
        for t in prompt.split(' '):
            onToken(t)
    def toolCall(self, name: str, args: dict) -> ToolResult:
        return self.toolsvc.invokeTool(name, args)

class TemplateRepository:
    def getAgentTemplate(self, name: str) -> AgentTemplate:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "agents"
        if name == "agentCard":
            template_path = data_dir / "agent-card.json"
        else:
            template_path = data_dir / "agent.json"

        if not template_path.exists():
            raise FileNotFoundError(f"Agent template not found at {template_path}")

        return AgentTemplate(name=name, template_path=str(template_path))
    def getWorkflowTemplate(self, name: str) -> Workflow:
        return Workflow(id=new_id('wf_'), name=name)

class LangGraphBuilder:
    def buildFromPrompt(self, prompt: str, template: AgentTemplate):
        return {'graph': 'built', 'prompt': prompt, 'template': template.name}
    def compile(self, graph) -> dict:
        return {'runnable': True}

class AgentService:
    def __init__(self, tpl_repo: TemplateRepository, graph_builder: LangGraphBuilder, cred: ICredentialStore | None = None, log: ILogger | None = None):
        self.tpl_repo = tpl_repo
        self.graph_builder = graph_builder
        self.agents: dict[str, Agent] = {}
        self.log = log
    def createFromPrompt(self, prompt: str, template: AgentTemplate) -> Agent:
        a = Agent(id=new_id('agt_'), name=template.name)
        self.agents[a.id] = a
        return a
    def createFromCanvasCard(self, cardJSON: dict, template: AgentTemplate) -> Agent:
        a = Agent(id=new_id('agt_'), name=template.name)
        self.agents[a.id] = a
        return a
    def validateA2A(self, agent: Agent) -> ValidationResult:
        return ValidationResult(ok=True)
    def listAgents(self) -> List[Agent]:
        return list(self.agents.values())

class WorkflowService:
    def __init__(self, agentsvc: AgentService, bus: IEventBus | None = None):
        self.agentsvc = agentsvc
        self.bus = bus
    def createFromPrompt(self, prompt: str, agents: List[Agent]) -> Workflow:
        wf = Workflow(id=new_id('wf_'), name='wf_from_prompt')
        return wf
    def createFromCanvas(self, canvasJSON: dict) -> Workflow:
        return Workflow(id=new_id('wf_'), name='wf_from_canvas')
    def validate(self, workflow: Workflow) -> ValidationResult:
        return ValidationResult(ok=True)
    def publish(self, workflow: Workflow) -> None:
        pass

class ConnectorManager:
    """Unified manager routing to API or MCP connectors."""
    
    def __init__(self):
        self.api = APIConnectorManager()
        self.mcp = MCPConnectorManager()
    
    def get_manager(self, connector_type: str):
        """Route to appropriate manager."""
        if connector_type == "api":
            return self.api
        elif connector_type == "mcp":
            return self.mcp
        else:
            raise ValueError("connector_type must be 'api' or 'mcp'")

        
class MCPConnectorManager:
    """Manages MCP connectors (HTTP/SSE/STDIO) with storage."""
    
    def __init__(self):
        self._connectors: Dict[str, Any] = {}
        self.storage = get_storage()
    
    def create(self, config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            normalized = validate_and_normalize(config)
            transport = normalized["transport_type"]
            
            if transport == "http":
                connector = HTTPMCPConnector(**{
                    k: v for k, v in normalized.items()
                    if k in HTTPMCPConnector.__init__.__code__.co_varnames
                })
            elif transport == "sse":
                connector = SSEMCPConnector(**{
                    k: v for k, v in normalized.items()
                    if k in SSEMCPConnector.__init__.__code__.co_varnames
                })
            elif transport == "stdio":
                connector = STDIOMCPConnector(**{
                    k: v for k, v in normalized.items()
                    if k in STDIOMCPConnector.__init__.__code__.co_varnames
                })
            else:
                raise ValueError(f"Unsupported transport: {transport}")
            
            is_valid, errors = connector.validate_config()
            if not is_valid:
                return {"success": False, "error": "Validation failed", "errors": errors}
            
            self.storage.save(connector.serialize())
            self._connectors[connector.connector_id] = connector
            
            return {
                "success": True,
                "connector_id": connector.connector_id,
                "name": connector.name,
                "transport_type": connector.transport_type.value
            }
            
        except ValidationError as e:
            return {"success": False, "error": "Validation error", "errors": e.errors}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def invoke(self, connector_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if connector_id not in self._connectors:
                data = self.storage.load(connector_id)
                if not data:
                    return {"success": False, "error": f"Connector not found: {connector_id}"}
                
                transport = data["transport_type"]
                if transport == "http":
                    connector = HTTPMCPConnector.from_dict(data)
                elif transport == "sse":
                    connector = SSEMCPConnector.from_dict(data)
                elif transport == "stdio":
                    connector = STDIOMCPConnector.from_dict(data)
                else:
                    return {"success": False, "error": "Invalid transport"}
                
                self._connectors[connector_id] = connector
            else:
                connector = self._connectors[connector_id]
            
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.ensure_future(connector.test(payload))
                while not future.done():
                    loop._run_once()
                result = future.result()
            except RuntimeError:
                result = asyncio.run(connector.test(payload))
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get(self, connector_id: str) -> Optional[Dict[str, Any]]:
        data = self.storage.load(connector_id)
        return data if data else None
    
    def list(self) -> Dict[str, Any]:
        connectors = self.storage.list_all()
        return {"success": True, "count": len(connectors), "connectors": connectors}
    
    def delete(self, connector_id: str) -> Dict[str, Any]:
        try:
            success = self.storage.delete(connector_id)
            if connector_id in self._connectors:
                del self._connectors[connector_id]
            
            return {
                "success": success,
                "message": f"Connector {connector_id} deleted" if success else "Deletion failed"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}



class APIConnectorManager:
    """Manages API connectors with storage."""
   
    def __init__(self):
        self.factory = ConnectorFactory
        self._connectors = {}
   
    def create(self, definition: Dict[str, Any]) -> Dict[str, Any]:
        """Create API connector from definition."""
        try:
            # Build auth config from route format
            auth_config = {
                "type": definition["auth_type"],
                **definition.get("auth_config", {})
            }
           
            config = ConnectorConfig(
                id=definition["name"],
                name=definition["name"],
                base_url=definition["base_url"],
                auth=auth_config,
                default_headers=definition.get("default_headers", {}),
                timeout=float(definition.get("timeout", 30)),
                verify_ssl=True
            )
           
            connector = self.factory.create(config)
           
            self._connectors[definition["name"]] = {
                "connector": connector,
                "endpoints": definition.get("endpoints", {}),
                "definition": definition
            }
           
            return {
                "success": True,
                "connector_name": definition["name"]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
   
    def invoke(self, connector_name: str, endpoint_name: str, params: Dict = None, body: Dict = None) -> Dict[str, Any]:
        """Invoke connector endpoint."""
        try:
            connector_data = self._connectors.get(connector_name)
            if not connector_data:
                return {"success": False, "error": f"Connector '{connector_name}' not found"}
           
            connector = connector_data["connector"]
            method = "POST" if body else "GET"
           
            result = connector.execute(
                method=method,
                endpoint=endpoint_name or "/",
                query_params=params,
                body=body
            )
           
            return {
                "success": result.success,
                "data": result.body,
                "status_code": result.status_code
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
   
    def get(self, connector_name: str) -> Optional[Dict[str, Any]]:
        """Get connector details."""
        try:
            connector_data = self._connectors.get(connector_name)
            if not connector_data:
                return None
           
            return {
                "success": True,
                "connector_name": connector_name,
                "base_url": connector_data["definition"].get("base_url"),
                "auth_type": connector_data["definition"].get("auth_type"),
                "endpoints": connector_data["endpoints"]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
   
    def list(self) -> Dict[str, Any]:
        """List all connectors."""
        try:
            connectors = [
                {
                    "connector_name": name,
                    "base_url": data["definition"].get("base_url"),
                    "auth_type": data["definition"].get("auth_type")
                }
                for name, data in self._connectors.items()
            ]
           
            return {
                "success": True,
                "count": len(connectors),
                "connectors": connectors
            }
        except Exception as e:
            return {"success": False, "error": str(e), "count": 0, "connectors": []}
   
    def delete(self, connector_name: str) -> Dict[str, Any]:
        """Delete connector."""
        try:
            if connector_name not in self._connectors:
                return {"success": False, "error": f"Connector '{connector_name}' not found"}
           
            del self._connectors[connector_name]
           
            return {
                "success": True,
                "message": f"Connector '{connector_name}' deleted successfully"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}



class CustomConnector:
    def __init__(self):
        self._cfg = {}
    def configure(self, config: dict) -> None:
        self._cfg = config
    def call(self, operation: str, payload: dict) -> ConnectorResult:
        return ConnectorResult(name=operation, result={'payload': payload})
    def healthCheck(self) -> Health:
        return Health(status='ok')

class ChatOrchestrator:
    def __init__(self, bus: IEventBus, llm: LLMService, rag: RAGService, wf: WorkflowService, ag: AgentService, conns: ConnectorManager, log: ILogger | None = None):
        self.bus = bus
        self.llm = llm
        self.rag = rag
        self.wf = wf
        self.ag = ag
        self.conns = conns
        self.log = log
    def orchestrate(self, msg: dict, session: Session) -> dict:
        ctx = self.rag.queryIndex(msg.get('content',''), {})
        out = self.llm.generate(msg.get('content',''), [])
        self.bus.publish('chat.events', Event(type='chat.completion', data={'text': out.text}))
        return {'reply': out.text, 'ctx_docs': [d.id for d in ctx.documents]}
    def routeToAgent(self, msg: dict) -> Agent:
        return Agent(id='agt_default', name='default')
    def routeToService(self, intent: str):
        return {'service': intent}
    def publishEvent(self, topic: str, event: Event) -> None:
        self.bus.publish(topic, event)
