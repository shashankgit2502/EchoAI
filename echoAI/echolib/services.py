
from typing import List, Callable
from .interfaces import ILogger, IEventBus, ICredentialStore
from .types import *
from .utils import new_id

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
        return AgentTemplate(name=name)
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
    def __init__(self, mcp, custom):
        self.mcp = mcp
        self.custom = custom
        self._conns: dict[str, ConnectorDef] = {}
    def register(self, conn: ConnectorDef) -> ConnectorRef:
        self._conns[conn.name] = conn
        return ConnectorRef(name=conn.name)
    def invoke(self, name: str, payload: dict) -> ConnectorResult:
        if name not in self._conns:
            raise ValueError('connector not found')
        return ConnectorResult(name=name, result={'payload': payload})
    def list(self) -> List[ConnectorRef]:
        return [ConnectorRef(name=n) for n in self._conns.keys()]

class MCPConnector:
    def connect(self, endpoint: str, token: str):
        return {'connected': True}
    def sendMessage(self, sessionId: str, content: dict) -> dict:
        return {'ok': True, 'sessionId': sessionId, 'content': content}
    def getSession(self, sessionId: str) -> dict:
        return {'sessionId': sessionId}

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
