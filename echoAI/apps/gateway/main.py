
from fastapi import FastAPI
from echolib.config import settings

# Import containers to register providers
from apps.session import container as _sess_container  # noqa: F401
from apps.rag import container as _rag_container  # noqa: F401
from apps.llm import container as _llm_container  # noqa: F401
from apps.tool import container as _tool_container  # noqa: F401
from apps.appmgr import container as _app_container  # noqa: F401
from apps.workflow import container as _wf_container  # noqa: F401
from apps.agent import container as _agent_container  # noqa: F401
from apps.connector import container as _conn_container  # noqa: F401
from apps.chat import container as _chat_container  # noqa: F401

# Routers
from apps.session.routes import router as session_router
from apps.rag.routes import router as rag_router
from apps.llm.routes import router as llm_router
from apps.tool.routes import router as tool_router
from apps.appmgr.routes import router as app_router
from apps.workflow.routes import router as wf_router
from apps.agent.routes import router as agent_router
from apps.connector.routes import router as conn_router
from apps.chat.routes import router as chat_router

app = FastAPI(title=f"{settings.app_name} (Gateway)")

@app.get('/healthz')
async def healthz():
    return {'status': 'ok', 'mode': settings.service_mode}

app.include_router(session_router)
app.include_router(rag_router)
app.include_router(llm_router)
app.include_router(tool_router)
app.include_router(app_router)
app.include_router(wf_router)
app.include_router(agent_router)
app.include_router(conn_router)
app.include_router(chat_router)
