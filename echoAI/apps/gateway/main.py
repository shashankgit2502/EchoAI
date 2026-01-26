
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from echolib.config import settings
import logging
import time

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tool System Logging Middleware
@app.middleware("http")
async def log_tool_operations(request: Request, call_next):
    """
    Middleware to log all tool-related operations for debugging and monitoring.
    Logs: discovery, invocation, registration, deletion, and errors.
    """
    start_time = time.time()
    path = request.url.path
    method = request.method

    # Check if this is a tool-related request
    is_tool_request = path.startswith("/tools")

    if is_tool_request:
        logger.info(f"🔧 Tool Request: {method} {path}")

        # Log request details for specific operations
        if "/tools/discover" in path:
            logger.info("🔍 Tool Discovery initiated")
        elif "/tools/invoke" in path:
            tool_identifier = path.split("/")[-1] if "/" in path else "unknown"
            logger.info(f"⚡ Tool Invocation: {tool_identifier}")
        elif "/tools/list" in path:
            logger.info("📋 Listing all registered tools")
        elif method == "POST" and path == "/tools/register":
            logger.info("➕ Tool Registration request")
        elif method == "DELETE":
            tool_id = path.split("/")[-1]
            logger.info(f"🗑️ Tool Deletion request: {tool_id}")

    # Process request
    try:
        response = await call_next(request)

        # Log completion with execution time
        if is_tool_request:
            duration = (time.time() - start_time) * 1000  # Convert to ms
            status = response.status_code

            if status >= 200 and status < 300:
                logger.info(f"✅ Tool Request completed: {method} {path} - Status: {status} - Duration: {duration:.2f}ms")
            elif status >= 400 and status < 500:
                logger.warning(f"⚠️ Tool Request client error: {method} {path} - Status: {status} - Duration: {duration:.2f}ms")
            elif status >= 500:
                logger.error(f"❌ Tool Request server error: {method} {path} - Status: {status} - Duration: {duration:.2f}ms")

        return response

    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(f"❌ Tool Request exception: {method} {path} - Error: {str(e)} - Duration: {duration:.2f}ms")
        raise

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
