from fastapi import APIRouter, Depends, HTTPException
from echolib.di import container
from echolib.security import user_context
from echolib.types import *

from echolib.services import ToolService
router = APIRouter(prefix='/tools', tags=['ToolApi'])

def svc() -> ToolService:
    return container.resolve('tool.service')

@router.post('/register')
async def register(tool: ToolDef):
    return svc().registerTool(tool).model_dump()

@router.get('/list')
async def list_tools():
    return [t.model_dump() for t in svc().listTools()]

@router.post('/invoke/{name}')
async def invoke(name: str, args: dict):
    return svc().invokeTool(name, args).model_dump()
