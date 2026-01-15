from fastapi import APIRouter, Depends, HTTPException
from echolib.di import container
from echolib.security import user_context
from echolib.types import *

from echolib.services import ConnectorManager
from echolib.types import ConnectorDef

router = APIRouter(prefix='/connectors', tags=['ConnectorApi'])

def mgr() -> ConnectorManager:
    return container.resolve('connector.manager')

@router.post('/register')
async def register(conn: ConnectorDef):
    return mgr().register(conn).model_dump()

@router.post('/invoke/{name}')
async def invoke(name: str, payload: dict):
    return mgr().invoke(name, payload).model_dump()

@router.get('/list')
async def list_connectors():
    return [c.model_dump() for c in mgr().list()]
