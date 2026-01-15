from fastapi import APIRouter, Depends, HTTPException
from echolib.di import container
from echolib.security import user_context
from echolib.types import *

from echolib.adapters import MemcachedSessionStore, AzureADAuth

router = APIRouter(prefix='/session', tags=['SessionApi'])

def store():
    return container.resolve('session.store')

def auth():
    return container.resolve('auth.provider')

@router.post('/login')
async def login(email: str):
    # Create a dev token and session
    from echolib.security import create_token
    tok = create_token('usr_dev', email)
    s = store().createSession('usr_dev', {'email': email})
    return {'token': tok, 'session': s.model_dump()}

@router.get('/me')
async def me(ctx: UserContext = Depends(user_context)):
    return ctx

@router.post('/logout')
async def logout(session_id: str):
    store().invalidateSession(session_id)
    return {'ok': True}
