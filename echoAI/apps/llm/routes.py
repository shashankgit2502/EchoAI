from fastapi import APIRouter, Depends, HTTPException
from echolib.di import container
from echolib.security import user_context
from echolib.types import *

from echolib.services import LLMService

def svc() -> LLMService:
    return container.resolve('llm.service')

router = APIRouter(prefix='/llm', tags=['LLMApi'])

@router.post('/generate')
async def generate(prompt: str):
    return svc().generate(prompt, []).model_dump()

@router.get('/stream')
async def stream(prompt: str):
    tokens = []
    svc().stream(prompt, lambda t: tokens.append(t))
    return {'tokens': tokens}
