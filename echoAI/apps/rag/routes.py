from fastapi import APIRouter, Depends, HTTPException
from echolib.di import container
from echolib.security import user_context
from echolib.types import *

from echolib.services import RAGService

def svc() -> RAGService:
    return container.resolve('rag.service')

router = APIRouter(prefix='/rag', tags=['RAGApi'])

@router.post('/index')
async def index(docs: list[Document]):
    return svc().indexDocs(docs).model_dump()

@router.get('/search')
async def search(q: str):
    return svc().queryIndex(q, {}).model_dump()
