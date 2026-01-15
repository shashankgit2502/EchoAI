from fastapi import APIRouter, Depends, HTTPException
from echolib.di import container
from echolib.security import user_context
from echolib.types import *

from echolib.types import AppDef, App, DeployResult
from echolib.utils import new_id

router = APIRouter(prefix='/apps', tags=['AppApi'])

def store():
    return container.resolve('app.store')

@router.post('/create')
async def create(defn: AppDef):
    app = App(id=new_id('app_'), name=defn.name, config=defn.config)
    store()[app.id] = app
    return app.model_dump()

@router.get('/list')
async def list_apps():
    return [a.model_dump() for a in store().values()]

@router.post('/deploy/{app_id}')
async def deploy(app_id: str, env: str):
    if app_id not in store():
        raise HTTPException(status_code=404, detail='Not found')
    return DeployResult(app_id=app_id, env=env, status='deployed').model_dump()
