
from echolib.di import container
from echolib.types import App
_app_store: dict[str, App] = {}

def app_store():
    return _app_store

container.register('app.store', app_store)
