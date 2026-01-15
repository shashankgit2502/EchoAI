
from echolib.di import container
from echolib.services import DocumentStore, RAGService

_store = DocumentStore()
container.register('rag.store', lambda: _store)
container.register('rag.service', lambda: RAGService(_store))
