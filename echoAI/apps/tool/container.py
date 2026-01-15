
from echolib.di import container
from echolib.services import ToolService
_tool = ToolService()
container.register('tool.service', lambda: _tool)
