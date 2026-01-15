
from echolib.di import container
from echolib.services import ConnectorManager, MCPConnector, CustomConnector


_mcp = MCPConnector()
_custom = CustomConnector()
_mgr = ConnectorManager(_mcp, _custom)

container.register('connector.manager', lambda: _mgr)
