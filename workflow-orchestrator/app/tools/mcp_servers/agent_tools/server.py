from mcp.server import Server
from mcp.server.stdio import StdioServerTransport
from mcp.types import Tool

from app.tools.mcp_servers.agent_tools.tools.web_search_tool import handle_web_search
from app.tools.mcp_servers.agent_tools.tools.calculator_tool import handle_calculator
from app.tools.mcp_servers.agent_tools.tools.file_reader_tool import handle_file_reader

server = Server(
    name="agent-tools",
    version="1.0.0",
)


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="web_search",
            description="Search the web using Bing (enabled), DuckDuckGo or Google (disabled by policy)",
            input_schema={
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "enum": ["bing", "duckduckgo", "google"]},
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["provider", "query"],
            },
        )
        ,
        Tool(
            name="calculator",
            description="Advanced calculator supporting arithmetic, statistics, vectors, matrices, and precision control",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "values": {"type": "array", "items": {"type": "number"}},
                    "vectors": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                    "matrices": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                    },
                    "precision": {"type": "integer"},
                    "rounding": {"type": "string", "enum": ["round", "floor", "ceil", "none"]},
                },
                "required": ["operation"],
            },
        ),
        Tool(
            name="file_reader",
            description="Read files from base64 input and support extract, query, and summarize modes",
            input_schema={
                "type": "object",
                "properties": {
                    "file_name": {"type": "string"},
                    "mime_type": {"type": "string"},
                    "content_base64": {"type": "string"},
                    "mode": {"type": "string", "enum": ["extract", "query", "summarize"]},
                    "query": {"type": "string"},
                    "stream": {"type": "boolean"},
                    "chunk_size": {"type": "integer"},
                    "chunk_overlap": {"type": "integer"},
                },
                "required": ["file_name", "mime_type", "content_base64"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "web_search":
        results = await handle_web_search(arguments)
        return {"content": [{"type": "json", "json": results}]}
    if name == "calculator":
        result = await handle_calculator(arguments)
        return {"content": [{"type": "json", "json": result}]}
    if name == "file_reader":
        result = await handle_file_reader(arguments)
        return {"content": [{"type": "json", "json": result}]}
    raise ValueError(f"Unknown tool: {name}")


async def main():
    transport = StdioServerTransport()
    await server.run(transport)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
