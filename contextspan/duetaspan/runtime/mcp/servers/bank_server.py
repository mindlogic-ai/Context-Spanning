"""Serve every executable bank tool over the MCP protocol, on stdio.

The four servers next to this one are FastMCP: one decorated Python function per
tool. That does not scale to 87 tools whose JSON Schemas already exist in
``mcp_tool_bank.json`` — writing 87 signatures by hand would only re-derive what
the bank already states, and would drift from it.

So this uses the low-level ``Server`` and hands the bank's own schemas straight
to ``list_tools``. One file, every supported tool, no boilerplate, and the
advertised schema is by construction the schema the router already sees.

Unsupported tools are not advertised — a client cannot call a browser tool on a
machine with no browser.

    python -m contextspan.duetaspan.runtime.mcp.servers.bank_server
"""
from __future__ import annotations

import asyncio
import logging

import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from contextspan.duetaspan.runtime.mcp.registry import get_registry

logging.basicConfig(level=logging.WARNING)   # stdout is the MCP channel — keep it clean

server = Server("bank")
_registry = get_registry()


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    tools = []
    for name in sorted(_registry.supported()):
        spec = _registry.bank[name]
        tools.append(types.Tool(
            name=name,
            description=spec.get("description") or name,
            inputSchema=_registry.input_schema(name),
        ))
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Run the tool for real, off the event loop — the adapters block on I/O."""
    result = await asyncio.to_thread(_registry.dispatch, name, arguments or {})
    return [types.TextContent(type="text", text=str(result))]


async def _main() -> None:
    async with stdio_server() as (read, write):
        await server.run(
            read, write,
            InitializationOptions(
                server_name="bank",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    # Finish the TLS handshakes before the first tool call arrives. Open-Elevation's first
    # request cost 9.7 s and its second 226 ms; no user should ever meet the first.
    from contextspan.duetaspan.runtime.mcp import adapters_maps

    adapters_maps.warm_connections(block=12.0)
    asyncio.run(_main())
