"""
ToolManager — Manages MCP server connections via FastMCP stdio transport.

Spawns MCP servers as child processes, discovers their tools, and routes
tool calls to the correct server. All config comes from CORTEX's own .env.

Usage:
    async with ToolManager(mcp_config) as tm:
        tools = await tm.list_all_tools()
        result = await tm.call_tool("notion", "API-post-search", {"query": "tasks"})
"""

import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from fastmcp import Client
from fastmcp.client.transports.stdio import StdioTransport

logger = logging.getLogger("cortex.tool_manager")


class ToolManager:
    """
    Manages MCP server connections and routes tool calls.
    """

    def __init__(self, mcp_config: Dict[str, Dict]):
        """
        Args:
            mcp_config: Dict of server configs, e.g.:
                {
                    "notion": {
                        "command": "node",
                        "args": ["C:/.../notion-mcp-server/bin/cli.mjs"],
                        "env": {"NOTION_TOKEN": "ntn_***"}
                    },
                    "slack": {
                        "command": "C:/.../slack-mcp-server.exe",
                        "args": ["--transport", "stdio"],
                        "env": {"SLACK_MCP_XOXP_TOKEN": "xoxp-***"}
                    }
                }
        """
        self.mcp_config = mcp_config
        self.clients: Dict[str, Client] = {}
        self._tool_map: Dict[str, Dict] = {}
        self._exit_stack: Optional[AsyncExitStack] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        return False

    async def connect(self):
        """Connect to all configured MCP servers."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for name, config in self.mcp_config.items():
            try:
                env = {**os.environ, **config.get("env", {})}

                transport = StdioTransport(
                    command=config["command"],
                    args=config.get("args", []),
                    env=env,
                )

                client = Client(transport=transport)
                connected_client = await self._exit_stack.enter_async_context(client)
                self.clients[name] = connected_client
                logger.info(f"Connected to MCP server: {name}")

                tools = await connected_client.list_tools()
                for tool in tools:
                    tool_name = tool.name if hasattr(tool, "name") else tool.get("name", "")
                    self._tool_map[f"{name}.{tool_name}"] = {
                        "server": name,
                        "tool": tool,
                    }
                logger.info(f"  Found {len(tools)} tools from {name}")

            except Exception as e:
                logger.error(f"Failed to connect to MCP server '{name}': {e}")
                # Continue connecting other servers instead of aborting everything
                continue

    async def list_all_tools(self) -> List[Dict[str, Any]]:
        """Return all tools from all connected MCP servers."""
        tools = []
        for qualified_name, info in self._tool_map.items():
            tool = info["tool"]
            tools.append({
                "qualified_name": qualified_name,
                "server": info["server"],
                "name": tool.name if hasattr(tool, "name") else tool.get("name", ""),
                "description": tool.description if hasattr(tool, "description") else tool.get("description", ""),
            })
        return tools

    async def list_tools_for_server(self, server_name: str) -> List[Dict[str, Any]]:
        """Return tools for a specific server."""
        return [t for t in await self.list_all_tools() if t["server"] == server_name]

    async def call_tool(self, server: str, tool_name: str, arguments: Dict[str, Any] = None) -> Any:
        """Call a tool on a specific MCP server."""
        if server not in self.clients:
            raise ValueError(f"Server '{server}' not connected. Available: {list(self.clients.keys())}")

        client = self.clients[server]
        logger.info(f"Calling {server}.{tool_name} with args: {json.dumps(arguments or {}, default=str)}")

        try:
            result = await client.call_tool(tool_name, arguments or {})
            logger.info(f"{server}.{tool_name} returned successfully")
            return result
        except Exception as e:
            logger.error(f"{server}.{tool_name} failed: {e}")
            raise

    async def disconnect(self):
        """Disconnect from all MCP servers."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self.clients.clear()
        self._tool_map.clear()
        logger.info("Disconnected from all MCP servers")

    def get_tools_description(self) -> str:
        """Human-readable tool list for LLM system prompts."""
        lines = ["Available MCP Tools:"]
        current_server = ""
        for qualified_name, info in self._tool_map.items():
            server = info["server"]
            if server != current_server:
                lines.append(f"\n[{server.upper()}]")
                current_server = server
            tool = info["tool"]
            name = tool.name if hasattr(tool, "name") else tool.get("name", "")
            desc = tool.description if hasattr(tool, "description") else tool.get("description", "")
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)

    @property
    def is_connected(self) -> bool:
        return len(self.clients) > 0

    @property
    def connected_servers(self) -> List[str]:
        return list(self.clients.keys())
