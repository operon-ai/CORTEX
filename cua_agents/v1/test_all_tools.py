"""
Test all tools from both MCP servers.
Run: uv run python -m cua_agents.v1.test_all_tools
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Only load from CORTEX .env
load_dotenv()

from cua_agents.v1.tools.tool_manager import ToolManager


MCP_SERVERS = os.getenv("MCP_SERVERS_PATH", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "mcp-servers")
))


def build_mcp_config() -> dict:
    """Build MCP server config from CORTEX .env vars."""
    config = {}

    # Notion MCP Server
    notion_token = os.getenv("NOTION_TOKEN")
    if notion_token:
        config["notion"] = {
            "command": "node",
            "args": [os.path.join(MCP_SERVERS, "notion-mcp-server", "bin", "cli.mjs")],
            "env": {"NOTION_TOKEN": notion_token},
        }
        print(f"  ✓ Notion server configured")
    else:
        print(f"  ✗ Notion: NOTION_TOKEN not set in .env")

    # Slack MCP Server
    slack_token = os.getenv("SLACK_MCP_XOXP_TOKEN")
    if slack_token:
        exe = os.path.join(MCP_SERVERS, "slack-mcp-server", "build", "slack-mcp-server.exe")
        if os.path.exists(exe):
            config["slack"] = {
                "command": exe,
                "args": ["--transport", "stdio"],
                "env": {
                    "SLACK_MCP_XOXP_TOKEN": slack_token,
                    "SLACK_MCP_ADD_MESSAGE_TOOL": os.getenv("SLACK_MCP_ADD_MESSAGE_TOOL", "true"),
                },
            }
            print(f"  ✓ Slack server configured")
        else:
            print(f"  ✗ Slack: binary not found at {exe}")
    else:
        print(f"  ✗ Slack: SLACK_MCP_XOXP_TOKEN not set in .env")

    return config


async def main():
    print("\n🔧 Building MCP config...\n")
    config = build_mcp_config()

    if not config:
        print("\n❌ No MCP servers configured. Add tokens to CORTEX .env file.")
        sys.exit(1)

    print(f"\n🔌 Connecting to {len(config)} MCP server(s)...\n")

    try:
        async with ToolManager(config) as tm:
            print(f"\n✅ Connected to: {tm.connected_servers}")

            tools = await tm.list_all_tools()
            print(f"\n📦 Total tools discovered: {len(tools)}\n")

            current_server = ""
            for i, tool in enumerate(tools, 1):
                if tool["server"] != current_server:
                    current_server = tool["server"]
                    print(f"\n  [{current_server.upper()}]")
                print(f"    {i:2d}. {tool['name']}")

            # ── Notion tests ──
            if "notion" in tm.connected_servers:
                print("\n\n🧪 TEST 1: Notion API-get-self...")
                r = await tm.call_tool("notion", "API-get-self", {})
                print(f"   {'✅ PASS' if not r.is_error else '❌ FAIL'}")

                print("\n🧪 TEST 2: Notion API-post-search...")
                r = await tm.call_tool("notion", "API-post-search", {"query": "test", "page_size": 3})
                print(f"   {'✅ PASS' if not r.is_error else '❌ FAIL'}")

            # ── Slack tests ──
            if "slack" in tm.connected_servers:
                print("\n🧪 TEST 3: Slack channels_list...")
                r = await tm.call_tool("slack", "channels_list", {})
                print(f"   {'✅ PASS' if not r.is_error else '❌ FAIL'}")

                print("\n🧪 TEST 4: Slack conversations_history (#general)...")
                r = await tm.call_tool("slack", "conversations_history", {"channel_id": "#general", "limit": "3"})
                print(f"   {'✅ PASS' if not r.is_error else '❌ FAIL'}")

                print("\n🧪 TEST 5: Slack conversations_unreads...")
                r = await tm.call_tool("slack", "conversations_unreads", {})
                print(f"   {'✅ PASS' if not r.is_error else '❌ FAIL'}")

                print("\n🧪 TEST 6: Slack users_search...")
                r = await tm.call_tool("slack", "users_search", {"query": "udit"})
                print(f"   {'✅ PASS' if not r.is_error else '❌ FAIL'}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n\n✅ All tests complete.")


if __name__ == "__main__":
    asyncio.run(main())
