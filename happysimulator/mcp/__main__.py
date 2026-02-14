"""Entry point for running the MCP server.

Usage:
    python -m happysimulator.mcp
"""

import asyncio

from happysimulator.mcp.server import run_server

if __name__ == "__main__":
    asyncio.run(run_server())
