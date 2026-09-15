"""FastMCP server exposing a get_time tool over stdio."""
from __future__ import annotations

from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("time")


@mcp.tool()
def get_time(timezone: str = "") -> str:
    """Get the current wall-clock time and date.

    Args:
        timezone: Optional IANA timezone name (e.g. "Asia/Seoul", "America/New_York").
                  If empty, the server's local time is used.
    """
    tz = None
    place = "your local time zone"
    if timezone:
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(timezone)
            place = timezone.replace("_", " ")
        except Exception:
            tz = None
            place = "your local time zone"

    now = datetime.now(tz)
    spoken = now.strftime("%A, %B %d, %Y at %I:%M %p").replace(" 0", " ")
    return f"(tool result) It is {spoken} in {place} right now."


if __name__ == "__main__":
    mcp.run(transport="stdio")
