"""Wall-clock and tool-call caps. Checked before every tool, not after a runaway loop."""

from __future__ import annotations

import time

from agent.config import Settings


class RunBudget:
    def __init__(self, settings: Settings) -> None:
        self.max_tool_calls = settings.max_tool_calls
        self.tool_calls = 0
        self.deadline = time.monotonic() + settings.total_wall_clock_s

    def remaining_s(self) -> float:
        return self.deadline - time.monotonic()

    def can_call_tool(self) -> bool:
        return self.tool_calls < self.max_tool_calls and self.remaining_s() > 1.0

    def record_tool(self) -> None:
        self.tool_calls += 1
