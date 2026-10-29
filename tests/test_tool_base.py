import pytest

from agent.tools.base import BaseTool, EmptyInput, EmptyOutput, ToolError
from agent.tracing import Tracer


class BoomTool(BaseTool[EmptyInput, EmptyOutput]):
    name = "boom"

    def __init__(self, failures: int, retryable: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.failures = failures
        self.retryable = retryable
        self.calls = 0

    async def _execute(self, payload: EmptyInput) -> EmptyOutput:
        self.calls += 1
        if self.calls <= self.failures:
            raise ToolError("transient", tool=self.name, retryable=self.retryable)
        return EmptyOutput(detail=payload.note)


async def test_retry_then_success(tmp_path) -> None:
    tracer = Tracer.create(tmp_path, "t1")
    tool = BoomTool(failures=2, tracer=tracer, timeout_s=5)
    result = await tool.run(EmptyInput(note="x"))
    assert result.detail == "x"
    assert tool.calls == 3
    assert tracer.totals.tool_calls == 1


async def test_non_retryable_fails_fast() -> None:
    tool = BoomTool(failures=5, retryable=False, timeout_s=5)
    with pytest.raises(ToolError) as exc:
        await tool.run(EmptyInput())
    assert tool.calls == 1
    assert exc.value.retryable is False


async def test_retry_exhausted() -> None:
    tool = BoomTool(failures=10, retryable=True, timeout_s=5)
    tool.max_attempts = 3
    with pytest.raises(ToolError):
        await tool.run(EmptyInput())
    assert tool.calls == 3
