from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from agent.tools.base import BaseTool, ToolError
from agent.tools.search import SearchHit


class In(BaseModel):
    q: str


class Out(BaseModel):
    value: int


class BadOutputTool(BaseTool[In, Out]):
    name = "bad_output"

    async def _execute(self, payload: In) -> Out:
        return Out.model_validate({"value": "not-an-int-and-not-coercible-maybe"})


async def test_malformed_output_becomes_toolerror() -> None:
    # Pydantic will coerce some strings; force a real ValidationError path.
    class StrictTool(BaseTool[In, Out]):
        name = "strict"

        async def _execute(self, payload: In) -> Out:
            raise ValidationError.from_exception_data(
                "Out",
                [{"type": "int_type", "loc": ("value",), "input": {"nested": True}, "msg": "bad"}],
            )

    tool = StrictTool(timeout_s=5)
    with pytest.raises(ToolError, match="malformed tool output"):
        await tool.run(In(q="x"))


def test_search_hit_rejects_non_http() -> None:
    with pytest.raises(ValidationError):
        SearchHit(title="x", url="not-a-url", snippet="", provider="exa")
