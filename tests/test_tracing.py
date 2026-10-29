from agent.tracing import Tracer, truncate


def test_truncate() -> None:
    assert truncate("short") == "short"
    long = "word " * 400
    preview = truncate(long, limit=40)
    assert preview is not None
    assert preview.endswith("chars]")
    assert len(preview) < len(long)


def test_tracer_appends_jsonl(tmp_path) -> None:
    tracer = Tracer.create(tmp_path, "rid")
    tracer.log("PLAN", model="m", input_preview="hello", input_tokens=10, output_tokens=4)
    tracer.log("tool", tool="web_search", error="timeout")
    lines = tracer.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert tracer.totals.llm_calls == 0  # log() does not imply an LLM call
