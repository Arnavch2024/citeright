from agent.conflicts import numeric_conflicts
from agent.llm import FakeLLM
from agent.schemas import Conflict, ConflictReport, ConflictPosition, Finding


def test_numeric_conflict_requires_same_entity_metric_and_different_sources() -> None:
    findings = [
        Finding(claim="Qdrant has 18,000 stars.", source_ids=["s1"]),
        Finding(claim="Qdrant has 20,000 stars.", source_ids=["s2"]),
        Finding(claim="Milvus has 20,000 stars.", source_ids=["s3"]),
    ]
    conflicts = numeric_conflicts(findings)
    assert len(conflicts) == 1
    assert conflicts[0].kind == "numeric"
    assert "qdrant stars" in conflicts[0].topic


def test_numeric_conflict_ignores_one_source_revisions() -> None:
    findings = [
        Finding(claim="Qdrant has 18,000 stars.", source_ids=["s1"]),
        Finding(claim="Qdrant has 20,000 stars.", source_ids=["s1"]),
    ]
    assert numeric_conflicts(findings) == []


async def test_textual_conflict_structured_output(tmp_path) -> None:
    from agent.conflicts import detect_conflicts
    from agent.config import Settings
    from agent.tracing import Tracer

    settings = Settings(llm_provider="fake", traces_dir=tmp_path / "trace", cache_dir=tmp_path / "cache")
    llm = FakeLLM(settings, Tracer.create(settings.traces_dir, "conflict-test"))
    conflict = Conflict(
        topic="license",
        positions=[
            ConflictPosition(claim="The project uses Apache-2.0.", source_ids=["s1"]),
            ConflictPosition(claim="The project uses AGPL-3.0.", source_ids=["s2"]),
        ],
    )
    llm.set("ConflictReport", ConflictReport(conflicts=[conflict]))
    out = await detect_conflicts(
        llm,
        [Finding(claim="License differs by source.", source_ids=["s1", "s2"])],
        {"s1": "Apache-2.0", "s2": "AGPL-3.0"},
    )
    assert out == [conflict]
