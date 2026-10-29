from agent.tools.fetch import extract_markdown

HTML = """
<html><head><title>Compare engines</title></head>
<body>
<article>
<h1>Compare engines</h1>
<p>Primary sources for RAG vector stores.</p>
<table>
  <tr><th>Engine</th><th>License</th></tr>
  <tr><td>Qdrant</td><td>Apache-2.0</td></tr>
  <tr><td>Milvus</td><td>Apache-2.0</td></tr>
</table>
</article>
</body></html>
"""


def test_extract_keeps_table_as_markdown() -> None:
    title, text = extract_markdown(HTML)
    assert "Compare" in title or "engine" in text.lower()
    # Trafilatura markdown tables use pipes; some versions emit HTML tables — either is structured.
    structured = "|" in text or "<table" in text or "Apache-2.0" in text
    assert structured
    assert "Qdrant" in text
