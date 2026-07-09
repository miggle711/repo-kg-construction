"""End-to-end test: RepoASTParser(infer_types=True) must emit 'uses' edges
for lowercase factory-function calls that the uppercase heuristic can't see
(e.g. `self.session = requests.session()`), and must NOT emit them when
infer_types is left at its default (False).

See docs/type_inference_plan.md for the full design, and
tests/test_factory_call_sites.py / tests/test_type_inference.py for the two
mechanisms this end-to-end test exercises together.
"""

from pathlib import Path

import pytest

from kg_construction.kg.builder import RepoASTParser
from kg_construction.kg.type_inference import is_available

pyright_required = pytest.mark.skipif(
    not is_available(),
    reason="pyright-langserver not installed (optional `types` extra)",
)


def _write_factory_repo(tmp_path: Path) -> Path:
    """A Client class whose only dependency on Session is via a lowercase
    factory function -- invisible to _get_instantiated_classes_in_class.
    """
    (tmp_path / "sessions.py").write_text(
        "class Session:\n"
        "    def __init__(self):\n"
        "        self.headers = {}\n"
        "\n"
        "\n"
        "def session():\n"
        "    return Session()\n"
    )
    (tmp_path / "client.py").write_text(
        "from sessions import session\n"
        "\n"
        "\n"
        "class Client:\n"
        "    def __init__(self):\n"
        "        self.s = session()\n"
        "\n"
        "    def request(self):\n"
        "        return self.s.headers\n"
    )
    return tmp_path


class TestPyrightUsesEdgeEnrichment:
    def test_infer_types_false_by_default_produces_no_uses_edge(self, tmp_path):
        """Baseline: without infer_types, this dependency is invisible."""
        repo_dir = _write_factory_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        uses_edges = [e for e in kg["edges"] if e["relation"] == "uses"]
        assert uses_edges == []

    @pyright_required
    def test_infer_types_true_resolves_factory_dependency(self, tmp_path):
        repo_dir = _write_factory_repo(tmp_path)
        parser = RepoASTParser(max_workers=1, infer_types=True)
        kg = parser.parse_repo("test/repo", repo_dir)

        client_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Client"
        )
        session_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Session"
        )

        uses_edges = [
            e for e in kg["edges"]
            if e["relation"] == "uses"
            and e["source"] == client_id
            and e["target"] == session_id
        ]
        assert len(uses_edges) == 1
        assert uses_edges[0]["metadata"]["confidence"] == "exact"
        assert uses_edges[0]["metadata"]["source"] == "pyright"

    @pyright_required
    def test_infer_types_true_gracefully_handles_repo_with_no_factory_calls(self, tmp_path):
        """A repo with nothing for the heuristic to record must not error."""
        (tmp_path / "plain.py").write_text(
            "class Widget:\n"
            "    def build(self):\n"
            "        return SubWidget()\n"
            "\n"
            "class SubWidget:\n"
            "    pass\n"
        )
        parser = RepoASTParser(max_workers=1, infer_types=True)
        kg = parser.parse_repo("test/repo", tmp_path)

        # The existing uppercase heuristic still works unaffected.
        uses_edges = [e for e in kg["edges"] if e["relation"] == "uses"]
        assert len(uses_edges) == 1
        assert uses_edges[0]["metadata"].get("source") != "pyright"
