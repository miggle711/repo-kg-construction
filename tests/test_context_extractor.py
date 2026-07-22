"""Regression test: TestContextExtractor.extract() must not crash.

KGQueryEngine.kg_extraction._bfs previously called self.engine.edges, an
attribute KGQueryEngine has never had (edges live at self.engine.kg['edges']).
Extract() has no test coverage anywhere, so this AttributeError was only
discovered when kg-test-generation actually tried to call it against a real
repo -- there had never been a synthetic end-to-end test exercising this path.
"""

from pathlib import Path

from kg_construction.kg.builder import RepoASTParser
from kg_construction.kg.query import KGQueryEngine
from kg_construction.extraction.context import TestContextExtractor


def _write_repo(tmp_path: Path) -> Path:
    (tmp_path / "mod.py").write_text(
        "class Widget:\n"
        "    def build(self):\n"
        "        return self.helper()\n"
        "\n"
        "    def helper(self):\n"
        "        return 42\n"
    )
    return tmp_path


class TestContextExtractorEndToEnd:
    def test_extract_does_not_crash_and_returns_seed(self, tmp_path):
        repo_dir = _write_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine)

        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -2,3 +2,3 @@\n"
            "     def build(self):\n"
            "-        return self.helper()\n"
            "+        return self.helper() + 1\n"
        )
        instance = {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

        context = extractor.extract(instance, depth=2)

        assert len(context.seeds) >= 1
        seed_labels = {s["label"] for s in context.seeds}
        assert "build" in seed_labels

        # The BFS must have actually traversed edges (helper is one hop away).
        context_labels = {n["label"] for n in context.context_nodes}
        assert "helper" in context_labels
