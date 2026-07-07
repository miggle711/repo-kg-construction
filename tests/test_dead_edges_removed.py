"""Regression test confirming reads/writes/returns edges are no longer emitted.

These edge types could never resolve in pass 2 (attribute/type names aren't
graph node IDs) and were unconditionally discarded by _resolve_edges, making
their emission in _emit_func_edges pure wasted computation. Because
_resolve_edges strips them before parse_repo() returns, the *final* KG output
is identical before and after this fix (that's what makes it safe) — so the
regression must be checked at the _parse_file layer, which is where the
unresolved edges existed pre-fix, not on parse_repo()'s resolved output.
"""

from pathlib import Path

from kg_construction.kg.builder import RepoASTParser, _parse_file


def _write_widget_file(tmp_path: Path) -> Path:
    mod_path = tmp_path / "mod.py"
    mod_path.write_text(
        "class Widget:\n"
        "    def __init__(self):\n"
        "        self.count = 0\n"
        "\n"
        "    def bump(self, amount):\n"
        "        current = self.count\n"
        "        self.count = current + amount\n"
        "        return self.count\n"
    )
    return mod_path


class TestDeadEdgesRemoved:
    """_parse_file must not emit reads/writes/returns edges at all."""

    def test_parse_file_emits_no_reads_writes_returns_edges(self, tmp_path):
        mod_path = _write_widget_file(tmp_path)
        result = _parse_file(("test/repo", "mod.py", str(mod_path)))

        edge_relations = {e["relation"] for e in result["edges"]}
        assert "reads" not in edge_relations
        assert "writes" not in edge_relations
        assert "returns" not in edge_relations

    def test_parse_file_still_emits_depends_on_and_tests(self, tmp_path):
        """Sanity check that _emit_func_edges' remaining edge types are untouched."""
        mod_path = tmp_path / "with_import.py"
        mod_path.write_text(
            "import os\n\n"
            "def get_cwd():\n"
            "    return os.getcwd()\n\n"
            "def test_get_cwd():\n"
            "    assert get_cwd()\n"
        )
        result = _parse_file(("test/repo", "with_import.py", str(mod_path)))

        edge_relations = {e["relation"] for e in result["edges"]}
        assert "depends_on" in edge_relations
        assert "tests" in edge_relations


class TestMetadataStillPopulated:
    """The equivalent information must still be available as node metadata."""

    def test_side_effects_and_data_flows_metadata_still_populated(self, tmp_path):
        repo_dir = tmp_path
        _write_widget_file(repo_dir)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        bump_node = next(
            n for n in kg["nodes"]
            if n["type"] == "method" and n["label"] == "bump"
        )
        meta = bump_node["metadata"]

        # side_effects still captures the self.count write
        assert "count" in meta["side_effects"]

        # data_flows still captures the return expression
        assert "self.count" in meta["data_flows"]["returns"]
