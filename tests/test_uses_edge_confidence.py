"""Regression test: 'uses' edges must downgrade to 'ambiguous' confidence
when the instantiated-class heuristic's candidate name also matches a
real function/method elsewhere in the repo.

_get_instantiated_classes uses a bare uppercase-first-letter heuristic to
spot constructor calls (e.g. `x = SomeClass()`). It can't distinguish a
real constructor call from a call to an unrelated uppercase-named
function that happens to share the name. Previously, if exactly one class
matched the candidate name, the edge was reported 'exact' even though the
callable at the actual call site might have been that unrelated function,
not the class constructor.
"""

from pathlib import Path

from kg_construction.kg.builder import RepoASTParser


def _write_repo(tmp_path: Path) -> Path:
    """A module-level function named 'Config' (unrelated, uppercase by
    coincidence) alongside an unrelated class also named 'Config'. A third
    class calls Config(...), which the heuristic can't tell apart from
    the real class's constructor.
    """
    (tmp_path / "mod.py").write_text(
        "def Config():\n"
        "    \"\"\"Not a class -- a factory-style function that happens\n"
        "    to be capitalized.\"\"\"\n"
        "    return {'key': 'value'}\n"
        "\n"
        "class Config:\n"
        "    def __init__(self):\n"
        "        self.key = None\n"
        "\n"
        "class App:\n"
        "    def setup(self):\n"
        "        self.config = Config()\n"
    )
    return tmp_path


class TestUsesEdgeConfidenceOnNameCollision:
    """A 'uses' edge whose candidate name collides with a real callable
    must be reported as 'ambiguous', not 'exact', even with one class match.
    """

    def test_uses_edge_downgraded_to_ambiguous_on_name_collision(self, tmp_path):
        repo_dir = _write_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        app_class_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "App"
        )
        config_class_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Config"
        )

        uses_edges = [
            e for e in kg["edges"]
            if e["relation"] == "uses"
            and e["source"] == app_class_id
            and e["target"] == config_class_id
        ]
        assert len(uses_edges) == 1
        assert uses_edges[0]["metadata"]["confidence"] == "ambiguous"

    def test_uses_edge_stays_exact_without_name_collision(self, tmp_path):
        """Sanity check: no collision, single class match -> still 'exact'."""
        (tmp_path / "mod2.py").write_text(
            "class Widget:\n"
            "    def __init__(self):\n"
            "        pass\n"
            "\n"
            "class Factory:\n"
            "    def build(self):\n"
            "        return Widget()\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo2", tmp_path)

        factory_class_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Factory"
        )
        widget_class_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Widget"
        )

        uses_edges = [
            e for e in kg["edges"]
            if e["relation"] == "uses"
            and e["source"] == factory_class_id
            and e["target"] == widget_class_id
        ]
        assert len(uses_edges) == 1
        assert uses_edges[0]["metadata"]["confidence"] == "exact"
