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


class TestPyrightUsesEdgeNonSelfReceiver:
    """other.attr = factory() (issue #29): the 'uses' edge source is the
    receiver's own resolved class, not the enclosing function's class (a
    plain function has none) and not the enclosing class of wherever the
    assignment happens to be written.
    """

    def _write_receiver_repo(self, tmp_path: Path) -> Path:
        (tmp_path / "parsers.py").write_text(
            "class Parser:\n"
            "    def __init__(self):\n"
            "        self.rules = []\n"
            "\n"
            "\n"
            "def build_parser():\n"
            "    return Parser()\n"
        )
        (tmp_path / "config.py").write_text(
            "class Config:\n"
            "    pass\n"
        )
        (tmp_path / "setup.py").write_text(
            "from config import Config\n"
            "from parsers import build_parser\n"
            "\n"
            "\n"
            "def configure(config: Config):\n"
            "    config.parser = build_parser()\n"
        )
        return tmp_path

    @pyright_required
    def test_resolves_uses_edge_from_receivers_own_type(self, tmp_path):
        repo_dir = self._write_receiver_repo(tmp_path)
        parser = RepoASTParser(max_workers=1, infer_types=True)
        kg = parser.parse_repo("test/repo", repo_dir)

        config_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Config"
        )
        parser_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Parser"
        )

        uses_edges = [
            e for e in kg["edges"]
            if e["relation"] == "uses"
            and e["source"] == config_id
            and e["target"] == parser_id
        ]
        assert len(uses_edges) == 1
        assert uses_edges[0]["metadata"]["source"] == "pyright"

    @pyright_required
    def test_uses_edge_source_ignores_later_reassignment(self, tmp_path):
        """Regression test for issue #31: a later reassignment of the
        receiver must not override the type it held at the factory-call
        site (previously, _collect_local_types' whole-function, last-write-
        wins summary caused this site to be misattributed to whatever class
        the receiver was reassigned to, anywhere in the function).
        """
        (tmp_path / "parsers.py").write_text(
            "class Parser:\n"
            "    def __init__(self):\n"
            "        self.rules = []\n"
            "\n"
            "\n"
            "def build_parser():\n"
            "    return Parser()\n"
        )
        (tmp_path / "config.py").write_text(
            "class Config:\n"
            "    pass\n"
            "\n"
            "\n"
            "class SomeUnrelatedClass:\n"
            "    pass\n"
        )
        (tmp_path / "setup.py").write_text(
            "from config import Config, SomeUnrelatedClass\n"
            "from parsers import build_parser\n"
            "\n"
            "\n"
            "def configure(config: Config):\n"
            "    config.parser = build_parser()\n"
            "    config = SomeUnrelatedClass()\n"
        )
        parser = RepoASTParser(max_workers=1, infer_types=True)
        kg = parser.parse_repo("test/repo", tmp_path)

        config_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Config"
        )
        unrelated_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "SomeUnrelatedClass"
        )
        parser_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Parser"
        )

        uses_edges = [e for e in kg["edges"] if e["relation"] == "uses"]

        assert any(
            e["source"] == config_id and e["target"] == parser_id
            for e in uses_edges
        ), "uses edge must be attributed to Config (the type at the call site)"
        assert not any(
            e["source"] == unrelated_id and e["target"] == parser_id
            for e in uses_edges
        ), "uses edge must NOT be attributed to SomeUnrelatedClass (a later reassignment)"

    @pyright_required
    def test_no_edge_when_receiver_type_unresolvable(self, tmp_path):
        """other.attr = factory() where other's type can't be determined
        (no annotation, no prior constructor call) must not produce an edge
        with a fabricated or missing source.
        """
        (tmp_path / "parsers.py").write_text(
            "class Parser:\n"
            "    pass\n"
            "\n"
            "\n"
            "def build_parser():\n"
            "    return Parser()\n"
        )
        (tmp_path / "setup.py").write_text(
            "from parsers import build_parser\n"
            "\n"
            "\n"
            "def configure(config):\n"  # no type annotation on config
            "    config.parser = build_parser()\n"
        )
        parser = RepoASTParser(max_workers=1, infer_types=True)
        kg = parser.parse_repo("test/repo", tmp_path)

        uses_edges = [e for e in kg["edges"] if e["relation"] == "uses"]
        assert uses_edges == []

    @pyright_required
    def test_source_collision_with_callable_downgrades_confidence(self, tmp_path):
        """Regression test for issue #32: a resolved source class name that
        collides with a real function/method name elsewhere in the repo
        must be reported 'ambiguous', not 'exact' -- the same downgrade the
        target side of a 'uses' edge already applies on a name collision.
        """
        (tmp_path / "mod.py").write_text(
            "class Config:\n"
            "    pass\n"
            "\n"
            "def Config():\n"
            "    \"\"\"Not a class -- a function sharing the class's name.\"\"\"\n"
            "    return None\n"
            "\n"
            "class Parser:\n"
            "    pass\n"
            "\n"
            "def build_parser():\n"
            "    return Parser()\n"
            "\n"
            "def configure(config: Config):\n"
            "    config.parser = build_parser()\n"
        )
        parser = RepoASTParser(max_workers=1, infer_types=True)
        kg = parser.parse_repo("test/repo", tmp_path)

        config_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Config"
        )
        parser_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "class" and n["label"] == "Parser"
        )

        uses_edges = [
            e for e in kg["edges"]
            if e["relation"] == "uses"
            and e["source"] == config_id
            and e["target"] == parser_id
        ]
        assert len(uses_edges) == 1
        assert uses_edges[0]["metadata"]["confidence"] == "ambiguous"
