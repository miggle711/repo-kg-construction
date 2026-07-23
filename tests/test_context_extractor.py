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


class TestSeedNeverIncludesTestFile:
    """kg_construction#54: extract() used to unconditionally add the test
    file's own node to seed_ids alongside the real target function.
    context.seeds' order comes from BFS visited-node order, not seed_ids'
    construction order, so it was non-deterministic which one landed at
    seeds[0] -- and LLMSerializer._build_seed_section trusts seeds[0]
    blindly. When the test file won that race, the LLM-augmented arm's
    entire seed section was the test file (empty signature/source_code)
    instead of the real function, discovered via kg-test-generation#49's
    investigation into prepare_body_2015's repeated collection failures.
    """

    def _write_repo_with_test_file(self, tmp_path: Path) -> Path:
        (tmp_path / "mod.py").write_text(
            "class Widget:\n"
            "    def build(self):\n"
            "        return self.helper()\n"
            "\n"
            "    def helper(self):\n"
            "        return 42\n"
        )
        (tmp_path / "test_mod.py").write_text(
            "from mod import Widget\n"
            "\n"
            "def test_build():\n"
            "    assert Widget().build() == 42\n"
        )
        return tmp_path

    def test_seeds_never_contains_a_test_file_node(self, tmp_path):
        repo_dir = self._write_repo_with_test_file(tmp_path)
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

        # Run several times -- the bug was non-deterministic (BFS visited-
        # node order), so a single run passing wouldn't rule it out.
        for _ in range(10):
            context = extractor.extract(instance, depth=2)

            seed_types = {s.get("type") for s in context.seeds}
            assert "test_file" not in seed_types

            assert context.seeds, "the real function must still be a seed"
            assert context.seeds[0]["label"] == "build"
            assert context.seeds[0].get("type") != "test_file"

    def test_test_function_still_reachable_via_tests_edge(self, tmp_path):
        """Removing the test file from seed_ids must not break test_nodes
        -- 'tests' edges are resolved by function-naming convention
        (test_<name> -> <name>), not via any relationship to the test
        file's own node, so BFS from the seed alone should still reach it.
        """
        repo_dir = self._write_repo_with_test_file(tmp_path)
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

        test_labels = {t["label"] for t in context.test_nodes}
        assert "test_build" in test_labels


class TestCallsBasedTestDetection:
    """kg_construction#57's follow-up investigation: the naming-convention
    'tests' edge (test_<name> -> <name>, exact match) assumes a test's name
    mechanically derives from the function it tests -- checked directly
    against psf/requests' real test suite and found true for only 1 of 159
    test functions; the rest use descriptive names with no derivable
    relationship to the function under test, so existing_tests/test_nodes
    was empty for 21 of 22 real benchmark instances even when a real,
    directly-relevant test existed. A test that actually CALLS the target
    function is a naming-independent signal already computable from the
    ordinary 'calls' edge resolution -- these tests cover deriving 'tests'
    edges from that instead of/in addition to the naming heuristic.
    """

    def _write_repo(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class Widget:\n"
            "    def build(self):\n"
            "        return self.helper()\n"
            "\n"
            "    def helper(self):\n"
            "        return 42\n"
            "\n"
            "    def unrelated(self):\n"
            "        return 0\n"
        )
        (tmp_path / "test_mod.py").write_text(
            "from mod import Widget\n"
            "\n"
            "def test_descriptive_name_with_no_relation_to_build():\n"
            "    assert Widget().build() == 42\n"
            "\n"
            "def test_covers_unrelated_function():\n"
            "    assert Widget().unrelated() == 0\n"
        )
        return tmp_path

    def _instance(self):
        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -2,3 +2,3 @@\n"
            "     def build(self):\n"
            "-        return self.helper()\n"
            "+        return self.helper() + 1\n"
        )
        return {
            "repo": "test/repo",
            "base_commit": "deadbeef",
            "patch": patch,
            "code_file": "mod.py",
            "test_file": "test_mod.py",
        }

    def test_descriptively_named_test_is_found_via_calls_not_naming(self, tmp_path):
        repo_dir = self._write_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine)
        context = extractor.extract(self._instance(), depth=2)

        test_labels = {t["label"] for t in context.test_nodes}
        assert "test_descriptive_name_with_no_relation_to_build" in test_labels

    def test_unrelated_functions_test_is_not_attributed_to_the_seed(self, tmp_path):
        """A test that calls some OTHER function reachable in the subgraph
        (here, a sibling method 'unrelated') must not be misattributed as
        an existing test FOR the seed -- the same scoping bug class as
        kg-test-generation#49's 'related' list, here for test_nodes.
        """
        repo_dir = self._write_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        engine = KGQueryEngine(kg)
        extractor = TestContextExtractor(engine)
        context = extractor.extract(self._instance(), depth=2)

        test_labels = {t["label"] for t in context.test_nodes}
        assert "test_covers_unrelated_function" not in test_labels
