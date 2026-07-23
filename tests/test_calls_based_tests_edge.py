"""kg_construction#57's follow-up: 'tests' edges were previously resolved
only by naming convention (test_<name> -> <name>, exact string match after
stripping the prefix) -- checked directly against psf/requests' real test
suite and found true for only 1 of 159 test functions, since real test
names are usually descriptive (test_prepared_request_hook), not derived
from the function under test. A test that actually CALLS the target
function is a naming-independent signal already computable from the
ordinary 'calls' edge resolution -- this derives 'tests' edges from that
too, in RepoASTParser.parse_repo's second pass (_derive_tests_edges_from_calls).
"""

from kg_construction.kg.builder import RepoASTParser


class TestCallsBasedTestsEdgeDerivation:
    def _build(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "def target_function():\n"
            "    return 1\n"
        )
        (tmp_path / "test_mod.py").write_text(
            "from mod import target_function\n"
            "\n"
            "def test_totally_unrelated_name():\n"
            "    assert target_function() == 1\n"
        )
        parser = RepoASTParser(max_workers=1)
        return parser.parse_repo("test/repo", tmp_path)

    def test_tests_edge_emitted_for_descriptively_named_test(self, tmp_path):
        kg = self._build(tmp_path)
        nodes_by_id = {n["id"]: n for n in kg["nodes"]}

        target = next(n for n in kg["nodes"] if n["label"] == "target_function")
        tests_edges = [
            e for e in kg["edges"]
            if e["relation"] == "tests" and e["target"] == target["id"]
        ]

        assert len(tests_edges) == 1
        source_label = nodes_by_id[tests_edges[0]["source"]]["label"]
        assert source_label == "test_totally_unrelated_name"

    def test_derived_tests_edge_is_tagged_with_its_origin(self, tmp_path):
        """Distinguishes a calls-derived 'tests' edge from a naming-derived
        one in metadata, in case a future consumer needs to weight/filter
        by how the relationship was established.
        """
        kg = self._build(tmp_path)
        target = next(n for n in kg["nodes"] if n["label"] == "target_function")
        tests_edges = [
            e for e in kg["edges"]
            if e["relation"] == "tests" and e["target"] == target["id"]
        ]

        assert tests_edges[0]["metadata"].get("derived_from") == "calls"

    def test_no_duplicate_tests_edge_when_naming_and_calls_both_match(self, tmp_path):
        """When a test's name DOES follow the naming convention AND it
        calls the target, only one 'tests' edge should exist (deduplicated
        via seen_edges), not two.
        """
        tmp_path.joinpath("mod.py").write_text(
            "def target_function():\n"
            "    return 1\n"
        )
        tmp_path.joinpath("test_mod.py").write_text(
            "from mod import target_function\n"
            "\n"
            "def test_target_function():\n"
            "    assert target_function() == 1\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        target = next(n for n in kg["nodes"] if n["label"] == "target_function")
        tests_edges = [
            e for e in kg["edges"]
            if e["relation"] == "tests" and e["target"] == target["id"]
        ]

        assert len(tests_edges) == 1
