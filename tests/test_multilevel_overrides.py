"""Regression tests for transitive (multi-level) overrides resolution.

Previously, 'overrides' resolution only checked whether a method's
*immediate* base class defined a same-named method (matching the base
name string literally). For `class C(B)`, `class B(A)`, if A defines
`foo` and neither B nor C redefine it except C.foo, the override
relationship to A.foo was invisible -- the check only ever matched a
direct parent, never a grandparent.

The fix walks the resolved 'inherits' chain transitively, finding the
*nearest* ancestor that defines the method -- matching Python's real
MRO/attribute-lookup semantics.
"""

from pathlib import Path

from kg_construction.kg.builder import RepoASTParser


def _method_id(kg, class_name, method_name):
    matches = [
        n["id"] for n in kg["nodes"]
        if n["type"] == "method"
        and n["label"] == method_name
        and n["metadata"].get("class") == class_name
    ]
    assert len(matches) == 1, f"expected exactly one {class_name}.{method_name}"
    return matches[0]


def _overrides_targets(kg, source_id):
    return {
        e["target"] for e in kg["edges"]
        if e["relation"] == "overrides" and e["source"] == source_id
    }


class TestMultiLevelOverrides:
    """A method must be able to override a grandparent's definition."""

    def test_grandparent_definition_is_found_when_parent_does_not_override(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class A:\n"
            "    def foo(self):\n"
            "        return 'A.foo'\n"
            "\n"
            "class B(A):\n"
            "    def bar(self):\n"
            "        return 'B.bar'\n"
            "\n"
            "class C(B):\n"
            "    def foo(self):\n"
            "        return 'C.foo'\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        c_foo_id = _method_id(kg, "C", "foo")
        a_foo_id = _method_id(kg, "A", "foo")

        targets = _overrides_targets(kg, c_foo_id)
        assert a_foo_id in targets, "C.foo should override A.foo (grandparent)"

    def test_nearest_ancestor_wins_when_intermediate_class_also_defines_it(self, tmp_path):
        """If B also defines foo, C.foo overrides B.foo only, not A.foo too --
        matching Python's actual MRO (nearest definition wins)."""
        (tmp_path / "mod2.py").write_text(
            "class A:\n"
            "    def foo(self):\n"
            "        return 'A.foo'\n"
            "\n"
            "class B(A):\n"
            "    def foo(self):\n"
            "        return 'B.foo'\n"
            "\n"
            "class C(B):\n"
            "    def foo(self):\n"
            "        return 'C.foo'\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo2", tmp_path)

        c_foo_id = _method_id(kg, "C", "foo")
        b_foo_id = _method_id(kg, "B", "foo")
        a_foo_id = _method_id(kg, "A", "foo")

        targets = _overrides_targets(kg, c_foo_id)
        assert b_foo_id in targets, "C.foo should override B.foo (nearest)"
        assert a_foo_id not in targets, "C.foo must not also link to A.foo"

    def test_direct_parent_override_still_works(self, tmp_path):
        """Sanity check: the original single-level case must be unaffected."""
        (tmp_path / "mod3.py").write_text(
            "class Base:\n"
            "    def greet(self):\n"
            "        return 'base'\n"
            "\n"
            "class Child(Base):\n"
            "    def greet(self):\n"
            "        return 'child'\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo3", tmp_path)

        child_greet_id = _method_id(kg, "Child", "greet")
        base_greet_id = _method_id(kg, "Base", "greet")

        targets = _overrides_targets(kg, child_greet_id)
        assert base_greet_id in targets

    def test_three_level_chain_finds_top_ancestor(self, tmp_path):
        """class D(C(B(A))), only A defines the method -- must resolve
        across three levels, not just one."""
        (tmp_path / "mod4.py").write_text(
            "class A:\n"
            "    def run(self):\n"
            "        return 'A'\n"
            "\n"
            "class B(A):\n"
            "    pass\n"
            "\n"
            "class C(B):\n"
            "    pass\n"
            "\n"
            "class D(C):\n"
            "    def run(self):\n"
            "        return 'D'\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo4", tmp_path)

        d_run_id = _method_id(kg, "D", "run")
        a_run_id = _method_id(kg, "A", "run")

        targets = _overrides_targets(kg, d_run_id)
        assert a_run_id in targets, "D.run should override A.run across a 3-level chain"
