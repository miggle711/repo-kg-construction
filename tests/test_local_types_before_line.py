"""Regression test for issue #31: _collect_local_types' before_line parameter.

Without before_line, _collect_local_types returns a single whole-function
summary dict with "last assignment wins" semantics -- correct for
_emit_call_edges' existing use (a later reassignment really should win,
see test_local_type_hints.py), but wrong when a caller needs "the type a
variable held at a SPECIFIC earlier call site," since a later reassignment
elsewhere in the function would otherwise silently override it.
"""

import ast

from kg_construction.ast.helpers import _collect_local_types


def _first_function(source: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    raise AssertionError("no function found in source")


class TestCollectLocalTypesBeforeLine:
    def test_default_behavior_unchanged_without_before_line(self):
        """No before_line: whole-function summary, last assignment wins."""
        func = _first_function(
            "def build(other: Config):\n"
            "    other.thing = make_thing()\n"
            "    other = SomeUnrelatedClass()\n"
        )
        assert _collect_local_types(func) == {"other": "SomeUnrelatedClass"}

    def test_before_line_excludes_later_reassignment(self):
        """With before_line set to the call site's line, a later
        reassignment must not be visible -- the annotated type should win.
        """
        func = _first_function(
            "def build(other: Config):\n"
            "    other.thing = make_thing()\n"  # line 2 (1-indexed)
            "    other = SomeUnrelatedClass()\n"  # line 3
        )
        types = _collect_local_types(func, before_line=2)
        assert types.get("other") == "Config"

    def test_before_line_includes_earlier_reassignment(self):
        """An earlier reassignment (before the query line) must still be
        picked up -- before_line only excludes assignments AT or AFTER it.
        """
        func = _first_function(
            "def build(other: Config):\n"
            "    other = EarlyClass()\n"       # line 2
            "    other.thing = make_thing()\n"  # line 3
            "    other = LateClass()\n"         # line 4
        )
        types = _collect_local_types(func, before_line=3)
        assert types.get("other") == "EarlyClass"

    def test_before_line_at_exact_assignment_line_excludes_it(self):
        """before_line is exclusive of the given line itself (an assignment
        ON that line doesn't count as 'before' it)."""
        func = _first_function(
            "def build():\n"
            "    x = SomeClass()\n"  # line 2
        )
        assert _collect_local_types(func, before_line=2) == {}
        assert _collect_local_types(func, before_line=3).get("x") == "SomeClass"
