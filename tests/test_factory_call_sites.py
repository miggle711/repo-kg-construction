"""Tests for _get_factory_call_sites: the AST-only detection half of the
pyright-backed 'uses' edge enrichment for lowercase factory-function calls
(see kg/type_inference.py and docs/type_inference_plan.md for the full
mechanism this feeds into).

_get_instantiated_classes only catches uppercase-named callees (SomeClass()).
_get_factory_call_sites is the complement: it records where a *lowercase*
callee's return value is assigned, so an external type checker can resolve
what it actually returns.
"""

import ast

from kg_construction.ast.helpers import _get_factory_call_sites


def _first_function(source: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    raise AssertionError("no function found in source")


class TestFactoryCallSiteDetection:
    def test_detects_bare_local_assignment_from_lowercase_call(self):
        func = _first_function(
            "def do_work():\n"
            "    s = requests.session()\n"
        )
        sites = _get_factory_call_sites(func)
        assert sites == [(1, 4)]

    def test_detects_self_attribute_assignment_from_lowercase_call(self):
        func = _first_function(
            "def __init__(self):\n"
            "    self.s = requests.session()\n"
        )
        sites = _get_factory_call_sites(func)
        # position must land on the attribute name 's', not on 'self'
        line, col = sites[0]
        source_line = "    self.s = requests.session()"
        assert source_line[col] == "s"
        assert source_line[col - 1] == "."

    def test_ignores_uppercase_callee_bare_name(self):
        func = _first_function(
            "def build():\n"
            "    x = SomeClass()\n"
        )
        assert _get_factory_call_sites(func) == []

    def test_ignores_uppercase_callee_attribute_call(self):
        func = _first_function(
            "def build():\n"
            "    x = module.SomeClass()\n"
        )
        assert _get_factory_call_sites(func) == []

    def test_detects_lowercase_attribute_call(self):
        func = _first_function(
            "def build():\n"
            "    x = pool.connection()\n"
        )
        sites = _get_factory_call_sites(func)
        assert sites == [(1, 4)]

    def test_ignores_tuple_unpacking_target(self):
        func = _first_function(
            "def build():\n"
            "    x, y = pair_factory()\n"
        )
        assert _get_factory_call_sites(func) == []

    def test_ignores_non_call_assignment(self):
        func = _first_function(
            "def build():\n"
            "    x = 5\n"
        )
        assert _get_factory_call_sites(func) == []

    def test_ignores_nested_function_locals(self):
        func = _first_function(
            "def outer():\n"
            "    def inner():\n"
            "        y = session()\n"
            "    return inner\n"
        )
        assert _get_factory_call_sites(func) == []

    def test_multiple_sites_in_one_function(self):
        func = _first_function(
            "def build():\n"
            "    a = make_a()\n"
            "    b = make_b()\n"
        )
        sites = _get_factory_call_sites(func)
        assert sites == [(1, 4), (2, 4)]

    def test_ignores_non_self_attribute_target(self):
        """other.attr = factory() -- not `self`, skipped (only self.x supported)."""
        func = _first_function(
            "def build(other):\n"
            "    other.thing = make_thing()\n"
        )
        assert _get_factory_call_sites(func) == []
