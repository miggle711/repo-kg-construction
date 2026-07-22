"""Tests for issue #51: _build_func_metadata never computed 'signature' or
'source_code' -- LLMSerializer has always expected both (they're rendered
directly into the KG-augmented test-generation prompt), but neither key
was ever set, so the model only ever saw a docstring and a param-name
list, never the seed function's actual code.
"""

import ast

from kg_construction.ast.helpers import _build_func_metadata


def _parse_function(source: str, name: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in source")


class TestSignatureAndSourceCode:
    def test_module_level_function_gets_signature_and_source(self):
        source = (
            "def add(a, b):\n"
            "    \"\"\"Add two numbers.\"\"\"\n"
            "    return a + b\n"
        )
        node = _parse_function(source, "add")
        meta = _build_func_metadata(
            node, "mod.py", "test/repo", source_lines=source.splitlines(keepends=True)
        )

        assert meta["signature"] == "def add(a, b):"
        assert meta["source_code"] == source

    def test_method_gets_signature_and_source(self):
        source = (
            "class Foo:\n"
            "    def bar(self, x):\n"
            "        return x * 2\n"
        )
        node = _parse_function(source, "bar")
        meta = _build_func_metadata(
            node, "mod.py", "test/repo", parent_class="Foo",
            source_lines=source.splitlines(keepends=True),
        )

        assert meta["signature"] == "    def bar(self, x):"
        assert "return x * 2" in meta["source_code"]

    def test_multiline_signature_is_captured_in_full(self):
        """A wrapped/multi-line signature (long param list) must have its
        full 'def ...:' span captured, not just the first physical line --
        the model needs the complete signature, not a truncated one.
        """
        source = (
            "def long_func(\n"
            "    a,\n"
            "    b,\n"
            "    c,\n"
            "):\n"
            "    return a + b + c\n"
        )
        node = _parse_function(source, "long_func")
        meta = _build_func_metadata(
            node, "mod.py", "test/repo", source_lines=source.splitlines(keepends=True)
        )

        assert "def long_func(" in meta["signature"]
        assert "a," in meta["signature"] and "b," in meta["signature"] and "c," in meta["signature"]
        assert "):" in meta["signature"]
        # Signature must stop before the body -- must not include the return statement.
        assert "return a + b + c" not in meta["signature"]

    def test_source_code_includes_docstring_and_full_body(self):
        source = (
            "def multi_line():\n"
            "    \"\"\"A docstring.\"\"\"\n"
            "    x = 1\n"
            "    y = 2\n"
            "    return x + y\n"
        )
        node = _parse_function(source, "multi_line")
        meta = _build_func_metadata(
            node, "mod.py", "test/repo", source_lines=source.splitlines(keepends=True)
        )

        assert meta["source_code"] == source

    def test_missing_source_lines_leaves_both_fields_empty(self):
        """No source text available (e.g. a caller with no file text handy)
        must not raise -- both fields are optional per LLMSerializer, which
        already defaults them to "" when absent.
        """
        source = "def f():\n    pass\n"
        node = _parse_function(source, "f")
        meta = _build_func_metadata(node, "mod.py", "test/repo")

        assert meta["signature"] == ""
        assert meta["source_code"] == ""

    def test_signature_and_source_do_not_affect_existing_fields(self):
        """Adding the two new fields must not disturb any existing metadata
        key -- a pure addition, not a refactor of what's already computed.
        """
        source = (
            "def f(x: int) -> int:\n"
            "    \"\"\"Doubles x.\"\"\"\n"
            "    return x * 2\n"
        )
        node = _parse_function(source, "f")
        meta = _build_func_metadata(
            node, "requests/api.py", "psf/requests",
            source_lines=source.splitlines(keepends=True),
        )

        assert meta["docstring"] == "Doubles x."
        assert meta["params"] == [{"name": "x", "annotation": "int"}]
        assert meta["returns"] == "int"
        assert meta["filepath"] == "requests/api.py"
        assert meta["repo"] == "psf/requests"
