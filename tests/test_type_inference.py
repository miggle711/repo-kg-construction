"""Tests for kg/type_inference.py: pyright-backed type resolution for
factory-call sites recorded by _get_factory_call_sites (see
tests/test_factory_call_sites.py for the pure-AST half).

Split into two groups:
  - Unit tests for hover-response parsing and graceful degradation, which
    don't need pyright installed at all.
  - Integration tests that drive a real `pyright-langserver` process against
    synthetic fixture files (no network access needed), skipped automatically
    if pyright isn't installed in the environment running the tests.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from kg_construction.kg.type_inference import (
    _parse_hover_type,
    is_available,
    resolve_types,
)

pyright_required = pytest.mark.skipif(
    not is_available(),
    reason="pyright-langserver not installed (optional `types` extra)",
)


class TestParseHoverType:
    """_parse_hover_type must extract a simple class name or return None,
    never guess at anything ambiguous (matches the 'honest unknown' contract
    described in kg/type_inference.py and the issue #19 research spike).
    """

    def test_extracts_simple_class_name(self):
        response = {"result": {"contents": {"value": "(variable) s: Session"}}}
        assert _parse_hover_type(response) == "Session"

    def test_returns_none_for_missing_result(self):
        assert _parse_hover_type({"result": None}) is None
        assert _parse_hover_type({}) is None
        assert _parse_hover_type(None) is None

    def test_returns_none_for_union_type(self):
        response = {"result": {"contents": {"value": "(variable) x: Session | ConnectionPool"}}}
        assert _parse_hover_type(response) is None

    def test_returns_none_for_unknown_type(self):
        response = {"result": {"contents": {"value": "(variable) s: Unknown"}}}
        assert _parse_hover_type(response) is None

    def test_returns_none_for_builtin_type(self):
        response = {"result": {"contents": {"value": "(variable) y: bool"}}}
        assert _parse_hover_type(response) is None

    def test_returns_none_for_none_type(self):
        response = {"result": {"contents": {"value": "(variable) z: None"}}}
        assert _parse_hover_type(response) is None

    def test_handles_string_contents_shape(self):
        response = {"result": {"contents": "(variable) s: Session"}}
        assert _parse_hover_type(response) == "Session"

    def test_strips_markdown_code_fences(self):
        response = {"result": {"contents": {"value": "(variable) s: `Session`"}}}
        assert _parse_hover_type(response) == "Session"


class TestResolveTypesDegradesGracefully:
    """resolve_types must never raise -- any failure means "no enrichment"."""

    def test_returns_empty_dict_when_pyright_unavailable(self):
        with patch("kg_construction.kg.type_inference.is_available", return_value=False):
            result = resolve_types(Path("/tmp"), {"foo.py": [(0, 0)]})
        assert result == {}

    def test_returns_empty_dict_for_empty_sites(self):
        result = resolve_types(Path("/tmp"), {})
        assert result == {}

    def test_returns_empty_dict_on_internal_exception(self):
        with patch("kg_construction.kg.type_inference.is_available", return_value=True):
            with patch(
                "kg_construction.kg.type_inference._resolve_types_inner",
                side_effect=RuntimeError("boom"),
            ):
                result = resolve_types(Path("/tmp"), {"foo.py": [(0, 0)]})
        assert result == {}


@pyright_required
class TestResolveTypesIntegration:
    """Real pyright-langserver invocation against synthetic fixtures.

    No network access required -- pyright analyzes local files only.
    """

    def _write_factory_fixture(self, tmp_path: Path) -> Path:
        (tmp_path / "sessions.py").write_text(
            "class Session:\n"
            "    def __init__(self):\n"
            "        self.headers = {}\n"
            "\n"
            "\n"
            "def session():\n"
            "    return Session()\n"
        )
        (tmp_path / "caller.py").write_text(
            "import sessions\n"
            "\n"
            "def do_work():\n"
            "    s = sessions.session()\n"
            "    return s\n"
        )
        return tmp_path

    def test_resolves_bare_local_factory_call(self, tmp_path):
        repo_dir = self._write_factory_fixture(tmp_path)
        # 's = sessions.session()' is on line 3 (0-indexed), 's' at column 4
        resolved = resolve_types(repo_dir, {"caller.py": [(3, 4)]})
        assert resolved == {("caller.py", 3, 4): "Session"}

    def test_resolves_self_attribute_factory_call(self, tmp_path):
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
        # 'self.s = session()' is on line 5 (0-indexed); attribute 's' at column 13
        resolved = resolve_types(tmp_path, {"client.py": [(5, 13)]})
        assert resolved == {("client.py", 5, 13): "Session"}

    def test_does_not_resolve_union_type(self, tmp_path):
        repo_dir = self._write_factory_fixture(tmp_path)
        (repo_dir / "sessions.py").write_text(
            (repo_dir / "sessions.py").read_text() +
            "\n\nclass ConnectionPool:\n    pass\n\n\n"
            "def branch_factory(cond):\n"
            "    if cond:\n"
            "        return Session()\n"
            "    return ConnectionPool()\n"
        )
        (repo_dir / "branch_caller.py").write_text(
            "import sessions\n"
            "\n"
            "def pick(cond):\n"
            "    x = sessions.branch_factory(cond)\n"
            "    return x\n"
        )
        resolved = resolve_types(repo_dir, {"branch_caller.py": [(3, 4)]})
        assert resolved == {}

    def test_does_not_resolve_builtin_type(self, tmp_path):
        (tmp_path / "caller.py").write_text(
            "def do_work():\n"
            "    y = bool('x')\n"
            "    return y\n"
        )
        resolved = resolve_types(tmp_path, {"caller.py": [(1, 4)]})
        assert resolved == {}
