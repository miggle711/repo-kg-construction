"""Tests for llm_serializer.py: converting flat subgraph JSON into the
hierarchical {seed, context, instructions} dict consumed by kg-test-generation.

Focus: the "module" field derived from each node's metadata["filepath"] --
without it, the LLM has no real import path for the seed function or its
callers/callees/related classes and can fabricate placeholder imports
(kg-test-generation issue #6).
"""

from kg_construction.llm.llm_serializer import LLMSerializer, _filepath_to_module


class TestFilepathToModule:
    def test_converts_nested_filepath(self):
        assert _filepath_to_module("requests/sessions.py") == "requests.sessions"

    def test_converts_top_level_filepath(self):
        assert _filepath_to_module("setup.py") == "setup"

    def test_empty_filepath_returns_empty_string(self):
        assert _filepath_to_module("") == ""


class TestSerializeSeedSection:
    def test_seed_includes_module_and_filepath(self):
        seed_node = {
            "id": "n1",
            "label": "send",
            "type": "method",
            "metadata": {
                "filepath": "requests/sessions.py",
                "signature": "def send(self, request, **kwargs)",
                "source_code": "def send(self, request, **kwargs):\n    ...",
            },
        }
        result = LLMSerializer(repo="psf/requests").serialize(
            {"seeds": [seed_node], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert result["seed"]["module"] == "requests.sessions"
        assert result["seed"]["filepath"] == "requests/sessions.py"
        assert result["seed"]["function_name"] == "send"

    def test_seed_with_no_filepath_metadata_gets_empty_module(self):
        seed_node = {"id": "n1", "label": "send", "type": "method", "metadata": {}}
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert result["seed"]["module"] == ""
        assert result["seed"]["filepath"] == ""

    def test_method_seed_includes_class_name(self):
        """A method's class name (e.g. "Session") must be surfaced so the
        LLM knows to import/instantiate the class rather than attempting a
        bare "from module import method_name" import, which doesn't exist
        for methods (see kg-test-generation issue #14).
        """
        seed_node = {
            "id": "n1",
            "label": "resolve_redirects",
            "type": "method",
            "metadata": {
                "filepath": "requests/sessions.py",
                "class": "Session",
            },
        }
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert result["seed"]["class_name"] == "Session"

    def test_function_seed_has_empty_class_name(self):
        seed_node = {
            "id": "n1",
            "label": "get",
            "type": "function",
            "metadata": {"filepath": "requests/api.py"},
        }
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert result["seed"]["class_name"] == ""

    def test_no_seeds_returns_empty_dict(self):
        result = LLMSerializer().serialize(
            {"seeds": [], "context_nodes": [], "edges": [], "test_nodes": []}
        )
        assert result["seed"] == {}


class TestSerializeContextSection:
    def _instance(self, seed_node, other_node, relation):
        return {
            "seeds": [seed_node],
            "context_nodes": [other_node],
            "edges": [{"source": other_node["id"], "target": seed_node["id"], "relation": relation}],
            "test_nodes": [],
        }

    def test_caller_includes_module(self):
        seed_node = {
            "id": "seed", "label": "send", "type": "method",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        caller_node = {
            "id": "caller", "label": "request", "type": "method",
            "metadata": {"filepath": "requests/sessions.py", "signature": "def request(...)"},
        }
        result = LLMSerializer().serialize(self._instance(seed_node, caller_node, "calls"))

        assert len(result["context"]["callers"]) == 1

    def test_caller_includes_class_name_when_a_method(self):
        seed_node = {
            "id": "seed", "label": "send", "type": "method",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        caller_node = {
            "id": "caller", "label": "request", "type": "method",
            "metadata": {"filepath": "requests/sessions.py", "class": "Session"},
        }
        result = LLMSerializer().serialize(self._instance(seed_node, caller_node, "calls"))

        assert result["context"]["callers"][0]["class_name"] == "Session"
        assert result["context"]["callers"][0]["type"] == "method"
        assert result["context"]["callers"][0]["module"] == "requests.sessions"
        assert result["context"]["callers"][0]["name"] == "request"

    def test_callee_includes_module(self):
        seed_node = {
            "id": "seed", "label": "send", "type": "method",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        callee_node = {
            "id": "callee", "label": "get_adapter", "type": "method",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        instance = {
            "seeds": [seed_node],
            "context_nodes": [callee_node],
            "edges": [{"source": seed_node["id"], "target": callee_node["id"], "relation": "calls"}],
            "test_nodes": [],
        }
        result = LLMSerializer().serialize(instance)

        assert len(result["context"]["callees"]) == 1
        assert result["context"]["callees"][0]["module"] == "requests.sessions"

    def test_parent_class_includes_module(self):
        seed_node = {
            "id": "seed", "label": "Session", "type": "class",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        parent_node = {
            "id": "parent", "label": "SessionRedirectMixin", "type": "class",
            "metadata": {"filepath": "requests/sessions.py", "source_code": "class SessionRedirectMixin: ..."},
        }
        instance = {
            "seeds": [seed_node],
            "context_nodes": [parent_node],
            "edges": [{"source": seed_node["id"], "target": parent_node["id"], "relation": "inherits"}],
            "test_nodes": [],
        }
        result = LLMSerializer().serialize(instance)

        related = result["context"]["related"]
        assert len(related) == 1
        assert related[0]["type"] == "parent_class"
        assert related[0]["module"] == "requests.sessions"

    def test_instantiation_includes_module(self):
        seed_node = {
            "id": "seed", "label": "send", "type": "method",
            "metadata": {"filepath": "requests/sessions.py"},
        }
        used_node = {
            "id": "used", "label": "HTTPAdapter", "type": "class",
            "metadata": {"filepath": "requests/adapters.py"},
        }
        instance = {
            "seeds": [seed_node],
            "context_nodes": [used_node],
            "edges": [{"source": seed_node["id"], "target": used_node["id"], "relation": "instantiates"}],
            "test_nodes": [],
        }
        result = LLMSerializer().serialize(instance)

        related = result["context"]["related"]
        assert len(related) == 1
        assert related[0]["type"] == "instantiation"
        assert related[0]["module"] == "requests.adapters"
