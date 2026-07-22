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


class TestSiblingMethods:
    """Issue #50: sibling methods reached via 'contains' BFS were present
    in context_nodes but silently dropped at serialization -- e.g. a
    method's own required setup (PreparedRequest.prepare()) was never
    visible to the model, only the method actually under test. This is
    context a flat single-function extraction (the baseline arm) can
    never provide, since it has no notion of "what else does this class
    define" -- see kg-test-generation#28's precondition-visibility gap.
    """

    def _class_seed_instance(self, sibling_relation="contains", sibling_target_is_seed=False):
        seed_node = {
            "id": "seed", "label": "prepare_content_length", "type": "method",
            "metadata": {"filepath": "requests/models.py", "class": "PreparedRequest"},
        }
        class_node = {
            "id": "class_preparedrequest", "label": "PreparedRequest", "type": "class",
            "metadata": {"filepath": "requests/models.py"},
        }
        sibling_node = {
            "id": "sibling", "label": "prepare", "type": "method",
            "metadata": {
                "filepath": "requests/models.py", "class": "PreparedRequest",
                "source_code": "def prepare(self, ...):\n    self.headers = {}\n",
            },
        }
        sibling_edge_target = seed_node["id"] if sibling_target_is_seed else sibling_node["id"]
        return {
            "seeds": [seed_node],
            "context_nodes": [class_node, sibling_node],
            "edges": [
                {"source": class_node["id"], "target": seed_node["id"], "relation": "contains"},
                {"source": class_node["id"], "target": sibling_edge_target, "relation": sibling_relation},
            ],
            "test_nodes": [],
        }

    def test_sibling_method_of_seed_class_is_included(self):
        result = LLMSerializer().serialize(self._class_seed_instance())

        siblings = result["context"]["sibling_methods"]
        assert len(siblings) == 1
        assert siblings[0]["name"] == "prepare"
        assert siblings[0]["module"] == "requests.models"
        assert "self.headers = {}" in siblings[0]["source_code"]

    def test_seed_does_not_list_itself_as_its_own_sibling(self):
        """The class->seed 'contains' edge itself must not cause the seed
        to appear in its own sibling_methods list.
        """
        result = LLMSerializer().serialize(
            self._class_seed_instance(sibling_target_is_seed=True)
        )

        # Only the class->seed edge exists in this instance (both edges
        # point at the seed) -- sibling_methods must be empty, not contain
        # the seed itself.
        assert result["context"]["sibling_methods"] == []

    def test_contains_edge_from_unrelated_class_is_not_treated_as_sibling(self):
        """A 'contains' edge from a DIFFERENT class (e.g. one reached via
        an unrelated 'related' relationship elsewhere in the subgraph)
        must not be mistaken for a sibling of the seed's own class.
        """
        seed_node = {
            "id": "seed", "label": "handle_401", "type": "method",
            "metadata": {"filepath": "requests/auth.py", "class": "HTTPDigestAuth"},
        }
        seed_class_node = {
            "id": "class_httpdigestauth", "label": "HTTPDigestAuth", "type": "class",
            "metadata": {"filepath": "requests/auth.py"},
        }
        unrelated_class_node = {
            "id": "class_unrelated", "label": "SomeOtherClass", "type": "class",
            "metadata": {"filepath": "requests/other.py"},
        }
        unrelated_method_node = {
            "id": "unrelated_method", "label": "some_method", "type": "method",
            "metadata": {"filepath": "requests/other.py", "class": "SomeOtherClass"},
        }
        instance = {
            "seeds": [seed_node],
            "context_nodes": [seed_class_node, unrelated_class_node, unrelated_method_node],
            "edges": [
                {"source": seed_class_node["id"], "target": seed_node["id"], "relation": "contains"},
                {"source": unrelated_class_node["id"], "target": unrelated_method_node["id"], "relation": "contains"},
            ],
            "test_nodes": [],
        }
        result = LLMSerializer().serialize(instance)

        assert result["context"]["sibling_methods"] == []

    def test_no_sibling_methods_is_an_empty_list_not_missing_key(self):
        seed_node = {"id": "seed", "label": "f", "type": "function", "metadata": {}}
        result = LLMSerializer().serialize(
            {"seeds": [seed_node], "context_nodes": [], "edges": [], "test_nodes": []}
        )

        assert result["context"]["sibling_methods"] == []
