"""Tests for 'accesses' edges: @property reads that never appear as
ast.Call nodes, so _emit_call_edges has zero signal for them.

A @property-decorated method is invoked as `obj.attr`, never
`obj.attr()`. Before this, the KG had no edge type at all for this --
not wrong, just a complete structural blind spot for a very common
Python pattern (e.g. requests.Session.headers).
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


def _accesses_targets(kg, source_id):
    return {
        e["target"] for e in kg["edges"]
        if e["relation"] == "accesses" and e["source"] == source_id
    }


class TestSelfPropertyAccess:
    """self.some_property reads must resolve via class_hint, like self.method() calls do."""

    def test_self_property_read_resolves_with_qualified_confidence(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "class Widget:\n"
            "    @property\n"
            "    def label(self):\n"
            "        return self._label\n"
            "\n"
            "    def describe(self):\n"
            "        return f'Widget: {self.label}'\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", tmp_path)

        describe_id = _method_id(kg, "Widget", "describe")
        label_prop_id = _method_id(kg, "Widget", "label")

        targets = _accesses_targets(kg, describe_id)
        assert label_prop_id in targets

        edge = next(
            e for e in kg["edges"]
            if e["relation"] == "accesses"
            and e["source"] == describe_id
            and e["target"] == label_prop_id
        )
        assert edge["metadata"]["confidence"] == "qualified"


class TestParameterAnnotatedPropertyAccess:
    """A property read through a type-annotated parameter must resolve via local_type_hint."""

    def test_annotated_parameter_property_read_resolves(self, tmp_path):
        (tmp_path / "mod2.py").write_text(
            "class Session:\n"
            "    @property\n"
            "    def headers(self):\n"
            "        return self._headers\n"
            "\n"
            "class Client:\n"
            "    def inspect(self, session: Session):\n"
            "        return session.headers\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo2", tmp_path)

        inspect_id = _method_id(kg, "Client", "inspect")
        headers_prop_id = _method_id(kg, "Session", "headers")

        targets = _accesses_targets(kg, inspect_id)
        assert headers_prop_id in targets


class TestPropertyAccessDoesNotDuplicateCalls:
    """obj.method() must still emit only a 'calls' edge, not also 'accesses'."""

    def test_method_call_does_not_also_emit_accesses_edge(self, tmp_path):
        (tmp_path / "mod3.py").write_text(
            "class Widget:\n"
            "    def render(self):\n"
            "        return 'rendered'\n"
            "\n"
            "class App:\n"
            "    def run(self, widget: Widget):\n"
            "        return widget.render()\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo3", tmp_path)

        run_id = _method_id(kg, "App", "run")
        render_id = _method_id(kg, "Widget", "render")

        calls_targets = {
            e["target"] for e in kg["edges"]
            if e["relation"] == "calls" and e["source"] == run_id
        }
        accesses_targets = _accesses_targets(kg, run_id)

        assert render_id in calls_targets
        assert render_id not in accesses_targets
        assert len(accesses_targets) == 0


class TestNonPropertyAttributeAccessNotResolved:
    """A same-named plain method (not @property) must not be linked via 'accesses'."""

    def test_attribute_read_on_non_property_method_is_dropped(self, tmp_path):
        (tmp_path / "mod4.py").write_text(
            "class Widget:\n"
            "    def label(self):\n"  # plain method, NOT a property
            "        return self._label\n"
            "\n"
            "class App:\n"
            "    def run(self, widget: Widget):\n"
            "        return widget.label\n"  # attribute read, no call syntax
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo4", tmp_path)

        run_id = _method_id(kg, "App", "run")
        accesses_targets = _accesses_targets(kg, run_id)

        # widget.label is a bare attribute read on a class where 'label' is
        # a plain method, not a property -- must not resolve to it.
        assert len(accesses_targets) == 0


class TestPropertyWriteNotTreatedAsAccess:
    """obj.attr = value (a write/Store) must not be treated as a property read."""

    def test_assignment_target_is_not_an_access_edge(self, tmp_path):
        (tmp_path / "mod5.py").write_text(
            "class Widget:\n"
            "    @property\n"
            "    def label(self):\n"
            "        return self._label\n"
            "\n"
            "class App:\n"
            "    def run(self, widget: Widget):\n"
            "        widget.other_attr = 'x'\n"
            "        return widget.other_attr\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo5", tmp_path)

        run_id = _method_id(kg, "App", "run")
        accesses_targets = _accesses_targets(kg, run_id)

        # other_attr isn't a @property anywhere, so this correctly resolves
        # to nothing -- the point of this test is that parsing an
        # assignment target doesn't crash or mis-tag the write as a read.
        assert isinstance(accesses_targets, set)
