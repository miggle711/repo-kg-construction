"""kg_construction#61: super().method() calls had no special handling in
_resolve_call, so they fell through to the bare-name fallback -- matching
the method name against EVERY same-named method anywhere in the repo.
Found via a django/django check that reported 141,972 mostly-bogus
dependency cycles, traced to this exact pattern (e.g. ErrorList.copy() ->
some unrelated class's copy() method).

Fixed by resolving super().method() against the caller's own (already-
resolved) inherits chain instead -- reusing the same nearest-ancestor
walk _resolve_overrides already uses for the 'overrides' relation.
"""

from kg_construction.kg.builder import RepoASTParser


class TestSuperCallResolution:
    def _build(self, tmp_path, code: str):
        (tmp_path / "mod.py").write_text(code)
        parser = RepoASTParser(max_workers=1)
        return parser.parse_repo("test/repo", tmp_path)

    def test_super_call_resolves_to_real_parent_method(self, tmp_path):
        kg = self._build(
            tmp_path,
            "class Base:\n"
            "    def greet(self):\n"
            "        return 'hi'\n"
            "\n"
            "class Child(Base):\n"
            "    def greet(self):\n"
            "        return super().greet() + '!'\n",
        )
        nodes_by_id = {n["id"]: n for n in kg["nodes"]}
        child_greet = next(
            n for n in kg["nodes"] if n["label"] == "greet" and n["metadata"].get("class") == "Child"
        )
        calls_out = [e for e in kg["edges"] if e["relation"] == "calls" and e["source"] == child_greet["id"]]

        assert len(calls_out) == 1
        target = nodes_by_id[calls_out[0]["target"]]
        assert target["metadata"].get("class") == "Base"
        assert calls_out[0]["metadata"].get("confidence") == "qualified"

    def test_super_call_does_not_bare_name_match_unrelated_class(self, tmp_path):
        """The exact bug this fix addresses: a same-named method on an
        unrelated class must never be the resolution target, even though
        it would be a valid bare-name match.
        """
        kg = self._build(
            tmp_path,
            "class Widget:\n"
            "    def copy(self):\n"
            "        return 1\n"
            "\n"
            "class ErrorList(list):\n"
            "    def copy(self):\n"
            "        return super().copy()\n",
        )
        nodes_by_id = {n["id"]: n for n in kg["nodes"]}
        error_list_copy = next(
            n for n in kg["nodes"] if n["label"] == "copy" and n["metadata"].get("class") == "ErrorList"
        )
        calls_out = [e for e in kg["edges"] if e["relation"] == "calls" and e["source"] == error_list_copy["id"]]

        # ErrorList's real base (list) is a builtin, never parsed into
        # this KG -- so the edge must be dropped, NOT bare-name-matched
        # to Widget.copy just because it shares the name.
        for e in calls_out:
            target = nodes_by_id[e["target"]]
            assert target["metadata"].get("class") != "Widget"
        assert calls_out == []

    def test_super_call_walks_past_a_non_defining_ancestor(self, tmp_path):
        """B doesn't define greet, so C's super().greet() must resolve to
        A's version, not stop at B or fail to find anything.
        """
        kg = self._build(
            tmp_path,
            "class A:\n"
            "    def greet(self):\n"
            "        return 'a'\n"
            "\n"
            "class B(A):\n"
            "    def other(self):\n"
            "        return 'b'\n"
            "\n"
            "class C(B):\n"
            "    def greet(self):\n"
            "        return super().greet()\n",
        )
        nodes_by_id = {n["id"]: n for n in kg["nodes"]}
        c_greet = next(
            n for n in kg["nodes"] if n["label"] == "greet" and n["metadata"].get("class") == "C"
        )
        calls_out = [e for e in kg["edges"] if e["relation"] == "calls" and e["source"] == c_greet["id"]]

        assert len(calls_out) == 1
        target = nodes_by_id[calls_out[0]["target"]]
        assert target["metadata"].get("class") == "A"

    def test_bare_super_construction_call_is_not_emitted_as_its_own_edge(self, tmp_path):
        """super().method() parses as two ast.Call nodes: the outer method
        call, and the inner call that constructs the super proxy itself
        (callee='super', receiver=None). The inner one is not a real call
        to anything named 'super' and must not become its own edge --
        found via django/django, where a real method literally named
        'super' existed elsewhere and was bare-name-matched by this
        artifact before the fix.
        """
        kg = self._build(
            tmp_path,
            "class Base:\n"
            "    def greet(self):\n"
            "        return 'hi'\n"
            "\n"
            "class Child(Base):\n"
            "    def greet(self):\n"
            "        return super().greet()\n",
        )
        super_named_calls = [
            e for e in kg["edges"]
            if e["relation"] == "calls" and e["target"] == "super"
        ]
        assert super_named_calls == []
