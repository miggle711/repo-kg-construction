"""Regression test: type-annotated parameters should resolve call edges
with 'qualified' confidence via local_type_hint, instead of falling
through to bare-name resolution and getting marked 'ambiguous' whenever
another class in the repo happens to share a method name.
"""

from pathlib import Path

from kg_construction.kg.builder import RepoASTParser


def _write_repo(tmp_path: Path) -> Path:
    """Two classes share a method name ('prepare'); a third class receives
    one of them as a type-annotated parameter and calls .prepare() on it.
    Without the parameter-annotation hint, that call is ambiguous between
    Request.prepare and Response.prepare.
    """
    (tmp_path / "mod.py").write_text(
        "class Request:\n"
        "    def prepare(self):\n"
        "        return 'request prepared'\n"
        "\n"
        "class Response:\n"
        "    def prepare(self):\n"
        "        return 'response prepared'\n"
        "\n"
        "class Session:\n"
        "    def send(self, request: Request):\n"
        "        return request.prepare()\n"
    )
    return tmp_path


class TestLocalTypeHintsFromAnnotations:
    """Type-annotated parameters must disambiguate call-edge resolution."""

    def test_annotated_parameter_resolves_with_qualified_confidence(self, tmp_path):
        repo_dir = _write_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        nodes_by_id = {n["id"]: n for n in kg["nodes"]}

        def method_id(class_name: str, method_name: str) -> str:
            matches = [
                n["id"] for n in kg["nodes"]
                if n["type"] == "method"
                and n["label"] == method_name
                and n["metadata"].get("class") == class_name
            ]
            assert len(matches) == 1, f"expected exactly one {class_name}.{method_name}"
            return matches[0]

        send_id = method_id("Session", "send")
        request_prepare_id = method_id("Request", "prepare")
        response_prepare_id = method_id("Response", "prepare")

        calls_from_send = [
            e for e in kg["edges"]
            if e["relation"] == "calls" and e["source"] == send_id
        ]

        # Must resolve to Request.prepare specifically, not Response.prepare
        targets = {e["target"] for e in calls_from_send}
        assert request_prepare_id in targets, "send() should call Request.prepare"
        assert response_prepare_id not in targets, (
            "send() must not ambiguously link to Response.prepare "
            "when the parameter annotation disambiguates it"
        )

        # And it should be qualified, not merely 'exact' by luck or 'ambiguous'
        edge_to_request = next(
            e for e in calls_from_send if e["target"] == request_prepare_id
        )
        assert edge_to_request["metadata"]["confidence"] == "qualified"

    def test_reassignment_still_overrides_annotation(self, tmp_path):
        """A constructor-call reassignment of an annotated parameter should
        still win, matching _collect_local_types' existing documented
        behavior of "reassignments overwrite earlier entries"."""
        (tmp_path / "mod2.py").write_text(
            "class Request:\n"
            "    def prepare(self):\n"
            "        return 'request'\n"
            "\n"
            "class MockRequest:\n"
            "    def prepare(self):\n"
            "        return 'mock'\n"
            "\n"
            "class Session:\n"
            "    def send(self, request: Request):\n"
            "        request = MockRequest()\n"
            "        return request.prepare()\n"
        )
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo2", tmp_path)

        send_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "method" and n["label"] == "send"
        )
        mock_prepare_id = next(
            n["id"] for n in kg["nodes"]
            if n["type"] == "method" and n["label"] == "prepare"
            and n["metadata"].get("class") == "MockRequest"
        )

        calls_from_send = [
            e for e in kg["edges"]
            if e["relation"] == "calls" and e["source"] == send_id
        ]
        targets = {e["target"] for e in calls_from_send}
        assert mock_prepare_id in targets, (
            "reassignment to MockRequest() should override the Request "
            "parameter annotation"
        )
