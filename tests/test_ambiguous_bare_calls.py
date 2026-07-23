"""kg_construction#65: the bare-name call fallback (used when no class/
local-type/import hint resolves a call's receiver) used to link an
ambiguous callee name to EVERY same-named function/method in the repo.
Measured on real KGs: 80.6% of psf/requests' calls edges and 92.56% of
django/django's hit this fallback, and a random sample showed most of
those links were simply wrong (e.g. a syndication-feed test's call
linked to an unrelated template-context method sharing the name 'get').

Fixed: an ambiguous bare-name match (>1 candidate, no receiver at all)
now prefers a same-file candidate when exactly one exists (a bare call
resolves via the caller's own module scope, e.g. api.py's post() calling
bare request(...) means api.py's own request() function, not an
unrelated method on a class in a different file). Otherwise, the call is
dropped rather than fanned out to every candidate.
"""

from kg_construction.kg.builder import RepoASTParser


class TestAmbiguousBareCalls:
    def _build(self, tmp_path, files: dict):
        for name, content in files.items():
            (tmp_path / name).write_text(content)
        parser = RepoASTParser(max_workers=1)
        return parser.parse_repo("test/repo", tmp_path)

    def test_ambiguous_bare_call_prefers_same_file_candidate(self, tmp_path):
        """The real psf/requests case that motivated this: api.py's post()
        calls bare request(...), meaning api.py's own request() function
        -- not the unrelated Session.request method in a different file.
        """
        kg = self._build(
            tmp_path,
            {
                "api.py": (
                    "def request(method, url):\n"
                    "    return method\n"
                    "\n"
                    "def post(url):\n"
                    "    return request('post', url)\n"
                ),
                "sessions.py": (
                    "class Session:\n"
                    "    def request(self, method, url):\n"
                    "        return method\n"
                ),
            },
        )
        nodes_by_id = {n["id"]: n for n in kg["nodes"]}
        post_fn = next(n for n in kg["nodes"] if n["label"] == "post")
        calls_out = [e for e in kg["edges"] if e["relation"] == "calls" and e["source"] == post_fn["id"]]

        assert len(calls_out) == 1
        target = nodes_by_id[calls_out[0]["target"]]
        assert target["metadata"].get("filepath") == "api.py"
        assert target["type"] == "function"

    def test_ambiguous_attribute_call_with_no_same_file_candidate_is_dropped(self, tmp_path):
        """An attribute call (obj.method()) whose receiver type can't be
        inferred, with multiple same-named candidates NONE of which are
        in the caller's own file, must be dropped -- not linked to an
        arbitrary one of them.
        """
        kg = self._build(
            tmp_path,
            {
                "caller.py": (
                    "def do_work(obj):\n"
                    "    return obj.get()\n"
                ),
                "a.py": (
                    "class Alpha:\n"
                    "    def get(self):\n"
                    "        return 1\n"
                ),
                "b.py": (
                    "class Beta:\n"
                    "    def get(self):\n"
                    "        return 2\n"
                ),
            },
        )
        do_work = next(n for n in kg["nodes"] if n["label"] == "do_work")
        calls_out = [e for e in kg["edges"] if e["relation"] == "calls" and e["source"] == do_work["id"]]
        assert calls_out == []

    def test_call_to_nonexistent_name_is_dropped_not_erroring(self, tmp_path):
        """A call to something not parsed anywhere in the repo (e.g. an
        external library function) must simply produce no edge.
        """
        kg = self._build(
            tmp_path,
            {"mod.py": "def caller():\n    return some_external_library_call()\n"},
        )
        caller = next(n for n in kg["nodes"] if n["label"] == "caller")
        calls_out = [e for e in kg["edges"] if e["relation"] == "calls" and e["source"] == caller["id"]]
        assert calls_out == []

    def test_unambiguous_bare_call_still_resolves_normally(self, tmp_path):
        """A callee name matching exactly one function anywhere in the
        repo must still resolve -- this fix only changes the >1-match
        case, not the single-match case.
        """
        kg = self._build(
            tmp_path,
            {
                "mod.py": (
                    "def helper():\n"
                    "    return 1\n"
                    "\n"
                    "def caller():\n"
                    "    return helper()\n"
                )
            },
        )
        nodes_by_id = {n["id"]: n for n in kg["nodes"]}
        caller = next(n for n in kg["nodes"] if n["label"] == "caller")
        calls_out = [e for e in kg["edges"] if e["relation"] == "calls" and e["source"] == caller["id"]]

        assert len(calls_out) == 1
        assert nodes_by_id[calls_out[0]["target"]]["label"] == "helper"
        assert calls_out[0]["metadata"].get("confidence") == "exact"
