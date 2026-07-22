"""Regression tests for issue #45: RepoKGBuilder's on-disk cache must be
keyed on (repo, commit), not repo alone, and must never silently serve a
KG from the wrong commit or an outdated schema.

Also covers a second bug found while fixing #45: pipeline._load_or_build
relied on RepoKGBuilder.load() raising FileNotFoundError on a cache miss,
but load() has always returned None instead -- so _load_or_build returned
None (never building anything) for ANY never-before-cached repo, not just
the wrong-commit case #45 originally described.
"""

from pathlib import Path
from unittest.mock import MagicMock

from kg_construction.kg.builder import RepoKGBuilder, SCHEMA_VERSION
from kg_construction.pipeline import _load_or_build


class TestRepoKGBuilderCache:
    def test_load_returns_none_before_any_build(self, tmp_path):
        builder = RepoKGBuilder(output_dir=tmp_path)
        assert builder.load("some/repo", "commit1") is None

    def test_load_after_save_returns_the_kg(self, tmp_path):
        builder = RepoKGBuilder(output_dir=tmp_path)
        kg = _fake_kg("commit1")
        builder.save("some/repo", kg)

        loaded = builder.load("some/repo", "commit1")
        assert loaded is not None
        assert loaded["metadata"]["base_commit"] == "commit1"

    def test_load_at_different_commit_does_not_return_wrong_commit_kg(self, tmp_path):
        """The core #45 bug: a cached KG at one commit must never be
        served for a request at a different commit of the same repo.
        """
        builder = RepoKGBuilder(output_dir=tmp_path)
        builder.save("some/repo", _fake_kg("commit1"))

        assert builder.load("some/repo", "commit2") is None

    def test_load_rejects_kg_with_stale_schema_version(self, tmp_path):
        """A cached file with a missing or outdated schema_version (e.g.
        from before this fix) must be treated as a cache miss, not served
        as if its shape still matches what current code expects.
        """
        import json

        builder = RepoKGBuilder(output_dir=tmp_path)
        old_kg = _fake_kg("commit1")
        del old_kg["metadata"]["schema_version"]  # simulate a pre-fix cached file
        cache_path = builder._cache_path("some/repo", "commit1")
        with open(cache_path, "w") as f:
            json.dump(old_kg, f)

        assert builder.load("some/repo", "commit1") is None

    def test_build_stamps_schema_version(self, monkeypatch, tmp_path):
        builder = RepoKGBuilder(output_dir=tmp_path)
        monkeypatch.setattr(
            builder.repo_manager, "extract_at_commit", lambda repo, commit, dest: None
        )
        monkeypatch.setattr(
            builder.ast_parser, "parse_repo",
            lambda repo, dest: {"nodes": [], "edges": [], "metadata": {}}
        )
        kg = builder.build("some/repo", "commit1")
        assert kg["metadata"]["schema_version"] == SCHEMA_VERSION


class TestLoadOrBuild:
    def test_builds_and_saves_on_cache_miss(self):
        """Regression test: _load_or_build previously relied on load()
        raising FileNotFoundError, but load() has always returned None
        on a miss -- so this path silently returned None instead of
        building, for every never-before-cached repo.
        """
        mock_builder = MagicMock()
        mock_builder.load.return_value = None
        mock_builder.build.return_value = _fake_kg("somecommit")

        result = _load_or_build(mock_builder, "some/repo", "somecommit")

        assert result is not None
        mock_builder.build.assert_called_once_with("some/repo", "somecommit")
        mock_builder.save.assert_called_once()

    def test_returns_cached_kg_without_rebuilding_on_cache_hit(self):
        mock_builder = MagicMock()
        cached = _fake_kg("somecommit")
        mock_builder.load.return_value = cached

        result = _load_or_build(mock_builder, "some/repo", "somecommit")

        assert result is cached
        mock_builder.build.assert_not_called()
        mock_builder.save.assert_not_called()


def _fake_kg(commit: str) -> dict:
    return {
        "nodes": [],
        "edges": [],
        "metadata": {
            "repo": "some/repo",
            "base_commit": commit,
            "file_count": 0,
            "parse_mode": "source",
            "schema_version": SCHEMA_VERSION,
        },
    }
