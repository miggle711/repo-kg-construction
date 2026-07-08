"""Tests for RepoManager cache-refresh behavior on git archive misses."""

import subprocess

import pytest

from kg_construction.kg.repo_manager import RepoManager


def _run(args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, timeout=30)


@pytest.fixture
def remote_repo(tmp_path):
    """A local (non-bare) git repo standing in for a GitHub remote."""
    remote = tmp_path / "remote"
    remote.mkdir()
    _run(["git", "init", "-q"], cwd=remote)
    _run(["git", "config", "user.email", "test@test.com"], cwd=remote)
    _run(["git", "config", "user.name", "test"], cwd=remote)
    (remote / "file.txt").write_text("v1")
    _run(["git", "add", "file.txt"], cwd=remote)
    _run(["git", "commit", "-q", "-m", "initial commit"], cwd=remote)
    return remote


def _head_commit(repo_path):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_path,
        check=True, capture_output=True, text=True, timeout=30,
    )
    return result.stdout.strip()


class LocalRepoManager(RepoManager):
    """RepoManager that clones from a local path instead of github.com."""

    def __init__(self, cache_dir, remote_path):
        super().__init__(cache_dir=cache_dir)
        self._remote_path = str(remote_path)

    def ensure_clone(self, repo):
        path = self._cache_path(repo)
        if not path.exists():
            subprocess.run(
                ["git", "clone", "--bare", self._remote_path, str(path)],
                check=True, timeout=30,
            )
        return path


class TestRepoManagerRefresh:
    """RepoManager.extract_at_commit must recover from a stale cached clone."""

    def test_extracts_commit_present_at_clone_time(self, tmp_path, remote_repo):
        first_commit = _head_commit(remote_repo)
        manager = LocalRepoManager(cache_dir=tmp_path / "cache", remote_path=remote_repo)

        dest = tmp_path / "dest"
        manager.extract_at_commit("test/repo", first_commit, dest)

        assert (dest / "file.txt").read_text() == "v1"

    def test_fetches_and_retries_for_commit_added_after_clone(self, tmp_path, remote_repo):
        manager = LocalRepoManager(cache_dir=tmp_path / "cache", remote_path=remote_repo)

        # Prime the cache before the second commit exists upstream.
        first_commit = _head_commit(remote_repo)
        manager.extract_at_commit("test/repo", first_commit, tmp_path / "dest1")

        # A new commit lands on the "remote" after the clone was cached.
        (remote_repo / "file.txt").write_text("v2")
        _run(["git", "add", "file.txt"], cwd=remote_repo)
        _run(["git", "commit", "-q", "-m", "second commit"], cwd=remote_repo)
        second_commit = _head_commit(remote_repo)

        dest2 = tmp_path / "dest2"
        # Without the fetch-and-retry fix, this raises CalledProcessError
        # because the cached bare clone predates this commit.
        manager.extract_at_commit("test/repo", second_commit, dest2)

        assert (dest2 / "file.txt").read_text() == "v2"

    def test_raises_clear_error_for_nonexistent_commit(self, tmp_path, remote_repo):
        manager = LocalRepoManager(cache_dir=tmp_path / "cache", remote_path=remote_repo)

        with pytest.raises(ValueError, match="not found in test/repo"):
            manager.extract_at_commit(
                "test/repo", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", tmp_path / "dest"
            )
