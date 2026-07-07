"""Regression test for module_depends_on edge resolution across same-named files.

Covers the bug where file_label_to_id was keyed by bare filename, causing
identically-named files in different packages (pkg_a/utils.py vs
pkg_b/utils.py) to collide and misattribute module_depends_on edges to
whichever file happened to be indexed last.
"""

from pathlib import Path

from kg_construction.kg.builder import RepoASTParser


def _write_repo(tmp_path: Path) -> Path:
    """Create a small repo with two same-named 'utils.py' files in different packages.

    pkg_a/main.py imports pkg_a.utils (should depend on pkg_a/utils.py).
    pkg_b/main.py imports pkg_b.utils (should depend on pkg_b/utils.py).
    """
    (tmp_path / "pkg_a").mkdir()
    (tmp_path / "pkg_b").mkdir()

    (tmp_path / "pkg_a" / "__init__.py").write_text("")
    (tmp_path / "pkg_b" / "__init__.py").write_text("")

    (tmp_path / "pkg_a" / "utils.py").write_text(
        "def helper_a():\n    return 'a'\n"
    )
    (tmp_path / "pkg_b" / "utils.py").write_text(
        "def helper_b():\n    return 'b'\n"
    )

    (tmp_path / "pkg_a" / "main.py").write_text(
        "from pkg_a.utils import helper_a\n\n"
        "def run():\n    return helper_a()\n"
    )
    (tmp_path / "pkg_b" / "main.py").write_text(
        "from pkg_b.utils import helper_b\n\n"
        "def run():\n    return helper_b()\n"
    )

    return tmp_path


class TestModuleDependsOnPathCollision:
    """module_depends_on edges must resolve to the correct same-named file."""

    def test_same_named_files_resolve_independently(self, tmp_path):
        repo_dir = _write_repo(tmp_path)
        parser = RepoASTParser(max_workers=1)
        kg = parser.parse_repo("test/repo", repo_dir)

        nodes_by_id = {n["id"]: n for n in kg["nodes"]}

        def file_id_for(rel_path: str) -> str:
            matches = [
                n["id"] for n in kg["nodes"]
                if n["type"] == "file" and n["metadata"].get("path") == rel_path
            ]
            assert len(matches) == 1, f"expected exactly one node for {rel_path}"
            return matches[0]

        main_a_id = file_id_for("pkg_a/main.py")
        main_b_id = file_id_for("pkg_b/main.py")
        utils_a_id = file_id_for("pkg_a/utils.py")
        utils_b_id = file_id_for("pkg_b/utils.py")

        module_depends_on = [
            e for e in kg["edges"] if e["relation"] == "module_depends_on"
        ]

        # pkg_a/main.py must depend on pkg_a/utils.py, not pkg_b/utils.py
        assert any(
            e["source"] == main_a_id and e["target"] == utils_a_id
            for e in module_depends_on
        ), "pkg_a/main.py should depend on pkg_a/utils.py"
        assert not any(
            e["source"] == main_a_id and e["target"] == utils_b_id
            for e in module_depends_on
        ), "pkg_a/main.py must not depend on pkg_b/utils.py"

        # pkg_b/main.py must depend on pkg_b/utils.py, not pkg_a/utils.py
        assert any(
            e["source"] == main_b_id and e["target"] == utils_b_id
            for e in module_depends_on
        ), "pkg_b/main.py should depend on pkg_b/utils.py"
        assert not any(
            e["source"] == main_b_id and e["target"] == utils_a_id
            for e in module_depends_on
        ), "pkg_b/main.py must not depend on pkg_a/utils.py"
