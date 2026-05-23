"""
patch.py

Unified diff parsing for code change extraction.

Parses GitHub-style unified diffs to identify which functions/classes were
modified in a given code file. Used in Phase 2 (Seed Identification) to find
which nodes in the KG correspond to the changed code.
"""

import re
from typing import Set, List


class PatchParser:
    """Parse unified diffs to extract changed function/class names."""

    @staticmethod
    def extract_changed_functions(patch: str, code_file: str) -> Set[str]:
        """Extract function and class names changed in a specific file's hunks.

        Parses unified diff hunks for code_file and identifies function/class
        definitions that appear in the changed lines (those with + or context
        lines near the hunk start). Returns a set of changed names.

        Args:
            patch: Unified diff string (multi-file).
            code_file: Relative path to the file to extract changes from
                      (e.g. 'requests/sessions.py').

        Returns:
            Set of function/class name strings that changed in code_file.
        """
        changed_names: Set[str] = set()

        current_file = None
        current_hunk = []

        lines = patch.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for file boundary: +++ b/path
            if line.startswith('+++'):
                # Process accumulated hunk for previous file before switching
                if current_hunk:
                    changed_names.update(
                        PatchParser._extract_defs_from_hunk(current_hunk)
                    )
                current_hunk = []
                # Extract the file path from '+++ b/path'
                match = re.match(r'^\+\+\+ b/(.+)$', line)
                if match:
                    current_file = match.group(1)

            # Within the target file, collect hunk lines
            if current_file == code_file:
                # Hunk header: @@ -start,count +start,count @@
                if line.startswith('@@'):
                    if current_hunk:
                        changed_names.update(
                            PatchParser._extract_defs_from_hunk(current_hunk)
                        )
                    current_hunk = []
                # Accumulate changed lines (+ or context, not -)
                elif line.startswith(('+', ' ')) and not line.startswith('+++'):
                    current_hunk.append(line)

            i += 1

        # Process final hunk
        if current_hunk:
            changed_names.update(
                PatchParser._extract_defs_from_hunk(current_hunk)
            )

        return changed_names

    @staticmethod
    def _extract_defs_from_hunk(hunk_lines: List[str]) -> Set[str]:
        """Extract function/class definitions from a list of hunk lines.

        Looks for lines matching:
          def function_name(...)
          async def function_name(...)
          class ClassName(...)

        Returns the set of names found.
        """
        names: Set[str] = set()

        for line in hunk_lines:
            # Strip the leading +/space marker
            content = line[1:] if line and line[0] in ('+', ' ') else line

            # Match: def/async def/class name(
            def_match = re.match(r'^\s*(async\s+)?def\s+(\w+)\s*\(', content)
            if def_match:
                names.add(def_match.group(2))

            class_match = re.match(r'^\s*class\s+(\w+)\s*[\(:]', content)
            if class_match:
                names.add(class_match.group(1))

        return names
