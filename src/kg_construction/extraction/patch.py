"""
patch.py

Unified diff parsing for code change extraction.

Parses GitHub-style unified diffs to identify which functions/classes were
modified in a given code file. Used in Phase 2 (Seed Identification) to find
which nodes in the KG correspond to the changed code.
"""

import re
from typing import List, NamedTuple, Optional, Set


class _HunkLine(NamedTuple):
    """A single hunk line, its diff marker preserved separately from its
    content so later analysis can tell "genuinely added/removed" apart
    from "unchanged context that happens to be visible in this hunk"."""
    marker: str  # '+', '-', or ' '
    content: str  # line text with the marker stripped
    indent: int  # leading whitespace width of content


class PatchParser:
    """Parse unified diffs to extract changed function/class names."""

    @staticmethod
    def extract_changed_functions(patch: str, code_file: str) -> Set[str]:
        """Extract function and class names genuinely changed in a specific
        file's hunks.

        Parses unified diff hunks for code_file and identifies function/class
        definitions where at least one line actually changed (was added or
        removed) -- either the def/class line itself, its decorator, or a
        line within its body -- as opposed to merely being visible in the
        hunk as unchanged context. A wide diff context window can otherwise
        sweep an entirely unmodified neighboring function's def line into the
        hunk; such a function must not be reported as changed.

        A changed line consisting only of a decorator (e.g. `@app.route(...)`)
        carries no name of its own, so it is tracked as "pending" and resolved
        against the next def/class line encountered, even if that def/class
        falls in a later hunk of the same file's diff (small context windows
        can put a hunk boundary between an added/modified decorator and the
        otherwise-unchanged function it decorates). A function reached via a
        pending changed decorator counts as changed even if its own def line
        and body are pure context.

        Args:
            patch: Unified diff string (multi-file).
            code_file: Relative path to the file to extract changes from
                      (e.g. 'requests/sessions.py').

        Returns:
            Set of function/class name strings genuinely changed in code_file.
        """
        changed_names: Set[str] = set()

        current_file = None
        current_hunk: List[_HunkLine] = []
        pending_decorator = False

        lines = patch.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for file boundary: +++ b/path
            if line.startswith('+++'):
                # Process accumulated hunk for previous file before switching
                if current_hunk:
                    pending_decorator = PatchParser._extract_defs_from_hunk(
                        current_hunk, changed_names, pending_decorator
                    )
                current_hunk = []
                pending_decorator = False
                # Extract the file path from '+++ b/path'
                match = re.match(r'^\+\+\+ b/(.+)$', line)
                if match:
                    current_file = match.group(1)

            # Within the target file, collect hunk lines
            if current_file == code_file:
                # Hunk header: @@ -start,count +start,count @@
                if line.startswith('@@'):
                    if current_hunk:
                        pending_decorator = PatchParser._extract_defs_from_hunk(
                            current_hunk, changed_names, pending_decorator
                        )
                    current_hunk = []
                # Accumulate hunk lines: added, removed, or context.
                # Removed ('-') lines are kept (unlike before) so body-change
                # detection can see them; they carry no post-patch line
                # number, but def/class matching only needs their text.
                elif line.startswith(('+', '-', ' ')) and not line.startswith(('+++', '---')):
                    marker = line[0]
                    content = line[1:]
                    indent = len(content) - len(content.lstrip())
                    current_hunk.append(_HunkLine(marker, content, indent))

            i += 1

        # Process final hunk
        if current_hunk:
            PatchParser._extract_defs_from_hunk(
                current_hunk, changed_names, pending_decorator
            )

        return changed_names

    @staticmethod
    def _extract_defs_from_hunk(
        hunk_lines: List[_HunkLine], names: Set[str], pending_decorator: bool
    ) -> bool:
        """Extract function/class definitions genuinely changed within a hunk.

        Looks for lines matching:
          def function_name(...)
          async def function_name(...)
          class ClassName(...)
          @decorator(...)

        A def/class is only added to `names` if it is genuinely changed:
        its own line was added/removed, a changed decorator immediately
        precedes it (including one carried over from a previous hunk via
        `pending_decorator`), or at least one line within its body (up to
        the next sibling def/class at the same or shallower indentation,
        or the end of the hunk) was added/removed. A def/class line that is
        pure unchanged context, with an unchanged body and no changed
        decorator, is NOT added -- it's just visible because of the hunk's
        context window, not because it changed.

        Returns:
            True if a decorator is still pending resolution at the end of
            this hunk (i.e. no def/class line followed it), so the caller
            can carry it into the next hunk of the same file.
        """
        def_pattern = re.compile(r'^\s*(async\s+)?def\s+(\w+)\s*\(')
        class_pattern = re.compile(r'^\s*class\s+(\w+)\s*[\(:]')
        decorator_pattern = re.compile(r'^\s*@\w')

        n = len(hunk_lines)
        for idx, hline in enumerate(hunk_lines):
            def_match = def_pattern.match(hline.content)
            class_match = class_pattern.match(hline.content)

            if not (def_match or class_match):
                if decorator_pattern.match(hline.content):
                    pending_decorator = pending_decorator or hline.marker != ' '
                continue

            name = def_match.group(2) if def_match else class_match.group(1)
            own_line_changed = hline.marker != ' '
            decorator_changed = pending_decorator
            body_changed = PatchParser._body_has_change(hunk_lines, idx, hline.indent)

            if own_line_changed or decorator_changed or body_changed:
                names.add(name)

            pending_decorator = False

        return pending_decorator

    @staticmethod
    def _body_has_change(hunk_lines: List[_HunkLine], def_idx: int, def_indent: int) -> bool:
        """Return True if any line strictly after hunk_lines[def_idx], up to
        the next sibling def/class at the same or shallower indentation (or
        the end of the hunk), was added or removed.

        Blank lines are skipped for the indentation check (they carry no
        real indent signal) but still checked for a changed marker.
        """
        for hline in hunk_lines[def_idx + 1:]:
            stripped = hline.content.strip()
            if stripped and hline.indent <= def_indent:
                break  # sibling or dedent: end of this def/class's body
            if hline.marker != ' ':
                return True
        return False
