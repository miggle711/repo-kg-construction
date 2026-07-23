"""
patch.py

Unified diff parsing for code change extraction.

Parses GitHub-style unified diffs to identify which functions/classes were
modified in a given code file. Used in Phase 2 (Seed Identification) to find
which nodes in the KG correspond to the changed code.
"""

import re
from typing import List, NamedTuple, Optional, Set, Tuple


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
            Bare names only -- callers needing to disambiguate a name that
            matches more than one class' method (kg_construction#63) should
            use extract_changed_functions_with_scope instead.
        """
        return {name for name, _class_name in PatchParser.extract_changed_functions_with_scope(
            patch, code_file
        )}

    @staticmethod
    def extract_changed_functions_with_scope(
        patch: str, code_file: str
    ) -> Set[Tuple[str, Optional[str]]]:
        """Same as extract_changed_functions, but also reports the
        enclosing class name for each changed method, when it can be
        determined.

        The enclosing class is only known when the hunk itself contains a
        'class X:' line at a shallower indentation than the changed
        def/class line -- true whenever the diff's context window happens
        to reach back that far, but not guaranteed (a hunk deep inside a
        long class body may never include its 'class X:' line at all).
        When unknown, the class element of the tuple is None -- callers
        must not assume every entry has a real class name, just that one
        is reported when the hunk provides enough information to know it.

        Added for kg_construction#63: TestContextExtractor.extract()'s
        seed lookup previously had no way to prefer 'AsyncClient.aclose'
        over an unrelated 'BoundAsyncStream.aclose' in the same file, once
        extract_changed_functions returned the bare name 'aclose' with no
        further information -- this gives it a real signal to disambiguate
        with, when the diff happens to make it available; when it isn't
        available, extract()/TestContextValidator's ambiguity check is
        still the correct fallback (fail loud, not guess).

        Returns:
            Set of (name, enclosing_class_or_None) tuples.
        """
        changed: Set[Tuple[str, Optional[str]]] = set()

        current_file = None
        current_hunk: List[_HunkLine] = []
        pending_decorator = False
        # git's own "which function/class is this hunk inside" hints, from
        # the trailing text on an '@@ ... @@' line (e.g.
        # '@@ -470,11 +470,11 @@ def slugify(value, allow_unicode=False):' or
        # '@@ -1976,10 +1976,11 @@ class AsyncClient(BaseClient):'). The
        # function-name hint is a fallback ONLY when the hunk body itself
        # contains no def/class line at all; the class-name hint is a
        # fallback enclosing-class for a def line that IS found in the
        # hunk body but has no preceding 'class X:' line within the same
        # hunk -- see _extract_defs_from_hunk for both.
        header_scope_name: Optional[str] = None
        header_scope_class: Optional[str] = None

        lines = patch.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for file boundary: +++ b/path
            if line.startswith('+++'):
                # Process accumulated hunk for previous file before switching
                if current_hunk:
                    pending_decorator = PatchParser._extract_defs_from_hunk(
                        current_hunk, changed, pending_decorator,
                        header_scope_name, header_scope_class,
                    )
                current_hunk = []
                pending_decorator = False
                header_scope_name = None
                header_scope_class = None
                # Extract the file path from '+++ b/path'
                match = re.match(r'^\+\+\+ b/(.+)$', line)
                if match:
                    current_file = match.group(1)

            # Within the target file, collect hunk lines
            if current_file == code_file:
                # Hunk header: @@ -start,count +start,count @@ [trailing context]
                if line.startswith('@@'):
                    if current_hunk:
                        pending_decorator = PatchParser._extract_defs_from_hunk(
                            current_hunk, changed, pending_decorator,
                            header_scope_name, header_scope_class,
                        )
                    current_hunk = []
                    header_scope_name = PatchParser._extract_header_scope_name(line)
                    header_scope_class = PatchParser._extract_header_scope_class(line)
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
                current_hunk, changed, pending_decorator,
                header_scope_name, header_scope_class,
            )

        return changed

    @staticmethod
    def _extract_header_scope_name(header_line: str) -> Optional[str]:
        """Extract the enclosing FUNCTION name git prints as trailing
        context on a hunk header line (e.g.
        '@@ -470,11 +470,11 @@ def slugify(value, allow_unicode=False):'
        -> 'slugify'), when git includes one. Returns None if the header's
        trailing context is a class line instead -- see
        _extract_header_scope_class for that case.

        This is git's own "which function is this hunk inside" hint,
        computed from its diff context algorithm -- independent of and a
        fallback for the def/class-line-matching this parser does over
        the hunk body itself, which fails silently whenever the changed
        lines are far enough into a long function that its own def line
        falls outside the (small) context window and never appears as a
        hunk body line at all. Confirmed via a real django/django patch
        (slugify(), kg_construction#60's second-repo check): identical
        semantic change, changed_functions came back empty purely because
        the def line wasn't within context, and only non-empty once a
        wider context window happened to include it.
        """
        # Only match a def-shaped trailing context, not arbitrary trailing
        # text (git falls back to the nearest preceding non-blank line for
        # other languages/heuristic misses, which isn't necessarily a
        # function signature at all).
        match = re.match(r'^@@[^@]*@@\s*(?:async\s+)?def\s+(\w+)\s*\(', header_line)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _extract_header_scope_class(header_line: str) -> Optional[str]:
        """Extract the enclosing CLASS name from a hunk header's trailing
        context (e.g. '@@ -1976,10 +1976,11 @@ class AsyncClient(BaseClient):'
        -> 'AsyncClient'), when git's diff context algorithm reports one.

        Used as a fallback enclosing-class hint for a def/class line found
        within the hunk body when no 'class X:' line precedes it in the
        SAME hunk (kg_construction#63) -- e.g. a hunk deep inside a class
        body whose own 'class AsyncClient(BaseClient):' line is outside
        the hunk's context window as a body line, but git's own header
        hint still reports it (confirmed on the real patch that surfaced
        #63: the class-scoped def wasn't visible as a hunk body line, but
        WAS visible as the header's trailing context).
        """
        match = re.match(r'^@@[^@]*@@\s*class\s+(\w+)\s*[\(:]', header_line)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _extract_defs_from_hunk(
        hunk_lines: List[_HunkLine],
        changed: Set[Tuple[str, Optional[str]]],
        pending_decorator: bool,
        header_scope_name: Optional[str] = None,
        header_scope_class: Optional[str] = None,
    ) -> bool:
        """Extract function/class definitions genuinely changed within a hunk.

        Looks for lines matching:
          def function_name(...)
          async def function_name(...)
          class ClassName(...)
          @decorator(...)

        A def/class is only added to `changed` if it is genuinely changed:
        its own line was added/removed, a changed decorator immediately
        precedes it (including one carried over from a previous hunk via
        `pending_decorator`), or at least one line within its body (up to
        the next sibling def/class at the same or shallower indentation,
        or the end of the hunk) was added/removed. A def/class line that is
        pure unchanged context, with an unchanged body and no changed
        decorator, is NOT added -- it's just visible because of the hunk's
        context window, not because it changed.

        If the hunk body contains NO def/class line at all (the changed
        lines are deep inside a function whose own def line falls outside
        this hunk's context window -- confirmed to happen on real patches,
        not just a theoretical edge case), header_scope_name (git's own
        hint from the '@@ ... @@ def name(...)' trailing text) is used
        instead: the hunk clearly changed SOMETHING, and this is the only
        available signal for what function/class it's inside.

        Each entry in `changed` is (name, enclosing_class_or_None):
        - a 'class X:' line at a shallower indentation than a def,
          appearing earlier in the SAME hunk, is the primary source for a
          def's enclosing class (kg_construction#63).
        - if no such in-hunk class line exists, header_scope_class (git's
          own hint from the '@@ ... @@ class X(...):' trailing context) is
          used instead -- confirmed necessary on the real patch that
          surfaced #63: the hunk was deep enough into AsyncClient's body
          that 'class AsyncClient(BaseClient):' never appeared as a hunk
          BODY line, only as the header's trailing context.
        - if neither is available, the class stays None (a bare name with
          no disambiguating information -- same as before this was added).

        Returns:
            True if a decorator is still pending resolution at the end of
            this hunk (i.e. no def/class line followed it), so the caller
            can carry it into the next hunk of the same file.
        """
        def_pattern = re.compile(r'^\s*(async\s+)?def\s+(\w+)\s*\(')
        class_pattern = re.compile(r'^\s*class\s+(\w+)\s*[\(:]')
        decorator_pattern = re.compile(r'^\s*@\w')

        # Stack of (indent, class_name) for classes seen so far in this
        # hunk, innermost last -- popped whenever a later line dedents to
        # or past that class's own indentation, so a def's enclosing class
        # is always the innermost still-open one at the def's indent.
        class_stack: List[Tuple[int, str]] = []

        saw_def_or_class = False
        for idx, hline in enumerate(hunk_lines):
            def_match = def_pattern.match(hline.content)
            class_match = class_pattern.match(hline.content)

            if not (def_match or class_match):
                if decorator_pattern.match(hline.content):
                    pending_decorator = pending_decorator or hline.marker != ' '
                continue

            # Close out any class whose body this line has dedented out of,
            # BEFORE checking enclosure -- a class line itself is handled
            # after (it may open a new scope, not belong to the old one).
            while class_stack and class_stack[-1][0] >= hline.indent:
                class_stack.pop()

            saw_def_or_class = True
            name = def_match.group(2) if def_match else class_match.group(1)
            own_line_changed = hline.marker != ' '
            decorator_changed = pending_decorator
            body_changed = PatchParser._body_has_change(hunk_lines, idx, hline.indent)

            if own_line_changed or decorator_changed or body_changed:
                if def_match:
                    enclosing_class = (
                        class_stack[-1][1] if class_stack else header_scope_class
                    )
                else:
                    enclosing_class = None
                changed.add((name, enclosing_class))

            if class_match:
                class_stack.append((hline.indent, name))

            pending_decorator = False

        if not saw_def_or_class and header_scope_name is not None:
            changed.add((header_scope_name, header_scope_class))

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
