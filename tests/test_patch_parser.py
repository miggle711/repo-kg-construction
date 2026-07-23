"""Smoke tests for unified diff parsing."""

import pytest
from kg_construction.extraction.patch import PatchParser


class TestPatchParser:
    """Test PatchParser.extract_changed_functions()."""

    def test_extract_single_function_added(self):
        """Extract function added in patch."""
        patch = """--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -100,6 +100,12 @@
     return response

+def new_function():
+    \"\"\"New function.\"\"\"
+    pass
+
 class Session:
     pass
"""
        changed = PatchParser.extract_changed_functions(patch, 'requests/sessions.py')
        assert 'new_function' in changed

    def test_extract_method_modified(self):
        """Extract method that was modified."""
        patch = """--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -455,6 +455,7 @@
 def send(self, method, url, **kwargs):
     response = self._request(url)
+    self.cache[url] = response
     return response
"""
        changed = PatchParser.extract_changed_functions(patch, 'requests/sessions.py')
        assert 'send' in changed

    def test_extract_class_added(self):
        """Extract class added in patch."""
        patch = """--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -500,6 +500,15 @@
     return response

+class NewSession:
+    \"\"\"New session class.\"\"\"
+    def __init__(self):
+        pass
+
+    def send(self):
+        pass
+
 def helper():
     pass
"""
        changed = PatchParser.extract_changed_functions(patch, 'requests/sessions.py')
        assert 'NewSession' in changed

    def test_ignore_changes_in_other_files(self):
        """Don't extract changes from other files."""
        patch = """--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -100,6 +100,7 @@
+def target_function():
+    pass

--- a/requests/adapters.py
+++ b/requests/adapters.py
@@ -50,6 +50,7 @@
+def other_function():
+    pass
"""
        changed = PatchParser.extract_changed_functions(patch, 'requests/sessions.py')
        assert 'target_function' in changed
        assert 'other_function' not in changed

    def test_multi_file_patch_final_hunk(self):
        """Process final hunk when target file is not last in patch."""
        patch = """--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -100,6 +100,7 @@
+def first_change():
+    pass

--- a/requests/adapters.py
+++ b/requests/adapters.py
@@ -50,6 +50,7 @@
+def second_change():
+    pass
"""
        # Extract from first file (not the last in patch)
        changed = PatchParser.extract_changed_functions(patch, 'requests/sessions.py')
        assert 'first_change' in changed
        # Ensure we processed the hunk before switching files
        assert len(changed) == 1

    def test_empty_patch(self):
        """Handle empty patch gracefully."""
        changed = PatchParser.extract_changed_functions('', 'requests/sessions.py')
        assert len(changed) == 0

    def test_patch_no_target_file(self):
        """Return empty set when target file not in patch."""
        patch = """--- a/requests/adapters.py
+++ b/requests/adapters.py
@@ -100,6 +100,7 @@
+def some_function():
+    pass
"""
        changed = PatchParser.extract_changed_functions(patch, 'requests/sessions.py')
        assert len(changed) == 0

    def test_decorator_only_change_same_hunk(self):
        """Detect the function when a changed decorator's def is later context in the same hunk."""
        patch = """--- a/app.py
+++ b/app.py
@@ -10,3 +10,3 @@
-@app.route("/old")
+@app.route("/new")
 def handler():
     pass
"""
        changed = PatchParser.extract_changed_functions(patch, 'app.py')
        assert 'handler' in changed

    def test_decorator_only_change_across_hunk_boundary(self):
        """Detect the function when its def falls in a later hunk than the changed decorator.

        A small diff context window can put a hunk boundary between a
        modified decorator and the otherwise-unchanged function it
        decorates, so the def line never appears in the same hunk as the
        decorator line. The pending decorator must carry across hunks
        within the same file.
        """
        patch = """--- a/app.py
+++ b/app.py
@@ -10,1 +10,1 @@
-@app.route("/old")
+@app.route("/new")
@@ -20,2 +20,2 @@
 def handler():
     pass
"""
        changed = PatchParser.extract_changed_functions(patch, 'app.py')
        assert 'handler' in changed

    def test_pending_decorator_does_not_leak_across_files(self):
        """A decorator with no following def/class in its own file resolves to nothing.

        It must not attach to the first def/class encountered in a
        different file later in the same multi-file patch.
        """
        patch = """--- a/other.py
+++ b/other.py
@@ -5,1 +5,1 @@
+@some_decorator
--- a/app.py
+++ b/app.py
@@ -20,2 +20,2 @@
 def unrelated():
     pass
"""
        changed_other = PatchParser.extract_changed_functions(patch, 'other.py')
        assert changed_other == set()

    def test_wide_context_does_not_sweep_in_unmodified_sibling(self):
        """Regression test for issue #43: a diff hunk wide enough to include
        an entirely unmodified neighboring function's def line must not
        report that function as changed -- only the function whose body (or
        decorator) actually changed.
        """
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,15 +1,15 @@
 def delete(self, url, **kwargs):
     kwargs.setdefault('allow_redirects', True)
     return self.request('DELETE', url, **kwargs)

 def send(self, request, **kwargs):
     kwargs.setdefault('stream', self.stream)
     kwargs.setdefault('verify', self.verify)
     kwargs.setdefault('cert', self.cert)
-    kwargs.setdefault('proxies', self.proxies)
+    kwargs.setdefault('proxies', self.rebuild_proxies(request, self.proxies))

     pass
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py')
        assert changed == {'send'}

    def test_body_change_detection_does_not_leak_into_siblings(self):
        """A changed line inside one function's body must not cause a
        following sibling function (pure context) to be reported as changed.
        """
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,10 +1,10 @@
 def unrelated():
     pass

 def target():
-    return 1
+    return 2

 def another_unrelated():
     pass
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py')
        assert changed == {'target'}

    def test_context_only_def_with_unchanged_body_not_reported(self):
        """A def/class line and its whole body appearing as pure unchanged
        context (no decorator change either) must not be reported --
        it's only visible because of the hunk's context window.
        """
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,8 +1,9 @@
 def untouched():
     return 1

 def changed():
-    return 2
+    return 3
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py')
        assert changed == {'changed'}
        assert 'untouched' not in changed

    def test_falls_back_to_header_scope_name_when_def_line_not_in_hunk(self):
        """kg_construction#60's second-repo check: when the changed lines
        are deep enough inside a long function that its own 'def' line
        falls outside the hunk's context window, the hunk body contains
        no def/class line at all -- confirmed on a real django/django
        patch (slugify()), where an identical semantic change was
        detected or silently missed depending only on whether the def
        line happened to be within context. git's own hunk-header trailing
        text ('@@ ... @@ def name(...)') is the fallback signal for this
        case.
        """
        patch = """--- a/mod.py
+++ b/mod.py
@@ -10,7 +10,7 @@ def long_function(value):
     step_one = value
     step_two = step_one + 1
     step_three = step_two * 2
-    return step_three
+    return step_three + 1
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py')
        assert changed == {'long_function'}

    def test_header_scope_name_ignored_when_def_line_is_present(self):
        """When the hunk body DOES contain a def/class line, the ordinary
        body-based detection is authoritative -- the header hint must not
        add a second, spurious name.
        """
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,5 +1,5 @@ def other_function():
 def other_function():
-    return 1
+    return 2
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py')
        assert changed == {'other_function'}

    def test_header_scope_name_not_used_when_hunk_body_genuinely_unchanged(self):
        """The header-hint fallback only applies when NO def/class line
        appears in the hunk body at all -- if a def/class line IS present
        but its body has no real change (pure context), the existing
        no-genuine-change logic must still win; the header hint must not
        override that by adding the name anyway via a different path.
        """
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,4 +1,4 @@ def untouched():
 def untouched():
     return 1
-# comment removed below, outside the function
+# comment replaced below, outside the function
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py')
        assert 'untouched' not in changed


class TestExtractChangedFunctionsWithScope:
    """kg_construction#63: a changed method name can match more than one
    class' same-named method in the same file (found via a real
    encode/httpx patch to AsyncClient.aclose, which also matched the
    unrelated BoundAsyncStream.aclose). extract_changed_functions_with_scope
    reports the enclosing class alongside each name, when it can be
    determined, so TestContextExtractor can disambiguate instead of
    adding every same-named match as a seed.
    """

    def test_class_line_in_hunk_body_is_tracked_as_enclosing_class(self):
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,6 +1,7 @@
 class Widget:
     def build(self):
         return self.helper()
+        # a comment
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py')
        assert ('build', 'Widget') in changed

    def test_header_class_hint_used_when_class_line_not_in_hunk_body(self):
        """kg_construction#63's actual reproducing shape: the hunk is deep
        enough into a class's body that 'class X:' never appears as a hunk
        BODY line, but git's header still reports it as trailing context.
        """
        patch = """--- a/mod.py
+++ b/mod.py
@@ -50,6 +50,7 @@ class Widget:
     def build(self):
         return self.helper()
+        return None
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py')
        assert ('build', 'Widget') in changed

    def test_no_class_information_available_reports_none(self):
        """A top-level function has no enclosing class at all -- must
        report None, not omit the entry or raise.
        """
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,3 +1,4 @@
 def standalone():
     return 1
+    # comment
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py')
        assert ('standalone', None) in changed

    def test_class_scope_does_not_leak_across_dedent(self):
        """A def AFTER the class body has ended (dedented back to module
        level) must not be attributed to that class.
        """
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,8 +1,9 @@
 class Widget:
     def build(self):
         return 1

 def standalone():
     return 2
+    # comment
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py')
        assert ('standalone', None) in changed
        assert ('standalone', 'Widget') not in changed

    def test_two_classes_same_method_name_get_different_scopes(self):
        """The exact ambiguity #63 was found from: two unrelated classes
        each define a method with the same name, in the same file/hunk.
        Both must be reported, each with its OWN correct enclosing class.
        """
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,10 +1,12 @@
 class Alpha:
     def aclose(self):
         return 1
+        # comment

 class Beta:
     def aclose(self):
         return 2
+        # comment
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py')
        assert ('aclose', 'Alpha') in changed
        assert ('aclose', 'Beta') in changed

    def test_flat_extract_changed_functions_still_returns_bare_names(self):
        """extract_changed_functions (the original, still-used-elsewhere
        method) must keep returning a flat Set[str] -- existing callers
        (resolve_target_function, build_baseline_context) depend on this
        shape and must not need to change. (Both 'Widget' and 'build' are
        expected here: a change nested inside a method's body is, by the
        existing body-change-detection logic, also a change within the
        enclosing class's own body -- unrelated to this fix, unchanged
        from before it.)
        """
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,3 +1,4 @@
 class Widget:
     def build(self):
         return 1
+        # comment
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py')
        assert changed == {'Widget', 'build'}
