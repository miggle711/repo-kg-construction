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
