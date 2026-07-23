"""Smoke tests for validators."""

import pytest
from kg_construction.kg.validator import KGValidator
from kg_construction.extraction.validator import TestContextValidator
from kg_construction.extraction.context import TestContext


class TestKGValidator:
    """Test KGValidator for knowledge graphs."""

    @pytest.fixture
    def valid_kg(self):
        """Create a valid minimal KG."""
        return {
            'nodes': [
                {'id': 'n1', 'label': 'func_a', 'type': 'function',
                 'metadata': {'filepath': 'a.py', 'signature': 'def func_a():'}},
                {'id': 'n2', 'label': 'func_b', 'type': 'function',
                 'metadata': {'filepath': 'b.py', 'signature': 'def func_b():'}},
            ],
            'edges': [
                {'source': 'n1', 'target': 'n2', 'relation': 'calls'},
            ],
            'metadata': {'repo': 'test/repo', 'commit': 'abc123'},
        }

    @pytest.fixture
    def kg_with_orphans(self):
        """Create KG with orphaned nodes."""
        return {
            'nodes': [
                {'id': 'n1', 'label': 'func_a', 'type': 'function', 'metadata': {'filepath': 'a.py'}},
                {'id': 'n2', 'label': 'func_b', 'type': 'function', 'metadata': {'filepath': 'b.py'}},
                {'id': 'n3', 'label': 'func_c', 'type': 'function', 'metadata': {'filepath': 'c.py'}},
            ],
            'edges': [
                {'source': 'n1', 'target': 'n2', 'relation': 'calls'},
            ],
            'metadata': {'repo': 'test/repo'},
        }

    def test_validate_valid_kg(self, valid_kg):
        """Validate a valid KG."""
        validator = KGValidator(valid_kg)
        is_valid, report = validator.validate()
        assert is_valid is True
        assert '✅' in report or 'passed' in report.lower()

    def test_validate_kg_with_orphans(self, kg_with_orphans):
        """Detect orphaned nodes."""
        validator = KGValidator(kg_with_orphans)
        is_valid, report = validator.validate()
        assert is_valid is False
        assert 'Orphaned' in report or 'orphan' in report.lower()

    def test_validate_self_loops(self):
        """Detect self-loops (non-recursive calls)."""
        kg = {
            'nodes': [
                {'id': 'n1', 'label': 'func_a', 'type': 'function', 'metadata': {'filepath': 'a.py'}},
            ],
            'edges': [
                {'source': 'n1', 'target': 'n1', 'relation': 'uses'},
            ],
            'metadata': {'repo': 'test/repo'},
        }
        validator = KGValidator(kg)
        is_valid, report = validator.validate()
        # Self-loop with non-'calls' relation should warn
        assert 'Self-loops' in report or 'self-loop' in report.lower()

    def test_report_includes_stats(self, valid_kg):
        """Report includes node/edge statistics."""
        validator = KGValidator(valid_kg)
        is_valid, report = validator.validate()
        assert 'Nodes:' in report or 'nodes' in report.lower()
        assert 'Edges:' in report or 'edges' in report.lower()

    def test_missing_signature_key_is_flagged(self):
        """kg-test-generation#51: signature/source_code were never computed
        for any function/method -- this must be caught, not silently
        passed, even though 'filepath' (the only key checked before this
        fix) was present the whole time #51 was live.
        """
        kg = {
            'nodes': [
                {'id': 'n1', 'label': 'func_a', 'type': 'function', 'metadata': {'filepath': 'a.py'}},
            ],
            'edges': [],
            'metadata': {'repo': 'test/repo'},
        }
        validator = KGValidator(kg)
        is_valid, report = validator.validate()
        assert 'signature' in report

    def test_empty_string_signature_is_flagged_same_as_missing(self):
        """A present-but-empty signature (metadata['signature'] == '') is
        exactly as useless to the LLM as a missing key -- both must be
        flagged, since key-presence alone doesn't mean real content.
        """
        kg = {
            'nodes': [
                {'id': 'n1', 'label': 'func_a', 'type': 'function',
                 'metadata': {'filepath': 'a.py', 'signature': ''}},
            ],
            'edges': [],
            'metadata': {'repo': 'test/repo'},
        }
        validator = KGValidator(kg)
        is_valid, report = validator.validate()
        assert 'signature' in report

    def test_single_affected_node_is_flagged_not_just_majority(self):
        """Previously only warned if >50% of nodes in a type were missing
        a critical key -- a bug affecting a MINORITY of nodes (e.g. one
        function out of ten with a real parsing edge case) would have
        been silently ignored. Any affected node must be reported.
        """
        kg = {
            'nodes': [
                {'id': 'n1', 'label': 'func_a', 'type': 'function',
                 'metadata': {'filepath': 'a.py', 'signature': 'def func_a():'}},
                {'id': 'n2', 'label': 'func_b', 'type': 'function',
                 'metadata': {'filepath': 'b.py', 'signature': 'def func_b():'}},
                {'id': 'n3', 'label': 'func_c', 'type': 'function',
                 'metadata': {'filepath': 'c.py'}},  # the one bad node, 1/3
            ],
            'edges': [],
            'metadata': {'repo': 'test/repo'},
        }
        validator = KGValidator(kg)
        is_valid, report = validator.validate()
        assert 'func_c' in report
        assert '1/3' in report

    def test_file_node_with_path_key_is_not_flagged(self):
        """file/test_file nodes use metadata['path'], not metadata['filepath']
        (a different convention than function/method/class nodes) --
        _expected_metadata_keys previously declared 'filepath' for these
        too, which would have flagged every real file/test_file node in
        every KG the moment this check was strengthened to actually look
        at real expected keys (caught via a real-dataset sweep).
        """
        kg = {
            'nodes': [
                {'id': 'n1', 'label': 'mod.py', 'type': 'file', 'metadata': {'path': 'mod.py'}},
                {'id': 'n2', 'label': 'test_mod.py', 'type': 'test_file', 'metadata': {'path': 'test_mod.py'}},
            ],
            'edges': [],
            'metadata': {'repo': 'test/repo'},
        }
        validator = KGValidator(kg)
        is_valid, report = validator.validate()
        assert 'file metadata missing' not in report
        assert 'test_file metadata missing' not in report

    def test_class_node_without_signature_is_not_flagged(self):
        """Classes aren't expected to have a 'signature' key
        (_expected_metadata_keys only requires 'filepath' for classes) --
        this must not produce a false-positive warning.
        """
        kg = {
            'nodes': [
                {'id': 'n1', 'label': 'MyClass', 'type': 'class', 'metadata': {'filepath': 'a.py'}},
            ],
            'edges': [],
            'metadata': {'repo': 'test/repo'},
        }
        validator = KGValidator(kg)
        is_valid, report = validator.validate()
        assert 'signature' not in report


class TestTestContextValidator:
    """Test TestContextValidator for subgraph validation."""

    @pytest.fixture
    def valid_context(self):
        """Create a valid TestContext."""
        return TestContext(
            seeds=[
                {'id': 's1', 'label': 'send', 'type': 'method', 'metadata': {'filepath': 'session.py'}},
            ],
            context_nodes=[
                {'id': 'c1', 'label': 'request', 'type': 'function', 'metadata': {'filepath': 'session.py'}},
            ],
            edges=[
                {'source': 's1', 'target': 'c1', 'relation': 'calls'},
            ],
            test_nodes=[
                {'id': 't1', 'label': 'test_send', 'type': 'test_function', 'metadata': {'filepath': 'test_session.py'}},
            ],
            repo='test/repo',
            base_commit='abc123def456',
        )

    @pytest.fixture
    def context_with_orphans(self):
        """Create TestContext with orphaned nodes."""
        return TestContext(
            seeds=[
                {'id': 's1', 'label': 'send', 'type': 'method', 'metadata': {'filepath': 'session.py'}},
            ],
            context_nodes=[
                {'id': 'c1', 'label': 'request', 'type': 'function', 'metadata': {'filepath': 'session.py'}},
                {'id': 'c2', 'label': 'orphan', 'type': 'function', 'metadata': {'filepath': 'other.py'}},
            ],
            edges=[
                {'source': 's1', 'target': 'c1', 'relation': 'calls'},
            ],
            test_nodes=[],
            repo='test/repo',
            base_commit='abc123def456',
        )

    def test_validate_valid_context(self, valid_context):
        """Validate a valid TestContext."""
        validator = TestContextValidator(valid_context)
        is_valid, report = validator.validate()
        assert is_valid is True

    def test_validate_context_with_orphans(self, context_with_orphans):
        """Detect orphaned nodes in TestContext."""
        validator = TestContextValidator(context_with_orphans)
        is_valid, report = validator.validate()
        assert is_valid is False
        assert 'Orphaned' in report or 'orphan' in report.lower()

    def test_validate_broken_edges(self):
        """Detect broken edges in TestContext."""
        context = TestContext(
            seeds=[{'id': 's1', 'label': 'send', 'type': 'method', 'metadata': {'filepath': 'session.py'}}],
            context_nodes=[],
            edges=[
                {'source': 's1', 'target': 'nonexistent', 'relation': 'calls'},
            ],
            test_nodes=[],
            repo='test/repo',
            base_commit='abc123def456',
        )
        validator = TestContextValidator(context)
        is_valid, report = validator.validate()
        assert is_valid is False

    def test_validate_disconnected_seeds(self):
        """Detect seeds with no outgoing edges."""
        context = TestContext(
            seeds=[
                {'id': 's1', 'label': 'send', 'type': 'method', 'metadata': {'filepath': 'session.py'}},
            ],
            context_nodes=[
                {'id': 'c1', 'label': 'request', 'type': 'function', 'metadata': {'filepath': 'session.py'}},
            ],
            edges=[],  # No edges
            test_nodes=[],
            repo='test/repo',
            base_commit='abc123def456',
        )
        validator = TestContextValidator(context)
        is_valid, report = validator.validate()
        assert is_valid is False
        assert 'Disconnected' in report or 'disconnected' in report.lower()

    def test_report_includes_context_stats(self, valid_context):
        """Report includes context statistics."""
        validator = TestContextValidator(valid_context)
        is_valid, report = validator.validate()
        assert 'Seeds:' in report or 'seeds' in report.lower()
        assert 'Context:' in report or 'context' in report.lower()
