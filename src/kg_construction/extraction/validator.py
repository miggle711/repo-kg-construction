"""
test_context_validator.py

Validate TestContext subgraphs for LLM test generation.

Runs after TestContextExtractor.extract() to ensure the subgraph is:
  ✓ Structurally valid (no orphans, no broken edges, closed graph)
  ✓ Semantically meaningful (seeds connected, test coverage, proper types)
  ✓ LLM-friendly (good density, diverse edges, documented)

Errors block use; warnings flag issues to monitor.
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict

from kg_construction.extraction.context import TestContext
from kg_construction.validation.base import ValidationBase


class TestContextValidator(ValidationBase):
    """Validate TestContext subgraphs for LLM test generation."""

    def __init__(self, context: TestContext):
        """
        Args:
            context: TestContext as returned by TestContextExtractor.extract().
        """
        super().__init__()
        self.context = context
        self.repo = context.repo
        self.base_commit = context.base_commit[:8]

        # Build lookup structures
        all_nodes = context.seeds + context.context_nodes
        self.all_nodes = all_nodes
        self.node_ids: Set[str] = {n['id'] for n in all_nodes}
        self.node_by_id: Dict[str, Dict] = {n['id']: n for n in all_nodes}
        self.node_types: Dict[str, str] = {n['id']: n['type'] for n in all_nodes}

        self.edges = context.edges
        self.edges_by_source: Dict[str, List[Dict]] = defaultdict(list)
        self.edges_by_target: Dict[str, List[Dict]] = defaultdict(list)
        self.edges_by_relation: Dict[str, List[Dict]] = defaultdict(list)

        for edge in self.edges:
            self.edges_by_source[edge['source']].append(edge)
            self.edges_by_target[edge['target']].append(edge)
            self.edges_by_relation[edge['relation']].append(edge)

    def validate(self) -> Tuple[bool, str]:
        """Run all validations and return (is_valid, report_string).

        Returns:
            (is_valid, report_string) where is_valid is True if no errors.
        """
        self.errors.clear()
        self.warnings.clear()

        # --- Must-haves (5) ---
        self._check_no_orphaned_nodes()
        self._check_no_broken_edges()
        self._check_seed_connectivity()
        self._check_closed_subgraph()
        self._check_no_duplicate_edges()

        # --- Should-haves (3) ---
        self._check_test_coverage()
        self._check_seed_types()
        self._check_context_coverage()

        return self._format_report()

    # --- Must-haves ---

    def _check_no_orphaned_nodes(self) -> None:
        """Flag nodes with no incoming or outgoing edges."""
        orphaned = []
        for node_id in self.node_ids:
            has_edges = bool(
                self.edges_by_source.get(node_id) or
                self.edges_by_target.get(node_id)
            )
            if not has_edges:
                node = self.node_by_id[node_id]
                orphaned.append(f"{node['label']} ({node['type']})")

        if orphaned:
            self.errors.append(
                f"Orphaned nodes ({len(orphaned)}): {', '.join(orphaned[:3])}"
                f"{'...' if len(orphaned) > 3 else ''}"
            )

    def _check_no_broken_edges(self) -> None:
        """Flag edges where source or target is not in the subgraph."""
        broken = []
        for edge in self.edges:
            if edge['source'] not in self.node_ids or edge['target'] not in self.node_ids:
                broken.append(f"{edge['source'][:4]}...→{edge['target'][:4]}... ({edge['relation']})")

        if broken:
            self.errors.append(
                f"Broken edges ({len(broken)}): {', '.join(broken[:3])}"
                f"{'...' if len(broken) > 3 else ''}"
            )

    def _check_seed_connectivity(self) -> None:
        """Flag seeds with no outgoing edges (no context)."""
        disconnected = []
        seed_ids = {n['id'] for n in self.context.seeds}

        for seed_id in seed_ids:
            outgoing = self.edges_by_source.get(seed_id, [])
            if not outgoing:
                node = self.node_by_id[seed_id]
                disconnected.append(node['label'])

        if disconnected:
            self.errors.append(
                f"Disconnected seeds ({len(disconnected)}): {', '.join(disconnected)}"
                " — no outgoing edges (no context)"
            )

    def _check_closed_subgraph(self) -> None:
        """Ensure all edge endpoints are in the node set."""
        dangling = set()
        for edge in self.edges:
            if edge['source'] not in self.node_ids:
                dangling.add(f"source {edge['source'][:8]}")
            if edge['target'] not in self.node_ids:
                dangling.add(f"target {edge['target'][:8]}")

        if dangling:
            self.errors.append(
                f"Dangling edges ({len(dangling)}): {', '.join(list(dangling)[:3])}"
            )

    def _check_no_duplicate_edges(self) -> None:
        """Flag duplicate edges (same source, target, relation)."""
        seen = set()
        duplicates = []

        for edge in self.edges:
            key = (edge['source'], edge['target'], edge['relation'])
            if key in seen:
                duplicates.append(f"{edge['relation']}")
            seen.add(key)

        if duplicates:
            self.errors.append(
                f"Duplicate edges ({len(duplicates)}): {', '.join(set(duplicates))}"
            )

    # --- Should-haves ---

    def _check_test_coverage(self) -> None:
        """Warn if code_file exists but test_file is missing."""
        if not self.context.test_nodes:
            self.warnings.append(
                "Test coverage: no test_nodes found — may indicate missing test file"
            )

    def _check_seed_types(self) -> None:
        """Warn if seeds are not functions/methods/classes."""
        bad_seeds = []
        for seed in self.context.seeds:
            if seed['type'] not in ('function', 'method', 'class', 'test_function'):
                bad_seeds.append(f"{seed['label']} ({seed['type']})")

        if bad_seeds:
            self.warnings.append(
                f"Seed types: {len(bad_seeds)} seed(s) are not function/class: "
                f"{', '.join(bad_seeds[:2])}"
            )

    def _check_context_coverage(self) -> None:
        """Warn if context is empty or very small."""
        if len(self.context.context_nodes) == 0:
            self.warnings.append(
                "Context coverage: no context nodes (only seeds) — very narrow view"
            )
        elif len(self.context.context_nodes) < 2:
            self.warnings.append(
                "Context coverage: only 1 context node — may be insufficient context"
            )

    def _format_report(self) -> Tuple[bool, str]:
        """Format validation report using base class formatter."""
        stats = (f"Seeds: {len(self.context.seeds)} | Context: {len(self.context.context_nodes)} "
                 f"| Edges: {len(self.edges)} | Tests: {len(self.context.test_nodes)}")
        return super()._format_report(
            title=f"TestContext Validation Report: {self.repo} @ {self.base_commit}",
            stats_line=stats,
        )
