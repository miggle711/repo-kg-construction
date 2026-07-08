"""
kg_validator.py

Post-extraction validation for Knowledge Graphs.

Runs after full KG construction but before subgraph extraction to catch
Phase I extraction bugs early:
  ✓ No orphaned nodes (indicates bad extraction)
  ✓ No self-loops (indicates edge artifact)
  ✓ No cycles in dependency graph (indicates circular imports)
  ✓ Consistent metadata (indicates uniform extraction)
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict

from kg_construction.validation.base import ValidationBase


class KGValidator(ValidationBase):
    """Validate Knowledge Graph structure and metadata consistency."""

    def __init__(self, kg: Dict):
        """
        Args:
            kg: KG dict as returned by RepoKGBuilder.build().
        """
        super().__init__()
        self.kg = kg
        self.nodes = kg.get('nodes', [])
        self.edges = kg.get('edges', [])
        self.repo = kg.get('metadata', {}).get('repo', 'unknown')

        # Build lookup structures
        self.node_ids: Set[str] = {n['id'] for n in self.nodes}
        self.node_types: Dict[str, str] = {n['id']: n['type'] for n in self.nodes}
        self.edges_by_source: Dict[str, List[Dict]] = defaultdict(list)
        self.edges_by_target: Dict[str, List[Dict]] = defaultdict(list)

        for edge in self.edges:
            self.edges_by_source[edge['source']].append(edge)
            self.edges_by_target[edge['target']].append(edge)

    def validate(self) -> Tuple[bool, str]:
        """Run all validations and return (success, report).

        Returns:
            (is_valid, report_string) where is_valid is True if all checks pass.
        """
        self.errors.clear()
        self.warnings.clear()

        self._check_orphaned_nodes()
        self._check_self_loops()
        self._check_cycles()
        self._check_metadata_consistency()

        return self._format_report()

    def _check_orphaned_nodes(self) -> None:
        """Flag nodes with no incoming or outgoing edges (except files/imports)."""
        exclude_types = {'file', 'test_file', 'import'}
        orphaned = []

        for node_id in self.node_ids:
            node_type = self.node_types[node_id]
            if node_type in exclude_types:
                continue

            has_edges = bool(self.edges_by_source.get(node_id) or
                           self.edges_by_target.get(node_id))
            if not has_edges:
                node = next((n for n in self.nodes if n['id'] == node_id), None)
                orphaned.append(f"{node['label']} ({node_type})")

        if orphaned:
            self.errors.append(
                f"Orphaned nodes ({len(orphaned)}): {', '.join(orphaned[:5])}"
                f"{'...' if len(orphaned) > 5 else ''}"
            )

    def _check_self_loops(self) -> None:
        """Flag unexpected self-loops (except recursive calls which are valid)."""
        # Self-loops are OK for 'calls' edges (recursive calls) but suspicious for others
        # Note: Some edge types may legitimately have self-loops in extracted code
        self_loops = [e for e in self.edges if e['source'] == e['target'] and e['relation'] != 'calls']
        if self_loops:
            labels = [self._node_label(e['source']) for e in self_loops[:3]]
            self.warnings.append(
                f"Self-loops ({len(self_loops)}, non-call): {', '.join(labels)}"
                f"{'...' if len(self_loops) > 3 else ''}"
            )

    def _check_cycles(self) -> None:
        """Detect cycles in dependency graph using DFS.

        Only checks edges: calls, accesses, inherits, uses, overrides,
        module_depends_on. Excludes 'contains' (parent-child, not circular),
        'tests' (orthogonal), 'depends_on' and 'imports' (allowed to form
        cycles, resolved at runtime).
        """
        dep_relations = {'calls', 'accesses', 'inherits', 'uses', 'overrides', 'module_depends_on'}

        # Build adjacency list for dependency edges
        dep_graph: Dict[str, List[str]] = defaultdict(list)
        for edge in self.edges:
            if edge['relation'] in dep_relations:
                dep_graph[edge['source']].append(edge['target'])

        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        cycles: List[List[str]] = []

        def dfs(node_id: str, path: List[str]) -> None:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for neighbor in dep_graph.get(node_id, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    cycle = path[path.index(neighbor):] + [neighbor]
                    cycles.append(cycle)

            rec_stack.discard(node_id)

        for node_id in dep_graph:
            if node_id not in visited:
                dfs(node_id, [])

        if cycles:
            cycle_strs = []
            for cycle in cycles[:3]:
                labels = [self._node_label(n) for n in cycle]
                cycle_strs.append(" → ".join(labels))
            self.warnings.append(
                f"Cycles in dependency graph ({len(cycles)} found): "
                f"{'; '.join(cycle_strs)}"
                f"{'...' if len(cycles) > 3 else ''}"
            )

    def _check_metadata_consistency(self) -> None:
        """Verify critical metadata keys are consistent within each node type."""
        by_type: Dict[str, List[Dict]] = defaultdict(list)
        for node in self.nodes:
            by_type[node['type']].append(node)

        # Only check critical fields (filepath is always expected)
        critical_keys = {'filepath'}

        for node_type, nodes in by_type.items():
            if not nodes:
                continue

            inconsistent = []
            for node in nodes:
                meta = node.get('metadata', {})
                missing = critical_keys - set(meta.keys())
                if missing:
                    inconsistent.append((node['label'], missing))

            if inconsistent and len(inconsistent) > len(nodes) * 0.5:
                # Warn only if >50% of nodes are missing critical fields
                labels_missing = [f"{l}: {m}" for l, m in inconsistent[:2]]
                self.warnings.append(
                    f"{node_type} metadata missing critical keys: {', '.join(labels_missing)}"
                )

    def _expected_metadata_keys(self, node_type: str) -> Set[str]:
        """Return expected metadata keys for a given node type."""
        expectations = {
            'function': {'signature', 'filepath'},
            'method': {'signature', 'filepath'},
            'test_function': {'signature', 'filepath'},
            'class': {'filepath'},
            'file': {'filepath'},
            'test_file': {'filepath'},
        }
        return expectations.get(node_type, set())

    def _node_label(self, node_id: str) -> str:
        """Get human-readable label for a node ID."""
        node = next((n for n in self.nodes if n['id'] == node_id), None)
        return node['label'] if node else node_id[:8]

    def _format_report(self) -> Tuple[bool, str]:
        """Format validation report using base class formatter."""
        stats = f"Nodes: {len(self.nodes)} | Edges: {len(self.edges)}"
        return super()._format_report(
            title=f"KG Validation Report: {self.repo}",
            stats_line=stats,
        )
