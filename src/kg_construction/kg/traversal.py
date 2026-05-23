"""
traversal.py

Generic graph traversal utilities for knowledge graphs.

Provides BFS traversal with configurable depth, edge filtering, and direction.
Used by subgraph extraction (Phase 3), test execution (Phase 6), and metrics (Phase 7).
"""

from typing import Dict, List, Set, Optional
from collections import deque


class GraphTraversal:
    """Generic BFS traversal on KGs with filtering."""

    def bfs(
        self,
        start_ids: List[str],
        kg_nodes: Dict[str, Dict],
        kg_edges: List[Dict],
        depth: int = 2,
        edge_filter: Optional[Set[str]] = None,
        directions: Set[str] = None,
    ) -> tuple[Set[str], List[Dict]]:
        """Breadth-first search from start nodes up to specified depth.

        Args:
            start_ids: List of starting node IDs (seeds)
            kg_nodes: Dict mapping node_id → node dict
            kg_edges: List of edge dicts with 'source', 'target', 'relation'
            depth: Maximum traversal depth (default 2)
            edge_filter: Set of edge relations to follow
                        (e.g., {'calls', 'inherits', 'contains'})
                        If None, follows all relations
            directions: Set containing 'outgoing' and/or 'incoming'
                       (default both)

        Returns:
            (visited_nodes_set, traversed_edges_list) where:
            - visited_nodes_set: Set of all node IDs visited
            - traversed_edges_list: List of edges traversed (filtered)
        """
        if directions is None:
            directions = {"outgoing", "incoming"}

        # Build edge indices for efficient lookup
        edges_by_source = {}
        edges_by_target = {}
        for edge in kg_edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src:
                if src not in edges_by_source:
                    edges_by_source[src] = []
                edges_by_source[src].append(edge)
            if tgt:
                if tgt not in edges_by_target:
                    edges_by_target[tgt] = []
                edges_by_target[tgt].append(edge)

        visited = set(start_ids)
        traversed_edges = []
        queue = deque([(node_id, 0) for node_id in start_ids])

        while queue:
            current_id, current_depth = queue.popleft()

            if current_depth >= depth:
                continue

            # Traverse outgoing edges (callees, children, etc.)
            if "outgoing" in directions:
                for edge in edges_by_source.get(current_id, []):
                    if edge_filter is None or edge.get("relation") in edge_filter:
                        target_id = edge.get("target")
                        if target_id and target_id in kg_nodes:
                            traversed_edges.append(edge)
                            if target_id not in visited:
                                visited.add(target_id)
                                queue.append((target_id, current_depth + 1))

            # Traverse incoming edges (callers, parents, etc.)
            if "incoming" in directions:
                for edge in edges_by_target.get(current_id, []):
                    if edge_filter is None or edge.get("relation") in edge_filter:
                        source_id = edge.get("source")
                        if source_id and source_id in kg_nodes:
                            traversed_edges.append(edge)
                            if source_id not in visited:
                                visited.add(source_id)
                                queue.append((source_id, current_depth + 1))

        return visited, traversed_edges
