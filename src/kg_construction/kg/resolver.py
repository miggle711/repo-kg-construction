"""
resolver.py

Edge resolution for knowledge graphs.

After parallel file parsing, edges reference functions/classes by name strings
(e.g., "send"). This module resolves those strings to node IDs using a global
name index, tagging confidence levels (exact/ambiguous/dropped).
"""

from typing import Dict, List, Set, Tuple, Optional


class EdgeResolver:
    """Resolves unresolved call edges using global name index."""

    def resolve(
        self,
        unresolved_edges: List[Dict],
        nodes_by_name: Dict[str, List[str]],
        nodes_by_id: Dict[str, Dict],
        confidence_threshold: str = "exact",
    ) -> List[Dict]:
        """Resolve unresolved edges by matching target names against index.

        Args:
            unresolved_edges: Edges with 'target' as name string (e.g., 'send')
            nodes_by_name: Index mapping names to list of node IDs
                          (e.g., {'send': ['abc123', 'def456']})
            nodes_by_id: Lookup map of node_id → node dict
            confidence_threshold: Minimum confidence to keep edge.
                                 One of: 'exact', 'ambiguous', 'dropped'

        Returns:
            List of resolved edges with 'target' as node ID and 'confidence' tag.
            Edges below threshold are dropped.
        """
        resolved_edges = []

        for edge in unresolved_edges:
            target_name = edge.get("target", "")
            if not target_name:
                continue

            source_id = edge.get("source", "")
            relation = edge.get("relation", "")

            # Only resolve call and inherits edges; others are already resolved
            if relation not in ("calls", "inherits"):
                resolved_edges.append(edge)
                continue

            # Resolve the target name to node ID(s)
            target_id, confidence = self._resolve_target(
                target_name, nodes_by_name, nodes_by_id, source_id
            )

            # Keep edge if confidence meets threshold
            if target_id and self._meets_threshold(confidence, confidence_threshold):
                edge_copy = edge.copy()
                edge_copy["target"] = target_id
                edge_copy["metadata"] = edge.get("metadata", {})
                edge_copy["metadata"]["confidence"] = confidence
                resolved_edges.append(edge_copy)

        return resolved_edges

    def _resolve_target(
        self,
        target_name: str,
        nodes_by_name: Dict[str, List[str]],
        nodes_by_id: Dict[str, Dict],
        source_id: str,
    ) -> Tuple[Optional[str], str]:
        """Resolve a single target name to node ID with confidence tag.

        Args:
            target_name: Function/class name to resolve (e.g., 'send')
            nodes_by_name: Index of name → [node_ids]
            nodes_by_id: Lookup of node_id → node dict
            source_id: Source node ID (for context-aware resolution)

        Returns:
            (resolved_node_id, confidence) tuple where:
            - resolved_node_id: str node ID or None if no match
            - confidence: 'exact' (1 match), 'ambiguous' (2+ matches), or 'dropped'
        """
        candidates = nodes_by_name.get(target_name, [])

        if not candidates:
            return None, "dropped"

        if len(candidates) == 1:
            return candidates[0], "exact"

        # Multiple candidates: ambiguous
        # Try to disambiguate by class context if available
        best_match = self._disambiguate_by_context(
            candidates, nodes_by_id, source_id
        )
        if best_match:
            return best_match, "ambiguous"

        # Keep first candidate, mark as ambiguous
        return candidates[0], "ambiguous"

    def _disambiguate_by_context(
        self,
        candidates: List[str],
        nodes_by_id: Dict[str, Dict],
        source_id: str,
    ) -> Optional[str]:
        """Try to pick the best candidate based on context.

        Heuristic: If source is a method in class A and a candidate is also
        a method in class A, prefer that one. Otherwise return None (no disambiguation).

        Args:
            candidates: List of node IDs with matching name
            nodes_by_id: Lookup of node_id → node dict
            source_id: Source node ID (caller)

        Returns:
            Best match node ID or None if no heuristic applies.
        """
        source_node = nodes_by_id.get(source_id, {})
        source_parent = source_node.get("metadata", {}).get("parent_id")

        if not source_parent:
            return None

        # Check if any candidate has the same parent
        for candidate_id in candidates:
            candidate = nodes_by_id.get(candidate_id, {})
            candidate_parent = candidate.get("metadata", {}).get("parent_id")
            if candidate_parent == source_parent:
                return candidate_id

        return None

    @staticmethod
    def _meets_threshold(confidence: str, threshold: str) -> bool:
        """Check if confidence meets or exceeds threshold.

        Confidence levels (best to worst): exact > ambiguous > dropped
        """
        levels = {"exact": 3, "ambiguous": 2, "dropped": 1}
        return levels.get(confidence, 0) >= levels.get(threshold, 0)
