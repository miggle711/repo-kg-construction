"""
base.py

Abstract base class for all validators in the kg_construction pipeline.

Provides common validation patterns:
- Error/warning collection
- Consistent report formatting
- Abstract validate() method for subclasses

Subclasses implement domain-specific checks (KG, TestContext, LLM output, etc.)
while inheriting consistent error handling and reporting.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List


class ValidationBase(ABC):
    """Abstract base class for all validators."""

    def __init__(self):
        """Initialize validator with empty error/warning lists."""
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @abstractmethod
    def validate(self) -> Tuple[bool, str]:
        """Run validation checks and return (is_valid, report_string).

        Subclasses implement specific validation logic.

        Returns:
            (is_valid, report_string) where:
            - is_valid: True if no blocking errors
            - report_string: Formatted report with errors and warnings
        """
        pass

    def _add_error(self, message: str) -> None:
        """Add a blocking error."""
        self.errors.append(message)

    def _add_warning(self, message: str) -> None:
        """Add a non-blocking warning."""
        self.warnings.append(message)

    def _format_report(
        self,
        title: str,
        stats_line: str = "",
        is_valid: bool = None,
    ) -> Tuple[bool, str]:
        """Format a standard validation report.

        Args:
            title: Report title (e.g., "KG Validation Report")
            stats_line: Single line of stats (e.g., "Nodes: 100 | Edges: 500")
            is_valid: Override is_valid check. If None, check for errors.

        Returns:
            (is_valid, formatted_report_string)
        """
        if is_valid is None:
            is_valid = len(self.errors) == 0

        lines = [f"\n{'='*70}"]
        lines.append(title)
        lines.append(f"{'='*70}")

        if stats_line:
            lines.append(stats_line)
            lines.append("")

        if self.errors:
            lines.append("❌ ERRORS:")
            for err in self.errors:
                lines.append(f"  - {err}")
            lines.append("")

        if self.warnings:
            lines.append("⚠️  WARNINGS:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")
            lines.append("")

        if not self.errors and not self.warnings:
            lines.append("✅ All checks passed")
            lines.append("")

        lines.append(f"{'='*70}\n")

        return is_valid, "\n".join(lines)
