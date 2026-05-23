"""
anthropic.py

Anthropic (Claude) based test generator.
"""

import os
from typing import Dict, Optional

from .base import LLMTestGenerator


class AnthropicTestGenerator(LLMTestGenerator):
    """Generates test code using Anthropic (Claude) API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Anthropic client.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.

        Raises:
            ValueError: If no API key is found.
            ImportError: If anthropic package not installed.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set via parameter or ANTHROPIC_API_KEY env var."
            )

        try:
            from anthropic import Anthropic

            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

    def generate(self, hierarchical_json: Dict, model: str = "claude-opus-4-1") -> str:
        """Generate test code using Claude.

        Args:
            hierarchical_json: Dict with 'seed', 'context', 'instructions' sections.
            model: Claude model to use (default: claude-opus-4-1).

        Returns:
            Generated test code as a string.

        Raises:
            ValueError: If API call fails.
        """
        prompt = self._build_prompt(hierarchical_json)

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=2000,
                system=self._system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            return response.content[0].text
        except Exception as e:
            raise ValueError(f"Anthropic API call failed: {e}")
