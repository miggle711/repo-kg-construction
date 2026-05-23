"""
groq.py

Groq-based test generator.
"""

import os
from typing import Dict, Optional

from .base import LLMTestGenerator


class GroqTestGenerator(LLMTestGenerator):
    """Generates test code using Groq API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq client.

        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.

        Raises:
            ValueError: If no API key is found.
            ImportError: If groq package not installed.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set via parameter or GROQ_API_KEY env var."
            )

        try:
            from groq import Groq

            self.client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError("groq package not installed. Install with: pip install groq")

    def generate(self, hierarchical_json: Dict, model: str = "mixtral-8x7b-32768") -> str:
        """Generate test code using Groq.

        Args:
            hierarchical_json: Dict with 'seed', 'context', 'instructions' sections.
            model: Groq model to use (default: mixtral-8x7b-32768).

        Returns:
            Generated test code as a string.

        Raises:
            ValueError: If API call fails.
        """
        prompt = self._build_prompt(hierarchical_json)

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": self._system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Groq API call failed: {e}")
