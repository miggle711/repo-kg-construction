"""
pipeline_config.py

Pipeline configuration and orchestration builder.

Provides PipelineBuilder for flexible phase execution with configurable parameters.
Enables selective phase execution, parameter tuning, and batch processing with
consistent configuration across runs.
"""

from typing import Set, Optional, Dict, Any
from dataclasses import dataclass

from kg_construction.kg.builder import RepoKGBuilder
from kg_construction.kg.query import KGQueryEngine
from kg_construction.kg.validator import KGValidator
from kg_construction.extraction.context import TestContextExtractor
from kg_construction.extraction.validator import TestContextValidator
from kg_construction.llm.llm_serializer import LLMSerializer
from kg_construction.llm.groq import GroqTestGenerator
from kg_construction.llm.anthropic import AnthropicTestGenerator


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    context: Optional[Any] = None
    report: Optional[str] = None
    generated_tests: Optional[str] = None
    is_valid: bool = False


class PipelineBuilder:
    """Flexible builder for kg_construction pipeline execution.

    Enables selective phase execution, parameter configuration, and consistent
    pipeline orchestration across different use cases.

    Example:
        result = (
            PipelineBuilder()
            .enable_phases(1, 2, 3, 4, 5)
            .set_depth(2)
            .set_llm("groq")
            .run(instance)
        )
    """

    def __init__(self):
        """Initialize pipeline builder with defaults."""
        self.phases: Set[int] = {1, 2, 3, 4, 5}
        self.depth: int = 2
        self.llm_provider: Optional[str] = None
        self.llm_api_key: Optional[str] = None
        self.verbose: bool = True
        self.edge_filter: Optional[Set[str]] = None
        self.validator_rules: Dict[str, Any] = {}

    def enable_phases(self, *phases: int) -> "PipelineBuilder":
        """Enable specific phases (1-5).

        Args:
            *phases: Phase numbers to execute (1=KG, 2=patch parsing, 3=extraction,
                    4=validation, 5=LLM generation)

        Returns:
            Self for method chaining.
        """
        self.phases = set(phases)
        return self

    def set_depth(self, depth: int) -> "PipelineBuilder":
        """Set BFS traversal depth for subgraph extraction.

        Args:
            depth: Maximum traversal depth (default 2)

        Returns:
            Self for method chaining.
        """
        self.depth = depth
        return self

    def set_edge_filter(self, edge_filter: Set[str]) -> "PipelineBuilder":
        """Set edge relations to include in BFS traversal.

        Args:
            edge_filter: Set of edge relation types to follow
                        (e.g., {'calls', 'inherits', 'tests'})

        Returns:
            Self for method chaining.
        """
        self.edge_filter = edge_filter
        return self

    def set_llm(self, provider: str, api_key: Optional[str] = None) -> "PipelineBuilder":
        """Configure LLM provider for test generation.

        Args:
            provider: LLM provider ("groq" or "anthropic")
            api_key: Optional API key (if not set, reads from env var)

        Returns:
            Self for method chaining.
        """
        if provider not in ("groq", "anthropic"):
            raise ValueError(f"Unknown LLM provider: {provider}")
        self.llm_provider = provider
        self.llm_api_key = api_key
        return self

    def set_verbose(self, verbose: bool) -> "PipelineBuilder":
        """Set verbosity of output.

        Args:
            verbose: Whether to print progress messages

        Returns:
            Self for method chaining.
        """
        self.verbose = verbose
        return self

    def run(self, instance: Dict) -> PipelineResult:
        """Execute the pipeline with configured phases.

        Args:
            instance: Dataset instance dict with keys:
                - repo: Repository name (e.g. 'psf/requests')
                - base_commit: Commit SHA
                - patch: Unified diff string
                - code_file: Relative path to code file
                - test_file: Relative path to test file

        Returns:
            PipelineResult with context, report, and optional generated_tests.

        Raises:
            ValueError: If required phases are missing for downstream phases.
        """
        result = PipelineResult()
        repo = instance['repo']
        commit = instance['base_commit']

        # Phase 1: KG Construction
        kg = None
        if 1 in self.phases:
            if self.verbose:
                print()
            builder = RepoKGBuilder()
            try:
                kg = builder.load(repo)
                if self.verbose:
                    print(f"✓ Loaded existing KG")
            except FileNotFoundError:
                if self.verbose:
                    print(f"Building KG for {repo} @ {commit[:8]}...", end=" ", flush=True)
                kg = builder.build(repo, commit)
                builder.save(repo, kg)
                if self.verbose:
                    print(f"✓ Built and saved")

            if self.verbose:
                print("Validating full KG...", end=" ", flush=True)
            kg_validator = KGValidator(kg)
            kg_valid, kg_report = kg_validator.validate()
            if self.verbose:
                if kg_valid:
                    print("✓")
                else:
                    print("\n" + kg_report)

        # Phase 2-4: Extraction and Validation
        context = None
        if 3 in self.phases or 4 in self.phases:
            if kg is None:
                raise ValueError("Phase 3 requires Phase 1 (KG Construction)")

            # Extract subgraph
            if self.verbose:
                print("Extracting subgraph...", end=" ", flush=True)
            engine = KGQueryEngine(kg)
            extractor = TestContextExtractor(engine)
            context = extractor.extract(instance, depth=self.depth, edge_filter=self.edge_filter)
            if self.verbose:
                print(f"✓ {len(context.seeds)} seeds, {len(context.context_nodes)} context nodes")

            # Validate subgraph
            if self.verbose:
                print("Validating subgraph...")
            validator = TestContextValidator(context)
            result.is_valid, result.report = validator.validate()
            if self.verbose:
                print(result.report)

        # Phase 5: LLM Test Generation
        if 5 in self.phases:
            if context is None:
                raise ValueError("Phase 5 requires Phase 3 (Subgraph Extraction)")
            if not result.is_valid:
                if self.verbose:
                    print("⊘ Phase 5 skipped: subgraph validation failed")
                return result

            if self.verbose:
                print("Phase 5: Generating tests...", end=" ", flush=True)

            try:
                serializer = LLMSerializer(repo=repo)
                context_dict = {
                    'repo': context.repo,
                    'base_commit': context.base_commit,
                    'seeds': context.seeds,
                    'context_nodes': context.context_nodes,
                    'edges': context.edges,
                    'test_nodes': context.test_nodes,
                }
                hierarchical_json = serializer.serialize(context_dict)

                # Initialize generator based on provider
                if self.llm_provider == "anthropic":
                    generator = AnthropicTestGenerator(self.llm_api_key)
                else:  # default to groq
                    generator = GroqTestGenerator(self.llm_api_key)

                result.generated_tests = generator.generate(hierarchical_json)
                if self.verbose:
                    print("✓")
            except Exception as e:
                error_msg = str(e)
                if self.verbose:
                    print(f"\n✗ Test generation failed: {error_msg}")
                if result.report:
                    result.report += f"\n\nPhase 5 (Test Generation): {error_msg}"

        result.context = context
        return result
