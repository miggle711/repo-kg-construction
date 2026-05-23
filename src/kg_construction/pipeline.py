"""
run.py

Generic KG builder + subgraph extractor + validator.

Supports multiple input modes:
  1. Interactive (default): prompts for repo, commit, patch file, code_file, test_file
  2. Programmatic: import extract_and_validate() and call with instance dict
  3. Any data source: as long as you can build an instance dict, you can use this

Usage:
  # Interactive mode
  python3 run.py

  # Programmatic mode (e.g., from a dataset)
  from run import extract_and_validate
  instance = {
      'repo': 'psf/requests',
      'base_commit': 'a0df2c12...',
      'patch': unified_diff_string,
      'code_file': 'requests/sessions.py',
      'test_file': 'tests/test_sessions.py',
  }
  context, report = extract_and_validate(instance, depth=2)
  context.save('output.json')
"""

import sys
import json
from pathlib import Path
from kg_construction.kg.builder import RepoKGBuilder
from kg_construction.kg.query import KGQueryEngine
from kg_construction.extraction.context import TestContextExtractor
from kg_construction.extraction.validator import TestContextValidator
from kg_construction.kg.validator import KGValidator
from kg_construction.llm.llm_serializer import LLMSerializer
from kg_construction.llm.groq import GroqTestGenerator


def _load_or_build(builder, repo, commit):
    """Load existing KG or build+save a new one."""
    try:
        kg = builder.load(repo)
        print(f"✓ Loaded existing KG")
        return kg
    except FileNotFoundError:
        print(f"Building KG for {repo} @ {commit[:8]}...", end=" ", flush=True)
        kg = builder.build(repo, commit)
        builder.save(repo, kg)
        print(f"✓ Built and saved")
        return kg


def extract_and_validate(instance, depth=2, verbose=True, generate_tests=False):
    """Extract and validate a subgraph from an instance dict.

    This is the core function that works with any patch source (dataset,
    file, stdin, etc.) as long as the instance dict is populated.

    Args:
        instance: Dict with keys:
            - repo: Repository name (e.g. 'psf/requests')
            - base_commit: Commit SHA
            - patch: Unified diff string
            - code_file: Relative path to code file
            - test_file: Relative path to test file
        depth: BFS depth for subgraph extraction (default 2)
        verbose: Print progress messages (default True)
        generate_tests: If True, generate tests using Groq API (Phase 5)

    Returns:
        If generate_tests=False:
            (context, report) where:
            - context: TestContext object (can be saved with context.save())
            - report: Validation report string (errors + warnings)
        If generate_tests=True:
            (context, report, generated_tests) where:
            - generated_tests: Generated test code as string

    Raises:
        ValueError: If code_file not found in KG or other extraction errors
    """
    repo = instance['repo']
    commit = instance['base_commit']

    # Build or load KG
    if verbose:
        print()
    builder = RepoKGBuilder()
    kg = _load_or_build(builder, repo, commit)

    # Validate full KG structure
    if verbose:
        print("Validating full KG...", end=" ", flush=True)
    kg_validator = KGValidator(kg)
    kg_valid, kg_report = kg_validator.validate()
    if verbose:
        if kg_valid:
            print("✓")
        else:
            print("\n" + kg_report)

    # Extract subgraph
    if verbose:
        print("Extracting subgraph...", end=" ", flush=True)
    engine = KGQueryEngine(kg)

    extractor = TestContextExtractor(engine)
    context = extractor.extract(instance, depth=depth)
    if verbose:
        print(f"✓ {len(context.seeds)} seeds, {len(context.context_nodes)} context nodes")

    # Validate
    if verbose:
        print("Validating subgraph...")
    validator = TestContextValidator(context)
    is_valid, report = validator.validate()
    if verbose:
        print(report)

    # Phase 5: Generate tests if requested
    generated_tests = None
    if generate_tests:
        if verbose:
            print("Phase 5: Generating tests with Groq...", end=" ", flush=True)
        try:
            # Serialize to hierarchical JSON
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

            # Generate tests
            generator = GroqTestGenerator()
            generated_tests = generator.generate(hierarchical_json)
            if verbose:
                print("✓")
        except Exception as e:
            if verbose:
                print(f"\n✗ Test generation failed: {e}")
            generated_tests = None

    if generate_tests:
        return context, report, generated_tests
    else:
        return context, report


def _interactive_mode():
    """Interactive mode: prompt for inputs."""
    print("=" * 70)
    print("KG Builder + Subgraph Extractor + Validator + Test Generator (Phase 5)")
    print("=" * 70)

    repo = input("\nRepo (e.g. psf/requests): ").strip()
    commit = input("Commit SHA: ").strip()
    patch_path = input("Path to patch file: ").strip()
    code_file = input("Code file (e.g. requests/sessions.py): ").strip()
    test_file = input("Test file (e.g. tests/test_sessions.py): ").strip()
    generate = input("Generate tests with Groq? (y/n, default: n): ").strip().lower() == 'y'

    patch = Path(patch_path).read_text()

    instance = {
        'repo': repo,
        'base_commit': commit,
        'patch': patch,
        'code_file': code_file,
        'test_file': test_file,
    }

    result = extract_and_validate(instance, depth=2, verbose=True, generate_tests=generate)

    if generate:
        context, report, generated_tests = result
    else:
        context, report = result
        generated_tests = None

    # Save subgraph
    repo_slug = repo.replace('/', '_')
    out_path = f"kg_output/{repo_slug}_{commit[:8]}_subgraph.json"
    context.save(out_path)
    print(f"✓ Saved subgraph to {out_path}")

    # Save generated tests if available
    if generated_tests:
        test_out_path = f"kg_output/{repo_slug}_{commit[:8]}_generated_tests.py"
        Path(test_out_path).write_text(generated_tests)
        print(f"✓ Saved generated tests to {test_out_path}")


def main():
    """Entry point."""
    _interactive_mode()
