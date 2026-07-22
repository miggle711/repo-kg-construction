"""
run.py

Generic KG builder + subgraph extractor + validator.
Test generation itself lives outside this repo (see kg-test-generation),
which consumes the hierarchical JSON produced by serialize_context().

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


def _load_or_build(builder, repo, commit):
    """Load an existing KG for (repo, commit) or build+save a new one.

    load() returns None (never raises) on a cache miss -- including a
    mismatched commit or an outdated schema_version, not just a missing
    file -- so a None result always means "build fresh", not "error".
    """
    kg = builder.load(repo, commit)
    if kg is not None:
        print(f"✓ Loaded existing KG")
        return kg

    print(f"Building KG for {repo} @ {commit[:8]}...", end=" ", flush=True)
    kg = builder.build(repo, commit)
    builder.save(repo, kg)
    print(f"✓ Built and saved")
    return kg


def serialize_context(context):
    """Serialize a TestContext into hierarchical JSON for LLM consumption.

    Args:
        context: TestContext object (e.g. returned by extract_and_validate)

    Returns:
        Hierarchical JSON dict ({seed, context, instructions}) suitable for
        passing to an LLM-based test generator.
    """
    serializer = LLMSerializer(repo=context.repo)
    context_dict = {
        'repo': context.repo,
        'base_commit': context.base_commit,
        'seeds': context.seeds,
        'context_nodes': context.context_nodes,
        'edges': context.edges,
        'test_nodes': context.test_nodes,
    }
    return serializer.serialize(context_dict)


def extract_and_validate(instance, depth=2, verbose=True):
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

    Returns:
        (context, report) where:
        - context: TestContext object (can be saved with context.save())
        - report: Validation report string (errors + warnings)

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

    return context, report


def _interactive_mode():
    """Interactive mode: prompt for inputs."""
    print("=" * 70)
    print("KG Builder + Subgraph Extractor + Validator")
    print("=" * 70)

    repo = input("\nRepo (e.g. psf/requests): ").strip()
    commit = input("Commit SHA: ").strip()
    patch_path = input("Path to patch file: ").strip()
    code_file = input("Code file (e.g. requests/sessions.py): ").strip()
    test_file = input("Test file (e.g. tests/test_sessions.py): ").strip()

    patch = Path(patch_path).read_text()

    instance = {
        'repo': repo,
        'base_commit': commit,
        'patch': patch,
        'code_file': code_file,
        'test_file': test_file,
    }

    context, report = extract_and_validate(instance, depth=2, verbose=True)

    # Save subgraph
    repo_slug = repo.replace('/', '_')
    out_path = f"kg_output/{repo_slug}_{commit[:8]}_subgraph.json"
    context.save(out_path)
    print(f"✓ Saved subgraph to {out_path}")


def main():
    """Entry point."""
    _interactive_mode()
