# Repository Knowledge Graph Construction

Build structural Knowledge Graphs (KGs) from Python repository source code. Given a GitHub repo and a commit SHA, the system clones the repo, parses every `.py` file with the AST module, and emits a queryable JSON graph of nodes and edges representing the code's structure.

Designed as a foundation for test generation using SWE-bench data.

## Quick Start

```bash
pip install -e ".[all]"  # or just `.` for core-only, see Installation below
python3 run.py
```

`run.py` is a CLI that guides you through: repo selection, commit SHA, patch file, and file paths. It extracts a KG subgraph and validates it for LLM test generation.

```python
from kg_construction.kg.builder import RepoKGBuilder
from kg_construction.kg.query import KGQueryEngine
from kg_construction.extraction.context import TestContextExtractor
from kg_construction.extraction.validator import TestContextValidator
import json

# Build from a specific commit
builder = RepoKGBuilder()
kg = builder.build('psf/requests', 'a0df2cbb')
builder.save('psf/requests', kg)

# Load and query
with open('kg_output/kg_psf_requests.json') as f:
    kg = json.load(f)

engine = KGQueryEngine(kg)

# Find a file and explore its contents
files = engine.find_file_by_path('sessions.py')
contents = engine.get_file_contents(files[0]['id'])
print(contents['classes'], contents['functions'])

# Find what calls a function
callers = engine.find_callers(contents['functions'][0]['id'])

# Extract a subgraph for test generation
instance = {
    'repo': 'psf/requests',
    'base_commit': 'a0df2cbb',
    'patch': '...',  # unified diff
    'code_file': 'requests/sessions.py',
    'test_file': 'tests/test_sessions.py',
}

extractor = TestContextExtractor(engine)
context = extractor.extract(instance, depth=2)

# Validate the subgraph
validator = TestContextValidator(context)
is_valid, report = validator.validate()
print(report)

# Visualise a subgraph in the browser
engine.visualize([files[0]['id']], depth=2, output_path='sessions.html')
```

## Package Structure

```text
src/kg_construction/
├── kg/
│   ├── builder.py       # Clone repo, parse AST, emit KG nodes and edges
│   ├── query.py         # In-memory query engine and pyvis visualisation
│   ├── validator.py     # Full KG validation (post-extraction sanity checks)
│   └── repo_manager.py  # Git clone and archive extraction
├── ast/
│   └── helpers.py       # Pure AST-in/data-out utilities (22 helper functions)
├── extraction/
│   ├── context.py       # Subgraph extraction (TestContext, TestContextExtractor)
│   └── validator.py     # Subgraph validation for LLM test generation
└── pipeline.py          # Extract and validate orchestration (extract_and_validate)

run.py                   # CLI shim (thin wrapper around pipeline.py)
tests/
├── unit/                # Unit tests for AST helpers and KG builder
├── integration/         # Integration tests with real KGs
└── e2e/                 # End-to-end pipeline tests
```

## Graph Structure

### Node types

| Type | Represents |
|------|-----------|
| `file` | A `.py` source file |
| `test_file` | A test file (`test_*.py` or `*_test.py`) |
| `class` | A class definition |
| `function` | A top-level function |
| `method` | A method inside a class |
| `test_function` | A `test_*` function or method |
| `import` | An imported module or name |

### Edge types

| Relation | Meaning |
|----------|---------|
| `contains` | File/class contains a class, function, or method |
| `imports` | File imports a module or name |
| `calls` | Function calls another function (confidence: `exact`/`ambiguous`/`qualified`) |
| `accesses` | Function reads an `@property`-decorated attribute (no call syntax; confidence: `qualified`) |
| `inherits` | Class inherits from another class |
| `tests` | Test function targets a specific function |
| `uses` | Class instantiates another class |
| `overrides` | Method overrides a parent class method |
| `depends_on` | Function uses a specific import |
| `module_depends_on` | File depends on another file via imports |

### Node metadata

Every function/method node carries: parameter list with defaults and annotations, return type annotation, decorators, docstring, raised and caught exceptions, branch count, and whether it is async. Test functions additionally store assert patterns. Class nodes include base classes, decorators, docstring, and class-level attributes. File nodes include module constants and `__all__` exports.

## Query Engine

```python
engine = KGQueryEngine(kg)

# Node accessors
engine.get_files()                          # all file/test_file nodes
engine.get_functions()                      # all function/method/test_function nodes

# Structural queries
engine.get_file_contents(file_id)           # {file, classes, functions}
engine.get_class_methods(class_id)          # list of method nodes

# Call graph
engine.find_callers(func_id)                # functions that call this one
engine.find_callees(func_id)                # functions this one calls
engine.find_test_functions_for(func_id)     # test functions covering this function

# Search
engine.find_file_by_path('sessions.py')     # substring match on path
engine.find_function_by_name('send')        # exact label match

# Export
engine.export_subgraph([node_id, ...])      # nodes + 1-hop edges as dict

# Visualise (requires pyvis)
engine.visualize([node_id], depth=2, output_path='graph.html')
```

## Running Tests

```bash
# Install in editable mode first
pip install -e .

# Run unit tests
python3 tests/unit/test_kg_builder.py -v
python3 tests/unit/test_subgraph_validator.py -v

# Run integration tests (requires pre-built KGs in kg_output/)
python3 tests/integration/test_subgraph_validator_integration.py -v

# Run all tests with pytest (if available)
pytest tests/ -v
```

**Unit tests** run entirely on synthetic Python source — no git clone or network access required. 35+ tests covering all AST helpers and edge types end-to-end through the KG builder.

**Integration tests** validate real subgraph extraction and validation with pre-built KGs (skipped gracefully if KGs not present).

## Installation

```bash
# Core install (KG building, extraction, validation — stdlib only)
pip install -e .

# Optional extras, as needed:
pip install -e ".[groq]"      # GroqTestGenerator
pip install -e ".[anthropic]" # AnthropicTestGenerator
pip install -e ".[llm]"       # both LLM backends
pip install -e ".[viz]"       # engine.visualize() (pyvis)
pip install -e ".[datasets]"  # SWE-bench dataset examples
pip install -e ".[all]"       # everything
```

Core functionality requires only Python 3.10+. No other dependencies beyond the standard library.

## See Also

- [SWE-bench](https://github.com/princeton-nlp/SWE-bench) — the benchmark dataset used as input
