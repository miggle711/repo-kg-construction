"""
ast_helpers.py

Pure AST-in, data-out helpers used by repo_kg_builder to extract node
metadata and edge targets from Python source trees. No external I/O,
no multiprocessing, no KG dataclass dependencies — safe to import from
any layer of the builder.

Organised into sections (in file order):
    - Identity & path helpers           (_make_id, _is_test_file)
    - AST unparsing                     (_safe_unparse)
    - Call-site extraction              (_extract_callee_name, _extract_call_receiver,
                                         _extract_property_accesses, _collect_local_types)
    - Function/method metadata          (_get_docstring, _get_decorators, _get_signature,
                                         _get_exceptions, _extract_conditions, _extract_data_flows,
                                         _count_branches, _get_assert_patterns, _get_return_types)
    - Class metadata                    (_get_base_names, _get_class_attributes,
                                         _get_instantiated_classes_in_class)
    - Function-body analysis            (_get_attribute_accesses, _get_used_imports,
                                         _get_instantiated_classes)
    - Test-target inference             (_get_test_target)
    - Aggregators                       (_build_func_metadata, _collect_file_level_info)
"""

import ast
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Identity & path helpers
# ---------------------------------------------------------------------------

def _make_id(text: str) -> str:
    """Generate a deterministic 8-character ID from a qualified name string.

    Uses MD5 truncated to 8 hex chars. Collision probability is negligible
    for the scale of a single repo KG (~10k entities).
    """
    return hashlib.md5(text.encode()).hexdigest()[:8]


def _is_test_file(filepath: str) -> bool:
    """Return True if the file path matches common Python test file conventions."""
    parts = Path(filepath).parts
    name = Path(filepath).name
    return (
        name.startswith('test_') or
        name.endswith('_test.py') or
        'tests' in parts or
        'test' in parts
    )


# ---------------------------------------------------------------------------
# AST unparsing
# ---------------------------------------------------------------------------

def _safe_unparse(node: ast.AST) -> Optional[str]:
    """Unparse an AST node to source string, returning None on failure.

    ast.unparse can fail on malformed or unusual AST nodes (e.g. from
    macro-generated code). Returning None rather than raising keeps the
    parse pipeline running.

    Used to capture:
    - Decorator expressions (e.g. @pytest.mark.parametrize("x", [1,2,3]))
    - Function signatures (parameter annotations and default values)
    - Exception expressions in raise statements (e.g. raise ValueError("bad input"))
    - Base classes in class definitions (e.g. class Foo(Bar, pkg.Base))
    """
    try:
        return ast.unparse(node)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Call-site extraction
# ---------------------------------------------------------------------------

def _extract_callee_name(call: ast.Call) -> Optional[str]:
    """Extract the function callee name from a Call node.

    TODO: This is a best-effort heuristic for simple call patterns. More complex patterns (e.g. dynamic calls, lambdas, calls via variables) are not handled, which may lead to missed edges but avoids false positives from unreliable inference.
    Handles:
        - Simple calls: foo() → 'foo'
        - Attribute calls: obj.method() → 'method'

    Does not handle:
        - Subscript calls: obj['key']() → None
        - Chained calls: foo()() → None
    """
    if isinstance(call.func, ast.Name):
        return call.func.id  # foo() → 'foo'
    if isinstance(call.func, ast.Attribute):
        return call.func.attr  # obj.method() → 'method'
    return None


def _extract_call_receiver(call: ast.Call) -> Optional[str]:
    """Extract the receiver expression of an attribute call as a string.

    For obj.method() returns 'obj'; for json.loads() returns 'json';
    for self.foo.bar() returns 'self.foo'. Returns None for non-attribute
    calls (bare functions) or anything _safe_unparse can't render.

    The receiver lets pass-2 resolution distinguish json.loads from
    pickle.loads, and self.method from any other class's method.
    """
    if isinstance(call.func, ast.Attribute):
        return _safe_unparse(call.func.value)
    return None


def _annotated_param_types(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> Dict[str, str]:
    """Map parameter names to their bare annotated type name, where unambiguous.

    Only direct Name/Attribute annotations are used (e.g. `x: Request` or
    `x: pkg.Request` -> 'Request'). Subscripted annotations like
    `Optional[Request]` or `List[Request]` are skipped: the parameter could
    be None or a container, so the "receiver is an instance of this type"
    assumption that _collect_local_types relies on doesn't hold.
    """
    types: Dict[str, str] = {}
    all_args = (
        func_node.args.posonlyargs
        + func_node.args.args
        + func_node.args.kwonlyargs
    )
    for arg in all_args:
        ann = arg.annotation
        if isinstance(ann, ast.Name):
            types[arg.arg] = ann.id
        elif isinstance(ann, ast.Attribute):
            types[arg.arg] = ann.attr
    return types


def _extract_property_accesses(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> List[Tuple[str, Optional[str]]]:
    """Find genuine (non-call) attribute reads: `obj.attr`, not `obj.attr()`.

    A @property-decorated method is invoked as `obj.attr`, never
    `obj.attr()` -- it never appears as an ast.Call node, so
    _extract_callee_name/_emit_call_edges have zero signal for it. This
    walks Attribute nodes directly and excludes:
      - Attribute nodes that are the .func of an enclosing Call (already
        covered by 'calls' edge extraction -- obj.method() is a call, not
        a property access)
      - Attribute nodes in Store/Del context (assignment targets, e.g.
        `obj.attr = x` -- a write, not a read)
      - Nested function scopes (their accesses belong to that scope)

    Returns a deduplicated list of (attr_name, receiver) pairs, e.g.
    `self.headers` -> ('headers', 'self'), `session.cookies` ->
    ('cookies', 'session'). receiver is None if _safe_unparse fails.
    """
    accesses: List[Tuple[str, Optional[str]]] = []
    seen: Set[Tuple[str, Optional[str]]] = set()
    call_funcs: Set[int] = set()

    def _collect_call_funcs(n: ast.AST):
        for child in ast.walk(n):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                call_funcs.add(id(child.func))

    _collect_call_funcs(func_node)

    def _walk_no_nested(n: ast.AST):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # nested fn -- its accesses belong to that scope
            if (isinstance(child, ast.Attribute)
                    and isinstance(child.ctx, ast.Load)
                    and id(child) not in call_funcs):
                receiver = _safe_unparse(child.value)
                key = (child.attr, receiver)
                if key not in seen:
                    seen.add(key)
                    accesses.append(key)
            _walk_no_nested(child)

    _walk_no_nested(func_node)
    return accesses


def _collect_local_types(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> Dict[str, str]:
    """Map local variable names to inferred class names from constructor calls
    and type-annotated parameters.

    Seeds the map from parameter annotations first (e.g. `def f(self, x: Request)`
    -> {'x': 'Request'}), then walks assignments of the form `x = SomeClass(...)`
    and records {x: SomeClass}, overwriting the annotation-derived entry if the
    same name is reassigned to a constructor call. Heuristics for constructor
    calls: only direct calls (capitalized name or attribute call whose final
    part is capitalized). Nested function bodies are skipped for assignments —
    their locals belong to that scope — but a nested function's own parameters
    are not in scope here either way.

    Used to resolve attribute calls like `x.save()` to `SomeClass.save` when x's
    type is known from a visible constructor call or parameter annotation.
    """
    types: Dict[str, str] = _annotated_param_types(func_node)

    def _walk(n: ast.AST):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # nested fn — its locals belong to that scope
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Call):
                call = child.value
                cls_name: Optional[str] = None
                if isinstance(call.func, ast.Name) and call.func.id and call.func.id[0].isupper():
                    cls_name = call.func.id
                elif isinstance(call.func, ast.Attribute) and call.func.attr and call.func.attr[0].isupper():
                    cls_name = call.func.attr
                if cls_name is not None:
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            types[target.id] = cls_name
            _walk(child)

    _walk(func_node)
    return types


# ---------------------------------------------------------------------------
# Function/method metadata
# ---------------------------------------------------------------------------

def _get_docstring(node: ast.AST) -> Optional[str]:
    """Extract the docstring from a function, class, or module node.

    Returns the string value of the first expression statement if it is
    a string constant, otherwise None.
    """
    body = getattr(node, 'body', [])
    # Docstring is typically the first statement in the body and must be a string constant.
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        val = body[0].value.value
        if isinstance(val, str):
            return val.strip()
    return None


def _get_decorators(node: ast.AST) -> List[str]:
    """Return the list of decorator expressions as unparsed strings.

    Handles plain names (@staticmethod), attribute access (@pytest.mark.skip),
    and decorator calls (@pytest.mark.parametrize("x", [1,2,3])).
    """
    return [
        _safe_unparse(d) or ''
        for d in getattr(node, 'decorator_list', [])
        if _safe_unparse(d)
    ]


def _get_signature(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict:
    """Extract the full parameter signature of a function or method.

    Returns a dict with:
        params: list of {name, annotation?, default?} dicts in declaration order.
            - Positional-only and regular args are listed first.
            - Keyword-only args follow.
            - *args and **kwargs appear last, prefixed with * and **.
        returns: unparsed return annotation string, if present.

    Defaults are right-aligned against the args list per Python semantics
    (i.e. def f(a, b=1) has defaults=[1] aligned to b, not a).
    """
    args = node.args

    # defaults is right-aligned: pad left with None for args that have no default
    all_args = args.posonlyargs + args.args

    # Align defaults to the end of all_args: if there are 2 args and 1 default, we want [None, default] not [default]
    defaults = [None] * (len(all_args) - len(args.defaults)) + list(args.defaults)

    params = []
    for i, arg in enumerate(all_args):
        param: Dict = {'name': arg.arg}
        if arg.annotation:
            param['annotation'] = _safe_unparse(arg.annotation)
        if defaults[i] is not None:
            param['default'] = _safe_unparse(defaults[i])
        params.append(param)

    # kw_defaults is parallel to kwonlyargs, entries are None where there's no default
    for i, arg in enumerate(args.kwonlyargs):
        param = {'name': arg.arg}
        if arg.annotation:
            param['annotation'] = _safe_unparse(arg.annotation)
        kw_default = args.kw_defaults[i] if i < len(args.kw_defaults) else None  # safety check in case of malformed AST where kw_defaults is shorter than kwonlyargs
        if kw_default is not None:
            param['default'] = _safe_unparse(kw_default)
        params.append(param)

    if args.vararg:  # *args
        param = {'name': f"*{args.vararg.arg}"}
        if args.vararg.annotation:
            param['annotation'] = _safe_unparse(args.vararg.annotation)
        params.append(param)

    if args.kwarg:  # **kwargs
        param = {'name': f"**{args.kwarg.arg}"}
        if args.kwarg.annotation:
            param['annotation'] = _safe_unparse(args.kwarg.annotation)
        params.append(param)

    result: Dict = {'params': params}
    if node.returns:
        result['returns'] = _safe_unparse(node.returns)
    return result


def _get_exceptions(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict:
    """Extract exception types raised and caught within a function body.

    Returns:
        raises: deduplicated list of unparsed raise expressions,
                e.g. ['ValueError("bad input")', 'KeyError(key)']
        catches: deduplicated list of unparsed except handler types,
                 e.g. ['KeyError', 'TypeError']

    Walks the entire function body including nested calls, so re-raises
    and chained handlers are all captured.
    """
    raises, catches = [], []
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and child.exc is not None:
            unparsed = _safe_unparse(child.exc)
            if unparsed:
                raises.append(unparsed)
        elif isinstance(child, ast.ExceptHandler) and child.type is not None:
            unparsed = _safe_unparse(child.type)
            if unparsed:
                catches.append(unparsed)
    return {'raises': list(set(raises)), 'catches': list(set(catches))}


def _extract_conditions(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict]:
    """Extract boundary conditions from if, while, and assert statements.

    Returns a list of {type, condition, lineno} dicts representing the conditions
    that guard behavior in the function. Used for test generation to identify
    edge cases and boundary values to test.

    Does NOT descend into nested function scopes to avoid capturing conditions
    from helper functions defined within the target function.

    Examples:
        if x < 0: ... → {type: 'if', condition: 'x < 0', lineno: 5}
        while y > 100: ... → {type: 'while', condition: 'y > 100', lineno: 10}
        assert z != 0 → {type: 'assert', condition: 'z != 0', lineno: 15}
    """
    conditions: List[Dict] = []
    seen: Set[Tuple[str, int]] = set()

    def _walk_no_nested(n: ast.AST) -> None:
        """Recursively extract conditions, stopping at nested function definitions."""
        for child in ast.iter_child_nodes(n):
            # Skip nested function scopes entirely
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            if isinstance(child, ast.If):
                unparsed = _safe_unparse(child.test)
                if unparsed:
                    key = ('if', child.lineno)
                    if key not in seen:
                        seen.add(key)
                        conditions.append({'type': 'if', 'condition': unparsed, 'lineno': child.lineno})

            elif isinstance(child, ast.While):
                unparsed = _safe_unparse(child.test)
                if unparsed:
                    key = ('while', child.lineno)
                    if key not in seen:
                        seen.add(key)
                        conditions.append({'type': 'while', 'condition': unparsed, 'lineno': child.lineno})

            elif isinstance(child, ast.Assert):
                unparsed = _safe_unparse(child.test)
                if unparsed:
                    key = ('assert', child.lineno)
                    if key not in seen:
                        seen.add(key)
                        conditions.append({'type': 'assert', 'condition': unparsed, 'lineno': child.lineno})

            # Recurse into this child (will skip nested functions)
            _walk_no_nested(child)

    _walk_no_nested(node)
    return conditions


def _extract_data_flows(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict:
    """Extract data flow information from a function body.

    Identifies how data flows through the function:
    - Return values flowing to callers
    - Parameter usage patterns
    - Attribute mutations on self

    Returns a dict with keys:
        returns: List of unparsed return value expressions
        mutates_attributes: {attr_name: [list of value expressions assigned to it]}
        parameter_usage: {param_name: [line numbers where it appears]}

    Example:
        def send(self, url, timeout=30):
            if timeout < 0:
                raise ValueError()
            response = self._request(url)
            self.cache[url] = response
            return response

        Returns:
        {
            'returns': ['response'],
            'mutates_attributes': {'cache': ['self._request(url)']},
            'parameter_usage': {'url': [4, 5], 'timeout': [3]}
        }
    """
    flows = {
        'returns': [],
        'mutates_attributes': {},
        'parameter_usage': {},
    }

    # Get parameter names for later lookup
    param_names = set()
    for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
        param_names.add(arg.arg)
    if node.args.vararg:
        param_names.add(node.args.vararg.arg)
    if node.args.kwarg:
        param_names.add(node.args.kwarg.arg)

    # Walk function body (skip nested function defs for mutations, but track param usage in them)
    def _walk_no_nested(n: ast.AST, allow_param_in_nested: bool = False) -> None:
        """Recursively extract data flows.

        Args:
            n: AST node to walk
            allow_param_in_nested: If True, we're inside a nested function and should track param usage
        """
        for child in ast.iter_child_nodes(n):
            # For nested functions, extract param usage but skip mutations/returns
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Recursively walk nested function only for parameter usage (closures)
                _walk_no_nested(child, allow_param_in_nested=True)
                continue

            # Track return values (only in direct scope, not nested functions)
            if isinstance(child, ast.Return) and child.value and not allow_param_in_nested:
                unparsed = _safe_unparse(child.value)
                if unparsed:
                    flows['returns'].append(unparsed)

            # Track attribute mutations (only in direct scope, not nested functions):
            # - Direct: self.x = value
            # - Subscript: self.x[key] = value
            elif isinstance(child, ast.Assign) and not allow_param_in_nested:
                for target in child.targets:
                    attr_name = None

                    # Direct attribute: self.x = value
                    if isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name) and target.value.id == 'self':
                            attr_name = target.attr

                    # Subscript mutation: self.x[key] = value
                    elif isinstance(target, ast.Subscript):
                        if isinstance(target.value, ast.Attribute):
                            if isinstance(target.value.value, ast.Name) and target.value.value.id == 'self':
                                attr_name = target.value.attr

                    if attr_name:
                        if attr_name not in flows['mutates_attributes']:
                            flows['mutates_attributes'][attr_name] = []
                        value_unparsed = _safe_unparse(child.value)
                        if value_unparsed:
                            flows['mutates_attributes'][attr_name].append(value_unparsed)

            # Track parameter usage (Name nodes in Load context)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id in param_names:
                    param = child.id
                    if param not in flows['parameter_usage']:
                        flows['parameter_usage'][param] = []
                    flows['parameter_usage'][param].append(child.lineno)

            # Recurse
            _walk_no_nested(child)

    _walk_no_nested(node)

    # Deduplicate and sort parameter usage lines
    for param in flows['parameter_usage']:
        flows['parameter_usage'][param] = sorted(list(set(flows['parameter_usage'][param])))

    # Deduplicate return values and mutations
    flows['returns'] = list(set(flows['returns']))
    for attr in flows['mutates_attributes']:
        flows['mutates_attributes'][attr] = list(set(flows['mutates_attributes'][attr]))

    return flows


def _count_branches(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
    """Count control flow branches (if/for/while) in this function only.

    Includes all branches in the function body
    """
    count = 0

    def _walk(n):
        nonlocal count
        for child in ast.iter_child_nodes(n):
            if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Count if, for, and while statements as branches. Other control flow constructs (try/except, with) could be added in the future if desired.
            if isinstance(child, (ast.If, ast.For, ast.While)):
                count += 1
            _walk(child)

    _walk(node)
    return count


def _get_assert_patterns(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
    """Extract assertion patterns from a test function.

    Captures two forms:
        - assert statements: `assert x == y` → 'x == y'
        - unittest-style method calls: `self.assertEqual(...)` → full call string

    These patterns are used by the test generator to understand what
    properties existing tests verify, informing generation of new assertions.
    """
    patterns = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            unparsed = _safe_unparse(child.test)
            if unparsed:
                patterns.append(unparsed)
        elif (isinstance(child, ast.Call) and
              isinstance(child.func, ast.Attribute) and
              child.func.attr.startswith('assert')):
            unparsed = _safe_unparse(child)
            if unparsed:
                patterns.append(unparsed)
    return patterns


def _get_return_types(func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
    """Extract the set of types actually returned by a function from return statements.

    Supplements the type annotation (which may be absent) by inspecting
    what the function actually returns. Useful for generating assertions
    when there is no return annotation.

    Examples:
        return None                        → 'None'
        return self                        → 'self'
        return []                          → 'list'
        return (a, b, c)                   → '(a, b, c)'
        return response                    → 'response' (name, resolved by test generator)
        return x > 5                       → 'bool'
        return (x for x in items)          → 'generator'
    """
    return_types: List[str] = []

    # Include explicit return annotation first (most authoritative)
    if func_node.returns is not None:
        unparsed = _safe_unparse(func_node.returns)
        if unparsed:
            return_types.append(unparsed)

    for child in ast.walk(func_node):
        if child is not func_node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(child, ast.Return):
            if child.value is None:
                return_types.append('None')
            elif isinstance(child.value, ast.Constant) and child.value.value is None:
                return_types.append('None')
            elif isinstance(child.value, ast.ListComp):
                return_types.append('list')
            elif isinstance(child.value, ast.List):
                unparsed = _safe_unparse(child.value)
                return_types.append(unparsed or 'list')
            elif isinstance(child.value, ast.DictComp):
                return_types.append('dict')
            elif isinstance(child.value, ast.Dict):
                unparsed = _safe_unparse(child.value)
                return_types.append(unparsed or 'dict')
            elif isinstance(child.value, ast.SetComp):
                return_types.append('set')
            elif isinstance(child.value, ast.Set):
                unparsed = _safe_unparse(child.value)
                return_types.append(unparsed or 'set')
            elif isinstance(child.value, ast.Tuple):
                unparsed = _safe_unparse(child.value)
                return_types.append(unparsed or 'tuple')
            elif isinstance(child.value, ast.Constant):
                return_types.append(type(child.value.value).__name__)
            elif isinstance(child.value, ast.Compare):
                return_types.append('bool')
            elif isinstance(child.value, ast.BoolOp):
                return_types.append('bool')
            elif isinstance(child.value, ast.UnaryOp) and isinstance(child.value.op, ast.Not):
                return_types.append('bool')
            elif isinstance(child.value, ast.GeneratorExp):
                return_types.append('generator')
            elif isinstance(child.value, ast.Lambda):
                return_types.append('function')
            elif isinstance(child.value, (ast.BinOp, ast.UnaryOp)):
                unparsed = _safe_unparse(child.value)
                return_types.append(unparsed or 'numeric')
            else:
                unparsed = _safe_unparse(child.value)
                if unparsed:
                    return_types.append(unparsed)
    return list(dict.fromkeys(return_types))


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

def _get_base_names(node: ast.ClassDef) -> List[str]:
    """Return the unparsed base class expressions for a class definition.

    e.g. class Foo(Bar, Mixin) → ['Bar', 'Mixin']
         class Foo(pkg.Base)   → ['pkg.Base']
    """
    return [_safe_unparse(base) for base in node.bases if _safe_unparse(base)]


def _get_class_attributes(node: ast.ClassDef) -> List[str]:
    """Extract instance variable names assigned in __init__.

    Covers both plain assignment (self.x = ...) and annotated assignment
    (self.x: int = ...). Returns names in assignment order, deduplicated.

    Only __init__ is scanned — attributes set in other methods are not
    included, as they may not always exist on the instance.
    """
    attrs = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == '__init__':
            for stmt in ast.walk(child):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (isinstance(target, ast.Attribute) and
                                isinstance(target.value, ast.Name) and
                                target.value.id == 'self'):
                            attrs.append(target.attr)
                elif (isinstance(stmt, ast.AnnAssign) and
                      isinstance(stmt.target, ast.Attribute) and
                      isinstance(stmt.target.value, ast.Name) and
                      stmt.target.value.id == 'self'):
                    attrs.append(stmt.target.attr)
    # dict.fromkeys preserves insertion order while deduplicating
    return list(dict.fromkeys(attrs))


def _get_instantiated_classes_in_class(class_node: ast.ClassDef) -> List[str]:
    """Collect unique class names instantiated across all methods of a class.

    Aggregates _get_instantiated_classes over every method body. Used to emit
    class→uses→class edges at the class level: if any method of class A
    instantiates class B, A is considered a direct dependency of B.

    Args:
        class_node: The ClassDef AST node to inspect.

    Returns:
        Deduplicated list of instantiated class name strings (PEP 8 uppercase
        convention). Names are unresolved — second-pass resolution maps them
        to class node IDs via class_label_to_ids.
    """
    seen: List[str] = []
    for child in ast.iter_child_nodes(class_node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for name in _get_instantiated_classes(child):
                if name not in seen:
                    seen.append(name)
    return seen


# ---------------------------------------------------------------------------
# Function-body analysis
# ---------------------------------------------------------------------------

def _get_attribute_accesses(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    class_name: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Extract self.x reads and self.x writes from a method body.

    Reads are any `self.attr` access not on the left side of an assignment.
    Writes are `self.attr = ...` assignments (both plain and annotated).
    Excludes nested function scopes to avoid false positives from closures.

    Args:
        func_node: The method AST node.
        class_name: Used only for scoping — not currently used but kept for
                    future qualified-attribute support.

    Returns:
        (reads, writes): deduplicated lists of attribute name strings.
    """
    reads: List[str] = []
    writes: List[str] = []
    written: Set[str] = set()

    def _walk_no_nested(node: ast.AST):
        """Walk AST nodes, stopping descent into nested function scopes."""
        yield node
        for child in ast.iter_child_nodes(node):
            if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            yield from _walk_no_nested(child)

    for child in _walk_no_nested(func_node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if (isinstance(target, ast.Attribute) and
                        isinstance(target.value, ast.Name) and
                        target.value.id == 'self'):
                    writes.append(target.attr)
                    written.add(target.attr)
                elif (isinstance(target, ast.Subscript) and
                      isinstance(target.value, ast.Attribute) and
                      isinstance(target.value.value, ast.Name) and
                      target.value.value.id == 'self'):
                    attr = target.value.attr
                    writes.append(attr)
                    written.add(attr)

        elif isinstance(child, ast.AnnAssign):
            if (isinstance(child.target, ast.Attribute) and
                    isinstance(child.target.value, ast.Name) and
                    child.target.value.id == 'self'):
                writes.append(child.target.attr)
                written.add(child.target.attr)

        elif isinstance(child, ast.Attribute):
            if (isinstance(child.value, ast.Name) and
                    child.value.id == 'self' and
                    child.attr not in written):
                reads.append(child.attr)

    return list(dict.fromkeys(reads)), list(dict.fromkeys(writes))


def _get_used_imports(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    import_map: Dict[str, str],
) -> List[str]:
    """Find which imported names are actually referenced inside a function.

    Walks the function body looking for Name nodes whose id appears in the
    file-level import_map. Returns the fully-qualified import paths for
    those names, which tells the test generator exactly what to mock.

    Args:
        func_node: The function AST node.
        import_map: {local_name: qualified_name} from _collect_file_level_info.

    Returns:
        List of fully-qualified import strings used in this function,
        e.g. ['requests.auth.HTTPBasicAuth', 'os.path'].
    """
    used: List[str] = []
    for child in ast.walk(func_node):
        if child is not func_node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(child, ast.Name) and child.id in import_map:
            qualified = import_map[child.id]
            if qualified not in used:
                used.append(qualified)
    return used


def _get_instantiated_classes(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> List[str]:
    """Find class names instantiated inside a function (e.g. Foo(...) calls).

    Detects patterns like `x = SomeClass(...)` by finding Call nodes where
    the callee is a Name starting with an uppercase letter (PEP 8 convention).
    Used to build `uses` edges from a class to its dependencies.

    Returns:
        List of class name strings instantiated in this function.
    """
    classes: List[str] = []
    for child in ast.walk(func_node):
        if child is not func_node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id[0].isupper():
                if child.func.id not in classes:
                    classes.append(child.func.id)
            elif isinstance(child.func, ast.Attribute) and child.func.attr[0].isupper():
                if child.func.attr not in classes:
                    classes.append(child.func.attr)
    return classes


# ---------------------------------------------------------------------------
# Test-target inference
# ---------------------------------------------------------------------------

def _get_test_target(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    label_to_names: Dict[str, str],
) -> Optional[str]:
    """Infer the function under test from a test function's name and body.

    Two strategies:
    1. Name convention: test_send → looks for 'send' in label_to_names
    2. Body inspection: the first non-assert, non-setup Call in the body
       whose callee matches a known function name

    Args:
        func_node: The test function AST node.
        label_to_names: Set of known function/method labels in the repo.
                        Passed in as a set for O(1) lookup.

    Returns:
        The inferred target function name, or None if not determinable.
    """
    # Strategy 1: strip 'test_' prefix and check if it matches a known function
    name = func_node.name
    if name.startswith('test_'):
        candidate = name[5:]  # strip 'test_'
        if candidate in label_to_names:
            return candidate
        # Handle test_send_request → try 'send', 'send_request'
        parts = candidate.split('_')
        for i in range(len(parts), 0, -1):
            joined = '_'.join(parts[:i])
            if joined in label_to_names:
                return joined

    # Strategy 2: first non-trivial call in body
    for child in ast.walk(func_node):
        if isinstance(child, ast.Call):
            callee = _extract_callee_name(child)
            if callee and callee in label_to_names and not callee.startswith('assert'):
                return callee

    return None


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------

def _get_annotation_type_names(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
    """Extract bare type names from all parameter annotations and the return annotation.

    Walks annotation AST nodes to find Name and Attribute nodes, extracting just the
    final component (e.g., pkg.MyClass → 'MyClass', Optional[MyClass] → 'Optional', 'MyClass').
    Used to identify external type dependencies from type hints.
    """
    names: List[str] = []

    def _collect(ann: ast.AST):
        if isinstance(ann, ast.Name):
            names.append(ann.id)
        elif isinstance(ann, ast.Attribute):
            names.append(ann.attr)
        for child in ast.iter_child_nodes(ann):
            _collect(child)

    for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
        if arg.annotation:
            _collect(arg.annotation)
    if node.returns:
        _collect(node.returns)

    return list(dict.fromkeys(names))


def _build_func_metadata(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    rel_path: str,
    repo: str,
    parent_class: Optional[str] = None,
    import_map: Optional[Dict[str, str]] = None,
) -> Dict:
    """Build the full metadata dict for a function or method node.

    Centralises metadata construction to avoid duplication between the
    top-level function and class method branches of _parse_file.

    Args:
        func_node: The AST function node.
        rel_path: Relative file path within the repo.
        repo: Repository name (e.g. 'psf/requests').
        parent_class: Class name if this is a method, else None.
        import_map: {local_name: fully_qualified_name} dict for resolving
                    external type dependencies. Defaults to {}.

    Returns:
        Dict with keys: filepath, repo, lineno, params, returns, decorators,
        docstring, raises, catches, is_async, branches, assert_patterns,
        external_deps, side_effects, and optionally 'class' if parent_class
        is provided.
    """
    sig = _get_signature(func_node)
    exc = _get_exceptions(func_node)
    _import_map = import_map or {}

    # external_deps: annotation types + instantiated classes that appear in import_map
    annotation_types = _get_annotation_type_names(func_node)
    instantiated = _get_instantiated_classes(func_node)
    all_type_names = list(dict.fromkeys(annotation_types + instantiated))
    external_deps = [
        _import_map[name] for name in all_type_names if name in _import_map
    ]

    # side_effects: attribute names written via self.*
    _, writes = _get_attribute_accesses(func_node, parent_class)

    # data_flows: return values, mutations, parameter usage
    data_flows = _extract_data_flows(func_node)

    meta: Dict = {
        'filepath': rel_path,
        'repo': repo,
        'lineno': func_node.lineno,
        'params': sig['params'],
        'returns': sig.get('returns'),
        'decorators': _get_decorators(func_node),
        'docstring': _get_docstring(func_node),
        'raises': exc['raises'],
        'catches': exc['catches'],
        'is_async': isinstance(func_node, ast.AsyncFunctionDef),
        'branches': _count_branches(func_node),
        'conditions': _extract_conditions(func_node),
        'data_flows': data_flows,
        # Populated for anything in a test file — covers test functions and
        # their helper classes (e.g. DomDocument.get_unique_child in pytest).
        # Production code is skipped to avoid noise from invariant asserts.
        'assert_patterns': _get_assert_patterns(func_node) if _is_test_file(rel_path) else [],
        'external_deps': external_deps,
        'side_effects': writes,
    }
    if parent_class is not None:
        meta['class'] = parent_class
    return meta


def _collect_file_level_info(tree: ast.Module) -> Tuple[Dict[str, str], List[str], Dict[str, str]]:
    """Collect imports, __all__ exports, and module-level constants in one pass.

    Combining these into a single traversal avoids walking the module tree
    three separate times.

    Returns:
        import_map: {local_name: fully_qualified_name} for all imports.
            Used to annotate call edges with the source module when the
            callee was imported (e.g. 'HTTPBasicAuth' → 'requests.auth.HTTPBasicAuth').
        exports: List of names in __all__, if defined.
        constants: {NAME: value_string} for UPPER_CASE module-level assignments.
    """
    import_map: Dict[str, str] = {}
    exports: List[str] = []
    constants: Dict[str, str] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # `import os.path` → local name is 'os', full name is 'os.path'
                local = alias.asname or alias.name.split('.')[0]
                import_map[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            for alias in node.names:
                local = alias.asname or alias.name
                import_map[local] = f"{mod}.{alias.name}" if mod else alias.name
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == '__all__' and isinstance(node.value, (ast.List, ast.Tuple)):
                        exports = [
                            elt.value for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
                    elif target.id.isupper():
                        unparsed = _safe_unparse(node.value)
                        if unparsed:
                            constants[target.id] = unparsed
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            # Annotated constant: MAX_SIZE: int = 100
            if node.target.id.isupper() and node.value:
                unparsed = _safe_unparse(node.value)
                if unparsed:
                    constants[node.target.id] = unparsed

    return import_map, exports, constants
