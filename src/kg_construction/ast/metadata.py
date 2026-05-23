"""
metadata.py

Metadata extraction from Python AST nodes.

Groups extraction logic for different node types (functions, classes, files)
into dedicated Extractor classes. Replaces flat helper functions from helpers.py
with organized, testable extractors.

Used during parallel AST parsing (Phase 1) to build node metadata.
"""

import ast
from typing import Dict, Set, List, Any, Optional
from kg_construction.ast.helpers import (
    _get_signature,
    _get_docstring,
    _get_decorators,
    _get_exceptions,
    _count_branches,
    _get_assert_patterns,
    _get_return_types,
    _get_base_names,
    _get_class_attributes,
    _get_instantiated_classes_in_class,
    _get_instantiated_classes,
    _collect_local_types,
)


class FunctionMetadataExtractor:
    """Extract metadata from function/method definitions."""

    @staticmethod
    def extract(func_node: ast.FunctionDef, is_test: bool = False) -> Dict[str, Any]:
        """Extract metadata from a function node.

        Args:
            func_node: ast.FunctionDef or ast.AsyncFunctionDef
            is_test: Whether this is a test function

        Returns:
            Dict with keys: signature, docstring, decorators, exceptions,
            return_types, branch_count, assert_patterns (if test)
        """
        metadata = {
            'signature': _get_signature(func_node),
            'docstring': _get_docstring(func_node),
            'decorators': _get_decorators(func_node),
            'exceptions': _get_exceptions(func_node),
            'return_types': _get_return_types(func_node),
            'branch_count': _count_branches(func_node),
        }

        if is_test:
            metadata['assert_patterns'] = _get_assert_patterns(func_node)

        return metadata


class ClassMetadataExtractor:
    """Extract metadata from class definitions."""

    @staticmethod
    def extract(class_node: ast.ClassDef) -> Dict[str, Any]:
        """Extract metadata from a class node.

        Args:
            class_node: ast.ClassDef

        Returns:
            Dict with keys: docstring, decorators, base_classes, attributes,
            instantiated_classes
        """
        return {
            'docstring': _get_docstring(class_node),
            'decorators': _get_decorators(class_node),
            'base_classes': _get_base_names(class_node),
            'attributes': _get_class_attributes(class_node),
            'instantiated_classes': _get_instantiated_classes_in_class(class_node),
        }


class FileMetadataExtractor:
    """Extract metadata from file/module level."""

    @staticmethod
    def extract(tree: ast.Module, filepath: str) -> Dict[str, Any]:
        """Extract metadata from a file's AST.

        Args:
            tree: ast.Module (parsed file)
            filepath: Relative path to the file

        Returns:
            Dict with keys: filepath, docstring, instantiated_classes
        """
        return {
            'filepath': filepath,
            'docstring': _get_docstring(tree),
            'instantiated_classes': _get_instantiated_classes(tree),
        }


class LocalTypeCollector:
    """Collect local type hints from function definitions."""

    @staticmethod
    def collect(func_node: ast.FunctionDef) -> Dict[str, str]:
        """Collect local variable type hints.

        Args:
            func_node: ast.FunctionDef

        Returns:
            Dict mapping variable names to type strings.
        """
        return _collect_local_types(func_node)
