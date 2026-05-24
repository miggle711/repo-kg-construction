"""Tests for data flow extraction."""

import ast
import pytest
from kg_construction.ast.helpers import _extract_data_flows


class TestDataFlowExtraction:
    """Test _extract_data_flows for capturing data transformations."""

    def test_extract_return_values(self):
        """Extract function return values."""
        code = """
def get_value():
    x = 10
    return x
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        assert 'x' in flows['returns']

    def test_extract_multiple_returns(self):
        """Extract multiple return statements."""
        code = """
def conditional_return(x):
    if x > 0:
        return x
    else:
        return -x
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        assert len(flows['returns']) >= 1

    def test_extract_attribute_mutations(self):
        """Extract mutations to self attributes."""
        code = """
def cache_response(self):
    self.cache = {}
    self.count = 0
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        assert 'cache' in flows['mutates_attributes']
        assert 'count' in flows['mutates_attributes']

    def test_extract_attribute_mutations_with_expressions(self):
        """Extract mutations with complex expressions."""
        code = """
def send(self, url, response):
    self.response = response
    self.log_message = f"Sent to {url}"
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        assert 'response' in flows['mutates_attributes']
        assert 'log_message' in flows['mutates_attributes']

    def test_extract_subscript_mutations(self):
        """Extract subscript mutations like self.cache[key] = value."""
        code = """
def cache_response(self, url, response):
    self.cache[url] = response
    self.data["key"] = {"value": 42}
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        # Subscript mutations should be captured
        assert 'cache' in flows['mutates_attributes']
        assert 'data' in flows['mutates_attributes']

    def test_extract_parameter_usage(self):
        """Track which parameters are used in function body."""
        code = """
def send(self, url, timeout):
    if timeout < 0:
        raise ValueError()
    response = self._request(url)
    return response
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        assert 'url' in flows['parameter_usage']
        assert 'timeout' in flows['parameter_usage']
        # url and timeout should have line numbers
        assert len(flows['parameter_usage']['url']) > 0
        assert len(flows['parameter_usage']['timeout']) > 0

    def test_extract_no_parameter_usage(self):
        """Parameters not used in function should not appear."""
        code = """
def unused_param(self, url):
    return "constant"
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        assert 'url' not in flows['parameter_usage']

    def test_extract_self_not_as_parameter(self):
        """self should not appear in parameter_usage (it's implicit)."""
        code = """
def method(self):
    return self.value
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        assert 'self' in flows['parameter_usage']  # self is technically a param

    def test_empty_function(self):
        """Empty function returns empty flows."""
        code = """
def empty():
    pass
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        assert len(flows['returns']) == 0
        assert len(flows['mutates_attributes']) == 0
        assert len(flows['parameter_usage']) == 0

    def test_skip_nested_function_mutations(self):
        """Mutations in nested functions should not be captured."""
        code = """
def outer(self, x):
    def inner():
        self.nested = True
        return x
    y = x  # Use x in outer scope
    inner()
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        # 'nested' should not be captured (it's in nested function)
        assert 'nested' not in flows['mutates_attributes']
        # But x should be captured in outer scope
        assert 'x' in flows['parameter_usage']

    def test_capture_closure_parameters(self):
        """Parameters used in nested functions (closures) should be captured."""
        code = """
def outer(self, url):
    def inner():
        return url  # Closure: url from outer scope
    return inner()
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        # url is used in nested function (closure)
        assert 'url' in flows['parameter_usage']

    def test_deduplication(self):
        """Duplicate returns/mutations should be deduplicated."""
        code = """
def repeated_return(x):
    if True:
        return x
    else:
        return x
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        # Should have 'x' only once
        assert flows['returns'].count('x') == 1

    def test_deduplication_parameter_lines(self):
        """Duplicate parameter usage lines should be deduplicated."""
        code = """
def use_twice(x):
    y = x
    z = x
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        # Lines should be deduplicated and sorted
        assert len(flows['parameter_usage']['x']) > 0
        assert flows['parameter_usage']['x'] == sorted(flows['parameter_usage']['x'])

    def test_async_function(self):
        """Data flows should work on async functions."""
        code = """
async def async_send(self, url):
    self.pending = True
    response = await self._request(url)
    return response
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        assert 'pending' in flows['mutates_attributes']
        assert len(flows['returns']) > 0
        assert 'url' in flows['parameter_usage']

    def test_complex_expression_mutations(self):
        """Complex expressions in mutations should be unparsed."""
        code = """
def process(self):
    self.result = 42 * 2 + 10
    self.data = {"key": "value"}
"""
        func_node = ast.parse(code).body[0]
        flows = _extract_data_flows(func_node)
        assert 'result' in flows['mutates_attributes']
        assert 'data' in flows['mutates_attributes']
        # Should have unparsed expressions
        assert len(flows['mutates_attributes']['result']) > 0
        assert len(flows['mutates_attributes']['data']) > 0
