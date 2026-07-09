"""
type_inference.py

Optional pyright-backed type resolution for factory-function call sites.

_get_factory_call_sites() (ast/helpers.py) records assignment sites where a
lowercase-named call's return value is invisible to the uppercase-heuristic
`uses`-edge detection (e.g. `session = requests.session()`). This module
answers "what type does that call actually return?" by driving a single
long-lived `pyright-langserver` process over stdio and asking it via the
standard `textDocument/hover` request — the same query an editor issues when
you hover a variable. No source files are rewritten; the extracted repo
tree is only ever read.

This is best-effort enrichment, not a required step: any failure (pyright
not installed, timeout, protocol error) results in an empty resolution map
rather than raising, so a KG build never fails because of this step.
"""

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Wall-clock budget for starting pyright and getting through initialize.
STARTUP_TIMEOUT = 30
# Wall-clock budget per hover request.
HOVER_TIMEOUT = 10


class PyrightUnavailableError(Exception):
    """Raised when infer_types=True is requested but pyright is not installed."""


def is_available() -> bool:
    """Return True if the pyright-langserver executable can be found on PATH."""
    return shutil.which("pyright-langserver") is not None


class _LSPClient:
    """Minimal JSON-RPC-over-stdio client for a single pyright-langserver session."""

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        self._next_id = 1
        self._lock = threading.Lock()

    def _send(self, message: Dict) -> None:
        body = json.dumps(message).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        self._proc.stdin.write(header + body)
        self._proc.stdin.flush()

    def _read_message(self) -> Optional[Dict]:
        stream = self._proc.stdout
        headers: Dict[str, str] = {}
        while True:
            line = stream.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if ":" in decoded:
                key, _, value = decoded.partition(":")
                headers[key.strip()] = value.strip()
        length = int(headers.get("Content-Length", 0))
        if length == 0:
            return None
        body = stream.read(length)
        return json.loads(body.decode("utf-8", errors="replace"))

    def notify(self, method: str, params: Dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: Dict) -> Optional[Dict]:
        """Send a request and read messages until the matching response arrives.

        Diagnostics and other server-initiated notifications may interleave
        before the response; those are silently skipped.
        """
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            for _ in range(50):
                message = self._read_message()
                if message is None:
                    return None
                if message.get("id") == request_id:
                    return message
            return None


def resolve_types(
    repo_dir: Path,
    sites_by_file: Dict[str, List[Tuple[int, int]]],
) -> Dict[Tuple[str, int, int], str]:
    """Resolve inferred types at recorded factory-call assignment sites.

    Args:
        repo_dir: Root of the extracted source tree (read-only).
        sites_by_file: Dict mapping repo-relative file path to a list of
            (line, col) tuples (0-indexed) to query, as produced by
            _get_factory_call_sites().

    Returns:
        Dict mapping (rel_path, line, col) -> inferred type name (e.g.
        'Session'). Sites that pyright can't resolve to a simple type name
        (unions, unknown, builtins) are omitted rather than guessed at.
        Returns an empty dict if pyright is unavailable or fails outright —
        callers should treat that as "no enrichment," not an error.
    """
    if not sites_by_file or not is_available():
        return {}

    try:
        return _resolve_types_inner(repo_dir, sites_by_file)
    except Exception:
        # Any failure here (timeout, protocol hiccup, crash) degrades to
        # "no enrichment" rather than failing the whole KG build.
        return {}


def _resolve_types_inner(
    repo_dir: Path,
    sites_by_file: Dict[str, List[Tuple[int, int]]],
) -> Dict[Tuple[str, int, int], str]:
    resolved: Dict[Tuple[str, int, int], str] = {}

    proc = subprocess.Popen(
        ["pyright-langserver", "--stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _drain_stderr_in_background(proc)
        client = _LSPClient(proc)

        # workspaceFolders + hover capabilities are required for pyright to
        # do full-project analysis (e.g. inferring an instance attribute's
        # type from an assignment in __init__). Without them pyright falls
        # back to shallow single-file analysis and reports such attributes
        # as "Unknown" even though its own CLI/reveal_type mode resolves
        # them correctly — confirmed by direct comparison during development.
        root_uri = repo_dir.resolve().as_uri()
        client.request("initialize", {
            "processId": None,
            "rootUri": root_uri,
            "workspaceFolders": [{"uri": root_uri, "name": repo_dir.name}],
            "capabilities": {
                "textDocument": {"hover": {"contentFormat": ["plaintext"]}},
                "workspace": {"workspaceFolders": True},
            },
        })
        client.notify("initialized", {})

        for rel_path, sites in sites_by_file.items():
            abs_path = repo_dir / rel_path
            try:
                text = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            uri = abs_path.resolve().as_uri()
            client.notify("textDocument/didOpen", {
                "textDocument": {
                    "uri": uri, "languageId": "python", "version": 1, "text": text,
                }
            })

            for line, col in sites:
                response = client.request("textDocument/hover", {
                    "textDocument": {"uri": uri},
                    "position": {"line": line, "character": col},
                })
                type_name = _parse_hover_type(response)
                if type_name:
                    resolved[(rel_path, line, col)] = type_name

            client.notify("textDocument/didClose", {"textDocument": {"uri": uri}})

        client.request("shutdown", {})
        client.notify("exit", {})
    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    return resolved


def _drain_stderr_in_background(proc: subprocess.Popen) -> None:
    """Consume stderr on a background thread so pyright never blocks on a full pipe."""
    def _drain():
        for _ in iter(proc.stderr.readline, b""):
            pass
    threading.Thread(target=_drain, daemon=True).start()


def _parse_hover_type(response: Optional[Dict]) -> Optional[str]:
    """Extract a simple class name from a pyright hover response, if any.

    Hover text looks like '(variable) s: Session'. Only single, simple
    (non-union, non-builtin) type names are returned — anything else
    (unions, 'Unknown', primitives) is treated as unresolved rather than
    guessed at, matching this module's "honest unknown" contract.
    """
    if not response or "result" not in response or response["result"] is None:
        return None

    contents = response["result"].get("contents")
    if isinstance(contents, dict):
        text = contents.get("value", "")
    elif isinstance(contents, str):
        text = contents
    else:
        return None

    if ":" not in text:
        return None
    type_part = text.rsplit(":", 1)[1].strip()
    # Strip markdown code fences pyright sometimes wraps hover text in.
    type_part = type_part.strip("`").strip()

    if not type_part or " " in type_part or "|" in type_part:
        return None  # union, unknown-with-qualifier, or otherwise not simple
    if type_part in ("Unknown", "None", "Any"):
        return None
    if type_part[0].islower():
        return None  # builtins (str, int, bool, ...) aren't repo classes

    return type_part
