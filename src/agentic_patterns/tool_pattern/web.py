# Synchronous, anonymous Parallel Search MCP tools for the agent loops.
import json
from contextvars import ContextVar
from functools import wraps
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request
from urllib.request import urlopen
from uuid import uuid4

from agentic_patterns.tool_pattern.tool import tool

ENDPOINT = "https://search.parallel.ai/mcp"
USER_AGENT = "agentic-patterns/1.0.0"
TIMEOUT = 30
MAX_RESPONSE_BYTES = 2_000_000
_search_context = ContextVar("search_context", default=None)


def search_run(fn):
    """Each run owns its conversation ID, including when agents run concurrently."""

    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        token = _search_context.set((str(uuid4()), self.model))
        try:
            return fn(self, *args, **kwargs)
        finally:
            _search_context.reset(token)

    return wrapped


def _response(response, request_id):
    """Read the JSON or SSE response supported by Streamable HTTP MCP."""
    if "text/event-stream" in response.headers.get("Content-Type", ""):
        data = []
        size = 0
        for line in response:
            size += len(line)
            if size > MAX_RESPONSE_BYTES:
                raise ValueError("MCP response is too large")
            line = line.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif not line and data:
                message = json.loads("\n".join(data))
                data = []
                if isinstance(message, dict) and message.get("id") == request_id:
                    return message
        raise ValueError("MCP stream ended without a result")
    body = response.read(MAX_RESPONSE_BYTES + 1)
    if len(body) > MAX_RESPONSE_BYTES:
        raise ValueError("MCP response is too large")
    return json.loads(body)


def _post(message, headers):
    request = Request(
        ENDPOINT,
        data=json.dumps(message).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=TIMEOUT) as response:
        session = response.headers.get("Mcp-Session-Id")
        if session:
            headers["Mcp-Session-Id"] = session
        if "id" not in message:
            return None
        result = _response(response, message["id"])
    if not isinstance(result, dict) or result.get("id") != message["id"]:
        raise ValueError("Invalid MCP response ID")
    if "error" in result:
        raise ValueError("MCP request failed: " + str(result["error"]))
    if not isinstance(result.get("result"), dict):
        raise ValueError("Missing MCP result")
    return result["result"]


def _call(name, arguments):
    session_id, model = _search_context.get() or (str(uuid4()), None)
    arguments = dict(arguments, session_id=session_id)
    if model:
        arguments["model_name"] = model
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": USER_AGENT,
    }
    try:
        initialized = _post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "agentic-patterns", "version": "1.0.0"},
                },
            },
            headers,
        )
        if initialized.get("protocolVersion") != "2025-03-26":
            raise ValueError("Unsupported MCP protocol version")
        headers["MCP-Protocol-Version"] = initialized["protocolVersion"]
        _post({"jsonrpc": "2.0", "method": "notifications/initialized"}, headers)
        result = _post(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": arguments,
                },
            },
            headers,
        )
        if result.get("isError"):
            return "Parallel tool error: " + json.dumps(result.get("content", []))
        if "structuredContent" in result:
            if not isinstance(result["structuredContent"], dict):
                raise ValueError("Invalid MCP structured content")
            return json.dumps(result["structuredContent"], ensure_ascii=False)
        content = result.get("content")
        if not isinstance(content, list) or any(
            not isinstance(block, dict) for block in content
        ):
            raise ValueError("Invalid MCP tool content")
        texts = []
        for block in content:
            if block.get("type") == "text":
                if not isinstance(block.get("text"), str):
                    raise ValueError("Invalid MCP text content")
                texts.append(block["text"])
        text = "\n".join(texts)
        if not text:
            raise ValueError("MCP tool returned no text")
        return text
    except HTTPError as error:
        error.close()
        return f"Parallel tool error: HTTP {error.code}. No automatic retry."
    except (OSError, HTTPException, ValueError) as error:
        return f"Parallel tool error: {error}"


@tool
def web_search(query: str):
    """Search the current web using a focused query of 3-6 words. Returns sources and excerpts."""
    if not query.strip():
        return "Parallel tool error: query must not be empty"
    return _call("web_search", {"objective": query, "search_queries": [query]})


@tool
def web_fetch(url: str):
    """Read excerpts from one HTTP/HTTPS page, usually a source URL returned by web_search."""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "Parallel tool error: provide a valid HTTP/HTTPS URL"
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        return "Parallel tool error: provide an HTTP/HTTPS URL without credentials"
    return _call("web_fetch", {"urls": [url]})


def resolve_tools(tools):
    """Omitted tools enable web access; an explicit selection, even [], wins."""
    if tools is None:
        return [web_search, web_fetch]
    return tools if isinstance(tools, list) else [tools]
