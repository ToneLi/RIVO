"""Stateful browser tools backed by the local deep-research search service."""

from __future__ import annotations

import asyncio
import html
import json
import textwrap
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import tiktoken

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

# Tool objects are separate instances, so cursor state is shared by request id.
_SESSIONS: dict[str, list[dict[str, Any]]] = {}
_VIEW_ENCODING = tiktoken.get_encoding("o200k_base")



def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach retrieval service {url}: {exc.reason}") from exc


def _lines(text: str, width: int = 80) -> list[str]:
    output = []
    for raw_line in str(text).split("\n"):
        output.extend(
            textwrap.wrap(
                raw_line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            if raw_line
            else [""]
        )
    return output


def _numbered_body(lines: list[str], start: int, end: int) -> str:
    return "\n".join(f"L{line_number}: {lines[line_number]}" for line_number in range(start, end))


def _token_view_end(lines: list[str], start: int, view_tokens: int = 2048) -> int:
    """Round up to the line containing the first token beyond the observation budget."""
    full_body = _numbered_body(lines, start, len(lines))
    if len(full_body) <= view_tokens:
        return len(lines)
    if len(_VIEW_ENCODING.encode(full_body, disallowed_special=())) <= view_tokens:
        return len(lines)

    low, high = start + 1, len(lines)
    first_over_budget = len(lines)
    while low <= high:
        middle = (low + high) // 2
        body = _numbered_body(lines, start, middle)
        if len(_VIEW_ENCODING.encode(body, disallowed_special=())) > view_tokens:
            first_over_budget = middle
            high = middle - 1
        else:
            low = middle + 1
    return first_over_budget


def _render_page(cursor: int, page: dict, loc: int = 0, num_lines: int = -1, view_tokens: int = 2048) -> str:
    lines = page.get("lines") or [""]
    start = max(0, min(int(loc), max(0, len(lines) - 1)))
    budget_end = _token_view_end(lines, start, view_tokens)
    if int(num_lines) <= 0:
        end = budget_end
    else:
        end = min(len(lines), start + int(num_lines), budget_end)
    title = page.get("title") or page.get("url") or "page"
    url = page.get("url", "")
    header = f"[{cursor}] {title}" + (f" ({url})" if url else "")
    header += f"\n**viewing lines [{start} - {max(start, end - 1)}] of {max(0, len(lines) - 1)}**\n\n"
    return header + _numbered_body(lines, start, end)


def _truncate_observation_tokens(text: str, max_tokens: int) -> str:
    tokens = _VIEW_ENCODING.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return _VIEW_ENCODING.decode(tokens[:max_tokens])


class DeepResearchBrowserTool(BaseTool):
    """Implements search/open/find while preserving cursor state per trajectory."""

    def __init__(self, config: dict, tool_schema):
        super().__init__(config, tool_schema)
        self.base_url = str(config["base_url"]).rstrip("/")
        self.operation = str(config.get("operation") or self.name).split(".")[-1]
        self.timeout = float(config.get("timeout", 60))
        self.default_topn = int(config.get("topn", 10))
        self.default_num_lines = int(config.get("num_lines", -1))
        self.max_observation_tokens = int(config.get("max_observation_tokens", 2048))
        if self.max_observation_tokens <= 0:
            raise ValueError("max_observation_tokens must be positive")

    async def create(self, instance_id: str | None = None, **kwargs):
        return instance_id or str(uuid4()), ToolResponse()

    def _session(self, kwargs: dict) -> tuple[str, list[dict[str, Any]]]:
        agent_data = kwargs.get("agent_data")
        request_id = getattr(agent_data, "request_id", None) or "default"
        return request_id, _SESSIONS.setdefault(request_id, [])

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        _, pages = self._session(kwargs)
        try:
            if self.operation == "search":
                text = await self._search(parameters, pages)
            elif self.operation == "open":
                text = await self._open(parameters, pages)
            elif self.operation == "find":
                text = self._find(parameters, pages)
            else:
                raise ValueError(f"Unsupported browser operation: {self.operation}")
            text = _truncate_observation_tokens(text, self.max_observation_tokens)
            return ToolResponse(text=text), 0.0, {"status": "success", "operation": self.operation}
        except Exception as exc:
            message = f"Browser {self.operation} failed: {exc}"
            message = _truncate_observation_tokens(message, self.max_observation_tokens)
            return ToolResponse(text=message), 0.0, {"status": "error", "error": str(exc)}

    async def _search(self, parameters: dict, pages: list[dict[str, Any]]) -> str:
        query_value = parameters.get("query", "")
        if isinstance(query_value, list):
            queries = [str(query).strip() for query in query_value if str(query).strip()]
        else:
            query = str(query_value).strip()
            queries = [query] if query else []
        if not queries:
            raise ValueError("'query' must be a non-empty string")

        # OpenResearcher's BrowserTool always retrieves its max_search_results
        # (20), even though the public schema exposes topn=10.
        topn = 20
        payloads = await asyncio.gather(
            *[
                asyncio.to_thread(
                    _post_json, f"{self.base_url}/search", {"query": query, "topn": topn}, self.timeout
                )
                for query in queries
            ]
        )
        results = [result for payload in payloads for result in payload.get("results", [])]
        if not results:
            raise ValueError(f"No results returned for any query: {queries}")

        cursor = len(pages)
        query_title = " | ".join(queries)
        search_url = f"web-search://ts={int(time.time())}"
        display = ["", "# Search Results", ""]
        links = []
        for index, result in enumerate(results):
            title = html.escape(str(result.get("title") or "Untitled"), quote=True)
            url = html.escape(str(result.get("url") or ""), quote=True)
            summary = html.escape(str(result.get("summary") or ""), quote=True)
            domain = urlparse(url).netloc or url
            links.append(result)
            display.append(f"  * 【{index}†{title}†{domain}】 {summary}")

        page = {
            "kind": "search",
            "title": query_title,
            "url": search_url,
            "links": links,
            "lines": _lines("\n".join(display)),
        }
        pages.append(page)
        return _render_page(cursor, page, 0, self.default_num_lines, self.max_observation_tokens)

    async def _open(self, parameters: dict, pages: list[dict[str, Any]]) -> str:
        target = parameters.get("id", parameters.get("url"))
        if isinstance(target, list):
            if not target:
                raise ValueError("'url' must contain at least one URL")
            rendered_pages = []
            for item in target:
                item_parameters = dict(parameters)
                item_parameters.pop("id", None)
                item_parameters["url"] = item
                rendered_pages.append(await self._open(item_parameters, pages))
            return "\n\n".join(rendered_pages)

        if isinstance(target, str):
            stripped_target = target.strip()
            if stripped_target.lstrip("-").isdigit():
                target = int(stripped_target)
            else:
                target = stripped_target
        if target == -1:
            target = None

        cursor_arg = parameters.get("cursor")
        if isinstance(cursor_arg, str) and cursor_arg.strip().lstrip("-").isdigit():
            cursor_arg = int(cursor_arg.strip())
        if cursor_arg == -1:
            cursor_arg = None

        loc = int(parameters.get("loc", -1))
        if loc < 0:
            loc = 0
        num_lines = int(parameters.get("num_lines", -1))
        if num_lines < 1:
            num_lines = self.default_num_lines

        url = None
        if isinstance(target, int):
            page_index = int(cursor_arg) if cursor_arg is not None else len(pages) - 1
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"cursor {page_index} is out of range")
            source_page = pages[page_index]
            links = source_page.get("links", [])
            if target < 0 or target >= len(links):
                raise ValueError(f"link id {target} is out of range")
            url = links[target].get("url")
        elif isinstance(target, str) and target:
            url = target
        else:
            page_index = int(cursor_arg) if cursor_arg is not None else len(pages) - 1
            if page_index < 0 or page_index >= len(pages):
                raise ValueError("No current page is available; call search first")
            return _render_page(page_index, pages[page_index], loc, num_lines, self.max_observation_tokens)

        payload = await asyncio.to_thread(
            _post_json, f"{self.base_url}/get_content", {"url": url}, self.timeout
        )
        content = payload.get("content", "")
        if not content:
            raise ValueError(f"No content returned for {url}")
        page = {
            "kind": "page",
            "title": payload.get("title") or url,
            "url": url,
            "lines": _lines(content),
        }
        cursor = len(pages)
        pages.append(page)
        return _render_page(cursor, page, loc, num_lines, self.max_observation_tokens)

    def _find(self, parameters: dict, pages: list[dict[str, Any]]) -> str:
        pattern = str(parameters.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("'pattern' must be a non-empty string")

        cursor_arg = parameters.get("cursor")
        if isinstance(cursor_arg, str) and cursor_arg.strip().lstrip("-").isdigit():
            cursor_arg = int(cursor_arg.strip())
        if cursor_arg is None or cursor_arg == -1:
            candidates = [index for index, page in enumerate(pages) if page.get("kind") == "page"]
            if not candidates:
                raise ValueError("No opened page is available; call open first")
            cursor = candidates[-1]
        else:
            cursor = int(cursor_arg)
        if cursor < 0 or cursor >= len(pages):
            raise ValueError(f"cursor {cursor} is out of range")
        if pages[cursor].get("kind") == "search":
            raise ValueError("Cannot run find on a search-results page")

        matches = [
            line_number
            for line_number, line in enumerate(pages[cursor].get("lines", []))
            if pattern.casefold() in line.casefold()
        ]
        if not matches:
            return f"Pattern {pattern!r} not found in cursor {cursor}."
        return "\n".join(
            f"【{cursor}†L{line_number}】 {pages[cursor]['lines'][line_number]}" for line_number in matches[:20]
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        # Cursor state belongs to the whole trajectory, not an individual call.
        return None

    async def cleanup_session(self, request_id: str) -> None:
        _SESSIONS.pop(request_id, None)
