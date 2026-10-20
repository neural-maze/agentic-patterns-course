# Exercise the real agent loops with scripted model and HTTP responses.
import copy
import io
import json
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.error import URLError
from uuid import UUID

from agentic_patterns.planning_pattern.react_agent import ReactAgent
from agentic_patterns.tool_pattern import web
from agentic_patterns.tool_pattern.tool import tool
from agentic_patterns.tool_pattern.tool_agent import ToolAgent

SOURCE = "https://example.org/release"


def call(name="web_search", arguments=None, number=0):
    if arguments is None:
        arguments = {"query": "latest Python release notes"}
    return (
        "<thought>Look up sources</thought><tool_call>"
        + json.dumps({"name": name, "arguments": arguments, "id": number})
        + "</tool_call>"
    )


class Response(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {"Content-Type": "application/json"}


class Server:
    def __init__(self, error=None, sse=False, barrier=None):
        self.requests = []
        self.responses = []
        self.error = error
        self.sse = sse
        self.barrier = barrier

    def __call__(self, request, timeout):
        message = json.loads(request.data)
        self.requests.append(
            (request.full_url, dict(request.header_items()), timeout, message)
        )
        headers = {"Content-Type": "application/json"}
        if message["method"] == "initialize":
            result = {"protocolVersion": "2025-03-26"}
            headers["Mcp-Session-Id"] = "transport-only"
        elif message["method"] == "notifications/initialized":
            response = Response(b"", headers)
            self.responses.append(response)
            return response
        else:
            if self.barrier:
                self.barrier.wait(timeout=5)
            if isinstance(self.error, BaseException):
                raise self.error
            result = self.error or {
                "content": [{"type": "text", "text": f"Python release: {SOURCE}"}]
            }
        envelope = {"jsonrpc": "2.0", "id": message["id"], "result": result}
        body = json.dumps(envelope).encode()
        if self.sse:
            headers["Content-Type"] = "text/event-stream"
            body = b": keepalive\n\nevent: message\ndata: " + body + b"\n\n"
        response = Response(body, headers)
        self.responses.append(response)
        return response

    def arguments(self):
        return [
            r[3]["params"]["arguments"]
            for r in self.requests
            if r[3]["method"] == "tools/call"
        ]


class Model:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, messages, model):
        self.messages.append(copy.deepcopy(messages))
        output = next(self.outputs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=output))]
        )


def agent_with_model(kind, outputs, **kwargs):
    model = Model(outputs)
    module = (
        "agentic_patterns.planning_pattern.react_agent"
        if kind is ReactAgent
        else "agentic_patterns.tool_pattern.tool_agent"
    )
    with patch(module + ".Groq", return_value=model):
        return kind(**kwargs), model


class WebAgentTests(unittest.TestCase):
    def setUp(self):
        # Neither ambient credentials nor saved .env files are needed for these scripted runs.
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_defaults_dispatch_and_result_consumption(self):
        for kind in (ToolAgent, ReactAgent):
            with self.subTest(kind=kind):
                outputs = [
                    call() + call("web_fetch", {"url": SOURCE}, 1),
                    "<response>Answer with sources</response>",
                ]
                agent, model = agent_with_model(kind, outputs)
                server = Server(sse=kind is ReactAgent)
                with patch.object(web, "urlopen", side_effect=server):
                    answer = agent.run("Find and read Python release notes")
                self.assertIn("Answer with sources", answer)
                self.assertEqual(set(agent.tools_dict), {"web_search", "web_fetch"})
                self.assertIn(SOURCE, str(model.messages[-1]))
                args = server.arguments()
                self.assertEqual(args[0]["session_id"], args[1]["session_id"])
                UUID(args[0]["session_id"])
                self.assertEqual(args[0]["model_name"], agent.model)
                self.assertEqual(
                    args[0]["search_queries"], ["latest Python release notes"]
                )
                self.assertEqual(args[1]["urls"], [SOURCE])
                for url, headers, timeout, message in server.requests:
                    self.assertEqual(url, "https://search.parallel.ai/mcp")
                    self.assertEqual(headers["User-agent"], web.USER_AGENT)
                    self.assertNotIn("Authorization", headers)
                    self.assertEqual(timeout, 30)
                    if message["method"] != "initialize":
                        self.assertEqual(headers["Mcp-session-id"], "transport-only")
                        self.assertEqual(headers["Mcp-protocol-version"], "2025-03-26")
                self.assertTrue(all(response.closed for response in server.responses))

    def test_explicit_tools_and_empty_list_are_authoritative(self):
        @tool
        def custom(query: str):
            return "custom provider result"

        for kind in (ToolAgent, ReactAgent):
            for selection in ([], custom, [custom]):
                with self.subTest(kind=kind, selection=selection):
                    agent, _ = agent_with_model(
                        kind, ["<response>Offline</response>"] * 2, tools=selection
                    )
                    with patch.object(web, "urlopen") as network:
                        agent.run("Hello")
                    self.assertEqual(agent.tools, [] if selection == [] else [custom])
                    network.assert_not_called()
            agent, _ = agent_with_model(kind, [], tools=[])
            with patch.object(web, "urlopen") as network:
                with self.assertRaises(KeyError):
                    agent.process_tool_calls(
                        [
                            json.dumps(
                                {
                                    "name": "web_search",
                                    "arguments": {"query": "anything"},
                                    "id": 0,
                                }
                            )
                        ]
                    )
            network.assert_not_called()

    def test_explicit_none_and_construction_are_lazy(self):
        for kind in (ToolAgent, ReactAgent):
            with patch.object(web, "urlopen") as network:
                agent, _ = agent_with_model(kind, [], tools=None)
            self.assertEqual(set(agent.tools_dict), {"web_search", "web_fetch"})
            network.assert_not_called()

    def test_custom_provider_failure_does_not_enable_web(self):
        @tool
        def custom(query: str):
            raise RuntimeError("explicit provider failed")

        agent, _ = agent_with_model(ReactAgent, [call("custom")], tools=[custom])
        with patch.object(web, "urlopen") as network:
            with self.assertRaisesRegex(RuntimeError, "explicit provider failed"):
                agent.run("Research")
        network.assert_not_called()

    def test_followups_and_new_runs_own_identity(self):
        outputs = [
            call(),
            call("web_fetch", {"url": SOURCE}),
            call(),
            "<response>Done</response>",
        ] * 2
        agent, _ = agent_with_model(ReactAgent, outputs)
        server = Server()
        with patch.object(web, "urlopen", side_effect=server):
            agent.run("First research run")
            agent.run("New research run")
        ids = [a["session_id"] for a in server.arguments()]
        self.assertEqual(len(set(ids[:3])), 1)
        self.assertEqual(len(set(ids[3:])), 1)
        self.assertNotEqual(ids[0], ids[3])
        self.assertIsNone(web._search_context.get())
        self.assertEqual(agent.system_prompt, "")

    def test_interleaved_agents_do_not_share_identity(self):
        agents = [
            agent_with_model(
                ReactAgent,
                [
                    call(),
                    call("web_fetch", {"url": SOURCE}),
                    "<response>Done</response>",
                ],
            )[0]
            for _ in range(2)
        ]
        server = Server(barrier=threading.Barrier(2))
        with patch.object(web, "urlopen", side_effect=server):
            with ThreadPoolExecutor(max_workers=2) as executor:
                self.assertEqual(
                    list(executor.map(lambda a: a.run("Research"), agents)),
                    ["Done", "Done"],
                )
        ids = [a["session_id"] for a in server.arguments()]
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(sorted(ids[:2]), sorted(ids[2:]))

    def test_errors_are_observations_and_do_not_retry_or_fallback(self):
        errors = [
            TimeoutError("timed out"),
            URLError("offline"),
            HTTPError(web.ENDPOINT, 429, "quota", {}, None),
            {"isError": True, "content": [{"type": "text", "text": "quota exhausted"}]},
        ]
        for error in errors:
            with self.subTest(error=error):
                agent, model = agent_with_model(
                    ReactAgent, [call(), "<response>Unavailable</response>"]
                )
                server = Server(error=error)
                with patch.object(web, "urlopen", side_effect=server):
                    self.assertEqual(agent.run("Research"), "Unavailable")
                self.assertIn("Parallel tool error", str(model.messages[-1]))
                self.assertEqual(len(server.arguments()), 1)

    def test_quota_followup_keeps_identity(self):
        agent, _ = agent_with_model(
            ReactAgent, [call(), call(), "<response>Done</response>"]
        )
        server = Server(
            error={
                "isError": True,
                "content": [{"type": "text", "text": "quota exhausted"}],
            }
        )
        with patch.object(web, "urlopen", side_effect=server):
            agent.run("Research")
        self.assertEqual(
            server.arguments()[0]["session_id"], server.arguments()[1]["session_id"]
        )

    def test_interrupt_propagates_and_clears_context(self):
        agent, _ = agent_with_model(
            ReactAgent, [call(), call(), "<response>Done</response>"]
        )
        server = Server(error=KeyboardInterrupt())
        with patch.object(web, "urlopen", side_effect=server):
            with self.assertRaises(KeyboardInterrupt):
                agent.run("Interrupted")
            self.assertIsNone(web._search_context.get())
            server.error = None
            agent.run("New run")
        self.assertNotEqual(
            server.arguments()[0]["session_id"], server.arguments()[1]["session_id"]
        )

    def test_invalid_inputs_do_not_send_network_requests(self):
        with patch.object(web, "urlopen") as network:
            for url in (
                "file:///etc/passwd",
                "javascript:alert(1)",
                "https://user:pass@example.org",
                "http://",
                "http://[",
            ):
                self.assertIn("Parallel tool error", web.web_fetch.run(url=url))
            self.assertIn("Parallel tool error", web.web_search.run(query=" "))
        network.assert_not_called()

    def test_malformed_transport_and_protocol_errors(self):
        for body in (
            b"not JSON",
            b'{"id":2,"result":{}}',
            b'{"id":1,"error":{"code":-1}}',
            b'{"id":1,"result":{}}',
        ):
            response = Response(body)
            with patch.object(web, "urlopen", return_value=response):
                self.assertIn(
                    "Parallel tool error",
                    web.web_search.run(query="Python release notes"),
                )
            self.assertTrue(response.closed)

    def test_bounded_response(self):
        with patch.object(web, "MAX_RESPONSE_BYTES", 10):
            response = Response(b"xxxxxxxxxxx")
            with patch.object(web, "urlopen", return_value=response):
                self.assertIn(
                    "too large", web.web_search.run(query="Python release notes")
                )


if __name__ == "__main__":
    unittest.main()
