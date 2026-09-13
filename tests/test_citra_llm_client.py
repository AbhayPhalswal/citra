"""
The Smart Path backends against a scripted fake HTTP session: the exact
request shape each backend sends, the tool-call round trip (model asks
for a tool -> we dispatch -> the result goes back -> the model answers),
every failure mode as a RouteResult rather than an exception, the
bounded round limit, and the shared conversation memory.

No LM Studio, no Gemini key, no network.
"""
import json

import pytest
import requests

import citra_llm_client
from citra_llm_client import (
    CONVERSATION_HISTORY_MAX_MESSAGES,
    MAX_TOOL_CALL_ROUNDS,
    ConversationMemory,
    GeminiClient,
    LMStudioClient,
    _looks_like_code_request,
    to_gemini_tools,
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "turn_on_relay",
        "description": "Turn on a relay.",
        "parameters": {
            "type": "object",
            "properties": {"relay_number": {"type": "integer", "enum": [1, 2, 3, 4]}},
        },
    },
}]


class FakeResponse:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}")

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """Replies in order; records every request. A reply may be an exception."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.posts: list[dict] = []
        self.gets: list[str] = []
        self.models_reply = FakeResponse(200, {"data": [{"id": "some-model"}]})

    def get(self, url, timeout=None, **kwargs):
        self.gets.append(url)
        reply = self.models_reply
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def post(self, url, json=None, headers=None, timeout=None, **kwargs):
        self.posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def dispatch_ok(tool_call: dict) -> str:
    """A stand-in for JarvisRouter._dispatch_tool_call."""
    dispatch_ok.calls.append(tool_call)
    return json.dumps({"success": True, "endpoint": "fake", "message": "OK", "data": None})


@pytest.fixture(autouse=True)
def _reset_dispatch():
    dispatch_ok.calls = []


def lm_text(answer: str) -> FakeResponse:
    return FakeResponse(200, {"choices": [{"message": {"role": "assistant", "content": answer}}]})


def lm_tool_call(name: str, arguments: dict) -> FakeResponse:
    return FakeResponse(200, {"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)}}],
    }}]})


# --------------------------------------------------------------------------
# ConversationMemory
# --------------------------------------------------------------------------
def test_memory_is_bounded_to_the_configured_number_of_messages():
    memory = ConversationMemory(max_messages=4)
    for i in range(5):
        memory.remember(f"q{i}", f"a{i}")
    assert len(memory) == 4
    assert [m["content"] for m in memory.history] == ["q3", "a3", "q4", "a4"]


def test_memory_expires_after_silence(monkeypatch):
    memory = ConversationMemory(timeout_seconds=300)
    memory.remember("q", "a")
    memory.last_time -= 301
    memory.expire_if_stale()
    assert len(memory) == 0 and memory.last_time is None


def test_memory_does_not_expire_before_the_timeout():
    memory = ConversationMemory(timeout_seconds=300)
    memory.remember("q", "a")
    memory.expire_if_stale()
    assert len(memory) == 2


def test_default_memory_matches_the_constants():
    assert ConversationMemory().history.maxlen == CONVERSATION_HISTORY_MAX_MESSAGES


# --------------------------------------------------------------------------
# LM Studio
# --------------------------------------------------------------------------
@pytest.fixture
def memory():
    return ConversationMemory()


def lm(session, memory, **kw) -> LMStudioClient:
    return LMStudioClient(session, memory, dispatch_ok, TOOLS,
                          base_url="http://lm.test/", model="test-model", timeout=7.0, **kw)


def test_lm_studio_sends_the_openai_chat_shape_and_remembers_the_exchange(memory):
    session = FakeSession(lm_text("It is 24 degrees."))
    result = lm(session, memory).complete("what temperature is it")

    assert result.path == "SMART" and result.success is True
    assert result.message == "It is 24 degrees."
    (post,) = session.posts
    assert post["url"] == "http://lm.test/v1/chat/completions"
    assert post["timeout"] == 7.0
    assert post["json"]["model"] == "test-model"
    assert post["json"]["tools"] == TOOLS
    assert post["json"]["messages"][0]["role"] == "system"
    assert post["json"]["messages"][-1] == {"role": "user", "content": "what temperature is it"}
    assert [m["content"] for m in memory.history] == ["what temperature is it", "It is 24 degrees."]
    # The preflight hit /v1/models first.
    assert session.gets == ["http://lm.test/v1/models"]


def test_lm_studio_history_is_sent_on_the_next_call(memory):
    session = FakeSession(lm_text("Paris."), lm_text("About two million."))
    client = lm(session, memory)
    client.complete("capital of France?")
    client.complete("and its population?")

    messages = session.posts[1]["json"]["messages"]
    assert [m["content"] for m in messages[1:]] == [
        "capital of France?", "Paris.", "and its population?",
    ]


def test_lm_studio_tool_call_round_trip(memory):
    session = FakeSession(
        lm_tool_call("turn_on_relay", {"relay_number": 2}),
        lm_text("Done, light 2 is on."),
    )
    result = lm(session, memory).complete("switch on the second light")

    assert result.success is True
    assert result.message == "Done, light 2 is on."
    assert dispatch_ok.calls[0]["function"]["name"] == "turn_on_relay"
    # Second request carries the assistant's tool-call turn AND our result.
    second = session.posts[1]["json"]["messages"]
    assert second[-2]["tool_calls"][0]["function"]["name"] == "turn_on_relay"
    assert second[-1]["role"] == "tool"
    assert second[-1]["tool_call_id"] == "call_1"
    assert json.loads(second[-1]["content"])["success"] is True
    # Only the prompt and the final answer are remembered, not the plumbing.
    assert len(memory) == 2


def test_lm_studio_gives_up_after_the_round_limit(memory):
    session = FakeSession(*[lm_tool_call("turn_on_relay", {"relay_number": 1})] * MAX_TOOL_CALL_ROUNDS)
    result = lm(session, memory).complete("loop forever")

    assert result.success is False
    assert "more steps" in result.message
    assert len(session.posts) == MAX_TOOL_CALL_ROUNDS
    assert len(memory) == 0


@pytest.mark.parametrize("reply, expected", [
    (requests.exceptions.Timeout(), "timed out after 7.0s"),
    (requests.exceptions.ConnectionError(), "Could not connect to LM Studio"),
    (requests.exceptions.InvalidURL("x"), "Unexpected LLM request failure"),
    (FakeResponse(200, None, text="<html>"), "non-JSON"),
    (FakeResponse(200, {"error": "no model loaded"}), "no choices. Server said: no model loaded"),
    (FakeResponse(200, {"choices": [{"message": {"content": "   "}}]}), "empty answer"),
])
def test_lm_studio_failures_are_results_not_exceptions(memory, reply, expected):
    result = lm(FakeSession(reply), memory).complete("hello")
    assert result.path == "SMART" and result.success is False
    assert expected in result.message
    assert len(memory) == 0


def test_lm_studio_preflight_catches_no_model_loaded(memory):
    session = FakeSession()
    session.models_reply = FakeResponse(200, {"data": []})
    client = lm(session, memory)

    assert "no model is loaded" in client.check_model_loaded()
    result = client.complete("hello")
    assert result.success is False and "no model is loaded" in result.message
    assert session.posts == []


def test_lm_studio_preflight_catches_server_down(memory):
    session = FakeSession()
    session.models_reply = requests.exceptions.ConnectionError()
    assert "Could not connect" in lm(session, memory).check_model_loaded()


def test_lm_studio_preflight_gets_out_of_the_way_on_other_errors(memory):
    session = FakeSession(lm_text("fine"))
    session.models_reply = requests.exceptions.Timeout()
    client = lm(session, memory)
    assert client.check_model_loaded() is None
    assert client.complete("hello").success is True


def test_lm_studio_expires_stale_memory_before_a_call(memory):
    memory.remember("old", "context")
    memory.last_time -= 10_000
    session = FakeSession(lm_text("fresh"))
    lm(session, memory).complete("new topic")
    assert [m["content"] for m in session.posts[0]["json"]["messages"][1:]] == ["new topic"]


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------
def gem_text(answer: str) -> FakeResponse:
    return FakeResponse(200, {"candidates": [{"content": {"role": "model", "parts": [{"text": answer}]}}]})


def gem_call(name: str, args: dict) -> FakeResponse:
    return FakeResponse(200, {"candidates": [{"content": {
        "role": "model", "parts": [{"functionCall": {"name": name, "args": args}}],
    }}]})


def gemini(session, memory, api_key="k") -> GeminiClient:
    return GeminiClient(session, memory, dispatch_ok, TOOLS, api_key=api_key, timeout=9.0,
                        base_url="https://gem.test/v1beta", model="test-gemini")


def test_gemini_without_a_key_refuses_before_any_request(memory):
    session = FakeSession()
    result = gemini(session, memory, api_key="").complete("hi")
    assert result.success is False
    assert "GEMINI_API_KEY" in result.message
    assert session.posts == []


def test_gemini_sends_its_own_wire_shape(memory):
    session = FakeSession(gem_text("Hello there."))
    result = gemini(session, memory).complete("hi")

    assert result.success is True and result.message == "Hello there."
    (post,) = session.posts
    assert post["url"] == "https://gem.test/v1beta/models/test-gemini:generateContent"
    assert post["headers"]["x-goog-api-key"] == "k"
    assert post["timeout"] == 9.0
    body = post["json"]
    assert "systemInstruction" in body
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert body["tools"] == to_gemini_tools(TOOLS)
    # The integer enum must be gone or Gemini 400s the whole request.
    props = body["tools"][0]["functionDeclarations"][0]["parameters"]["properties"]
    assert "enum" not in props["relay_number"]


def test_gemini_converts_shared_memory_to_model_and_user_roles(memory):
    memory.remember("q", "a")
    session = FakeSession(gem_text("ok"))
    gemini(session, memory).complete("next")
    contents = session.posts[0]["json"]["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]


def test_gemini_function_call_round_trip(memory):
    session = FakeSession(gem_call("turn_on_relay", {"relay_number": 3}), gem_text("Light 3 is on."))
    result = gemini(session, memory).complete("third light on please")

    assert result.success is True
    assert json.loads(dispatch_ok.calls[0]["function"]["arguments"]) == {"relay_number": 3}
    second = session.posts[1]["json"]["contents"]
    assert second[-2]["role"] == "model" and "functionCall" in second[-2]["parts"][0]
    assert second[-1]["role"] == "user"
    assert second[-1]["parts"][0]["functionResponse"]["name"] == "turn_on_relay"
    assert second[-1]["parts"][0]["functionResponse"]["response"]["success"] is True


def test_gemini_retries_a_503_then_succeeds(memory, monkeypatch):
    slept = []
    monkeypatch.setattr(citra_llm_client.time, "sleep", slept.append)
    session = FakeSession(FakeResponse(503, {}), gem_text("back"))
    result = gemini(session, memory).complete("hi")

    assert result.success is True and result.message == "back"
    assert len(session.posts) == 2
    assert slept == [citra_llm_client.GEMINI_503_RETRY_BACKOFF_SECONDS]


def test_gemini_rate_limit_is_not_retried(memory):
    session = FakeSession(FakeResponse(429, {}))
    result = gemini(session, memory).complete("hi")
    assert result.success is False
    assert "rate limit" in result.message
    assert len(session.posts) == 1


@pytest.mark.parametrize("reply, expected", [
    (requests.exceptions.Timeout(), "timed out after 9.0s"),
    (requests.exceptions.ConnectionError(), "check your internet"),
    (FakeResponse(400, {"error": {"message": "bad schema"}}), "Gemini API error (400). Server said: bad schema"),
    (FakeResponse(200, None, text="<html>"), "non-JSON"),
    (FakeResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}}), "no response (blocked: SAFETY)"),
    (gem_text("   "), "empty answer"),
])
def test_gemini_failures_are_results_not_exceptions(memory, reply, expected):
    result = gemini(FakeSession(reply), memory).complete("hello")
    assert result.path == "SMART" and result.success is False
    assert expected in result.message
    assert len(memory) == 0


def test_gemini_forces_the_code_tool_once_when_it_narrates_instead(memory):
    # Round 0: text-only reply to a code-shaped request. Round 1 must be
    # re-sent with toolConfig forcing write_python_code - exactly once.
    session = FakeSession(gem_text("I'm writing that now."), gem_text("Opened in Notepad."))
    result = gemini(session, memory).complete("write me a python script that renames photos")

    assert result.success is True
    assert "toolConfig" not in session.posts[0]["json"]
    forced = session.posts[1]["json"]["toolConfig"]["functionCallingConfig"]
    assert forced == {"mode": "ANY", "allowedFunctionNames": ["write_python_code"]}
    # The narration turn is NOT appended: contents must still end on the user prompt.
    assert session.posts[1]["json"]["contents"][-1]["role"] == "user"


@pytest.mark.parametrize("text, expected", [
    ("write me a python script that renames files", True),
    ("create a program to sort numbers", True),
    ("make me a website", True),
    ("what is the weather", False),
    ("write a poem", False),          # no code noun
    ("open the script folder", False),  # no code verb
])
def test_looks_like_code_request(text, expected):
    assert _looks_like_code_request(text) is expected
