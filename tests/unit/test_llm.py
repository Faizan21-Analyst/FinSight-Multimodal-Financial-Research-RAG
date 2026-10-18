"""Text LLM client tests. No network: httpx.MockTransport stands in for Groq."""
import json

import httpx
import pytest
from pydantic import BaseModel

from finsight.core.caching import DiskCache
from finsight.core.config import Settings
from finsight.core.exceptions import ConfigError
from finsight.llm.client import GroqLLM, LLMError, build_llm, parse_json_object, strip_thinking


def ok(content="hello", usage=(10, 5), finish="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
        },
    )


def make(handler, **kw):
    kw.setdefault("sleep", lambda s: None)
    return GroqLLM("m", "key", client=httpx.Client(transport=httpx.MockTransport(handler)), **kw)


# ---------- helpers ----------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("plain answer", "plain answer"),
        ("<think>hmm</think>The answer.", "The answer."),
        ("<think>line1\nline2</think>\n\nFinal", "Final"),
        ("Answer<think>cut off mid thought", "Answer"),
        ("reasoning without opening tag</think>Final", "Final"),
    ],
)
def test_strip_thinking(raw, expected):
    assert strip_thinking(raw) == expected


def test_parse_json_object():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": {"b": 2}} done') == {"a": {"b": 2}}
    for bad in ("no json", "{oops}", "[1, 2]"):
        with pytest.raises(LLMError):
            parse_json_object(bad)


# ---------- complete() ----------

def test_request_shape_and_stats():
    seen = []

    def handler(req):
        seen.append((req, json.loads(req.content)))
        return ok("hi", usage=(12, 3))

    llm = make(handler, extra_body={"reasoning_effort": "none"}, temperature=0.1, max_tokens=77)
    res = llm.complete("question", system="be brief")
    req, body = seen[0]
    assert res.text == "hi" and not res.cached and (res.prompt_tokens, res.completion_tokens) == (12, 3)
    assert req.headers["authorization"] == "Bearer key"
    assert body["messages"] == [{"role": "system", "content": "be brief"},
                                {"role": "user", "content": "question"}]
    assert body["temperature"] == 0.1 and body["max_tokens"] == 77
    assert body["reasoning_effort"] == "none" and "response_format" not in body
    assert (llm.stats.calls, llm.stats.prompt_tokens, llm.stats.completion_tokens) == (1, 12, 3)


def test_per_call_overrides():
    bodies = []
    llm = make(lambda r: (bodies.append(json.loads(r.content)), ok())[1])
    llm.complete("q", temperature=0.9, max_tokens=5)
    assert bodies[0]["temperature"] == 0.9 and bodies[0]["max_tokens"] == 5


def test_thinking_is_stripped_from_replies():
    llm = make(lambda r: ok("<think>let me see</think>42"))
    assert llm.complete("q").text == "42"


def test_disk_cache_keyed_on_full_request(tmp_path):
    calls = []
    llm = make(lambda r: (calls.append(1), ok("cached?"))[1], cache=DiskCache(tmp_path))
    first = llm.complete("same prompt")
    second = llm.complete("same prompt")
    assert (first.cached, second.cached) == (False, True) and second.text == "cached?"
    assert len(calls) == 1 and llm.stats.calls == 1 and llm.stats.cache_hits == 1
    llm.complete("same prompt", temperature=0.9)   # different params -> new request
    llm.complete("different prompt")
    assert len(calls) == 3


def test_retries_429_and_5xx_then_succeeds():
    replies = [httpx.Response(429, headers={"retry-after": "2"}), httpx.Response(503), ok("fine")]
    slept = []
    llm = make(lambda r: replies.pop(0), sleep=slept.append)
    assert llm.complete("q").text == "fine"
    assert slept == [2.0, 4.0]


def test_errors():
    def always(status):
        return make(lambda r: httpx.Response(status, text="nope"), retries=1)

    with pytest.raises(LLMError, match="500"):
        always(500).complete("q")
    with pytest.raises(ConfigError, match="401"):
        always(401).complete("q")
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        GroqLLM("m", "").complete("q")
    with pytest.raises(LLMError, match="empty reply"):
        make(lambda r: ok("<think>only thoughts</think>", finish="length")).complete("q")


def test_custom_base_url_needs_no_key():
    urls = []
    llm = GroqLLM("local", "", base_url="http://localhost:11434/v1/",
                  client=httpx.Client(transport=httpx.MockTransport(
                      lambda r: (urls.append(str(r.url)), ok())[1])))
    llm.complete("q")
    assert urls == ["http://localhost:11434/v1/chat/completions"]


# ---------- JSON ----------

def test_json_mode_sets_response_format_and_adds_the_json_hint_once():
    bodies = []
    llm = make(lambda r: (bodies.append(json.loads(r.content)), ok('{"a": 1}'))[1])
    assert llm.complete_json("classify this") == {"a": 1}
    assert bodies[0]["response_format"] == {"type": "json_object"}
    assert bodies[0]["messages"][-1]["content"].endswith("Respond with a single JSON object.")
    llm.complete_json("return JSON with key a")
    assert "Respond with a single JSON object." not in bodies[1]["messages"][-1]["content"]


class Grade(BaseModel):
    score: float
    reason: str = ""


def test_complete_json_validates_with_pydantic():
    llm = make(lambda r: ok('```json\n{"score": 0.8, "reason": "relevant"}\n```'))
    g = llm.complete_json("grade it", model=Grade)
    assert isinstance(g, Grade) and g.score == 0.8


def test_complete_json_retries_once_with_the_error():
    replies = [ok("I think the score is high"), ok('{"score": 0.5}')]
    prompts = []

    def handler(req):
        prompts.append(json.loads(req.content)["messages"][-1]["content"])
        return replies.pop(0)

    g = make(handler).complete_json("grade it", model=Grade)
    assert g.score == 0.5 and len(prompts) == 2
    assert "previous reply was rejected" in prompts[1]


def test_complete_json_gives_up():
    with pytest.raises(LLMError, match="no valid JSON"):
        make(lambda r: ok('{"wrong_field": 1}')).complete_json("grade it", model=Grade)


# ---------- factory ----------

def test_build_llm_from_config(tmp_path):
    cfg = {
        "models": {"llm_default": "big-model", "llm_small": "small-model"},
        "llm": {"extra_body": {"reasoning_effort": "none"}, "temperature": 0.1, "max_tokens": 50},
    }
    s = Settings(_env_file=None, groq_api_key="k", cache_dir=tmp_path)
    big, small = build_llm("default", cfg, s), build_llm("small", cfg, s, max_tokens=9)
    assert (big.model, small.model) == ("big-model", "small-model")
    assert big.extra_body == {"reasoning_effort": "none"} and big.temperature == 0.1
    assert (big.max_tokens, small.max_tokens) == (50, 9)
    assert big.api_key == "k" and big.cache is not None
