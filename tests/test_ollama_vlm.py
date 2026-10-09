import json

import httpx
import pytest
from PIL import Image

from autolabel.vlm.base import VlmError
from autolabel.vlm.ollama import OllamaVlm


def vlm_with(handler):
    return OllamaVlm("m", "http://ollama", transport=httpx.MockTransport(handler), retries=0)


def reply(content="A", logprobs=None):
    body = {"message": {"role": "assistant", "content": content}}
    if logprobs is not None:
        body["logprobs"] = logprobs
    return httpx.Response(200, json=body)


IMG = Image.new("RGB", (40, 40), "red")


def test_probabilities_come_from_the_first_letter_logprobs():
    import math

    def handler(request):
        return reply("B", [{"token": "B", "logprob": math.log(0.6), "top_logprobs": [
            {"token": "B", "logprob": math.log(0.6)},
            {"token": "A", "logprob": math.log(0.3)},
            {"token": "The", "logprob": math.log(0.1)},        # no es una opción: se ignora
        ]}])

    probs = vlm_with(handler).choose([IMG], "q", ["uno", "dos", "tres"])
    assert probs["dos"] == pytest.approx(0.6 / 0.9)
    assert probs["uno"] == pytest.approx(0.3 / 0.9)
    assert probs["tres"] == 0.0
    assert sum(probs.values()) == pytest.approx(1.0)


def test_a_thinking_model_is_reported_instead_of_inventing_probabilities():
    # El primer token es el marcador de pensamiento y la letra solo aparece como alternativa remota
    thinking = [{"token": "<think>", "logprob": 0.0, "top_logprobs": [
        {"token": "<think>", "logprob": 0.0}, {"token": "B", "logprob": -18.0},
    ]}]
    with pytest.raises(VlmError, match="razonamiento"):
        vlm_with(lambda r: reply("", thinking)).choose([IMG], "q", ["uno", "dos"])


def test_tokens_before_the_letter_are_skipped():
    import math

    logprobs = [
        {"token": " ", "logprob": 0.0, "top_logprobs": []},
        {"token": "A", "logprob": math.log(0.8), "top_logprobs": [
            {"token": "A", "logprob": math.log(0.8)}, {"token": "B", "logprob": math.log(0.2)},
        ]},
    ]
    probs = vlm_with(lambda r: reply("A", logprobs)).choose([IMG], "q", ["uno", "dos"])
    assert probs == pytest.approx({"uno": 0.8, "dos": 0.2})


def test_without_logprobs_the_chosen_letter_gets_all_the_probability():
    probs = vlm_with(lambda r: reply("b")).choose([IMG], "q", ["uno", "dos"])
    assert probs == {"uno": 0.0, "dos": 1.0}


def test_unreadable_answer_raises():
    with pytest.raises(VlmError):
        vlm_with(lambda r: reply("no sé")).choose([IMG], "q", ["uno", "dos"])


def test_request_is_deterministic_without_thinking_and_lists_the_options():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return reply("A")

    vlm_with(handler).choose([IMG], "pregunta", ["uno", "dos"], system="sistema")
    assert seen["think"] is False and seen["options"]["temperature"] == 0
    assert seen["logprobs"] is True
    assert seen["messages"][0] == {"role": "system", "content": "sistema"}
    user = seen["messages"][1]
    assert "A. uno" in user["content"] and "B. dos" in user["content"] and len(user["images"]) == 1


def test_choose_validates_the_number_of_options():
    with pytest.raises(ValueError):
        vlm_with(lambda r: reply()).choose([IMG], "q", ["solo"])


def test_ask_json_parses_and_reports_bad_json():
    ok = vlm_with(lambda r: reply('{"verdict": "keep"}')).ask_json([IMG], "q", {"type": "object"})
    assert ok == {"verdict": "keep"}
    with pytest.raises(VlmError):
        vlm_with(lambda r: reply("no es json")).ask_json([IMG], "q", {"type": "object"})


def test_server_errors_become_vlm_errors():
    with pytest.raises(VlmError):
        vlm_with(lambda r: httpx.Response(500)).choose([IMG], "q", ["a", "b"])
