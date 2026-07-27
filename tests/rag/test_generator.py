import json

import httpx
import pytest

from src.rag.exceptions import GenerationError
from src.rag.generator import LLMGenerator


def test_generate_returns_answer_and_sends_correct_payload() -> None:
    captured_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(
            status_code=200,
            json={"response": "  Test cevabı  "},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        generator = LLMGenerator(
            base_url="http://fake:11434/",
            model_name="test-model",
            temperature=0.1,
            max_tokens=512,
            client=client,
        )

        answer = generator.generate(
            system_prompt="Sistem talimatı",
            user_prompt="Kullanıcı sorusu",
        )

    assert answer == "Test cevabı"
    assert captured_payload["model"] == "test-model"
    assert captured_payload["system"] == "Sistem talimatı"
    assert captured_payload["prompt"] == "Kullanıcı sorusu"
    assert captured_payload["stream"] is False
    assert captured_payload["options"] == {
        "temperature": 0.1,
        "num_predict": 512,
    }


@pytest.mark.parametrize(
    "response_body",
    [
        {"response": ""},
        {"response": "   "},
        {"response": 123},
        {},
    ],
)
def test_generate_rejects_invalid_answer(response_body: dict[str, object]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_body, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        generator = LLMGenerator(
            base_url="http://fake",
            model_name="test-model",
            client=client,
        )

        with pytest.raises(GenerationError):
            generator.generate("system", "user")


def test_generate_wraps_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"{invalid-json",
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        generator = LLMGenerator(
            base_url="http://fake",
            model_name="test-model",
            client=client,
        )

        with pytest.raises(GenerationError) as captured:
            generator.generate("system", "user")

    assert isinstance(captured.value.__cause__, ValueError)


@pytest.mark.parametrize("failure_type", ["connection", "timeout"])
def test_generate_wraps_transport_errors(failure_type: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure_type == "connection":
            raise httpx.ConnectError("connection failed", request=request)

        raise httpx.ReadTimeout("generation timed out", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        generator = LLMGenerator(
            base_url="http://fake",
            model_name="test-model",
            client=client,
        )

        with pytest.raises(GenerationError) as captured:
            generator.generate("system", "user")

    assert isinstance(captured.value.__cause__, httpx.HTTPError)


@pytest.mark.parametrize(
    (
        "base_url",
        "model_name",
        "temperature",
        "max_tokens",
        "expected_message",
    ),
    [
        ("   ", "model", 0.1, 100, "base_url cannot be empty"),
        ("http://fake", "   ", 0.1, 100, "model_name cannot be empty"),
        ("http://fake", "model", -0.1, 100, "temperature cannot be negative"),
        ("http://fake", "model", 0.1, 0, "max_tokens must be positive"),
    ],
)
def test_constructor_rejects_invalid_configuration(
    base_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        LLMGenerator(
            base_url=base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )


@pytest.mark.parametrize(
    ("system_prompt", "user_prompt", "expected_message"),
    [
        ("   ", "user", "System prompt cannot be empty"),
        ("system", "   ", "User prompt cannot be empty"),
    ],
)
def test_generate_rejects_empty_prompt_before_http_call(
    system_prompt: str,
    user_prompt: str,
    expected_message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("Empty prompt must not reach the HTTP transport.")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        generator = LLMGenerator(
            base_url="http://fake",
            model_name="test-model",
            client=client,
        )

        with pytest.raises(GenerationError, match=expected_message):
            generator.generate(system_prompt, user_prompt)
