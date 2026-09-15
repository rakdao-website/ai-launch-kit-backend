"""Mock-transport tests for the OpenRouter adapter."""

import asyncio
import json

import httpx
import pytest

from launchkit.adapters.llm_queue import RequestQueue
from launchkit.adapters.openrouter import OpenRouterAdapter, coerce_profile_payload, strip_code_fence
from launchkit.core.exceptions import ConfigurationError, ProviderError


def adapter_for(
    handler: httpx.MockTransport,
    *,
    attempts: int = 1,
    sleeps: list[float] | None = None,
) -> OpenRouterAdapter:
    async def sleep(delay: float) -> None:
        if sleeps is not None:
            sleeps.append(delay)

    return OpenRouterAdapter(
        httpx.AsyncClient(transport=handler),
        RequestQueue(min_gap_seconds=0),
        api_key="secret",
        max_attempts=attempts,
        sleep=sleep,
        jitter=lambda: 0.0,
    )


def response(message: dict[str, object]) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": message}]})


def test_adapter_requires_key_and_strips_supported_fences() -> None:
    with pytest.raises(ConfigurationError):
        OpenRouterAdapter(httpx.AsyncClient(), RequestQueue(), api_key="")
    assert strip_code_fence("```tsx\n<div />\n```") == "<div />"


def test_coerce_profile_payload_flattens_list_and_dict_fields() -> None:
    coerced = coerce_profile_payload(
        {
            "fields": {
                "companyName": "Velora Atelier",
                "testimonials": [
                    "The Caspian Weekender survived 18 months of DXB hops. — Maya Al-Hassan",
                    "Booked a Friday fitting. — Sara Noureddine",
                ],
                "socials": {
                    "Instagram": "@velora.atelier",
                    "LinkedIn": "linkedin.com/company/velora-atelier",
                },
            },
            "designHints": {"tagline": "Hand-finished leather", "cta": ["Book a fitting"]},
        }
    )
    fields = coerced["fields"]
    assert fields["companyName"] == "Velora Atelier"
    assert "Maya Al-Hassan" in fields["testimonials"]
    assert "Sara Noureddine" in fields["testimonials"]
    assert isinstance(fields["testimonials"], str)
    assert "Instagram: @velora.atelier" in fields["socials"]
    assert isinstance(fields["socials"], str)
    assert coerced["designHints"]["cta"] == "Book a fitting"


def test_extract_profile_fields_accepts_list_and_dict_values() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return response(
            {
                "content": json.dumps(
                    {
                        "fields": {
                            "companyName": "Velora Atelier",
                            "testimonials": ["Quote one", "Quote two"],
                            "socials": {"Instagram": "@velora.atelier"},
                        },
                        "designHints": {"tagline": "Gulf pace of life", "cta": "Book a fitting"},
                    }
                )
            }
        )

    adapter = adapter_for(httpx.MockTransport(handle))
    result = asyncio.run(adapter.extract_profile_fields("profile"))
    assert result.fields.company_name == "Velora Atelier"
    assert "Quote one" in (result.fields.testimonials or "")
    assert "Instagram: @velora.atelier" in (result.fields.socials or "")


def test_generate_text_sends_normalized_request() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response({"content": "hello"})

    adapter = adapter_for(httpx.MockTransport(handle))
    result = asyncio.run(
        adapter.generate_text("prompt", system="rules", max_tokens=50, temperature=0.2)
    )
    body = json.loads(requests[0].content)

    assert result == "hello"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert body["messages"][0] == {"role": "system", "content": "rules"}
    assert body["temperature"] == 0.2


def test_generate_json_image_label_and_profile_fields() -> None:
    replies = iter(
        [
            response({"content": '```json\n{"value": 2}\n```'}),
            response({"images": [{"image_url": {"url": "data:image/png;base64,abc"}}]}),
            response({"content": "logo"}),
            response(
                {
                    "content": '{"fields":{"companyName":"Acme"},'
                    '"designHints":{"tagline":"Clear","cta":"Call"}}'
                }
            ),
            response(
                {
                    "content": '```json\n{"fields":{"companyName":"Visual Co"},'
                    '"designHints":{"tagline":"","cta":"Visit"}}\n```'
                }
            ),
        ]
    )
    adapter = adapter_for(httpx.MockTransport(lambda _request: next(replies)))

    async def scenario() -> tuple[object, str, str, str | None, str | None]:
        payload = await adapter.generate_json("json")
        image = await adapter.generate_image("image")
        label = await adapter.label_image("data:image/png;base64,abc")
        fields = await adapter.extract_profile_fields("profile")
        visual = await adapter.extract_profile_image_fields("data:image/png;base64,abc")
        return payload, image, label, fields.fields.company_name, visual.fields.company_name

    assert asyncio.run(scenario()) == (
        {"value": 2},
        "data:image/png;base64,abc",
        "logo",
        "Acme",
        "Visual Co",
    )


def test_adapter_retries_429_using_retry_after_and_shared_queue() -> None:
    calls = 0
    sleeps: list[float] = []

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "2"},
                json={"error": {"message": "limited", "code": 429}},
            )
        return response({"content": "ok"})

    adapter = adapter_for(httpx.MockTransport(handle), attempts=2, sleeps=sleeps)

    assert asyncio.run(adapter.generate_text("prompt")) == "ok"
    assert calls == 2
    assert sleeps == [2.0]


@pytest.mark.parametrize(
    "provider_response",
    [
        httpx.Response(401, json={"error": {"message": "bad key", "code": 401}}),
        httpx.Response(200, json={"error": {"message": "bad model", "code": 400}}),
        httpx.Response(200, text="not-json"),
    ],
)
def test_adapter_normalizes_terminal_provider_errors(provider_response: httpx.Response) -> None:
    adapter = adapter_for(httpx.MockTransport(lambda _request: provider_response))

    with pytest.raises(ProviderError):
        asyncio.run(adapter.generate_text("prompt"))


def test_generate_json_retries_invalid_payload_then_succeeds() -> None:
    calls = 0
    sleeps: list[float] = []

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            # Truncated mid-string — the failure mode seen on UAT AI Summary.
            return response({"content": '{"fields":{"companyName":"Diners Fire Engi'})
        return response(
            {
                "content": (
                    '{"fields":{"companyName":"Diners Fire Engineers"},'
                    '"designHints":{"tagline":"Fire Safety That Never Waits","cta":"Explore"}}'
                )
            }
        )

    adapter = adapter_for(httpx.MockTransport(handle), attempts=3, sleeps=sleeps)
    fields = asyncio.run(adapter.extract_profile_fields("profile text"))

    assert fields.fields.company_name == "Diners Fire Engineers"
    assert fields.design_hints.tagline == "Fire Safety That Never Waits"
    assert calls == 2
    assert sleeps == [1.0]


def test_generate_json_sends_json_object_response_format() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response({"content": '{"value":1}'})

    adapter = adapter_for(httpx.MockTransport(handle))
    assert asyncio.run(adapter.generate_json("return json")) == {"value": 1}
    body = json.loads(requests[0].content)
    assert body["response_format"] == {"type": "json_object"}


def test_message_text_joins_content_parts() -> None:
    from launchkit.adapters.openrouter import message_text

    assert message_text(" plain ") == "plain"
    assert message_text([{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]) == (
        "hello\nworld"
    )


@pytest.mark.parametrize(
    "content",
    ["", "[]", "not-json"],
)
def test_generate_json_rejects_empty_non_object_or_invalid_content(content: str) -> None:
    adapter = adapter_for(httpx.MockTransport(lambda _request: response({"content": content})))

    with pytest.raises(ProviderError):
        asyncio.run(adapter.generate_json("prompt"))


def test_image_and_profile_validation_errors_are_normalized() -> None:
    replies = iter(
        [
            response({"images": []}),
            response({"images": [{}]}),
            response({"content": '{"fields":{"unknown":true},"designHints":{}}'}),
        ]
    )
    adapter = adapter_for(httpx.MockTransport(lambda _request: next(replies)))

    with pytest.raises(ProviderError):
        asyncio.run(adapter.generate_image("prompt"))
    with pytest.raises(ProviderError):
        asyncio.run(adapter.generate_image("prompt"))
    with pytest.raises(ProviderError):
        asyncio.run(adapter.extract_profile_fields("profile"))
