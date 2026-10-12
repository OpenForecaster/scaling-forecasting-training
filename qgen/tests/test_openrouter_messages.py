import asyncio
import json
from unittest.mock import AsyncMock

import httpx

from qgen.inference import openrouter_inference


def test_string_and_multiturn_prompts_use_existing_openrouter_client(monkeypatch):
    monkeypatch.setattr(openrouter_inference, 'OPENROUTER_API_KEY', 'test-key')
    monkeypatch.setattr(asyncio, 'sleep', AsyncMock())
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': '<answer>1</answer>'}}]})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs))
    engine = openrouter_inference.OpenRouterInference('example/model')
    conversation = [{'role': 'user', 'content': 'Find the dates.'},
                    {'role': 'assistant', 'content': 'Initial answer.'},
                    {'role': 'user', 'content': 'Recheck the dates.'}]
    results = asyncio.run(engine.generate(['Return <answer>1</answer>', conversation], batch_size=2))
    assert len(results) == 2
    assert requests[0]['messages'] == [{'role': 'user', 'content': 'Return <answer>1</answer>'}]
    assert requests[1]['messages'] == conversation
    assert all(request['model'] == 'example/model' for request in requests)


def test_json_verifier_does_not_retry_for_answer_tags_in_input(monkeypatch):
    monkeypatch.setattr(openrouter_inference, 'OPENROUTER_API_KEY', 'test-key')
    monkeypatch.setattr(asyncio, 'sleep', AsyncMock())
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"violations": []}'}}]})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs))
    engine = openrouter_inference.OpenRouterInference('example/model', require_answer_tags=False)
    result = asyncio.run(engine.generate(['Check <answer>Ada Vale</answer>. Return JSON.']))
    assert result[0]['response'] == '{"violations": []}'
    assert len(requests) == 1
