import httpx
import asyncio

# Import OpenRouter API Key
from inference.open_router_key import API_KEY
API_URL = "https://openrouter.ai/api/v1/chat/completions"

async def async_query_openrouter(model, prompt, max_tokens=1024):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(API_URL, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()  # Raise an error for 4xx/5xx responses
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            print(f"Request failed: {e}")
        return None

# Run an async test
async def test():
    response = await async_query_openrouter("meta-llama/llama-3.1-8b-instruct", "Hello! I am meeting obama tomorrow. Can you help me with some talking points?")
    print(response)

asyncio.run(test())
