import os
import json
import logging
import asyncio
from typing import List, Dict, Any
from tqdm import tqdm
import httpx

logger = logging.getLogger(__name__)

# Import OpenRouter API Key if available
try:
    from qgen.config.openrouter_key import API_KEY as OPENROUTER_API_KEY
except ImportError:
    # Fallback to environment variable
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    if not OPENROUTER_API_KEY:
        logger.warning("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable or create config/openrouter_key.py with API_KEY variable")

from qgen.inference.base_inference import BaseInference

class OpenRouterInference(BaseInference):
    """OpenRouter-based inference engine"""
    
    def __init__(self, model: str, max_tokens: int = 2048, temperature: float = 0.7):
        """
        Initialize OpenRouter inference engine
        
        Args:
            model: Model name on OpenRouter
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
        """
        self.model = model
        self.max_tokens = max_tokens
        
        if "r1" in model or "qwen3" in model or "openai" in model or "2.5" in model or "pro" in model or "flash" in model or "grok" in model or "kimi" in model:
            self.temperature = 0.6
        else:
            self.temperature = temperature
        
        if not OPENROUTER_API_KEY:
            raise ValueError("OpenRouter API key not found")
            
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate completions for prompts using OpenRouter
        
        Args:
            prompts: List of prompts to generate completions for
            **kwargs: Additional arguments for generation
            
        Returns:
            List of generated completions
        """
        batch_size = kwargs.get('batch_size', 5)
        results = []
        
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[i:i+batch_size]
            tasks = []
            
            for prompt in batch:
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                }
                
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "reasoning": {
                        "effort": "high",
                    },
                    "max_tokens": self.max_tokens,
                }
                
                tasks.append(self._make_request(headers, data, prompt))
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Sleep to avoid rate limits
            await asyncio.sleep(1)
    
        return results
    
    async def _make_request(self, headers, data, prompt = None):
        """Make a request to the OpenRouter API with retries."""
        max_retries = 5
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                # Create a client for each request with an appropriate timeout
                async with httpx.AsyncClient(timeout=900.0) as client:
                    response = await client.post(self.api_url, headers=headers, json=data)
                    response.raise_for_status()

                    try:
                        resp_json = response.json()
                    except json.decoder.JSONDecodeError:
                        logger.error(f"Failed to decode JSON. Response text: {response.text}")
                        # Consider delaying before retry if JSON decode fails
                        await asyncio.sleep(retry_delay)
                        continue  # Go to next attempt

                    message = resp_json.get("choices", [{}])[0].get("message", {})
                    content = message.get("content")
                    answer_tags_required = False
                    if prompt:
                        answer_tags_required = "<answer>" in prompt
                    
                    if content is not None and len(content) > 0:  # Check if content is actually present
                        reasoning = message.get("reasoning", "")
                        finish_reason = resp_json.get("choices", [{}])[0].get("finish_reason", "")
                        usage = resp_json.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        # print("Response: ", content)
                        result = {
                            "response": content,
                            "finish_reason": finish_reason,
                            "prompt_tokens": prompt_tokens,
                            "reasoning": reasoning,
                            "completion_tokens": completion_tokens
                        }
                        
                        if answer_tags_required:
                            if "<answer>" in content and "</answer>" in content:
                                return result
                            else:
                                if attempt >= 2:
                                    return result
                                logger.warning(f"Received successful response but no <answer> tags for model {data.get('model')}. Response: {resp_json}")
                        else:
                            return result
                    
                    # if content is not None:  # Check if content is actually present
                    #     if reasoning and len(reasoning) > 5:  # Only add reasoning if it's long enough
                    #         content = f"{reasoning}\n{content}"
                    #     return content
                    
                    else:
                        logger.warning(f"Received successful response but no content for model {data.get('model')}. Response: {resp_json}")
                        # Treat as failure if content is None or empty after success
                        # Fall through to retry logic
                        if resp_json["error"]["code"] == 429:
                            logger.warning(f"Rate limit hit. Retrying after 60 seconds...")
                            await asyncio.sleep(60)

            except httpx.HTTPStatusError as e:
                # Specific handling for rate limits or server errors if needed
                if e.response.status_code == 429:
                    logger.warning(f"Rate limit hit. Retrying after longer delay...")
                    await asyncio.sleep(retry_delay * (attempt + 2))  # Exponential backoff
                elif e.response.status_code >= 500:
                    logger.warning(f"Server error ({e.response.status_code}). Retrying...")
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                    # Don't retry for client errors like 400 Bad Request immediately
                    if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                        logger.error("Client error detected, not retrying.")
                        break  # Exit retry loop for client errors

            except httpx.RequestError as e:
                logger.error(f"Request failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during API request: {str(e)}")

            # Wait before retrying, unless it was the last attempt or a non-retriable error
            if attempt < max_retries - 1:
                logger.info(f"Retrying request (attempt {attempt + 2}/{max_retries})...")
                await asyncio.sleep(retry_delay * (attempt + 1))  # Increase delay slightly

        # If all retries failed or a non-retriable error occurred, return None
        logger.error(f"Failed to get completion for prompt after {max_retries} attempts.")
        return None