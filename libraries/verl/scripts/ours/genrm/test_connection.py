#!/usr/bin/env python3
"""
Simple script to test vLLM server connection without proxy interference.
"""

import os
import requests
import json

# Disable all proxy settings
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    if key in os.environ:
        del os.environ[key]

def test_vllm_connection():
    """Test connection to vLLM server."""
    base_url = "http://127.0.0.1:30000"
    
    # Create session with no proxy
    session = requests.Session()
    session.trust_env = False
    session.proxies = {}
    
    try:
        # Test models endpoint
        print("Testing models endpoint...")
        response = session.get(f"{base_url}/v1/models", timeout=10)
        print(f"Models endpoint status: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"Available models: {[m.get('id') for m in models.get('data', [])]}")
        else:
            print(f"Error response: {response.text}")
            return False
        
        # Test chat completions endpoint
        print("\nTesting chat completions endpoint...")
        test_data = {
            "model": "qwen3-4b-non-think",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        
        response = session.post(
            f"{base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=30
        )
        
        print(f"Chat completions status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("Chat completions working!")
            print(f"Response: {result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
            return True
        else:
            print(f"Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Connection error: {e}")
        return False

if __name__ == "__main__":
    if test_vllm_connection():
        print("\n✅ vLLM server is working correctly!")
    else:
        print("\n❌ vLLM server connection failed!")
        print("Troubleshooting steps:")
        print("1. Check if vLLM server is running: ps aux | grep vllm")
        print("2. Check if port 30000 is open: netstat -tlnp | grep 30000")
        print("3. Try direct curl: curl --noproxy '*' http://127.0.0.1:30000/v1/models") 