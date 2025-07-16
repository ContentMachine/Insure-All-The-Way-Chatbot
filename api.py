import requests
import os
from dotenv import load_dotenv
load_dotenv()
headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
payload = {
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [{"role": "user", "content": "What is insurance?"}],
    "max_tokens": 50
}
response = requests.post(
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions",
    headers=headers,
    json=payload
)
print(response.json())