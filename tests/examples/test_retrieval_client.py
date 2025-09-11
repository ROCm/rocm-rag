from openai import OpenAI
from rocm_rag import config
import pytest


if config.ROCM_RAG_RETRIEVAL_FRAMEWORK == "haystack":
    model = "ROCm-RAG-Haystack"
    rag_server_chat_endpoint = f"http://localhost:{config.ROCM_RAG_HAYSTACK_SERVER_PORT}/v1"
elif config.ROCM_RAG_RETRIEVAL_FRAMEWORK == "langgraph":
    model = "ROCm-RAG-Langgraph"
    rag_server_chat_endpoint = f"http://localhost:{config.ROCM_RAG_LANGGRAPH_SERVER_PORT}/v1"
else:
    raise ValueError(f"Unsupported retrieval framework: {config.ROCM_RAG_RETRIEVAL_FRAMEWORK}")

# Configure the OpenAI client to point to your vLLM server
client = OpenAI(
    api_key="",  # leave empty if vLLM does not require a key
    base_url= rag_server_chat_endpoint # vLLM server endpoint
)


@pytest.mark.parametrize("prompt,expected_keywords", [
    ("What is ROCm-DS?", ["ROCm-DS", "data science", "AMD"])
])
def test_llm_inference(prompt, expected_keywords):    


    response = client.chat.completions.create(
        model=model,  # replace with your model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128, #compare first 50 tokens
        temperature=0,
        stream=False
    )
    print("Response:", response)
    
    # Check that the response has content
    output_text = response.choices[0].message.content.strip()
    
    # Check that the output contains expected keywords
    for kw in expected_keywords:
        assert kw.lower() in output_text.lower(), f"Missing keyword: {kw}"
