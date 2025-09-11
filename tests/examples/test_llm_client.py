from openai import OpenAI
from rocm_rag import config
import pytest

vLLM_server_endpoint = f"{config.ROCM_RAG_LLM_API_BASE_URL}:{config.ROCM_RAG_LLM_API_PORT}/v1"
client = OpenAI(
    api_key="",  # leave empty if vLLM does not require a key
    base_url= vLLM_server_endpoint
)


@pytest.mark.parametrize("prompt,expected_output", [
    ("Write 'Hello, world!'", "Hello, world!"),
    ("What is 2 + 2? Answer with a single number.", "4"),
    ("Respond only with the word YES", "YES"),
    ("Spell the word 'cat' backwards. Respond only with the word", "tac"),
])
def test_llm_exact_match(prompt, expected_output):
    """Ensure deterministic responses match exactly expected strings."""
    response = client.chat.completions.create(
        model=config.ROCM_RAG_LLM_MODEL,   # adjust if different
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0,  # deterministic
        top_p=1
    )

    output_text = response.choices[0].message.content.strip()

    # Exact match check
    assert output_text == expected_output, f"Expected '{expected_output}', got '{output_text}'"
