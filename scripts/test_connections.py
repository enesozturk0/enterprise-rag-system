"""
scripts/test_ollama.py

Simple smoke test that verifies the Ollama LLM (generate) and
embedding endpoints are working correctly.

Usage:
    python scripts/test_ollama.py
"""

import sys

import httpx

sys.path.insert(0, ".")

from src.config.settings import get_settings

settings = get_settings()


def test_generate() -> bool:
    """Tests the /api/generate endpoint."""
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": settings.llm_model_name,
        "prompt": "Merhaba, sadece 'ok' yaz.",
        "stream": False,
    }

    print(f"Testing LLM: {url} (model: {settings.llm_model_name})")
    try:
        response = httpx.post(url, json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        text = data.get("response", "")
        if not text.strip():
            print("LLM test failed: empty response received")
            return False

        print(f"LLM test passed — response: {text.strip()[:80]!r}")
        return True

    except httpx.ConnectError:
        print(
            f"LLM test failed: could not connect to {settings.ollama_base_url} (is Ollama running?)"
        )
        return False
    except httpx.HTTPStatusError as e:
        print(f"LLM test failed: HTTP {e.response.status_code} — {e.response.text[:200]}")
        return False
    except Exception as e:
        print(f"LLM test failed: {type(e).__name__}: {e}")
        return False


def test_embed() -> bool:
    """Tests the /api/embed endpoint and validates the embedding dimension."""
    url = f"{settings.ollama_base_url}/api/embed"
    payload = {
        "model": settings.embedding_model_name,
        "input": "Bu bir test cümlesidir.",
    }

    print(f"Testing embeddings: {url} (model: {settings.embedding_model_name})")
    try:
        response = httpx.post(url, json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        # Ollama /api/embed response format: {"embeddings": [[...]]}
        embeddings = data.get("embeddings")
        if not embeddings or not embeddings[0]:
            print("Embedding test failed: no embedding vector returned")
            return False

        vector = embeddings[0]
        dim = len(vector)

        if dim != 1024:
            print(f"Embedding test failed: dimension {dim}, expected 1024")
            return False

        print(f"Embedding test passed — dimension: {dim}")
        return True

    except httpx.ConnectError:
        print(f"Embedding test failed: could not connect to {settings.ollama_base_url}")
        return False
    except httpx.HTTPStatusError as e:
        print(f"Embedding test failed: HTTP {e.response.status_code} — {e.response.text[:200]}")
        return False
    except Exception as e:
        print(f"Embedding test failed: {type(e).__name__}: {e}")
        return False


def main() -> None:
    print(f"Ollama base URL: {settings.ollama_base_url}\n")

    llm_ok = test_generate()
    print()
    embed_ok = test_embed()

    print("\n--- Summary ---")
    print(f"LLM       : {'PASS' if llm_ok else 'FAIL'}")
    print(f"Embedding : {'PASS' if embed_ok else 'FAIL'}")

    if not (llm_ok and embed_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
