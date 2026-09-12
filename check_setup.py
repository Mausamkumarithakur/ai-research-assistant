"""
Run this BEFORE `streamlit run app.py` to confirm your setup is correct.

    python check_setup.py

It checks:
    1. Is Ollama running?
    2. Is the 'llama3' model pulled?
    3. Is the 'nomic-embed-text' embedding model pulled?
    4. Can it actually generate a response and an embedding?
"""

import sys
import requests

OLLAMA_BASE = "http://localhost:11434"
REQUIRED_MODELS = ["llama3", "nomic-embed-text"]


def check_ollama_running() -> bool:
    try:
        requests.get(OLLAMA_BASE, timeout=5)
        return True
    except requests.exceptions.ConnectionError:
        return False


def get_installed_models() -> list[str]:
    response = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    response.raise_for_status()
    return [m["name"].split(":")[0] for m in response.json().get("models", [])]


def test_generation() -> bool:
    response = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": "llama3", "prompt": "Say OK", "stream": False},
        timeout=60,
    )
    return response.status_code == 200 and "response" in response.json()


def test_embedding() -> bool:
    response = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": "test"},
        timeout=60,
    )
    return response.status_code == 200 and "embedding" in response.json()


def main():
    print("Checking Ollama setup...\n")

    if not check_ollama_running():
        print("❌ Ollama is not running.")
        print("   Fix: open a terminal and run: ollama serve")
        sys.exit(1)
    print("✅ Ollama is running.")

    installed = get_installed_models()
    missing = [m for m in REQUIRED_MODELS if m not in installed]
    if missing:
        print(f"❌ Missing model(s): {', '.join(missing)}")
        for m in missing:
            print(f"   Fix: ollama pull {m}")
        sys.exit(1)
    print(f"✅ Required models installed: {', '.join(REQUIRED_MODELS)}")

    print("Testing generation (llama3)...")
    if not test_generation():
        print("❌ Generation test failed.")
        sys.exit(1)
    print("✅ Generation works.")

    print("Testing embeddings (nomic-embed-text)...")
    if not test_embedding():
        print("❌ Embedding test failed.")
        sys.exit(1)
    print("✅ Embeddings work.")

    print("\nAll checks passed. You're good to run: streamlit run app.py")


if __name__ == "__main__":
    main()
