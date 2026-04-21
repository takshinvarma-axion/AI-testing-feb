"""
Local RAG integration test (no FastAPI server required).

This script imports the core pipeline from app.py and validates that:
1) ingest() stores chunks in memory, and
2) retrieve() returns an answer + retrieved context.

Required environment variables (same as app.py):
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_API_VERSION (optional, default handled by app.py)
- AZURE_CHAT_DEPLOYMENT (optional if default deployment exists)
- AZURE_EMBED_DEPLOYMENT (optional if default embedding exists)
"""

from app import ingest, retrieve, _knowledge_base


def main() -> None:
    # Ensure a clean in-memory store for deterministic test behavior.
    _knowledge_base.clear()

    docs = [
        {
            "text": (
                "Retrieval-Augmented Generation (RAG) combines a retriever with a language "
                "model. The retriever finds relevant chunks, and the LLM answers using that context."
            ),
            "source": "rag_basics",
        },
        {
            "text": (
                "Tokenization converts text into tokens that language models process as numerical ids."
            ),
            "source": "llm_fundamentals",
        },
    ]

    ingest(docs)
    assert len(_knowledge_base) > 0, "Ingest failed: knowledge base is empty."

    result = retrieve("How does a RAG pipeline work?", top_k=2)
    assert isinstance(result, dict), "retrieve() did not return a dict."
    assert "answer" in result, "retrieve() output missing 'answer'."
    assert "context" in result, "retrieve() output missing 'context'."
    assert isinstance(result["context"], list), "'context' should be a list."
    assert len(result["context"]) > 0, "No context retrieved from local KB."

    answer_text = str(result["answer"]).strip()
    assert answer_text, "Answer is empty."

    print("Local RAG integration test passed.")


if __name__ == "__main__":
    main()
