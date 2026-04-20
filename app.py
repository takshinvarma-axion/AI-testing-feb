"""
Azure OpenAI RAG (Retrieval-Augmented Generation) - Single File Implementation
===============================================================================
PUBLIC API — two ways to populate the knowledge base, one way to chat:

  ingest(documents)       → add plain-text document dicts
  ingest_pdfs(pdf_paths)  → add one or more PDF file paths   ← NEW
  retrieve(query)         → semantic search + GPT-4o grounded answer

Requirements:
    pip install openai pypdf python-dotenv

Environment variables (.env or shell):
    AZURE_OPENAI_ENDPOINT      = https://<your-resource>.openai.azure.com/
    AZURE_OPENAI_API_KEY       = <your-api-key>
    AZURE_OPENAI_API_VERSION   = 2024-02-01          (default)
    AZURE_EMBED_DEPLOYMENT     = text-embedding-ada-002
    AZURE_CHAT_DEPLOYMENT      = gpt-4o
    RAG_TOP_K                  = 3                   (chunks returned per query)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO INGEST DOCUMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── Option A: Ingest PDF files (NEW) ─────────────────────────

  from rag_azure import ingest_pdfs, retrieve

  # Single PDF
  ingest_pdfs(["company_policy.pdf"])

  # Multiple PDFs at once
  ingest_pdfs([
      "reports/Q1_2024.pdf",
      "reports/Q2_2024.pdf",
      "manuals/user_guide.pdf",
  ])

  answer = retrieve("What is the refund policy?")
  print(answer)

── Option B: Ingest plain-text document dicts ───────────────

  from rag_azure import ingest, retrieve

  ingest([
      {"text": "Azure OpenAI supports GPT-4o ...", "source": "azure_docs.txt"},
      {"text": "RAG combines retrieval with generation ...", "source": "paper.txt"},
  ])

  answer = retrieve("What models does Azure OpenAI support?")
  print(answer)

── Option C: Mix PDFs and text dicts in one session ─────────

  from rag_azure import ingest, ingest_pdfs, retrieve

  ingest_pdfs(["legal_terms.pdf", "faq.pdf"])
  ingest([{"text": "Extra context from internal wiki ...", "source": "wiki"}])

  answer = retrieve("What are the legal terms for cancellation?")
  print(answer)

── Option D: CLI — PDF files ────────────────────────────────

  python rag_azure.py --pdf report.pdf manual.pdf

── Option E: CLI — text/JSON files ──────────────────────────

  python rag_azure.py --ingest notes.txt data.json

── Option F: CLI — mix both ─────────────────────────────────

  python rag_azure.py --pdf guide.pdf --ingest extra.txt

── Option G: CLI — quick smoke-test ─────────────────────────

  python rag_azure.py --demo

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
import math
import textwrap
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI

# pypdf is only required when using ingest_pdfs(); imported lazily inside that function.

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()


client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)

EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-ada-002")
CHAT_DEPLOYMENT  = os.getenv("AZURE_CHAT_DEPLOYMENT",  "gpt-4o")
TOP_K            = int(os.getenv("RAG_TOP_K", "3"))

# ─────────────────────────────────────────────────────────────────────────────
# In-memory vector store  (swap for Azure AI Search / Cosmos DB in production)
# ─────────────────────────────────────────────────────────────────────────────

_knowledge_base: list[dict[str, Any]] = []


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    """Call the Azure OpenAI Embeddings API and return the vector."""
    response = client.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input=text.replace("\n", " "),
    )
    return response.data[0].embedding


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Pure-Python cosine similarity — no numpy dependency."""
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-level chunks so each fits inside the
    embedding model's context window.

    Parameters
    ----------
    text       : Raw text to split.
    chunk_size : Max words per chunk (500 words ≈ 750 tokens — safe for ada-002).
    overlap    : Words shared between consecutive chunks for context continuity.
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def _extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all readable text from a PDF file using pypdf.

    Parameters
    ----------
    pdf_path : Path to the PDF file (absolute or relative).

    Returns
    -------
    str : Concatenated text from every page, separated by newlines.
          Returns an empty string and prints a warning on failure.

    Notes
    -----
    Scanned / image-only PDFs yield no text. Pre-process them with an OCR
    tool (e.g. pdf2image + pytesseract) before calling this function.
    """
    try:
        from pypdf import PdfReader          # lazy import — only when PDFs are used
    except ImportError:
        raise ImportError(
            "pypdf is required to read PDF files.\n"
            "Install it with:  pip install pypdf"
        )

    if not os.path.exists(pdf_path):
        print(f"[pdf] ⚠️  File not found: {pdf_path}")
        return ""

    reader     = PdfReader(pdf_path)
    page_texts = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            if text.strip():
                page_texts.append(text)
        except Exception as exc:
            print(f"[pdf] ⚠️  Could not read page {page_num} of '{pdf_path}': {exc}")

    full_text = "\n".join(page_texts)

    if not full_text.strip():
        print(
            f"[pdf] ⚠️  No extractable text found in '{pdf_path}'.\n"
            "       The file may be scanned/image-only. "
            "Consider using an OCR tool first."
        )

    return full_text


def _ingest_document_list(documents: list[dict[str, str]]) -> int:
    """
    Shared embedding + storage logic used by both ingest() and ingest_pdfs().
    Returns the number of chunks added to the knowledge base.
    """
    total_chunks = 0

    for doc in documents:
        if "text" not in doc:
            print(f"[ingest] ⚠️  Skipping entry without 'text' key: {doc}")
            continue

        raw_text = doc["text"].strip()
        if not raw_text:
            print(f"[ingest] ⚠️  Skipping empty document: {doc.get('source', '?')}")
            continue

        metadata = {k: v for k, v in doc.items() if k != "text"}
        chunks   = _chunk_text(raw_text)

        for i, chunk in enumerate(chunks):
            print(
                f"[ingest] Embedding chunk {i + 1}/{len(chunks)} "
                f"from '{metadata.get('source', 'unknown')}' ..."
            )
            embedding = _embed(chunk)
            _knowledge_base.append({
                "text":      chunk,
                "embedding": embedding,
                "metadata":  {**metadata, "chunk_index": i},
            })
            total_chunks += 1

    return total_chunks


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def ingest(documents: list[dict[str, str]]) -> None:
    """
    Embed and store plain-text document dicts in the knowledge base.

    Parameters
    ----------
    documents : list of dicts. Each dict must have:
        "text"    (required) — raw content to embed and index.
        "source"  (optional) — label used in answer citations.
        Any other keys are preserved as metadata.

    Example
    -------
    ingest([
        {"text": "Azure OpenAI supports GPT-4o ...", "source": "azure_docs.txt"},
        {"text": "RAG combines retrieval with generation ...", "source": "paper.txt"},
    ])
    """
    if not documents:
        print("[ingest] No documents provided.")
        return

    total_chunks = _ingest_document_list(documents)

    print(
        f"\n[ingest] ✓ Done — {len(documents)} document(s) → {total_chunks} chunk(s) added. "
        f"Knowledge base total: {len(_knowledge_base)} chunks.\n"
    )


def ingest_pdfs(pdf_paths: list[str]) -> None:
    """
    Extract text from PDF files and embed them into the knowledge base.
    Follows the exact same chunking and embedding pipeline as ingest().

    Requires:  pip install pypdf

    Parameters
    ----------
    pdf_paths : List of paths to .pdf files (absolute or relative).

    Notes
    -----
    • Text is extracted page-by-page via pypdf, concatenated, then split
      into overlapping 500-word chunks before embedding — identical to the
      plain-text ingest() pipeline.
    • The "source" metadata for every chunk is set to the PDF filename so
      answers will cite the correct file (e.g. "user_guide.pdf [2]").
    • Scanned / image-only PDFs yield no text. Run them through an OCR
      tool first (e.g. pdf2image + pytesseract) if needed.

    Example
    -------
    ingest_pdfs([
        "reports/Q1_2024.pdf",
        "manuals/user_guide.pdf",
    ])
    """
    if not pdf_paths:
        print("[ingest_pdfs] No PDF paths provided.")
        return

    docs: list[dict[str, str]] = []

    for path in pdf_paths:
        print(f"[ingest_pdfs] Reading '{path}' ...")
        text = _extract_text_from_pdf(path)
        if text.strip():
            docs.append({
                "text":   text,
                "source": os.path.basename(path),   # e.g. "user_guide.pdf"
                "type":   "pdf",
            })
        else:
            print(f"[ingest_pdfs] ⚠️  Skipping '{path}' — no text extracted.")

    if not docs:
        print("[ingest_pdfs] No content could be extracted from the provided PDFs.\n")
        return

    total_chunks = _ingest_document_list(docs)

    print(
        f"\n[ingest_pdfs] ✓ Done — {len(docs)} PDF(s) → {total_chunks} chunk(s) added. "
        f"Knowledge base total: {len(_knowledge_base)} chunks.\n"
    )


def retrieve(
    query: str,
    top_k: int = TOP_K,
    chat_history: list[dict] | None = None
) -> dict:
    """
    Semantic search over the knowledge base and return
    structured JSON response with answer + context.
    """

    if not _knowledge_base:
        return {
            "answer": "Knowledge base is empty. Please ingest documents first.",
            "context": []
        }

    # 1. Embed query
    print("[retrieve] Embedding query ...")
    query_vec = _embed(query)

    # 2. Similarity scoring
    scored = [
        (chunk, _cosine_similarity(query_vec, chunk["embedding"]))
        for chunk in _knowledge_base
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_chunks = scored[:top_k]

    print(
        f"[retrieve] Top-{top_k} chunks retrieved "
        f"(scores: {[round(s, 4) for _, s in top_chunks]})"
    )

    # 3. Build context list
    context_list = []
    context_parts = []

    for rank, (chunk, score) in enumerate(top_chunks, 1):
        source = chunk["metadata"].get("source", "unknown")

        chunk_text = (
            f"[{rank}] (source: {source}, score: {score:.4f})\n"
            f"{chunk['text']}"
        )

        context_list.append(chunk_text)
        context_parts.append(chunk_text)

    context = "\n\n---\n\n".join(context_parts)

    # 4. Prompt
    system_prompt = textwrap.dedent("""
        You are a helpful assistant.
        Answer ONLY from the given context.

        If answer is not found, say:
        "I don't have enough information to answer that."

        Context:
        {context}
    """).strip().format(context=context)

    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": query})

    # 5. Generate answer
    print(f"[retrieve] Calling {CHAT_DEPLOYMENT} ...")

    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "context": context_list
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_text_files(file_paths: list[str]) -> list[dict[str, str]]:
    """Load .txt and .json files into document dicts for ingest()."""
    docs: list[dict[str, str]] = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"[load] ⚠️  File not found: {path}")
            continue
        if path.lower().endswith(".json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            docs.extend(data) if isinstance(data, list) else docs.append(data)
        else:
            with open(path, encoding="utf-8") as f:
                docs.append({"text": f.read(), "source": os.path.basename(path)})
    return docs


def _interactive_chat() -> None:
    """Simple multi-turn REPL for chatting with the loaded knowledge base."""
    print("\n" + "=" * 62)
    print("  Azure OpenAI RAG — Interactive Chat")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 62 + "\n")

    chat_history: list[dict] = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] Goodbye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("[chat] Goodbye!")
            break

        answer = retrieve(query, chat_history=chat_history)
        print(f"\nAssistant: {answer}\n")

        chat_history.append({"role": "user",      "content": query})
        chat_history.append({"role": "assistant",  "content": answer})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Azure OpenAI RAG — ingest PDFs / text files then chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python rag_azure.py --pdf report.pdf guide.pdf
              python rag_azure.py --ingest notes.txt data.json
              python rag_azure.py --pdf manual.pdf --ingest extra.txt
              python rag_azure.py --demo
        """),
    )
    parser.add_argument(
        "--pdf", nargs="+", metavar="PDF_FILE",
        help="Path(s) to PDF file(s) to ingest into the knowledge base",
    )
    parser.add_argument(
        "--ingest", nargs="+", metavar="FILE",
        help="Path(s) to .txt or .json file(s) to ingest into the knowledge base",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Seed built-in demo documents and start chatting (no files required)",
    )
    args = parser.parse_args()

    anything_loaded = False

    # ── PDF ingestion ─────────────────────────────────────────────────────────
    if args.pdf:
        ingest_pdfs(args.pdf)
        anything_loaded = True

    # ── Text / JSON file ingestion ────────────────────────────────────────────
    if args.ingest:
        docs = _load_text_files(args.ingest)
        if docs:
            ingest(docs)
            anything_loaded = True
        else:
            print("[main] ⚠️  No content loaded from specified text files.")

    # ── Built-in demo content ─────────────────────────────────────────────────
    if args.demo:
        ingest([
            {
                "text": (
                    "Azure OpenAI Service provides REST API access to OpenAI's powerful "
                    "language models including GPT-4o, GPT-4, GPT-35-Turbo, and Embeddings. "
                    "It is fully managed, enterprise-ready, and integrates with Azure's "
                    "security and compliance features."
                ),
                "source": "azure_openai_overview",
            },
            {
                "text": (
                    "Retrieval-Augmented Generation (RAG) combines information retrieval with "
                    "language model generation. A user query is embedded, matched against a "
                    "vector store, and the top-k relevant chunks are injected into the prompt "
                    "as grounding context before the LLM generates an answer. This reduces "
                    "hallucinations and keeps responses factual."
                ),
                "source": "rag_explainer",
            },
            {
                "text": (
                    "Azure AI Search supports vector search, hybrid search, and semantic "
                    "ranking. It is the recommended vector store for production RAG pipelines "
                    "on Azure, used alongside Azure OpenAI Embeddings for indexing and retrieval."
                ),
                "source": "azure_ai_search",
            },
            {
                "text": (
                    "text-embedding-ada-002 produces 1,536-dimensional vectors and is the "
                    "standard embedding model for RAG on Azure OpenAI. Newer options like "
                    "text-embedding-3-small and text-embedding-3-large offer improved "
                    "performance at lower cost."
                ),
                "source": "embedding_models",
            },
        ])
        anything_loaded = True

    if not anything_loaded:
        print(
            "\nNo documents specified. Usage:\n\n"
            "  python rag_azure.py --pdf  file1.pdf file2.pdf\n"
            "  python rag_azure.py --ingest notes.txt data.json\n"
            "  python rag_azure.py --pdf guide.pdf --ingest extra.txt\n"
            "  python rag_azure.py --demo\n"
        )
        raise SystemExit(0)

    _interactive_chat()