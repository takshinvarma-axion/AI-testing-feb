from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil

# Import core functions from your existing file
from app import ingest_pdfs, ingest, retrieve

app = FastAPI(title="RAG API", version="1.0.0")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
_startup_ingest_summary = "Startup ingestion not run yet."

# -----------------------------
# Request Models
# -----------------------------
class IngestDocument(BaseModel):
    text: str
    source: Optional[str] = "api"


class IngestRequest(BaseModel):
    documents: List[IngestDocument]


class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


def _ingest_existing_uploads() -> dict:
    """Ingest existing .pdf/.txt files from uploads directory."""
    files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
    pdf_paths: List[str] = []
    text_docs: List[dict] = []
    skipped: List[str] = []

    for filename in files:
        full_path = os.path.join(UPLOAD_DIR, filename)
        lower_name = filename.lower()
        if lower_name.endswith(".pdf"):
            pdf_paths.append(full_path)
        elif lower_name.endswith(".txt"):
            with open(full_path, "r", encoding="utf-8") as f:
                text_docs.append({"text": f.read(), "source": filename})
        else:
            skipped.append(filename)

    if pdf_paths:
        ingest_pdfs(pdf_paths)
    if text_docs:
        ingest(text_docs)

    return {
        "pdf_count": len(pdf_paths),
        "txt_count": len(text_docs),
        "skipped": skipped,
    }


@app.on_event("startup")
def startup_ingest_uploads() -> None:
    global _startup_ingest_summary
    result = _ingest_existing_uploads()
    _startup_ingest_summary = (
        f"Auto-ingested on startup: {result['pdf_count']} PDF(s), "
        f"{result['txt_count']} TXT file(s)."
    )
    if result["skipped"]:
        _startup_ingest_summary += f" Skipped unsupported: {', '.join(result['skipped'])}."

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health():
    return {
        "status": "healthy",
        "message": "RAG FastAPI is running",
        "startup_ingest": _startup_ingest_summary,
    }


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>RAG Chatbot</title>
      <style>
        body {
          margin: 0;
          background: #f3f4f6;
          color: #111827;
          font-family: Inter, Arial, sans-serif;
        }
        .app {
          max-width: 900px;
          height: 92vh;
          margin: 2vh auto;
          background: #ffffff;
          border: 1px solid #e5e7eb;
          border-radius: 14px;
          display: flex;
          flex-direction: column;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
        }
        .header {
          padding: 16px 18px;
          border-bottom: 1px solid #e5e7eb;
        }
        .header h1 {
          margin: 0 0 6px 0;
          font-size: 20px;
        }
        .muted {
          margin: 0;
          color: #6b7280;
          font-size: 13px;
        }
        #chat {
          flex: 1;
          overflow-y: auto;
          padding: 16px;
          background: linear-gradient(180deg, #fafafa 0%, #f9fafb 100%);
        }
        .msg {
          max-width: 80%;
          margin: 10px 0;
          padding: 10px 12px;
          border-radius: 12px;
          line-height: 1.45;
          white-space: pre-wrap;
          word-break: break-word;
        }
        .user {
          margin-left: auto;
          background: #2563eb;
          color: #ffffff;
          border-bottom-right-radius: 4px;
        }
        .bot {
          background: #ffffff;
          color: #111827;
          border: 1px solid #e5e7eb;
          border-bottom-left-radius: 4px;
        }
        .meta {
          font-size: 12px;
          color: #6b7280;
          margin-top: 6px;
        }
        .composer {
          border-top: 1px solid #e5e7eb;
          padding: 12px;
          display: grid;
          grid-template-columns: 110px 1fr 120px;
          gap: 10px;
          align-items: end;
        }
        .composer input,
        .composer textarea {
          width: 100%;
          box-sizing: border-box;
          border: 1px solid #d1d5db;
          border-radius: 10px;
          padding: 10px;
          font-size: 14px;
        }
        .composer textarea {
          min-height: 46px;
          max-height: 120px;
          resize: vertical;
        }
        .composer button {
          height: 44px;
          border: none;
          border-radius: 10px;
          background: #111827;
          color: #ffffff;
          font-weight: 600;
          cursor: pointer;
        }
        .composer button:disabled {
          opacity: 0.65;
          cursor: not-allowed;
        }
        details {
          margin-top: 8px;
        }
        summary {
          cursor: pointer;
          color: #374151;
          font-size: 13px;
        }
        .ctx {
          margin-top: 6px;
          background: #f9fafb;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          padding: 8px;
          font-size: 12px;
        }
      </style>
    </head>
    <body>
      <div class="app">
        <div class="header">
          <h1>RAG Chatbot</h1>
          <p class="muted">Knowledge is loaded from uploads/ at server startup.</p>
        </div>

        <div id="chat">
          <div class="msg bot">
            Hello! Ask anything about your ingested documents.
          </div>
        </div>

        <div class="composer">
          <input id="topK" type="number" min="1" value="3" title="Top K chunks" />
          <textarea id="query" placeholder="Ask a question..." onkeydown="onKey(event)"></textarea>
          <button id="askBtn" onclick="askQuestion()">Send</button>
        </div>
      </div>

      <script>
        function addMessage(text, role, context = []) {
          const chat = document.getElementById("chat");
          const bubble = document.createElement("div");
          bubble.className = `msg ${role}`;
          bubble.textContent = text;

          if (role === "bot" && context.length) {
            const details = document.createElement("details");
            const summary = document.createElement("summary");
            summary.textContent = `Show retrieved context (${context.length})`;
            details.appendChild(summary);

            const ctx = document.createElement("div");
            ctx.className = "ctx";
            ctx.textContent = context.join("\\n\\n----------------\\n\\n");
            details.appendChild(ctx);
            bubble.appendChild(details);
          }

          chat.appendChild(bubble);
          chat.scrollTop = chat.scrollHeight;
        }

        function onKey(event) {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            askQuestion();
          }
        }

        async function askQuestion() {
          const btn = document.getElementById("askBtn");
          const queryInput = document.getElementById("query");
          const query = queryInput.value.trim();
          const topK = Number(document.getElementById("topK").value || 3);

          if (!query) {
            return;
          }

          addMessage(query, "user");
          queryInput.value = "";
          btn.disabled = true;
          addMessage("Thinking...", "bot");
          const chat = document.getElementById("chat");
          const typingBubble = chat.lastElementChild;

          try {
            const response = await fetch("/retrieve", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ query, top_k: topK })
            });
            const data = await response.json();
            if (!response.ok) {
              throw new Error(data.detail || "Retrieve failed");
            }

            const answer = data.answer || {};
            const ctx = Array.isArray(answer.context) ? answer.context : [];
            typingBubble.remove();
            addMessage(answer.answer || "No answer returned.", "bot", ctx);
          } catch (err) {
            typingBubble.remove();
            addMessage("Error: " + err.message, "bot");
          } finally {
            btn.disabled = false;
            queryInput.focus();
          }
        }
      </script>
    </body>
    </html>
    """


# -----------------------------
# Ingest File Upload Endpoint
# -----------------------------
@app.post("/ingest")
async def ingest_files(files: List[UploadFile] = File(...)):
    try:
        pdf_paths = []
        text_docs = []

        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)

            # Save uploaded file locally
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Handle PDF files
            if file.filename.lower().endswith(".pdf"):
                pdf_paths.append(file_path)

            # Handle text files
            elif file.filename.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text_docs.append({
                        "text": f.read(),
                        "source": file.filename
                    })

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}"
                )

        # Ingest PDFs
        if pdf_paths:
            ingest_pdfs(pdf_paths)

        # Ingest text docs
        if text_docs:
            ingest(text_docs)

        return {
            "status": "success",
            "message": f"{len(files)} file(s) ingested successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Retrieve Endpoint
# -----------------------------
@app.post("/retrieve")
def retrieve_documents(request: RetrieveRequest):
    try:
        answer = retrieve(
            query=request.query,
            top_k=request.top_k
        )

        return {
            "status": "success",
            "query": request.query,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))