from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil

# Import core functions from your existing file
from app import ingest_pdfs, ingest, retrieve

app = FastAPI(title="RAG API", version="1.0.0")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health():
    return {"status": "healthy", "message": "RAG FastAPI is running"}


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