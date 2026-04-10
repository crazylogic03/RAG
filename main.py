import os
import shutil
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

load_dotenv()

app = FastAPI(title="RAG + Summarizer Application")

# CORS — allow Streamlit frontend to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

FAISS_PATH = Path("faiss_index")

vector_store = None
query_cache: dict = {}


# -------- REMOVED STARTUP BLOCK --------


# =====================================================================
#  HELPERS
# =====================================================================

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_document(file_path: str):
    """Load a PDF or TXT file and return LangChain Documents."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Only .pdf and .txt files are supported.")
    return loader.load()


def build_vector_store(documents):
    """Chunk documents, embed them, persist FAISS index, return store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    store = FAISS.from_documents(chunks, embeddings)
    store.save_local("faiss_index")
    return store


def get_qa_chain(store, k: int = 3):
    """Build a RetrievalQA chain with a PromptTemplate."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""",
    )

    retriever = store.as_retriever(search_kwargs={"k": k})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain


# =====================================================================
#  SUMMARIZATION HELPERS
# =====================================================================

SUMMARY_TEMPLATES = {
    "Brief": PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text briefly:\n\n{text}",
    ),
    "Detailed": PromptTemplate(
        input_variables=["text"],
        template="Provide a detailed summary of the following text:\n\n{text}",
    ),
    "Bullet Points": PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in bullet points:\n\n{text}",
    ),
}


def generate_summary(text: str, summary_type: str) -> str:
    """Generate a summary using ChatGroq and a PromptTemplate."""
    if summary_type not in SUMMARY_TEMPLATES:
        raise ValueError(
            f"Invalid summary_type '{summary_type}'. "
            f"Choose from: {list(SUMMARY_TEMPLATES.keys())}"
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
    )

    prompt_template = SUMMARY_TEMPLATES[summary_type]
    formatted_prompt = prompt_template.format(text=text)

    response = llm.invoke(formatted_prompt)
    return response.content


# =====================================================================
#  REQUEST / RESPONSE MODELS
# =====================================================================

class QueryRequest(BaseModel):
    question: str
    k: int = 3


class SummarizeRequest(BaseModel):
    text: str
    summary_type: str = "Brief"  # Brief | Detailed | Bullet Points


# =====================================================================
#  ROUTES
# =====================================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = Path("static/index.html")
    if html_path.exists():
        return html_path.read_text()
    return "<h1>AI Hub</h1><p>Welcome to the Unified Document Intelligence Hub.</p>"


@app.get("/rag", response_class=HTMLResponse)
async def rag_ui():
    html_path = Path("static/rag.html")
    if html_path.exists():
        return html_path.read_text()
    return "<h1>RAG Application</h1><p>rag.html not found.</p>"


@app.get("/summarizer", response_class=HTMLResponse)
async def summarizer_ui():
    html_path = Path("static/summarizer.html")
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Summarizer Application</h1><p>summarizer.html not found.</p>"


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files allowed")

    safe_name = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_name

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        documents = load_document(str(file_path))
        new_store = build_vector_store(documents)

        if vector_store:
            vector_store.merge_from(new_store)
            vector_store.save_local("faiss_index")
        else:
            vector_store = new_store

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": f"'{safe_name}' uploaded and indexed successfully.",
        "pages": len(documents),
    }


@app.post("/query")
async def query_document(req: QueryRequest):
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a document first.",
        )

    key = req.question.strip().lower()
    if key in query_cache:
        return query_cache[key]

    try:
        chain = get_qa_chain(vector_store, k=min(req.k, 10))
        result = chain.invoke({"query": req.question})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "content": doc.page_content[:300],
            "metadata": doc.metadata,
        })

    response = {
        "answer": result["result"],
        "sources": sources,
    }

    query_cache[key] = response
    return response


@app.get("/documents")
async def list_documents():
    files = sorted([f.name for f in UPLOAD_DIR.iterdir() if f.is_file()])
    return {"documents": files}


@app.post("/summarize")
async def summarize_text(req: SummarizeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        summary = generate_summary(req.text.strip(), req.summary_type)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return {"summary": summary, "summary_type": req.summary_type}