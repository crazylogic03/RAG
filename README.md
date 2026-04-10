# 🧠 Unified AI Platform

A production-ready platform combining **RAG (Retrieval-Augmented Generation)** and **AI Summarization** into a single FastAPI-powered web application.

## 🚀 Local Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Set Environment Variables**:
   Create a `.env` file and add:
   ```env
   GROQ_API_KEY=your_key_here
   ```
3. **Run the Server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
4. **Access**:
   - Web Hub: `http://localhost:8000`
   - API Docs: `http://localhost:8000/docs`

---

## ☁️ Deployment Guide

### Option 1: Render / Railway (Recommended)
1. **Push your code to GitHub**.
2. **Create a new Web Service** and connect your repository.
3. **Set Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set Start Command**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
5. **Add Environment Variables**:
   Add `GROQ_API_KEY` to the service's settings.

### Option 2: Docker
1. **Create a `Dockerfile`**:
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
2. **Build and Run**:
   ```bash
   docker build -t ai-hub .
   docker run -p 8000:8000 --env-file .env ai-hub
   ```

---

## 🛠️ Tech Stack
- **FastAPI**: Backend and UI hosting.
- **LangChain**: AI orchestration and RAG.
- **FAISS**: Local vector database for document search.
- **HuggingFace**: Sentence embeddings.
- **Groq**: LLaMA-3 inference.
