<div align="center">

# Agentic Adaptive RAG

**A self-correcting, security-aware Retrieval-Augmented Generation system with multimodal image understanding.**

Built with LangGraph · FastAPI · Next.js · ChromaDB · Docker

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C)
![Next.js](https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-33%20passed-brightgreen)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Security](#security)
- [Evaluation & Observability](#evaluation--observability)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Running with Docker](#running-with-docker)
- [API](#api)
- [Testing](#testing)
- [Roadmap](#roadmap)

---

## Overview

Most RAG pipelines follow a fixed path: embed the question, search a vector store, and pass whatever comes back to an LLM. **Agentic Adaptive RAG** replaces that fixed path with a stateful agent that makes decisions at every step:

- **Routes** each question to local document retrieval or web search.
- **Grades** retrieved documents for relevance before using them.
- **Checks** generated answers for hallucinations and for whether they actually answer the question.
- **Retries** generation when a quality gate fails.
- **Protects** the pipeline against prompt injection, jailbreaks and secret leakage.
- **Understands images**: upload an image and ask questions grounded in what the vision model actually sees.
- **Records** evaluation and observability data for every run.

## Features

| Area | Capability |
|---|---|
| Adaptive retrieval | Query router chooses between the local vector store and web search |
| Quality gates | Retrieval grader, hallucination grader and answer grader with automatic retries |
| Human-in-the-loop | Optional approval step before any web search is executed |
| Image understanding | Vision-model analysis with OCR as supporting evidence; optional related-image search |
| Document ingestion | PDF, DOCX, TXT and Markdown uploads, chunked and stored in ChromaDB |
| Security | Prompt-injection, jailbreak and document-injection detection; output redaction |
| Evaluation | Benchmark runner and RAGAS-based evaluation with generated reports |
| Observability | Per-run route, latency, retries, grounding and source metrics (LangSmith supported) |
| Deployment | Dockerfiles and Docker Compose for backend and frontend |

## How It Works

```text
                    User Question
                         │
                         ▼
                  Security Checks
                         │
                         ▼
                   Query Router
                    /         \
                   ▼           ▼
          Local Retrieval     Web Search ◄── (human approval)
                   │              │
                   ▼              │
           Retrieval Grader       │
                   │              │
        relevant ──┴── not relevant
              │                   │
              └─────────┬─────────┘
                        ▼
                    Generation
                        │
                        ▼
              Hallucination Grader
                        │
                        ▼
                  Answer Grader
                        │
                 ┌──────┴──────┐
               Pass          Fail
                 │             │
                 ▼             ▼
         Output Security    Retry ──► Generation
                 │
                 ▼
              Response
```

1. **Query routing** – decides whether uploaded knowledge is enough or external information is needed.
2. **Retrieval grading** – discards irrelevant chunks and falls back to web search when local evidence is weak.
3. **Generation** – produces an answer from trusted context only.
4. **Hallucination grading** – verifies the answer is grounded in the supplied context.
5. **Answer grading** – verifies the answer addresses the original question.
6. **Output security** – redacts secrets and enforces output limits before the response is returned.

### Image Question Answering

Upload an image and either ask a specific question or submit it without one. The vision model analyses the image itself, extracts readable text (OCR) as supporting evidence, and answers from what is actually in the picture. Related-image search is an optional enrichment that only runs when the user explicitly asks to find or show similar images.

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                          Next.js UI                          │
│        Chat · Document & Image Upload · Evaluation           │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                       │
│        API · Uploads · Chat · Evaluation · Observability     │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                        Security Layer                        │
│  Prompt Injection · Jailbreak · Document Injection · Redaction│
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                      LangGraph Workflow                      │
│  Router → Retrieve → Grade → Generate → Validate → Retry     │
└───────────────┬──────────────────────────────┬───────────────┘
                ▼                              ▼
        ChromaDB (vectors)            Web / Image Search
                └──────────────┬───────────────┘
                               ▼
                    LLM (Groq) · Vision (Gemini)
```

## Security

Security is built into the workflow rather than bolted on:

- **Prompt-injection and jailbreak detection** on user input
- **System-prompt extraction protection**
- **Document-injection detection**: uploaded and retrieved content is treated as untrusted data, never as instructions
- **Untrusted-content wrapping** to separate external text from application instructions
- **Output guard** that redacts API keys and other secrets
- **Output length limits**

## Evaluation & Observability

**Evaluation** (`backend/evaluation/`) provides a benchmark dataset and runner, RAGAS metrics, and a report generator covering retrieval quality, grounding, answer relevance and routing behaviour.

**Observability** records, for each run, the selected route, retrieved-document count, grounding result, retry count, latency, source counts and human-approval usage, with optional LangSmith tracing.

## Tech Stack

| Layer | Technologies |
|---|---|
| Backend | Python, FastAPI, Pydantic, Uvicorn |
| Agent / LLM | LangGraph, LangChain, Groq, Google Gemini (embeddings + vision), Tavily |
| Storage | ChromaDB |
| Frontend | Next.js, React, TypeScript |
| Evaluation | RAGAS, custom benchmarks |
| DevOps | Docker, Docker Compose |
| Testing | Pytest |

## Project Structure

```text
.
├── backend/
│   ├── api.py                 # FastAPI application
│   ├── config.py              # Environment-driven settings
│   ├── ingestion.py           # Document loading, chunking, indexing
│   ├── retrieval.py           # Vector retrieval
│   ├── image_search.py        # Image analysis / related-image search
│   ├── evaluation.py          # Evaluation tracking
│   ├── observability.py       # Run metrics
│   ├── graph/
│   │   ├── graph.py           # LangGraph workflow
│   │   ├── state.py
│   │   ├── chains/            # Router, graders, generation
│   │   └── nodes/             # Retrieve, grade, generate, web search, security…
│   ├── security/              # Prompt, content and output guards
│   ├── evaluation/            # Benchmarks, RAGAS, reports
│   └── documents/             # Sample knowledge-base documents
├── frontend/                  # Next.js application
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+
- API keys: [Groq](https://console.groq.com), [Google AI Studio](https://aistudio.google.com) and optionally [Tavily](https://tavily.com)

### Installation

```bash
git clone https://github.com/Chittalasailu/agentic-rag-image-understanding.git
cd agentic-rag-image-understanding

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then add your API keys
```

### Run the backend

```bash
uvicorn backend.api:api --reload
```

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Run the frontend

```bash
cd frontend
npm install
npm run dev
```

The app is available at `http://localhost:3000`.

## Configuration

Copy `.env.example` to `.env` and fill in the values. **Never commit real keys.**

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | LLM provider | – |
| `LLM_MODEL` | Chat model | `openai/gpt-oss-120b` |
| `GOOGLE_API_KEY` | Embeddings and vision | – |
| `EMBEDDING_MODEL` | Embedding model | `gemini-embedding-001` |
| `VISION_MODEL` | Image understanding model | `gemini-2.5-flash` |
| `TAVILY_API_KEY` | Web and related-image search (optional) | – |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | Tracing (optional) | `false` |
| `CHROMA_PATH` / `COLLECTION_NAME` | Vector store location | `./.chroma` / `rag-chroma` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `RETRIEVAL_K` | RAG tuning | `250` / `0` / `4` |
| `MAX_UPLOAD_SIZE_MB` | Upload limit | `10` |
| `FRONTEND_URL` | CORS origin | `http://localhost:3000` |

## Running with Docker

```bash
docker compose build
docker compose up -d
```

| Service | URL |
|---|---|
| Frontend | `http://localhost:3000` |
| Backend | `http://localhost:8000` |
| API docs | `http://localhost:8000/docs` |

```bash
docker compose logs -f     # view logs
docker compose down        # stop
```

## API

The FastAPI backend exposes endpoints for health checks, document and image upload, document management, chat / query execution, evaluation and observability. Browse the full interactive reference at `/docs`.

## Testing

```bash
python -m pytest -v
```

The suite (33 tests) covers retrieval, hallucination and answer grading, routing, the local-RAG and web-fallback paths, generation retries, response validation, evaluation tracking and the full security layer (prompt injection, jailbreaks, document injection, secret redaction).

## Roadmap

- Authentication, authorization and API rate limiting
- Document malware scanning and stricter file-type validation
- Production vector database and background ingestion
- Streaming responses and persistent conversation memory
- Multi-user support and cloud deployment
- Dependency and container image scanning

## Author

**Chittala Sailu** · [GitHub](https://github.com/Chittalasailu)

Machine Learning Engineer · Generative AI · Agentic AI · LLM Systems
