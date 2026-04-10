# AI Codebase Assistant

AI Codebase Assistant is a backend-first Generative AI project for understanding software repositories with retrieval-augmented generation.

## Problem Statement

Developers spend a lot of time searching files, tracing flows, and understanding unfamiliar repositories. Plain chat assistants can help, but they are often opaque and hard to evaluate.

This project solves that by building a grounded code intelligence system that can:

- ingest a repository
- chunk code into meaningful units
- generate embeddings
- retrieve relevant context
- answer developer questions with grounded citations
- support agentic workflows for code exploration

## Architecture

```text
User / API Client
  ->
FastAPI Backend
  -> Repository Ingestion
  -> Code Parsing and Chunking
  -> Embedding Generation
  -> FAISS Vector Index
  -> Metadata-Aware Retrieval
  -> LLM Answer Generation
  -> Agent Workflows
  -> Evaluation and Observability
```

## Project Layout

```text
app/
  api/
  agents/
  chunking/
  core/
  db/
  embeddings/
  eval/
  ingestion/
  llm/
  retrieval/
  main.py
requirements.txt
```

## Features

- local repository ingestion with filtering rules
- Python-aware chunking plus line-based fallback chunking
- embeddings using `sentence-transformers`
- vector indexing with FAISS
- metadata-aware retrieval and lightweight hybrid reranking
- grounded Q&A with citations
- LangChain-powered prompt templating and output parsing in the LLM layer
- LlamaIndex-powered alternate retrieval path for side-by-side RAG comparison
- Vertex AI / Gemini-backed answer generation with fallback mode
- React UI for a premium operator-style repository intelligence workspace
- Gradio UI for visual testing and demos across repository workflows
- agent workflows:
  - explain flow
  - compare files
  - trace symbol references
  - cleanup candidate detection
- evaluation support:
  - hit@k
  - stored eval history
  - answer feedback
  - retrieval and generation latency tracking

## Tech Stack

- Python
- FastAPI
- sentence-transformers
- FAISS
- LangChain Core
- LlamaIndex
- Google Vertex AI / Gemini
- Pydantic
- React
- Vite
- Gradio
- Docker

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
uvicorn app.main:app --reload
```

4. Open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

## React UI

The project now includes a React + Vite frontend for a cleaner interactive experience on top of the FastAPI backend.

Frontend capabilities include:

- grounded repository Q&A
- custom FAISS vs LlamaIndex backend comparison
- flow analysis
- file comparison
- symbol tracing
- cleanup review
- eval execution and telemetry inspection
- raw JSON inspection for API responses

To run the React UI:

```bash
cd frontend
npm install
npm run dev
```

Then open:

- `http://localhost:5173`

The React frontend talks to FastAPI on `http://127.0.0.1:8000`, and the backend already includes CORS support for the Vite dev server.

## Docker Run

Build and run with Docker:

```bash
docker compose up --build
```

Then open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

## Gradio Demo

To launch the Gradio UI:

```bash
python -m app.gradio_ui
```

Then open:

- `http://127.0.0.1:7860`

The Gradio app currently includes tabs for:

- embedding a repository
- asking grounded questions with the custom FAISS path
- asking grounded questions with the LlamaIndex path
- comparing custom vs LlamaIndex RAG side by side
- explaining repository flows
- comparing files
- tracing symbol references
- reviewing cleanup candidates
- running evaluation and viewing history
- saving answer feedback

Note:

- React UI runs on `http://localhost:5173`
- Gradio UI runs on `http://127.0.0.1:7860`
- FastAPI Swagger remains on `http://127.0.0.1:8000/docs`

## LLM Setup

To enable Vertex AI answers for `POST /query/ask`, create a `.env` file in the project root with:

```env
LLM_PROVIDER=vertex
LLM_MODEL_NAME=gemini-2.5-flash
VERTEX_API_KEY=your_vertex_api_key_here
```

If you prefer Google AI Studio instead, use:

```env
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-2.5-flash
GEMINI_API_KEY=your_gemini_api_key_here
```

If `.env` is missing, the app still works in fallback answer mode.

## Example Requests

Embed the repository:

```json
{
  "repo_path": "D:\\AI Codebase Assistant"
}
```

Ask a grounded question:

```json
{
  "repo_path": "D:\\AI Codebase Assistant",
  "question": "Where is repository ingestion implemented?",
  "top_k": 5,
  "language": "python",
  "chunk_types": ["function", "class"],
  "file_path_contains": "ingestion"
}
```

Ask with the alternate LlamaIndex retrieval path:

```json
{
  "repo_path": "D:\\AI Codebase Assistant",
  "question": "Where is repository ingestion implemented?",
  "top_k": 5,
  "language": "python",
  "chunk_types": ["function", "class"],
  "file_path_contains": "ingestion"
}
```

Run evaluation:

```json
{
  "repo_path": "D:\\AI Codebase Assistant",
  "top_k": 5
}
```

## Current API

- `GET /health`
- `POST /repos/index`
- `POST /repos/chunks`
- `POST /repos/embed`
- `POST /query/search`
- `POST /query/ask`
- `POST /query/search-llamaindex`
- `POST /query/ask-llamaindex`
- `POST /query/explain-flow`
- `POST /query/compare-files`
- `POST /query/trace-symbol`
- `POST /query/cleanup-candidates`
- `POST /query/compare-retrieval`
- `POST /eval/run`
- `GET /eval/results`
- `POST /eval/feedback`

`POST /query/search` and `POST /query/ask` also support optional metadata-aware filters:

- `language`
- `chunk_types`
- `file_path_contains`

## Evaluation and Observability

The project currently supports:

- retrieval hit@k evaluation
- persisted eval runs
- answer feedback capture
- retrieval latency tracking
- generation latency tracking
- comparison of raw vs filtered vs reranked retrieval

## RAG Comparison Findings

The project now includes two grounded answer paths:

- a custom FAISS-based retrieval pipeline
- a LlamaIndex-based alternate retrieval pipeline

In local testing on the same repository question, both paths returned correct grounded answers with citations, but the custom FAISS path was significantly faster than the LlamaIndex path. This gives the project a strong comparison story:

- the custom path is faster and more controllable
- the framework path is useful for demonstrating ecosystem familiarity and alternate RAG abstractions

## Notes

- runtime data such as eval history and feedback is stored under `data/`
- local personal planning files are not part of the tracked project content
- frontend dependency and build output directories are ignored via `.gitignore`
