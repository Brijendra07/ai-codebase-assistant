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

## Local Notes

Personal planning and CV files are kept in `local_docs/`, which is gitignored.

## Features

- local repository ingestion with filtering rules
- Python-aware chunking plus line-based fallback chunking
- embeddings using `sentence-transformers`
- vector indexing with FAISS
- metadata-aware retrieval and lightweight hybrid reranking
- grounded Q&A with citations
- Vertex AI / Gemini-backed answer generation with fallback mode
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
- Google Vertex AI / Gemini
- Pydantic
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

## Sample Workflow

1. Embed the repository:

```json
{
  "repo_path": "D:\\AI Codebase Assistant"
}
```

2. Ask a grounded question:

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

3. Run evaluation:

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

## Roadmap Status

- Phase 0: complete
- Phase 1: complete
- Phase 2: complete
- Phase 3: complete
- Phase 4: complete
- Phase 5: complete
- Phase 6: planned

This means the current project scope is portfolio-ready and covers the roadmap end to end.

## Next Extension

The next planned extension is a lightweight Gradio UI for:

- asking repository questions in a more natural interface
- running explain-flow, compare-files, trace-symbol, and cleanup workflows visually
- reviewing eval history and outputs more easily
- making demos and manual testing faster than Swagger alone

This UI phase is intended to strengthen testing, demos, screenshots, and overall portfolio presentation without changing the completed backend roadmap phases.
