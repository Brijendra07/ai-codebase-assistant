# AI Codebase Assistant

AI Codebase Assistant is a backend-first Generative AI project for understanding software repositories with retrieval-augmented generation.

## Phase 0 Scope

This starter sets up:

- FastAPI backend skeleton
- health endpoint
- basic config management
- structured logging
- placeholder package layout for ingestion, chunking, embeddings, retrieval, LLM, agents, eval, and DB layers

## Project Goal

Build a production-style AI system that can:

- ingest a repository
- chunk code into meaningful units
- generate embeddings
- retrieve relevant context
- answer developer questions with grounded citations
- support agentic workflows for code exploration

## Starter Layout

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

## Local Run

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

## Next Steps

- add repository ingestion
- add code chunking
- add embedding pipeline
- add vector search
- add query endpoint with citations

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

`POST /query/search` and `POST /query/ask` also support optional metadata-aware filters:

- `language`
- `chunk_types`
- `file_path_contains`
