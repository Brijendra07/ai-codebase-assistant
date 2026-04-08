import json

import gradio as gr

from app.agents.orchestrator import (
    run_cleanup_candidates,
    run_compare_files,
    run_explain_flow,
    run_trace_symbol,
)
from app.db.models import (
    AnswerFeedbackRequest,
    RetrievalSettings,
)
from app.eval.runner import load_eval_runs, run_retrieval_eval, save_answer_feedback
from app.retrieval.retriever import (
    answer_repository_question,
    answer_repository_question_llamaindex,
    index_repository_embeddings,
)


def _pretty(data) -> str:
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    return json.dumps(data, indent=2, ensure_ascii=False)


def _format_citations(citations: list[dict]) -> str:
    if not citations:
        return "- None"

    lines: list[str] = []
    for citation in citations:
        symbol = f" ({citation['symbol_name']})" if citation.get("symbol_name") else ""
        lines.append(
            f"- `{citation['file_path']}:{citation['start_line']}-{citation['end_line']}` "
            f"[{citation['chunk_type']}{symbol}]"
        )
    return "\n".join(lines)


def _format_answer_card(title: str, response_data: dict) -> str:
    return (
        f"### {title}\n"
        f"- Retrieval backend: `{response_data['retrieval_backend']}`\n"
        f"- Answer mode: `{response_data['answer_mode']}`\n"
        f"- Retrieval latency: `{response_data['retrieval_latency_ms']} ms`\n"
        f"- Generation latency: `{response_data['generation_latency_ms']} ms`\n"
        f"- Total latency: `{response_data['latency_ms']} ms`\n\n"
        f"**Answer**\n\n{response_data['answer']}\n\n"
        f"**Citations**\n{_format_citations(response_data['citations'])}"
    )


def _build_comparison_summary(custom_response, llamaindex_response) -> dict:
    custom_data = custom_response.model_dump(mode="json")
    llamaindex_data = llamaindex_response.model_dump(mode="json")
    return {
        "question": custom_data["question"],
        "custom_backend": {
            "retrieval_backend": custom_data["retrieval_backend"],
            "answer_mode": custom_data["answer_mode"],
            "retrieval_latency_ms": custom_data["retrieval_latency_ms"],
            "generation_latency_ms": custom_data["generation_latency_ms"],
            "latency_ms": custom_data["latency_ms"],
            "citations": custom_data["citations"],
            "answer": custom_data["answer"],
        },
        "llamaindex_backend": {
            "retrieval_backend": llamaindex_data["retrieval_backend"],
            "answer_mode": llamaindex_data["answer_mode"],
            "retrieval_latency_ms": llamaindex_data["retrieval_latency_ms"],
            "generation_latency_ms": llamaindex_data["generation_latency_ms"],
            "latency_ms": llamaindex_data["latency_ms"],
            "citations": llamaindex_data["citations"],
            "answer": llamaindex_data["answer"],
        },
        "latency_delta_ms": round(
            llamaindex_data["latency_ms"] - custom_data["latency_ms"],
            2,
        ),
    }


def _build_ask_markdown(response_data: dict) -> str:
    return _format_answer_card("Answer Summary", response_data)


def embed_repo(repo_path: str) -> str:
    response = index_repository_embeddings(repo_path)
    return _pretty(response)


def ask_repo(
    repo_path: str,
    question: str,
    top_k: int,
    language: str,
    chunk_types: list[str],
    file_path_contains: str,
) -> tuple[str, str]:
    response = answer_repository_question(
        repo_path=repo_path,
        question=question,
        top_k=top_k,
        language=language or None,
        chunk_types=chunk_types or None,
        file_path_contains=file_path_contains or None,
    )
    response_data = response.model_dump(mode="json")
    return _build_ask_markdown(response_data), _pretty(response_data)


def explain_flow(
    repo_path: str,
    question: str,
    top_k: int,
    language: str,
    chunk_types: list[str],
    file_path_contains: str,
) -> tuple[str, str]:
    response = run_explain_flow(
        repo_path=repo_path,
        question=question,
        retrieval_settings=RetrievalSettings(
            top_k=top_k,
            language=language or None,
            chunk_types=chunk_types or None,
            file_path_contains=file_path_contains or None,
        ),
    )
    response_data = response.model_dump(mode="json")
    markdown = (
        f"### Flow Summary\n"
        f"- Answer mode: `{response_data['answer_mode']}`\n"
        f"- Retrieval latency: `{response_data['retrieval_latency_ms']} ms`\n"
        f"- Generation latency: `{response_data['generation_latency_ms']} ms`\n"
        f"- Total latency: `{response_data['latency_ms']} ms`\n\n"
        f"{response_data['flow_summary']}\n\n"
        f"**Citations**\n{_format_citations(response_data['citations'])}"
    )
    return markdown, _pretty(response_data)


def ask_repo_llamaindex(
    repo_path: str,
    question: str,
    top_k: int,
    language: str,
    chunk_types: list[str],
    file_path_contains: str,
) -> tuple[str, str]:
    response = answer_repository_question_llamaindex(
        repo_path=repo_path,
        question=question,
        top_k=top_k,
        language=language or None,
        chunk_types=chunk_types or None,
        file_path_contains=file_path_contains or None,
    )
    response_data = response.model_dump(mode="json")
    return _build_ask_markdown(response_data), _pretty(response_data)


def compare_rag_backends(
    repo_path: str,
    question: str,
    top_k: int,
    language: str,
    chunk_types: list[str],
    file_path_contains: str,
) -> tuple[str, str]:
    custom_response = answer_repository_question(
        repo_path=repo_path,
        question=question,
        top_k=top_k,
        language=language or None,
        chunk_types=chunk_types or None,
        file_path_contains=file_path_contains or None,
    )
    llamaindex_response = answer_repository_question_llamaindex(
        repo_path=repo_path,
        question=question,
        top_k=top_k,
        language=language or None,
        chunk_types=chunk_types or None,
        file_path_contains=file_path_contains or None,
    )
    comparison = _build_comparison_summary(custom_response, llamaindex_response)
    faster_backend = "custom FAISS" if comparison["latency_delta_ms"] > 0 else "LlamaIndex"
    markdown = (
        "## RAG Backend Comparison\n"
        f"- Question: `{comparison['question']}`\n"
        f"- Faster backend: `{faster_backend}`\n"
        f"- Latency delta: `{comparison['latency_delta_ms']} ms`\n\n"
        f"{_format_answer_card('Custom Backend', comparison['custom_backend'])}\n\n"
        f"{_format_answer_card('LlamaIndex Backend', comparison['llamaindex_backend'])}"
    )
    return markdown, _pretty(comparison)


def compare_files(repo_path: str, file_path_a: str, file_path_b: str) -> str:
    response = run_compare_files(
        repo_path=repo_path,
        file_path_a=file_path_a,
        file_path_b=file_path_b,
    )
    return _pretty(response)


def trace_symbol(repo_path: str, symbol: str, top_k: int) -> str:
    response = run_trace_symbol(
        repo_path=repo_path,
        symbol=symbol,
        top_k=top_k,
    )
    return _pretty(response)


def cleanup_candidates(repo_path: str, top_k: int) -> str:
    response = run_cleanup_candidates(
        repo_path=repo_path,
        top_k=top_k,
    )
    return _pretty(response)


def run_eval(repo_path: str, top_k: int) -> str:
    response = run_retrieval_eval(
        repo_path=repo_path,
        top_k=top_k,
    )
    return _pretty(response)


def load_eval_history(limit: int) -> str:
    response = load_eval_runs(limit=limit)
    return _pretty(response)


def save_feedback(
    repo_path: str,
    question: str,
    answer_mode: str,
    rating: int,
    comments: str,
) -> str:
    response = save_answer_feedback(
        AnswerFeedbackRequest(
            repo_path=repo_path,
            question=question,
            answer_mode=answer_mode,
            rating=rating,
            comments=comments or None,
        )
    )
    return _pretty(response)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AI Codebase Assistant") as demo:
        gr.Markdown(
            """
            # AI Codebase Assistant
            Backend-powered Gradio UI for grounded repository Q&A, agent workflows, and evaluation.
            Start by embedding a repository, then use the tabs below to test the system visually.
            """
        )

        with gr.Tab("Embed"):
            embed_repo_path = gr.Textbox(label="Repository Path", value="D:\\AI Codebase Assistant")
            embed_button = gr.Button("Embed Repository", variant="primary")
            embed_output = gr.Code(label="Embed Response", language="json")
            embed_button.click(embed_repo, inputs=[embed_repo_path], outputs=[embed_output])

        with gr.Tab("Ask"):
            ask_repo_path = gr.Textbox(label="Repository Path", value="D:\\AI Codebase Assistant")
            ask_question = gr.Textbox(label="Question", value="Where is repository ingestion implemented?")
            ask_top_k = gr.Slider(label="Top K", minimum=1, maximum=20, value=5, step=1)
            ask_language = gr.Textbox(label="Language Filter", value="python")
            ask_chunk_types = gr.CheckboxGroup(
                label="Chunk Types",
                choices=["function", "class", "block"],
                value=["function", "class"],
            )
            ask_path_filter = gr.Textbox(label="File Path Contains", value="ingestion")
            ask_button = gr.Button("Ask", variant="primary")
            ask_summary = gr.Markdown(label="Ask Summary")
            ask_output = gr.Code(label="Ask Raw JSON", language="json")
            ask_button.click(
                ask_repo,
                inputs=[
                    ask_repo_path,
                    ask_question,
                    ask_top_k,
                    ask_language,
                    ask_chunk_types,
                    ask_path_filter,
                ],
                outputs=[ask_summary, ask_output],
            )

        with gr.Tab("Ask (LlamaIndex)"):
            ask_li_repo_path = gr.Textbox(label="Repository Path", value="D:\\AI Codebase Assistant")
            ask_li_question = gr.Textbox(label="Question", value="Where is repository ingestion implemented?")
            ask_li_top_k = gr.Slider(label="Top K", minimum=1, maximum=20, value=5, step=1)
            ask_li_language = gr.Textbox(label="Language Filter", value="python")
            ask_li_chunk_types = gr.CheckboxGroup(
                label="Chunk Types",
                choices=["function", "class", "block"],
                value=["function", "class"],
            )
            ask_li_path_filter = gr.Textbox(label="File Path Contains", value="ingestion")
            ask_li_button = gr.Button("Ask with LlamaIndex", variant="primary")
            ask_li_summary = gr.Markdown(label="Ask LlamaIndex Summary")
            ask_li_output = gr.Code(label="Ask LlamaIndex Raw JSON", language="json")
            ask_li_button.click(
                ask_repo_llamaindex,
                inputs=[
                    ask_li_repo_path,
                    ask_li_question,
                    ask_li_top_k,
                    ask_li_language,
                    ask_li_chunk_types,
                    ask_li_path_filter,
                ],
                outputs=[ask_li_summary, ask_li_output],
            )

        with gr.Tab("Compare RAG"):
            compare_rag_repo_path = gr.Textbox(label="Repository Path", value="D:\\AI Codebase Assistant")
            compare_rag_question = gr.Textbox(label="Question", value="Where is repository ingestion implemented?")
            compare_rag_top_k = gr.Slider(label="Top K", minimum=1, maximum=20, value=5, step=1)
            compare_rag_language = gr.Textbox(label="Language Filter", value="python")
            compare_rag_chunk_types = gr.CheckboxGroup(
                label="Chunk Types",
                choices=["function", "class", "block"],
                value=["function", "class"],
            )
            compare_rag_path_filter = gr.Textbox(label="File Path Contains", value="ingestion")
            compare_rag_button = gr.Button("Compare Custom vs LlamaIndex", variant="primary")
            compare_rag_summary = gr.Markdown(label="Comparison Summary")
            compare_rag_output = gr.Code(label="RAG Comparison Raw JSON", language="json")
            compare_rag_button.click(
                compare_rag_backends,
                inputs=[
                    compare_rag_repo_path,
                    compare_rag_question,
                    compare_rag_top_k,
                    compare_rag_language,
                    compare_rag_chunk_types,
                    compare_rag_path_filter,
                ],
                outputs=[compare_rag_summary, compare_rag_output],
            )

        with gr.Tab("Explain Flow"):
            flow_repo_path = gr.Textbox(label="Repository Path", value="D:\\AI Codebase Assistant")
            flow_question = gr.Textbox(label="Flow Question", value="Explain the repository ingestion flow")
            flow_top_k = gr.Slider(label="Top K", minimum=1, maximum=20, value=5, step=1)
            flow_language = gr.Textbox(label="Language Filter", value="python")
            flow_chunk_types = gr.CheckboxGroup(
                label="Chunk Types",
                choices=["function", "class", "block"],
                value=["function", "class"],
            )
            flow_path_filter = gr.Textbox(label="File Path Contains", value="ingestion")
            flow_button = gr.Button("Explain Flow", variant="primary")
            flow_summary_output = gr.Markdown(label="Explain Flow Summary")
            flow_output = gr.Code(label="Explain Flow Raw JSON", language="json")
            flow_button.click(
                explain_flow,
                inputs=[
                    flow_repo_path,
                    flow_question,
                    flow_top_k,
                    flow_language,
                    flow_chunk_types,
                    flow_path_filter,
                ],
                outputs=[flow_summary_output, flow_output],
            )

        with gr.Tab("Compare Files"):
            compare_repo_path = gr.Textbox(label="Repository Path", value="D:\\AI Codebase Assistant")
            compare_file_a = gr.Textbox(label="File Path A", value="app/ingestion/repo_loader.py")
            compare_file_b = gr.Textbox(label="File Path B", value="app/ingestion/file_filter.py")
            compare_button = gr.Button("Compare Files", variant="primary")
            compare_output = gr.Code(label="Compare Files Response", language="json")
            compare_button.click(
                compare_files,
                inputs=[compare_repo_path, compare_file_a, compare_file_b],
                outputs=[compare_output],
            )

        with gr.Tab("Trace Symbol"):
            trace_repo_path = gr.Textbox(label="Repository Path", value="D:\\AI Codebase Assistant")
            trace_symbol_name = gr.Textbox(label="Symbol", value="load_local_repository")
            trace_top_k = gr.Slider(label="Top K", minimum=1, maximum=50, value=10, step=1)
            trace_button = gr.Button("Trace Symbol", variant="primary")
            trace_output = gr.Code(label="Trace Symbol Response", language="json")
            trace_button.click(
                trace_symbol,
                inputs=[trace_repo_path, trace_symbol_name, trace_top_k],
                outputs=[trace_output],
            )

        with gr.Tab("Cleanup"):
            cleanup_repo_path = gr.Textbox(label="Repository Path", value="D:\\AI Codebase Assistant")
            cleanup_top_k = gr.Slider(label="Top K", minimum=1, maximum=50, value=10, step=1)
            cleanup_button = gr.Button("Find Cleanup Candidates", variant="primary")
            cleanup_output = gr.Code(label="Cleanup Candidates Response", language="json")
            cleanup_button.click(
                cleanup_candidates,
                inputs=[cleanup_repo_path, cleanup_top_k],
                outputs=[cleanup_output],
            )

        with gr.Tab("Evaluation"):
            eval_repo_path = gr.Textbox(label="Repository Path", value="D:\\AI Codebase Assistant")
            eval_top_k = gr.Slider(label="Top K", minimum=1, maximum=20, value=5, step=1)
            eval_button = gr.Button("Run Evaluation", variant="primary")
            eval_output = gr.Code(label="Eval Run Response", language="json")
            eval_button.click(run_eval, inputs=[eval_repo_path, eval_top_k], outputs=[eval_output])

            history_limit = gr.Slider(label="History Limit", minimum=1, maximum=100, value=20, step=1)
            history_button = gr.Button("Load Eval History")
            history_output = gr.Code(label="Eval History", language="json")
            history_button.click(load_eval_history, inputs=[history_limit], outputs=[history_output])

        with gr.Tab("Feedback"):
            feedback_repo_path = gr.Textbox(label="Repository Path", value="D:\\AI Codebase Assistant")
            feedback_question = gr.Textbox(label="Question", value="Where is repository ingestion implemented?")
            feedback_mode = gr.Dropdown(label="Answer Mode", choices=["llm", "fallback", "tool"], value="llm")
            feedback_rating = gr.Slider(label="Rating", minimum=1, maximum=5, value=5, step=1)
            feedback_comments = gr.Textbox(label="Comments", value="Good grounded answer with correct citations.")
            feedback_button = gr.Button("Save Feedback", variant="primary")
            feedback_output = gr.Code(label="Feedback Response", language="json")
            feedback_button.click(
                save_feedback,
                inputs=[
                    feedback_repo_path,
                    feedback_question,
                    feedback_mode,
                    feedback_rating,
                    feedback_comments,
                ],
                outputs=[feedback_output],
            )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
