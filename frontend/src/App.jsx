import { useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

const DEFAULT_REPO_PATH = "D:\\AI Codebase Assistant";
const DEFAULT_QUESTION = "Where is repository ingestion implemented?";

const VIEWS = [
  {
    id: "ask",
    label: "Grounded Answer",
    kicker: "Primary workflow",
    title: "Ask the repository with citations and latency metrics.",
  },
  {
    id: "compare",
    label: "Backend Matchup",
    kicker: "RAG comparison",
    title: "Run the custom FAISS path and LlamaIndex path side by side.",
  },
  {
    id: "flow",
    label: "Flow Analysis",
    kicker: "System reasoning",
    title: "Explain how a feature moves through the codebase.",
  },
  {
    id: "files",
    label: "File Comparison",
    kicker: "Code inspection",
    title: "Compare two files and summarize their roles.",
  },
  {
    id: "symbol",
    label: "Symbol Trace",
    kicker: "Reference lookup",
    title: "Track where a symbol appears across the repository.",
  },
  {
    id: "cleanup",
    label: "Cleanup Review",
    kicker: "Maintenance lens",
    title: "Surface oversized files and cleanup candidates.",
  },
  {
    id: "eval",
    label: "Eval Run",
    kicker: "Quality check",
    title: "Measure retrieval hit rate and latency across benchmark queries.",
  },
];

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function citationsToLines(citations = []) {
  if (!citations.length) {
    return ["No citations returned."];
  }

  return citations.map((citation) => {
    const symbol = citation.symbol_name ? ` (${citation.symbol_name})` : "";
    return `${citation.file_path}:${citation.start_line}-${citation.end_line} [${citation.chunk_type}${symbol}]`;
  });
}

function formatDisplayText(text) {
  if (!text) {
    return "";
  }

  return text
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^\s*[*-]\s+/gm, "• ")
    .replace(/\r\n/g, "\n")
    .trim();
}

function renderTextBlocks(text) {
  return formatDisplayText(text)
    .split(/\n{2,}/)
    .filter(Boolean)
    .map((block, index) => (
      <p key={`${block.slice(0, 24)}-${index}`}>{block}</p>
    ));
}

function toolStepsToLines(toolSteps = []) {
  if (!toolSteps.length) {
    return ["No tool steps returned."];
  }

  return toolSteps.map(
    (step) => `${step.tool_name}: ${step.description}${step.output_summary ? ` -> ${step.output_summary}` : ""}`,
  );
}

async function postJson(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }
  return data;
}

async function getJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }
  return data;
}

function formatMs(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "--";
  }

  return `${Number(value).toFixed(2)} ms`;
}

function App() {
  const [activeView, setActiveView] = useState("compare");
  const [repoPath, setRepoPath] = useState(DEFAULT_REPO_PATH);
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [topK, setTopK] = useState(5);
  const [language, setLanguage] = useState("python");
  const [chunkTypes, setChunkTypes] = useState(["function", "class"]);
  const [pathFilter, setPathFilter] = useState("ingestion");
  const [symbol, setSymbol] = useState("load_local_repository");
  const [cleanupTopK, setCleanupTopK] = useState(10);
  const [filePathA, setFilePathA] = useState("app/ingestion/repo_loader.py");
  const [filePathB, setFilePathB] = useState("app/ingestion/file_filter.py");
  const [status, setStatus] = useState("Workspace ready.");
  const [isBusy, setIsBusy] = useState(false);
  const [embedResult, setEmbedResult] = useState(null);
  const [result, setResult] = useState(null);
  const [rawJson, setRawJson] = useState("{}");

  const currentView = VIEWS.find((view) => view.id === activeView) || VIEWS[0];

  const askPayload = {
    repo_path: repoPath,
    question,
    top_k: topK,
    language,
    chunk_types: chunkTypes,
    file_path_contains: pathFilter || null,
  };

  async function withTask(label, task) {
    setIsBusy(true);
    setStatus(label);
    try {
      await task();
    } catch (error) {
      const message = error.message || "Something went wrong.";
      setStatus(message);
      setRawJson(prettyJson({ error: message }));
    } finally {
      setIsBusy(false);
    }
  }

  async function handleEmbed() {
    await withTask("Embedding repository...", async () => {
      const data = await postJson("/repos/embed", { repo_path: repoPath });
      setEmbedResult(data);
      setRawJson(prettyJson(data));
      setStatus(`Embedded ${data.total_chunks_indexed} chunks with ${data.backend}.`);
    });
  }

  async function handleRunView() {
    const handlers = {
      ask: runAsk,
      compare: runCompare,
      flow: runFlow,
      files: runFileCompare,
      symbol: runSymbolTrace,
      cleanup: runCleanup,
      eval: runEval,
    };

    const handler = handlers[activeView];
    if (handler) {
      await handler();
    }
  }

  async function runAsk() {
    await withTask("Running grounded answer workflow...", async () => {
      const data = await postJson("/query/ask", askPayload);
      setResult({ kind: "ask", data });
      setRawJson(prettyJson(data));
      setStatus(`Answer ready using ${data.retrieval_backend}.`);
    });
  }

  async function runCompare() {
    await withTask("Comparing retrieval backends...", async () => {
      const [custom, llamaindex] = await Promise.all([
        postJson("/query/ask", askPayload),
        postJson("/query/ask-llamaindex", askPayload),
      ]);

      const data = {
        question,
        custom_backend: custom,
        llamaindex_backend: llamaindex,
        latency_delta_ms: Number((llamaindex.latency_ms - custom.latency_ms).toFixed(2)),
      };

      setResult({ kind: "compare", data });
      setRawJson(prettyJson(data));
      setStatus("Backend comparison complete.");
    });
  }

  async function runFlow() {
    await withTask("Explaining repository flow...", async () => {
      const data = await postJson("/query/explain-flow", {
        repo_path: repoPath,
        question: question || "Explain the repository flow",
        top_k: topK,
        language,
        chunk_types: chunkTypes,
        file_path_contains: pathFilter || null,
      });
      setResult({ kind: "flow", data });
      setRawJson(prettyJson(data));
      setStatus("Flow explanation ready.");
    });
  }

  async function runFileCompare() {
    await withTask("Comparing files...", async () => {
      const data = await postJson("/query/compare-files", {
        repo_path: repoPath,
        file_path_a: filePathA,
        file_path_b: filePathB,
      });
      setResult({ kind: "files", data });
      setRawJson(prettyJson(data));
      setStatus("File comparison ready.");
    });
  }

  async function runSymbolTrace() {
    await withTask("Tracing symbol references...", async () => {
      const data = await postJson("/query/trace-symbol", {
        repo_path: repoPath,
        symbol,
        top_k: 10,
      });
      setResult({ kind: "symbol", data });
      setRawJson(prettyJson(data));
      setStatus("Symbol trace complete.");
    });
  }

  async function runCleanup() {
    await withTask("Scanning cleanup candidates...", async () => {
      const data = await postJson("/query/cleanup-candidates", {
        repo_path: repoPath,
        top_k: cleanupTopK,
      });
      setResult({ kind: "cleanup", data });
      setRawJson(prettyJson(data));
      setStatus("Cleanup scan complete.");
    });
  }

  async function runEval() {
    await withTask("Running retrieval evaluation...", async () => {
      const run = await postJson("/eval/run", {
        repo_path: repoPath,
        top_k: topK,
      });
      const history = await getJson("/eval/results?limit=5");
      const data = { run, history };
      setResult({ kind: "eval", data });
      setRawJson(prettyJson(data));
      setStatus(`Eval finished with ${(run.hit_rate * 100).toFixed(0)}% hit rate.`);
    });
  }

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <header className="app-header">
        <div className="brand-cluster">
          <div className="brand-badge">AC</div>
          <div>
            <div className="micro-label">Repository Intelligence</div>
            <h1>AI Codebase Assistant</h1>
            <p>
              Grounded code search, RAG comparison, workflow explanation, inspection, and eval in one
              operator workspace.
            </p>
          </div>
        </div>

        <div className="header-actions">
          <button className="secondary-button" onClick={handleEmbed} disabled={isBusy}>
            Embed Repository
          </button>
          <div className={`status-pill ${isBusy ? "busy" : ""}`}>
            <span className="status-dot" />
            {isBusy ? "Running" : status}
          </div>
        </div>
      </header>

      <main className="workspace-grid">
        <aside className="control-column">
          <section className="surface control-panel">
            <div className="section-head">
              <span className="micro-label">Repository</span>
              <h2>Context</h2>
            </div>

            <div className="field-stack">
              <label>
                Repository path
                <input value={repoPath} onChange={(event) => setRepoPath(event.target.value)} />
              </label>

              <label>
                Main question
                <textarea
                  rows={3}
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                />
              </label>

              <div className="two-up">
                <label>
                  Top K
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={topK}
                    onChange={(event) => setTopK(Number(event.target.value))}
                  />
                </label>

                <label>
                  Language
                  <input value={language} onChange={(event) => setLanguage(event.target.value)} />
                </label>
              </div>

              <label>
                File path filter
                <input value={pathFilter} onChange={(event) => setPathFilter(event.target.value)} />
              </label>

              <div className="field-block">
                <span className="field-title">Chunk types</span>
                <div className="chip-row">
                  {["function", "class", "block"].map((type) => {
                    const active = chunkTypes.includes(type);
                    return (
                      <button
                        key={type}
                        type="button"
                        className={`chip ${active ? "active" : ""}`}
                        onClick={() =>
                          setChunkTypes((current) =>
                            current.includes(type)
                              ? current.filter((item) => item !== type)
                              : [...current, type],
                          )
                        }
                      >
                        {type}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </section>
        </aside>

        <section className="main-column">
          <section className="surface hero-panel">
            <div className="hero-copy">
              <span className="micro-label">Active Workflow</span>
              <h2>{currentView.label}</h2>
              <p>{currentView.title}</p>
            </div>

            <div className="hero-actions">
              <button className="primary-button" onClick={handleRunView} disabled={isBusy}>
                Run Workflow
              </button>
            </div>
          </section>

          <section className="surface tabs-panel">
            <div className="tabs-bar">
              {VIEWS.map((view) => (
                <button
                  key={view.id}
                  className={`tab-button ${activeView === view.id ? "active" : ""}`}
                  onClick={() => setActiveView(view.id)}
                >
                  <span>{view.kicker}</span>
                  <strong>{view.label}</strong>
                </button>
              ))}
            </div>
          </section>

          <section className="content-grid">
            <div className="surface input-stage">
              <div className="section-head compact">
                <span className="micro-label">Inputs</span>
                <h2>Run Setup</h2>
              </div>
              <DynamicInputs
                activeView={activeView}
                symbol={symbol}
                setSymbol={setSymbol}
                cleanupTopK={cleanupTopK}
                setCleanupTopK={setCleanupTopK}
                filePathA={filePathA}
                setFilePathA={setFilePathA}
                filePathB={filePathB}
                setFilePathB={setFilePathB}
              />
            </div>

            <div className="surface result-stage">
              <div className="section-head compact">
                <span className="micro-label">Output</span>
                <h2>Live Result</h2>
              </div>
              <ResultView result={result} embedResult={embedResult} />
            </div>

            <div className="surface insight-stage">
              <div className="section-head compact">
                <span className="micro-label">Telemetry</span>
                <h2>Inspector</h2>
              </div>

              <div className="insight-stack">
                <SummaryStrip result={result} embedResult={embedResult} />
                <div className="json-card">
                  <div className="json-title">Raw JSON</div>
                  <pre>{rawJson}</pre>
                </div>
              </div>
            </div>
          </section>
        </section>
      </main>
    </div>
  );
}

function DynamicInputs({
  activeView,
  symbol,
  setSymbol,
  cleanupTopK,
  setCleanupTopK,
  filePathA,
  setFilePathA,
  filePathB,
  setFilePathB,
}) {
  if (activeView === "files") {
    return (
      <div className="field-stack">
        <label>
          File A
          <input value={filePathA} onChange={(event) => setFilePathA(event.target.value)} />
        </label>
        <label>
          File B
          <input value={filePathB} onChange={(event) => setFilePathB(event.target.value)} />
        </label>
      </div>
    );
  }

  if (activeView === "symbol") {
    return (
      <div className="field-stack">
        <label>
          Symbol name
          <input value={symbol} onChange={(event) => setSymbol(event.target.value)} />
        </label>
      </div>
    );
  }

  if (activeView === "cleanup") {
    return (
      <div className="field-stack">
        <label>
          Cleanup Top K
          <input
            type="number"
            min="1"
            max="50"
            value={cleanupTopK}
            onChange={(event) => setCleanupTopK(Number(event.target.value))}
          />
        </label>
      </div>
    );
  }

  return (
    <div className="input-note">
      This workflow uses the shared repository question and retrieval filters from the left control rail.
    </div>
  );
}

function ResultView({ result, embedResult }) {
  if (!result) {
    return (
      <div className="empty-panel">
        <div className="empty-orb" />
        <h3>Run a workflow to inspect the repository.</h3>
        <p>
          Start with <strong>Embed Repository</strong>, then launch grounded answer, comparison, flow,
          trace, cleanup, or eval.
        </p>
        {embedResult ? (
          <small>
            Current index: {embedResult.total_chunks_indexed} chunks via {embedResult.backend}
          </small>
        ) : null}
      </div>
    );
  }

  if (result.kind === "ask") {
    return <AnswerResult title="Grounded Answer" data={result.data} />;
  }

  if (result.kind === "compare") {
    return <CompareResult data={result.data} />;
  }

  if (result.kind === "flow") {
    return (
      <NarrativeResult
        title="Flow Summary"
        body={result.data.flow_summary}
        citations={result.data.citations}
        metrics={[
          ["Retrieval", formatMs(result.data.retrieval_latency_ms)],
          ["Generation", formatMs(result.data.generation_latency_ms)],
          ["Total", formatMs(result.data.latency_ms)],
        ]}
        extraTitle="Tool Steps"
        extraLines={toolStepsToLines(result.data.tool_steps)}
      />
    );
  }

  if (result.kind === "files") {
    return (
      <NarrativeResult
        title="File Comparison"
        body={result.data.comparison_summary}
        citations={result.data.citations}
        metrics={[
          ["Mode", result.data.answer_mode],
          ["Generation", formatMs(result.data.generation_latency_ms)],
          ["Total", formatMs(result.data.latency_ms)],
        ]}
        extraTitle="Tool Steps"
        extraLines={toolStepsToLines(result.data.tool_steps)}
      />
    );
  }

  if (result.kind === "symbol") {
    return (
      <NarrativeResult
        title="Symbol Trace"
        body={result.data.summary}
        citations={result.data.citations}
        metrics={[
          ["Mode", result.data.answer_mode],
          ["Latency", formatMs(result.data.latency_ms)],
        ]}
        extraTitle="Tool Steps"
        extraLines={toolStepsToLines(result.data.tool_steps)}
      />
    );
  }

  if (result.kind === "cleanup") {
    return (
      <NarrativeResult
        title="Cleanup Candidates"
        body={result.data.summary}
        citations={result.data.citations}
        metrics={[
          ["Mode", result.data.answer_mode],
          ["Latency", formatMs(result.data.latency_ms)],
        ]}
        extraTitle="Signals"
        extraLines={toolStepsToLines(result.data.tool_steps)}
      />
    );
  }

  if (result.kind === "eval") {
    return <EvalResult data={result.data} />;
  }

  return null;
}

function AnswerResult({ title, data }) {
  return (
    <div className="result-stack">
      <div className="metric-strip">
        <MetricCard label="Backend" value={data.retrieval_backend} />
        <MetricCard label="Retrieval" value={formatMs(data.retrieval_latency_ms)} />
        <MetricCard label="Generation" value={formatMs(data.generation_latency_ms)} />
        <MetricCard label="Total" value={formatMs(data.latency_ms)} />
      </div>

      <article className="narrative-card">
        <div className="card-title">{title}</div>
        {renderTextBlocks(data.answer)}
      </article>

      <ReferenceList title="Citations" lines={citationsToLines(data.citations)} />
    </div>
  );
}

function CompareResult({ data }) {
  const faster = data.latency_delta_ms > 0 ? "Custom FAISS" : "LlamaIndex";

  return (
    <div className="result-stack">
      <div className="compare-banner">
        <div>
          <span className="micro-label">Winner</span>
          <h3>{faster}</h3>
        </div>
        <div className="delta-chip">{Math.abs(data.latency_delta_ms).toFixed(2)} ms delta</div>
      </div>

      <div className="compare-grid">
        <BackendCard title="Custom Backend" data={data.custom_backend} />
        <BackendCard title="LlamaIndex Backend" data={data.llamaindex_backend} />
      </div>
    </div>
  );
}

function BackendCard({ title, data }) {
  return (
    <div className="backend-card">
      <div className="backend-head">
        <div>
          <span className="micro-label">{title}</span>
          <h3>{data.retrieval_backend}</h3>
        </div>
        <div className="backend-total">{formatMs(data.latency_ms)}</div>
      </div>
      <div className="backend-metrics">
        <MetricCard label="Retrieval" value={formatMs(data.retrieval_latency_ms)} />
        <MetricCard label="Generation" value={formatMs(data.generation_latency_ms)} />
      </div>
      <div className="backend-answer">{renderTextBlocks(data.answer)}</div>
      <ReferenceList title="Citations" lines={citationsToLines(data.citations)} compact />
    </div>
  );
}

function NarrativeResult({ title, body, citations, metrics, extraTitle, extraLines }) {
  return (
    <div className="result-stack">
      <div className="metric-strip">
        {metrics.map(([label, value]) => (
          <MetricCard key={label} label={label} value={value} />
        ))}
      </div>

      <article className="narrative-card">
        <div className="card-title">{title}</div>
        {renderTextBlocks(body)}
      </article>

      <ReferenceList title={extraTitle} lines={extraLines} />
      <ReferenceList title="Citations" lines={citationsToLines(citations)} />
    </div>
  );
}

function EvalResult({ data }) {
  return (
    <div className="result-stack">
      <div className="metric-strip">
        <MetricCard label="Run ID" value={data.run.run_id || "latest"} />
        <MetricCard label="Hit Rate" value={`${(data.run.hit_rate * 100).toFixed(0)}%`} />
        <MetricCard label="Hits" value={`${data.run.hits}/${data.run.total_cases}`} />
        <MetricCard label="Total" value={formatMs(data.run.latency_ms)} />
      </div>

      <div className="eval-list">
        {data.run.results.map((item) => (
          <div className="eval-item" key={item.name}>
            <div className="eval-head">
              <strong>{item.name}</strong>
              <span className={`eval-badge ${item.hit ? "hit" : "miss"}`}>
                {item.hit ? "Hit" : "Miss"}
              </span>
            </div>
            <p>{item.query}</p>
            <small>
              hit@k: {item.hit_at_k} | retrieval: {formatMs(item.retrieval_latency_ms)}
            </small>
          </div>
        ))}
      </div>
    </div>
  );
}

function ReferenceList({ title, lines, compact = false }) {
  return (
    <section className={`reference-section ${compact ? "compact" : ""}`}>
      <div className="card-title">{title}</div>
      <div className="reference-list">
        {lines.map((line) => (
          <div className="reference-line" key={line}>
            {line}
          </div>
        ))}
      </div>
    </section>
  );
}

function SummaryStrip({ result, embedResult }) {
  const cards = [];

  if (embedResult) {
    cards.push({
      label: "Index",
      value: `${embedResult.total_chunks_indexed} chunks`,
    });
    cards.push({
      label: "Vector Store",
      value: embedResult.backend,
    });
  }

  if (result?.kind === "ask") {
    cards.push({ label: "Mode", value: result.data.answer_mode });
    cards.push({ label: "Backend", value: result.data.retrieval_backend });
  }

  if (result?.kind === "compare") {
    cards.push({ label: "Comparison", value: "FAISS vs LlamaIndex" });
    cards.push({ label: "Faster", value: result.data.latency_delta_ms > 0 ? "FAISS" : "LlamaIndex" });
  }

  if (result?.kind === "eval") {
    cards.push({ label: "Latest Run", value: result.data.run.run_id || "eval" });
    cards.push({ label: "History", value: `${result.data.history.total_runs} stored` });
  }

  if (!cards.length) {
    cards.push({ label: "Status", value: "Ready" });
    cards.push({ label: "Backend", value: "FAISS / LlamaIndex" });
  }

  return (
    <div className="summary-strip">
      {cards.map((card) => (
        <MetricCard key={`${card.label}-${card.value}`} label={card.label} value={card.value} />
      ))}
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
