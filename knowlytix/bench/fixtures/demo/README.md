# Demo-mode fixtures

Bundled read-only artifacts loaded when `KNOWLYTIX_DEMO_MODE=1`. Kept
small on purpose — target <5 MB total across all files. v0.1.0 ships
stubs (empty JSON / missing binary); real content lands with the
tutorials repo (Phase 7.5 GMS-204).

| File | Shape | Purpose |
|---|---|---|
| `questions.json` | list of scored benchmark questions | bench scoring demo |
| `agent_trace.json` | dict, one MockHarness run | replay / regression demo |
| `document.json` | dict, pre-ingested doc metadata | query / retrieval demo |
| `llm_responses.json` | dict, SHA256(prompt)→response | `DemoLLMClient` canned responses |
| `knowledge_graph.pkl` | pickle, small GeometricKnowledgeGraph | read-only KG query demo (not shipped in v0.1.0) |

Consumers call `knowlytix.bench.fixtures.demo.load()` for everything.
