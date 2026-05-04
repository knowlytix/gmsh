# Demo-mode fixtures

Bundled read-only artifacts loaded when `KNOWLYTIX_DEMO_MODE=1`. Kept
small on purpose — target <5 MB total across all files. The current
release ships stubs (empty JSON / missing binary); fuller fixture
content lands alongside the public tutorial mirror.

| File | Shape | Purpose |
|---|---|---|
| `questions.json` | list of scored benchmark questions | bench scoring demo |
| `agent_trace.json` | dict, one MockHarness run | replay / regression demo |
| `document.json` | dict, pre-ingested doc metadata | query / retrieval demo |
| `llm_responses.json` | dict, SHA256(prompt)→response | `DemoLLMClient` canned responses |
| `knowledge_graph.pkl` | pickle, small GeometricKnowledgeGraph | read-only KG query demo (stub in current release) |

Consumers call `knowlytix.benchmark.fixtures.demo.load()` for everything.
