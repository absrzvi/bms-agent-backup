# BMS Agent Specification

## Overview
- **Purpose**: Deliver a retrieval-augmented assistant for railway network documentation, deployed on a single RunPod pod with persistent storage.
- **Scope**: Document ingestion (PDF, CSV, XLSX, TXT), semantic search, Slack integration, OpenWebUI custom tool, API endpoints for internal services.
- **Constitution Alignment**: Implements Railway IT standards (EN50155, EN45545), RAG architecture mandates, n8n workflow authentication (JWT + API key), code quality & testing minimums, availability/monitoring targets.

## User Stories
- As a network engineer, I need to search railway documentation via Slack
- As a system admin, I need to upload and process new documentation
- As a RunPod operator, I need health and monitoring endpoints to validate uptime targets
- As a QA engineer, I need automated tests and metrics to verify retrieval accuracy

## Requirements
- **Document Processing**
  - R1.1: Ingest PDF, CSV, XLSX, and TXT files up to 1 GB with rejection for unsupported types.
    - *Acceptance*: Upload endpoint streams and processes 1 GB test fixtures without memory errors; returns HTTP 400/413 with descriptive errors for invalid types or oversize payloads.
  - R1.2: Chunk documents using hierarchical strategy (1,500 tokens, 200 overlap) and store embeddings in Qdrant collection `nomad_bms_documents`.
    - *Acceptance*: `scripts/test_processor.py` logs chunk counts; Qdrant collection contains payload metadata per chunk.

- **Search & Retrieval**
  - R2.1: Provide semantic search via `/api/v1/search/semantic` returning responses ≤100 ms p95 latency under 1,000 concurrent requests.
    - *Acceptance*: `tests/performance/test_performance.py` (and load tests) record ≤100 ms p95 latency with 1,000 simulated clients.
  - R2.2: Achieve ≥95 % top-5 retrieval accuracy on curated validation set (`data/evaluation/ground_truth.jsonl`).
    - *Acceptance*: `scripts/evaluate_retrieval.py` reports accuracy ≥95 %.
  - R2.3: Support hybrid retrieval (semantic + keyword/BM25) with query-time fusion.
    - *Acceptance*: Hybrid search endpoint returns both dense and sparse scores; integration tests verify BM25 keywords stored in Qdrant payload and exposed via API.

- **Security & Compliance**
  - R3.1: Protect all API and webhook endpoints with JWT (RS256) plus internal API key as per constitution §3; tokens validated against the public key provided in `BMS_JWT_PUBLIC_KEY` (signing key managed by upstream identity service).
  - R3.2: Enforce rate limiting (default 60 requests/min per JWT subject) without external dependencies.
  - R3.3: Expose OpenAPI 3.0 documentation at `/openapi.json` with correct security schemes.

  - R4.1: Provide `/health` and `/health/detailed` endpoints including Qdrant, Ollama, n8n, OpenWebUI status checks.
  - R4.2: Provide `/metrics/uplink` endpoint publishing latency, throughput, and recent errors for monitoring.
  - R4.3: Maintain 99.99 % availability target via documented monitoring + incident response playbook (`DEPLOYMENT_CHECKLIST.md`).

- **Testing & Quality**
  - R5.1: Maintain ≥80% coverage for core logic; integrate CI (GitHub Actions) with security scans (Bandit, Safety) and performance benchmarks.
  - R5.2: Include regression, performance, and security tests invoked by `pytest` markers.
  - R5.3: Enforce pre-commit tooling (Black, Ruff, mypy) and NumPy-style docstrings on all public interfaces; CI must fail if formatting, linting, or typing checks regress.

- **Workflow & Change Management**
  - R6.1: Adopt Git flow branching for feature development (e.g., `feature/<name>`, `release/<version>`) with semantic commit messages.
  - R6.2: Document database/data store changes through a migration log (`docs/migrations.md`) even when applying manual steps during the MVP.
  - R6.3: Produce container images for the API service, follow semantic versioning, and automate database migrations as part of the release workflow.
  
- **Observability & Operations (Post-MVP)**
  - R7.1: Expose Prometheus-compatible metrics and ship a Grafana dashboard with 99.99 % availability alerts (latency/error budgets) per constitution §8 (scheduled for post-MVP delivery).

## Acceptance Criteria Summary
- Supported file types (≤1 GB) ingest successfully; invalid files rejected with specific errors.
- Semantic/hybrid search responds ≤100 ms p95 under 1,000 concurrent users.
- Retrieval evaluation script reports ≥95 % accuracy.
- JWT + API key enforcement validated by automated tests; unauthorized access denied.
- Monitoring endpoints return status objects and metrics; referenced in operations playbook.
- CI pipeline passes tests, coverage, and security scans on main branch.
