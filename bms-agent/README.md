# BMS Agent

Retrieval-augmented question answering system for railway documentation. This guide
covers local setup, environment configuration, core workflows, and deployment
automation on RunPod.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Local Setup](#local-setup)
4. [Environment Configuration](#environment-configuration)
5. [Running the Services](#running-the-services)
6. [Document Ingestion Workflow](#document-ingestion-workflow)
7. [API Usage](#api-usage)
8. [Testing](#testing)
9. [Deployment Automation](#deployment-automation)
10. [Development Workflow](#development-workflow)
11. [Operations Playbook](#operations-playbook)
12. [Additional Documentation](#additional-documentation)

## Architecture Overview

- **FastAPI** service exposing upload and semantic search endpoints.
- **EnhancedDocumentProcessor** for chunking, quality validation, and metadata
  enrichment (`bms-agent/api/processor_wrapper.py`).
- **Qdrant** vector database storing multi-vector embeddings in the
  `nomad_bms_documents` collection.
- **Ollama** models (`snowflake-arctic-embed2`, `mistral-nemo:12b-instruct`) for
  embeddings and summarisation.
- **n8n** workflow providing Slack integration (see `n8n/workflows/`).
- **OpenWebUI** custom tool (`~/.openwebui/tools/bms_search.py`) for search from
  the LLM interface.

## Prerequisites

- Python 3.11+
- Qdrant binary available at `/usr/local/bin/qdrant`
- Ollama running with access to required models
- RunPod pod (8-16 vCPU / 32-64GB RAM) with persistent storage mounted at
  `~/persistent`
- Slack credentials and n8n instance (optional integrations)

## Local Setup

```bash
git clone https://github.com/absrzvi/bms-agent.git
cd bms-agent

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r reqs/requirements.txt
pip install -r requirements-test.txt  # Optional: enable full test suite
```

### Metrics Uplink

```bash
curl -X GET http://localhost:8000/metrics/uplink \
     -H "X-API-Key: ${BMS_API_KEY}"
```

Returns aggregated counters collected during ingestion/search (e.g., `documents_processed`, `last_search_latency`).

## Environment Configuration

Set environment variables (consider adding them to `config/env.sh`). Core values shown below; override any `BMS_*` variable to tune the processor without code changes:

| Variable | Purpose | Default |
| --- | --- | --- |
| `BMS_API_KEY` | Optional API key required for `/api/v1/*` routes | unset (anonymous allowed) |
| `QDRANT_HOST` / `QDRANT_PORT` | Qdrant connection details | `localhost` / `6333` |
| `QDRANT_COLLECTION` | Target collection for embeddings | `nomad_bms_documents` |
| `EMBEDDING_MODEL` | Ollama embedding model used by the wrapper | `snowflake-arctic-embed2` |
| `EMBEDDING_URL` | Ollama embeddings endpoint | `http://localhost:11434/api/embeddings` |
| `BMS_PROCESSING_PROFILE` | Processing profile (`RAILWAY`, `TECHNICAL`, …) | `RAILWAY` |
| `BMS_CHUNK_SIZE` / `BMS_CHUNK_OVERLAP` | Chunking controls for EnhancedDocumentProcessor | `1500` / `200` |
| `BMS_ENABLE_HYBRID_SEARCH` | Enable keyword + vector payload enrichment | `true` |
| `BMS_ENABLE_QUALITY_VALIDATION` | Evaluate chunks with RAGAS-style metrics | `true` |
| `BMS_UPLOAD_MAX_BYTES` | Max upload size for `/documents/upload` | `104857600` (100 MB) |

All other processor toggles documented in `bms-agent/api/processor_wrapper.py` follow the same `BMS_*` prefix (e.g., `BMS_ENABLE_OCR`, `BMS_VECTOR_WEIGHT`).

Create the spec baseline if missing (`tasks.md` → `T000`).

## Running the Services

From the project root:

```bash
# Start Qdrant with persistent storage
./scripts/start_qdrant.sh

# Launch API (hot reload for development)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Convenience wrapper (start/stop/status)
./scripts/manage_services.sh start
./scripts/manage_services.sh status
```

## Document Ingestion Workflow

1. Place source files under `~/persistent/bms_data/uploads/` or upload via API.
2. `DocumentProcessorWrapper.process_document()` handles chunking, embedding, and
   storage in Qdrant (`nomad_bms_documents`).
3. Verify ingestion using
   `python scripts/test_processor.py` (created in `tasks.md` `T007`).

## API Usage

Interactive docs are available at `http://localhost:8000/docs` once the server is
running.

### Semantic Search

```bash
curl -X POST http://localhost:8000/api/v1/search/semantic \
     -H "Content-Type: application/json" \
     -H "X-API-Key: ${BMS_API_KEY}" \
     -d '{"query": "railway redundancy", "limit": 5}'
```

### Document Upload

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
     -H "X-API-Key: ${BMS_API_KEY}" \
     -F "file=@data/test_documents/test_railway_doc.txt"
```

**Response payload** includes enriched processing metadata from `DocumentProcessorWrapper.process_document()`:

```json
{
  "status": "success",
  "filename": "test_railway_doc.txt",
  "bytes_stored": 5421,
  "processing": {
    "document_id": "test_railway_doc",
    "chunks_indexed": 18,
    "collection_name": "nomad_bms_documents",
    "embedding_model": "snowflake-arctic-embed2",
    "quality_report": {"average_quality": 0.91, ...},
    "indexed_chunks": [
      {
        "chunk_key": 0,
        "metadata": {"hierarchy": "parent", "has_context": true},
        "quality": {"overall_score": 0.95},
        "term_frequencies": {"railway": 0.04, ...}
      }
    ]
  }
}
```

## Testing

Core testing guidance lives in `TESTING.md`. Common commands:

```bash
pytest -v --cov=./ --cov-report=term-missing
pytest tests/test_basic.py::test_search_endpoint
pytest tests/performance/test_performance.py -m performance
```

## Deployment Automation

Automated pipeline defined in `.github/workflows/ci-cd.yml`.

- **Secrets required**: `BMS_API_KEY`, optionally `CODECOV_TOKEN`,
  `SAFETY_API_KEY`.
- **Jobs**:
  - `test`: installs dependencies, runs full pytest suite with coverage.
  - `security`: executes Bandit and Safety scans.
  - `deploy`: placeholder for RunPod automation (extend with SSH or API calls).

Manual deployment steps for RunPod (until automation is completed):

```bash
rsync -avz ./bms-agent <user>@<runpod-ip>:~/bms-agent
ssh <user>@<runpod-ip> 'cd ~/bms-agent && ./scripts/manage_services.sh restart'
ssh <user>@<runpod-ip> '~/bms-agent/scripts/health_check.sh'
```

## Development Workflow

- **Branching**: Follow Git flow (`feature/<name>`, `release/<version>`, `hotfix/<issue>`). Keep commits semantic (e.g., `feat(api): ...`).
- **Migrations Log**: Record schema/data adjustments in `docs/migrations.md` with date, summary, and related task/PR references.
- **Pull Requests**: Prefer small, reviewable PRs mapped to the tasks in `tasks.md`.

## Operations Playbook

- **Health Checks**: `./scripts/health_check.sh`
- **Logs**: `~/persistent/logs/api.log`, `~/persistent/logs/qdrant.log`
- **Backups**: snapshot `~/persistent/qdrant_storage` and
  `~/persistent/bms_data`
- **Scaling**: adjust RunPod resources or shard Qdrant instance

## Additional Documentation

- `tasks.md`: Dependency-ordered task tracking for RunPod deployment.
- `TESTING.md`: Detailed QA strategy, fixtures, and performance targets.
- `reports/performance-baseline.md`: Latest latency benchmarks.
- `docs/migrations.md`: Running log for manual schema or data changes.
- `.github/workflows/ci-cd.yml`: CI/CD pipeline definition.
