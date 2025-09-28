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

## Environment Configuration

Set environment variables (consider adding them to `config/env.sh`):

```bash
export BMS_API_KEY="change-me"               # Align with CI secrets
export QDRANT_URL="http://localhost:6333"
export OLLAMA_URL="http://localhost:11434"
export COLLECTION_NAME="nomad_bms_documents"
```

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
