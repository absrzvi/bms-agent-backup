# Post-MVP Backlog

Constitution-aligned enhancements to address after the MVP delivery.

- **Hybrid search support**
  - Add BM25 keyword pipeline and combined semantic + keyword endpoint.
  - Update Qdrant schema and API contract accordingly.

- **Large file ingestion (1 GB)**
  - Extend processor streaming pipeline and storage configuration for gigabyte-scale files.
  - Stress-test retry logic and chunking performance.

- **Observability stack**
  - Integrate Prometheus exporter for metrics scraping.
  - Provision Grafana dashboards and alert rules per constitution §8.

- **Security hardening**
  - Implement encryption at rest for Qdrant and persisted uploads.
  - Add RBAC layers and audit logging for administrative actions.

- **Performance & scaling upgrades**
  - Optimize for ≤100 ms latency and 1000+ concurrent requests.
  - Introduce caching and connection pooling strategies.

- **CI/CD enhancements**
  - Add pre-commit hooks (Black, Ruff, mypy) and container build pipeline.
  - Document semantic versioning and release process.
