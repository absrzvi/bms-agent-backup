# BMS Agent Project Constitution

## Preamble
This constitution establishes the governing principles and technical standards for the BMS (Building Management System) Agent project, a Proof of Concept (POC) for RAG (Retrieval-Augmented Generation) implementation focused on railway connectivity infrastructure. This document ensures consistency, quality, and maintainability across all aspects of the project.

## 1. Railway IT Infrastructure Standards (MUST)
- **Compliance with Standards**: All implementations MUST adhere to:
  - EN50155 (Railway applications - Electronic equipment used in rolling stock)
  - EN45545 (Fire protection of railway vehicles)
  - TSI (Technical Specifications for Interoperability) specifications
  - ÖBB (Austrian Federal Railways) technical requirements
- **Availability**: Systems MUST maintain 99.99% uptime
- **Network Architecture**:
  - Support for high-availability configurations
  - Integration with Nomad Connect Routers
  - 10Gbps network infrastructure
  - Enterprise-grade WiFi access points
- **Performance**:
  - Real-time data processing with <100ms latency for moving train data
  - Resilience patterns for tunnel connectivity and high-speed handovers

## 2. Data Processing & RAG Architecture (MUST)
- **Core Technologies**:
  - Python 3.11+ with type hints for all public APIs
  - Pandas/Openpyxl for Excel/CSV processing (100k+ rows)
  - Qdrant vector database (v4+ schema)
- **Document Processing**:
  - Enhanced document processor architecture
  - Memory-efficient processing for files up to 1GB
  - Optimized chunking strategies
- **Search Capabilities**:
  - Hybrid search (semantic + keyword)
  - Vector embeddings using Mistral/OpenAI models

## 3. n8n Workflow Integration (MUST)
- **API Design**:
  - RESTful principles with OpenAPI 3.0
  - Webhook endpoints with JWT authentication
- **Data Handling**:
  - JSON schema validation
  - Idempotent operations
  - Structured error handling
  - Rate limiting implementation

## 4. Code Quality & Testing (MUST)
- **Test Coverage**:
  - Minimum 80% test coverage for core logic
  - Unit tests for data transformations
  - Integration tests for DB and API
- **Code Quality**:
  - Pre-commit hooks (Black, Ruff, mypy)
  - NumPy style docstrings
  - Performance benchmarking

## 5. Security & Compliance (MUST)
- **Data Protection**:
  - GDPR compliance
  - Encryption at rest
  - Input validation
- **Access Control**:
  - JWT/API key authentication
  - RBAC implementation
  - Audit logging
- **Security Scanning**:
  - Regular Bandit scans
  - Dependency vulnerability checks

## 6. Documentation (MUST)
- **Code Documentation**:
  - NumPy style docstrings
  - API documentation (auto-generated)
  - Type hints for all public interfaces
- **Project Documentation**:
  - Comprehensive README
  - Architecture overview
  - Setup and deployment guides
  - Troubleshooting guides
  - Changelog (Keep a Changelog format)

## 7. Performance & Scalability (MUST)
- **Scaling**:
  - Stateless design
  - Horizontal scaling support
  - Connection pooling
- **Optimization**:
  - Caching strategies
  - Async operations
  - Memory profiling
- **Targets**:
  - 1000+ concurrent requests
  - Sub-100ms response times

## 8. Monitoring & Observability (MUST)
- **Logging**:
  - Structured format
  - Correlation IDs
  - Log levels
- **Metrics**:
  - Prometheus integration
  - Health checks
  - Performance metrics
- **Visualization**:
  - Grafana dashboards
  - Alerting rules

## 9. Development Workflow (MUST)
- **Version Control**:
  - Git flow
  - Semantic versioning
  - PR reviews
- **CI/CD**:
  - Automated testing
  - Container builds
  - Environment management
- **Database**:
  - Alembic migrations
  - Schema versioning

## 10. Architecture (MUST)
- **Design Patterns**:
  - Modular architecture
  - Event-driven components
  - Repository pattern
  - Service layer
  - Factory pattern
  - Strategy pattern
- **Separation of Concerns**:
  - Clear module boundaries
  - Dependency injection
  - Interface segregation

## 11. AI/LLM Architecture (MUST)
- **Deployment**:
  - On-premises/local LLM hosting with zero external dependencies
  - Support for multiple local LLM backends (Ollama, vLLM, etc.)
  - Open-source models (Mistral, Llama) with size constraints based on hardware
  - Full air-gapped operation capability
- **Security & Data Sovereignty**:
  - Strictly no external API calls for inference or processing
  - Complete data sovereignty with no railway data leaving organizational boundaries
  - Model versioning with cryptographic hashing
  - Fallback mechanisms for local processing
  - Reproducible embedding generation with version control
- **RAG Pipeline**:
  - Self-contained operation with no external dependencies
  - Support for air-gapped deployment scenarios
  - Local embedding generation with versioned models
  - Documented data flow with clear boundaries

## 12. Vector Database (MUST)
- **Implementation**:
  - Self-hosted Qdrant
  - Hybrid retrieval
  - Optimized chunking
- **Management**:
  - Collection strategies
  - Embedding versioning
  - Schema management

## 13. n8n Integration (MUST)
- **Workflow Design**:
  - Agent orchestration
  - Stateless processing
  - Webhook-driven
- **Reliability**:
  - Error recovery
  - Monitoring hooks
  - Standardized schemas

## Governance

### Version Control
- Version: 1.0.0
- Ratified: 2025-09-28
- Last Amended: 2025-09-28

### Amendment Process
1. Propose changes via Pull Request
2. Required: 2+ approvals from maintainers
3. Update version according to:
   - MAJOR: Breaking changes
   - MINOR: Backward-compatible additions
   - PATCH: Backward-compatible fixes
4. Update `Last Amended` date

### Compliance
- All code must pass automated checks before merge
- Exceptions require documented approval
- Regular architecture reviews required

## Enforcement
- Automated checks for code style and quality
- Documentation requirements enforced in PRs
- Security scanning in CI/CD pipeline
- Performance gates in deployment process
