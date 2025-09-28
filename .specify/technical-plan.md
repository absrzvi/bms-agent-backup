/plan

Implement technical enhancements to the existing BMS Agent using the current codebase and architecture.

## Existing Codebase Structure
The repository already contains:
- `/bms-agent/scr/` - Core processing scripts
  - `enhanced_document_processor.py` - Main v4.0 processor with all engines
  - `ultimate_processor_v4.py` - Enterprise RAG edition with complete pipeline
  - `qdrant_schema_v4.py` - Multi-vector Qdrant integration
- `/bms-agent/reqs/` - Requirements and documentation
  - Complete pipeline examples
  - n8n-Qdrant retrieval guides
  - Feature mapping tables
- `/bms-agent/n8n/` - Workflow definitions
- Test data files (Excel, CSV) with railway configurations

## Technical Stack (Already Implemented)

### Core Technologies
- **Python 3.11+** with type hints throughout
- **Document Processing**: 
  - Ultimate Document Processor v4.0 with modular engines
  - NLTK for NLP (punkt, stopwords, wordnet)
  - pandas/openpyxl for Excel processing (200k+ rows)
  - BeautifulSoup4 for HTML parsing
  - PyPDF2/pdfplumber for PDF extraction
  - Optional: Docling for advanced document processing

### Vector Database
- **Qdrant** (self-hosted on localhost:6333)
  - 4 dense vectors per chunk (768 dimensions each):
    - chunk_embedding (main retrieval)
    - parent_embedding (hierarchical parent)
    - child_embedding (hierarchical child)  
    - full_doc_embedding (late chunking)
  - 1 sparse vector for BM25 keyword matching
  - Comprehensive payload structure with railway metadata
  - Collections: railway_documents_v4

### Embedding & AI
- **SentenceTransformer** models:
  - Primary: sentence-transformers/all-mpnet-base-v2
  - Alternative: sentence-transformers/all-MiniLM-L6-v2
- **Local LLMs** (optional via Ollama):
  - Mistral 7B Instruct for query understanding
  - Llama 2 for response generation
  - No external API dependencies

### Workflow Orchestration
- **n8n** (self-hosted)
  - Nodes required: 
    - nodes-langchain.vectorStoreQdrant
    - nodes-langchain.agent
    - HTTP Request nodes for custom Qdrant operations
    - Code nodes for advanced processing
    - Switch nodes for routing strategies
  - Webhook endpoints for document ingestion
  - AI Agent with system prompts for railway expertise

### API Layer
- **FastAPI** for REST endpoints
- **Pydantic** for request/response validation
- **asyncio** for concurrent processing
- **httpx** for async HTTP requests to Qdrant

### Deployment
- **Docker** containers for each component:
  - Python processor service
  - Qdrant database
  - n8n workflow engine
  - FastAPI application
- **Docker Compose** for local development
- **Kubernetes** manifests for production:
  - Deployments with resource limits
  - Services for internal communication
  - ConfigMaps for configuration
  - Secrets for credentials
  - HorizontalPodAutoscaler for scaling

## Implementation Architecture

### 1. Document Processing Pipeline
```python
# Existing modules to enhance:
- EnhancedDocumentProcessor (enhanced_document_processor.py)
  - ContextualRetrievalEngine (adds <context> tags)
  - HierarchicalChunkingEngine (parent-child relationships)
  - LateChunkingEngine (full document embeddings)
  - QualityValidationEngine (faithfulness scores)
  - HybridSearchPreparator (term frequencies)
  - RailwayDocumentProcessor (ÖBB-specific)
  
# Configuration:
ProcessingConfig(
    processing_profile=ProcessingProfile.RAILWAY,
    chunk_size=1500,
    chunk_overlap=200,
    chunking_strategy=ChunkingStrategy.HIERARCHICAL,
    enable_contextual_retrieval=True,  # -67% failures
    enable_quality_validation=True,
    quality_threshold=85.0
)
2. Qdrant Integration Architecture
python# Existing schema implementation:
- QdrantSchemaV4 (qdrant_schema_v4.py)
  - Multi-vector configuration
  - Railway-optimized indexes
  - Batch upload with progress tracking
  - Collection management
  
# Vector spaces:
vectors_config = {
    "chunk_embedding": VectorParams(size=768, distance=Distance.COSINE),
    "parent_embedding": VectorParams(size=768, distance=Distance.COSINE),
    "child_embedding": VectorParams(size=768, distance=Distance.COSINE),
    "full_doc_embedding": VectorParams(size=768, distance=Distance.COSINE)
}

sparse_vectors_config = {
    "keyword_sparse": SparseVectorParams()  # BM25
}
3. n8n Workflow Implementation
yaml# Workflow structure:
1. Webhook Trigger (document upload)
2. File Parser (Excel/CSV/PDF detection)
3. Document Processor (Code node calling Python)
4. Embedding Generator (SentenceTransformer)
5. Qdrant Indexer (vectorStore node)
6. Quality Validator (Code node)
7. Notification (success/failure)

# AI Agent Configuration:
- Tool: Qdrant Vector Store
- System Prompt: Railway expert with EN standards
- Search strategies: hierarchical, hybrid, quality-filtered
- Response formatting with citations
4. Search Implementation Patterns
python# Hierarchical Search (search children, return parents):
child_results = search(vector="child_embedding", filter={"is_child": True})
parent_ids = [r.payload['parent_chunk_id'] for r in child_results]
parents = retrieve(ids=parent_ids)

# Hybrid Search with RRF:
dense_results = search(vector="chunk_embedding")
sparse_results = search(vector="keyword_sparse")
combined = reciprocal_rank_fusion(dense_results, sparse_results, alpha=0.6)

# Quality-Filtered:
search(filter={"quality_score": {"gte": 85.0}, "has_context": True})
5. Railway-Specific Processing
python# Parse configurations from uploaded files:
- CCU configurations (R4600-2Ax, R4600-3Ax, R5001C)
- Network topology (VLANs: 101 passenger, bond0 interfaces)
- Multicast settings (239.0.0.1:5000)
- DHCP configurations (10.{{ train_id }}.0.0/24)
- Redundancy mechanisms (master/slave CCU)
- IP addressing (unit_id based allocation)

# Extract from documents:
- Fleet types: Railjet, Cityjet
- Standards: EN50155, EN45545, TSI
- Components: CCU count, VLAN structures
- Network patterns: /etc/NetworkManager/system-connections
6. API Endpoints Structure
python# FastAPI routes:
POST /documents/upload          # Upload and process documents
POST /documents/batch           # Batch processing
GET  /documents/{id}           # Retrieve document metadata
POST /search/vector            # Vector search
POST /search/hybrid            # Hybrid search with RRF
POST /search/hierarchical      # Parent-child search
GET  /health                   # Health check
GET  /metrics                  # Prometheus metrics
POST /webhooks/n8n             # n8n webhook endpoint
7. Data Flow Architecture
Excel/CSV/PDF → Document Processor → Chunks with Context
    ↓                                      ↓
Railway Metadata                    Quality Scores
    ↓                                      ↓
Embeddings (4 dense + 1 sparse) → Qdrant Storage
    ↓                                      ↓
n8n Workflows ← API Layer ← Search Strategies
    ↓                                      ↓
AI Agent → User Response with Citations
8. Performance Optimizations

Batch Processing: 25 documents per batch for 1000+ files
Memory Management: Generator patterns for streaming
Caching: Redis for frequent queries (TTL: 1 hour)
Connection Pooling: Qdrant client pool (size: 10)
Async Processing: asyncio for I/O operations
Quantization: INT8 for vector storage (-75% memory)

9. Monitoring & Observability

Logging: Structured JSON logs with correlation IDs
Metrics: Prometheus exporters for:

Document processing rate
Query latency (P50, P95, P99)
Retrieval accuracy metrics
Error rates by component


Tracing: OpenTelemetry for distributed tracing
Dashboards: Grafana for visualization

10. Testing Strategy

Unit Tests: pytest for all processing engines
Integration Tests: Full pipeline tests
Load Tests: Locust for API performance
Quality Tests: Retrieval accuracy validation
Railway Tests: ÖBB-specific document samples

11. Deployment Configuration
yaml# Docker Compose for development:
services:
  processor:
    build: ./processor
    environment:
      - QDRANT_URL=qdrant:6333
      - N8N_URL=n8n:5678
  
  qdrant:
    image: qdrant/qdrant
    volumes:
      - qdrant_data:/qdrant/storage
  
  n8n:
    image: n8nio/n8n
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
  
  api:
    build: ./api
    ports:
      - "8000:8000"
12. Environment Configuration
env# .env file structure:
QDRANT_URL=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=railway_documents_v4
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
CHUNK_SIZE=1500
QUALITY_THRESHOLD=85.0
N8N_WEBHOOK_URL=http://localhost:5678/webhook/
OLLAMA_URL=http://localhost:11434  # Optional
MODEL_NAME=mistral:7b-instruct  # Optional
Migration Path from Existing Code

Keep all existing processor engines intact
Complete truncated functions in ultimate_processor_v4.py
Fix NLTK data download in initialization
Complete DocumentPipelineV4 class implementation
Add missing n8n workflow configurations
Implement missing search strategies
Add FastAPI wrapper around existing functions
Containerize without breaking existing functionality

Critical Dependencies to Install
txt# requirements.txt (existing + new):
nltk==3.8.1
pandas==2.0.3
openpyxl==3.1.2
beautifulsoup4==4.12.2
pdfplumber==0.9.0
sentence-transformers==2.2.2
qdrant-client==1.7.0
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.0
httpx==0.26.0
redis==5.0.1
prometheus-client==0.19.0
pytest==7.4.0
python-multipart==0.0.6
This technical plan preserves your existing sophisticated codebase while filling in the missing pieces and adding production-ready infrastructure.

This comprehensive `/plan` command incorporates:
1. All the existing code structure from your repository
2. The specific implementations you already have (v4.0 processors, Qdrant schema)
3. Railway-specific details from the documentation (CCU configs, network topology)
4. The complete technical stack with specific versions
5. Detailed implementation patterns from your requirement docs
6. The actual data flow and architecture already defined
7. Specific configuration values from your context (chunk sizes, quality thresholds)
8. The missing pieces that need to be completed

The plan respects your existing codebase and focuses on completing and integrating what's already there rather than rebuilding from scratch.