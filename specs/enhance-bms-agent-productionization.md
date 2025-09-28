# Feature Specification: BMS Agent Productionization

## Overview
Enhance and productionize the BMS (Building Management System) Agent, an advanced RAG pipeline that has demonstrated a 67% reduction in retrieval failures through its Ultimate Document Processor v4.0 and Qdrant integration.

## MVP Goals
- Create functional POC of BMS Agent with core RAG capabilities
- Support common office document formats (DOCX, PDF, PPTX, CSV, XLSX)
- Process department-specific documents (QHSE, BID, Project, HR, Finance)
- Enable basic document search and retrieval
- Focus on core functionality before optimization

## Technical Requirements

### 1. Document Processing (MVP Focus)
- **Supported Formats**:
  - Office Documents: DOCX, PDF, PPTX, XLSX, CSV
  - Maximum document size: 10MB
  - Maximum rows for spreadsheets: 10,000
  - **MVP Focus**: Office documents only (no network configurations or specialized formats)

- **Processing Pipeline**:
  - Extract text and metadata
  - Advanced semantic chunking with hierarchy
  - Quality validation with 95%+ accuracy
  - Automatic format detection
  - No OCR or image processing

### 2. Qdrant Setup (MVP)
- **Vector Database**:
  - Single collection setup
  - Basic dense vectors only
  - No sharding required
  - Local deployment
- **Vector Schema v4.0**
  - Optimize 4 dense vectors:
    - chunk_embedding
    - parent_embedding
    - child_embedding
    - full_doc_embedding
  - Enhance sparse vector implementation for BM25
  - Add collection sharding for 1M+ chunks

- **Query Performance**
  - Implement automatic index optimization
  - Add reciprocal rank fusion (RRF)
  - Support cross-collection joins

- **Document Metadata**
  - Extract standard metadata:
    - Title, author, creation date
    - Page count (where applicable)
    - File type and size
  - Preserve document structure and hierarchy

### 3. User Interaction (MVP)
- **Slack Integration**:
  - Search and query interface
  - Document upload notifications
  - Basic bot commands for status

- **OpenWebUI + Ollama**:
  - Primary chat interface
  - Document context in responses
  - Conversation history
- **Document Processing Pipeline**
  - Webhook → Document Processor → Qdrant
  - AI Agent configuration for Qdrant
  - Hierarchical search implementation

- **Search Capabilities**
  - Hybrid search with RRF fusion
  - Quality-based filtering
  - Monitoring dashboard for metrics

### 4. Single-Container Architecture
- **Deployment**:
  - Single RunPod container
  - All components co-located
  - Persistent volume at `/data` for documents and indexes
  - Automatic recovery on restart

- **Component Interaction**:
  1. Documents placed in local `/data/uploads`
  2. File watcher triggers processing
  3. Documents processed and indexed in Qdrant
  4. Queries handled via OpenWebUI/Slack
  5. Ollama provides LLM capabilities

## MVP Targets
- **Document Volume**: Up to 1,000 documents
- **Query Latency**: 
  - Simple queries: <2 seconds
  - Complex searches: <5 seconds
- **Users**: Support for small team (5-10 users)
- **Storage**: Persistent volume at `/data`
- **Uptime**: Best effort (automatic recovery enabled)

## Success Metrics
- 100% completion of truncated functions
- End-to-end n8n workflow execution
- 1M+ document handling in Qdrant
- 99.9% production uptime
- 95%+ query accuracy
- 80%+ test coverage

## Dependencies (MVP)
- Qdrant (container)
- Ollama (container)
- OpenWebUI (container)
- Python 3.11+
- File system watcher
- Slack bot

## Error Handling Strategy
1. **Document Processing Errors**:
   - Invalid formats: Log warning, skip file, notify in Slack
   - Processing failures: Retry 3x, then quarantine file
   - Corrupt documents: Move to `/data/quarantine` with timestamp

2. **System Errors**:
   - Qdrant failures: Automatic restart with backoff
   - LLM timeouts: Auto-retry with exponential backoff
   - Storage issues: Alert when disk space < 10%

3. **User Feedback**:
   - Clear error messages in UI
   - Processing status in Slack
   - Error codes with resolution steps

## MVP Timeline
1. **Week 1**: Container Setup
   - Configure RunPod container
   - Set up local storage
   - Install core components

2. **Week 2**: Core Functionality
   - Implement file watcher
   - Basic document processing
   - OpenWebUI integration

3. **Week 3**: User Interaction
   - Add Slack bot
   - Implement search
   - Basic testing

## Clarified Requirements

### Document Processing
- **MVP Scope**:
  - Office documents only (DOCX, PDF, PPTX, XLSX, CSV)
  - No network configurations or specialized formats
  - Emphasis on text extraction and metadata handling

### Security (MVP)
- No authentication required for MVP
- All endpoints will be open
- Security features will be added in future phases

### Data Management (MVP)
- No retention policies for MVP
- All documents will be kept indefinitely
- Manual cleanup may be required between test cycles

### Monitoring (MVP)
- Basic logging to console
- No advanced monitoring required
- Debug mode available for troubleshooting

### Disaster Recovery (MVP)
- No formal disaster recovery for MVP
- Regular manual backups recommended
- Focus on core functionality first

## Next Steps
1. Review and approve specification
2. Set up development environment
3. Begin implementation of Phase 1
