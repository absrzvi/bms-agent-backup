# BMS Agent RunPod Deployment Plan

## Overview
Single-pod deployment on RunPod.io with direct binary installations (no Docker) for maximum performance and simplicity.

## Priority Order
1. Qdrant vector database setup
2. Document processor integration
3. Slack bot implementation
4. OpenWebUI integration

## Deployment Environment
- **Platform**: Runpod.io single pod (no Docker/Kubernetes)
- **Hardware**: 8-16 vCPUs, 32-64GB RAM, 200-500GB NVMe SSD
- **Existing Services**: Ollama, n8n, OpenWebUI (already installed)
- **To Install**: Qdrant vector database
- **Persistent Storage**: ~/persistent/ for all data

## Tech Stack & Models

### Core Services
- **Qdrant v1.7.4**: Direct binary installation (no Docker)
  - Multi-vector schema: chunk_embedding, parent_embedding (1024d for snowflake-arctic)
  - Sparse vectors for BM25 keyword search
  - On-disk storage for memory efficiency
  - Collection: railway_documents_v4

- **Ollama Models**:
  - Embeddings: snowflake-arctic-embed2 (1024 dimensions)
  - Generation: mistral-nemo:12b-instruct (7GB RAM, optimal performance)
  - Alternative: qwen2.5:14b or llama3.1:8b if needed

- **Python 3.11+**: Direct installation with venv
  - No containerization, runs as system process
  - FastAPI on port 8000 for API endpoints
  - Persistent data in ~/persistent/bms_data/

- **n8n Workflows**: 
  - Slack bot integration (Priority 1)
  - Document processing pipeline
  - Webhook endpoints for ingestion

- **OpenWebUI**: 
  - Custom Qdrant tool integration (Priority 2)
  - Railway expertise prompts

## Implementation Plan

### 1. Qdrant Installation & Setup (Day 1)
```bash
# Install Qdrant binary
wget https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
sudo mv qdrant /usr/local/bin/

# Configure persistent storage
mkdir -p ~/persistent/{qdrant_storage,qdrant_config,bms_data}

# Start Qdrant with optimized settings
nohup qdrant --storage-dir ~/persistent/qdrant_storage \
             --on-disk-payload true \
             --on-disk-vectors true \
             --log-level info \
             > ~/persistent/qdrant.log 2>&1 &
### 2. Initialize Qdrant Schema
```python
# setup_qdrant.py
from qdrant_client import QdrantClient, models

client = QdrantClient("localhost", port=6333)

# Create collection with 1024d vectors for snowflake-arctic-embed2
client.create_collection(
    collection_name="railway_docs",
    vectors_config={
        "chunk_embedding": models.VectorParams(
            size=1024,  # snowflake-arctic-embed2 dimension
            distance=models.Distance.COSINE,
            on_disk=True
        ),
        "parent_embedding": models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE,
            on_disk=True
        )
    },
    sparse_vectors_config={
        "keyword": models.SparseVectorParams()
    }
)

# Create payload indexes
client.create_payload_index(
    collection_name="railway_docs",
    field_name="fleet_type",
    field_schema="keyword"
)

client.create_payload_index(
    collection_name="railway_docs",
    field_name="standard_compliance",
    field_schema="keyword"
)
```

### 3. Document Processor Setup (Day 1-2)
```bash
# Set up Python environment
cd ~
git clone https://github.com/absrzvi/bms-agent.git
cd bms-agent
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install qdrant-client==1.7.0 fastapi uvicorn python-multipart
pip install nltk beautifulsoup4 pandas openpyxl

# Download NLTK data
python -m nltk.downloader punkt stopwords wordnet

# Create minimal processor
cat > mvp_processor.py << 'EOL'
import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
import httpx
import json

class MVPProcessor:
    def __init__(self):
        self.qdrant = QdrantClient("localhost", port=6333)
        self.ollama_url = "http://localhost:11434/api/embeddings"
        
    async def get_embedding(self, text: str) -> List[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.ollama_url,
                json={"model": "snowflake-arctic-embed2", "prompt": text}
            )
            return response.json()["embedding"]
    
    async def process_document(self, file_path: str):
        # TODO: Implement document processing logic
        pass

# Initialize processor
processor = MVPProcessor()
EOL

### 4. Test Ingestion (Day 2)
```python
# test_ingestion.py
import asyncio
from mvp_processor import processor

async def test_ingestion():
    test_doc = {
        "text": "Sample railway configuration for CCU R4600-2Ax.",
        "metadata": {
            "fleet_type": "Railjet",
            "standard_compliance": "EN50155",
            "source": "test_document.pdf"
        }
    }
    
    # Generate embeddings
    embedding = await processor.get_embedding(test_doc["text"])
    
    # Store in Qdrant
    points = [{
        "id": "test_doc_1",
        "vector": {"chunk_embedding": embedding},
        "payload": {
            "text": test_doc["text"],
            **test_doc["metadata"]
        }
    }]
    
    await processor.qdrant.upsert(
        collection_name="railway_docs",
        points=points
    )
    print("Test document ingested successfully!")

if __name__ == "__main__":
    asyncio.run(test_ingestion())
```

### 5. Slack Bot Integration (Day 3)

1. **n8n Workflow Setup**
   - Trigger: Slack Message
   - Action: HTTP Request to Ollama for embeddings
   - Search: Qdrant vector search
   - Generate: Ollama mistral-nemo:12b response
   - Reply: Post to Slack thread

2. **Required n8n Nodes**:
   - Slack Trigger
   - HTTP Request (Ollama embeddings)
   - Function (Qdrant query builder)
   - HTTP Request (Qdrant search)
   - HTTP Request (Ollama completion)
   - Slack Send Message

3. **Environment Variables**:
   ```
   OLLAMA_URL=http://localhost:11434
   QDRANT_URL=http://localhost:6333
   SLACK_BOT_TOKEN=xoxb-your-token
   ```

### 6. OpenWebUI Integration (Day 4)

1. **Create Tool Script** (`~/.openwebui/tools/qdrant_search.py`):
```python
def search_railway_docs(query: str, limit: int = 5) -> str:
    """Search railway documentation using Qdrant."""
    from qdrant_client import QdrantClient
    
    client = QdrantClient("localhost", port=6333)
    
    # Get query embedding from Ollama
    embedding = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "snowflake-arctic-embed2", "prompt": query}
    ).json()["embedding"]
    
    # Search Qdrant
    results = client.search(
        collection_name="railway_docs",
        query_vector=("chunk_embedding", embedding),
        limit=limit,
        with_vectors=False,
        with_payload=True
    )
    
    # Format results
    return "\n\n".join(
        f"Source: {r.payload.get('source', 'Unknown')}\n"
        f"Relevance: {r.score:.2f}\n"
        f"Content: {r.payload.get('text', '')[:500]}..."
        for r in results
    )
```

2. **Register Tool** in OpenWebUI configuration:
```yaml
tools:
  - name: search_railway_docs
    description: Search railway documentation
    parameters:
      type: object
      properties:
        query:
          type: string
          description: Search query
        limit:
          type: integer
          description: Maximum results to return
          default: 5
      required: ["query"]
```

## File Structure
```
~
├── persistent/
│   ├── qdrant_storage/        # Qdrant data
│   ├── bms_data/              # Processed documents
│   │   ├── uploads/           # User uploads
│   │   └── processed/         # Processed chunks
│   └── logs/                  # Application logs
│
└── bms-agent/                 # Git repository
    ├── venv/                  # Python virtual env
    ├── setup_qdrant.py        # Schema setup
    ├── mvp_processor.py       # Core processor
    ├── api/                   # FastAPI app
    │   ├── main.py
    │   └── endpoints/
    └── scripts/
        ├── start_services.sh  # Service manager
        └── backup.sh          # Backup utility
```
## Service Management

### Startup Script (`start_services.sh`)
```bash
#!/bin/bash

# Qdrant
if ! pgrep -x "qdrant" > /dev/null; then
    nohup qdrant --storage-dir ~/persistent/qdrant_storage \
                --on-disk-payload true \
                --on-disk-vectors true \
                > ~/persistent/logs/qdrant.log 2>&1 &
    echo "Started Qdrant"
fi

# API Server
cd ~/bms-agent
source venv/bin/activate
if ! pgrep -f "uvicorn" > /dev/null; then
    nohup uvicorn api.main:app \
                --host 0.0.0.0 \
                --port 8000 \
                > ~/persistent/logs/api.log 2>&1 &
    echo "Started API server"
fi

echo "All services started"
```
## Configuration (`.env`)
```env
# Core
PERSISTENT_PATH=~/persistent
LOG_LEVEL=INFO

# Qdrant
QDRANT_URL=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=railway_docs

# Ollama
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=snowflake-arctic-embed2
GENERATION_MODEL=mistral-nemo:12b-instruct

# Slack
SLACK_BOT_TOKEN=your-token-here
SLACK_SIGNING_SECRET=your-secret-here

# OpenWebUI
OPENWEBUI_URL=http://localhost:8080
```
## API Endpoints

### Document Ingestion
- `POST /api/v1/documents` - Upload and process document
- `POST /api/v1/documents/batch` - Process multiple documents
- `GET /api/v1/documents/{doc_id}` - Get document status

### Search
- `POST /api/v1/search/semantic` - Vector similarity search
- `POST /api/v1/search/hybrid` - Combined vector + keyword search
- `POST /api/v1/search/keyword` - BM25 keyword search

### System
- `GET /health` - Service health check
- `GET /metrics` - Performance metrics
- `GET /version` - API version info
Testing & Validation

Service Health: Verify all services running
Ingestion Test: Process sample railway doc
Retrieval Test: Query for CCU configurations
Slack Bot Test: Ask about VLAN settings
OpenWebUI Test: Full conversation about railway specs

Performance Targets (MVP)

Document processing: 10 docs/minute
Query latency: < 200ms
Concurrent users: 10
Storage efficiency: < 100GB for 10k documents
Memory usage: < 24GB total

Backup Strategy
bash# Daily cron job
0 2 * * * tar -czf ~/persistent/backups/bms_$(date +%Y%m%d).tar.gz \
  ~/persistent/qdrant_storage ~/persistent/bms_data
Monitoring

Simple bash script checking service health every 5 minutes
Log rotation for application logs
Resource usage tracking (CPU, memory, disk)
No external monitoring tools needed

Success Criteria

Qdrant running with schema initialized
Test document successfully ingested and retrieved
Slack bot responding to railway queries
OpenWebUI using Qdrant knowledge base
All services survive pod restart
Data persists in ~/persistent/

Priority Order

Qdrant installation and schema setup
Document processor with test ingestion
n8n Slack bot workflow
OpenWebUI Qdrant tool
Production optimizations

This plan focuses on getting a working MVP without Docker, using direct installations on the Runpod pod with persistent storage.

This `/plan` command is ready to paste into Windsurf. It's comprehensive yet focused on your specific requirements: single pod deployment, no Docker, integration with existing services, and the priority order you specified (Qdrant → Document processor → Slack bot → OpenWebUI).RetryAwhich file should i copy this in and how do i update the /plan command to use the file for inputEditPlotted file storage and command referencing strategy.The user wants to know how to save this large plan and reference it in Windsurf, similar to what I explained earlier. They need to:

Create a file to store the plan
Reference it with a shorter /plan command

I should provide clear instructions on where to save it and how to reference it.Where to Save the Plan
Option 1: Save in the .specify folder (Recommended)
bash# Create the file in your project's .specify folder
cd ~/bms-agent
mkdir -p .specify/plans
nano .specify/plans/runpod-deployment-plan.md

# Or create directly:
cat > .specify/plans/runpod-deployment-plan.md << 'EOF'
[Paste the entire plan content here]
EOF
Option 2: Save in the current feature spec folder
bash# If Windsurf already created a feature folder (like 001-bms-enhancement)
cd ~/bms-agent/.specify/specs/001-bms-enhancement/
nano deployment-plan.md
How to Reference in Windsurf
After saving the file, use this shortened /plan command in Windsurf:
Short Reference Command
/plan

Implement the comprehensive technical plan detailed in: .specify/plans/runpod-deployment-plan.md

Key points:
- Single Runpod.io pod deployment (no Docker)
- Integrate with existing Ollama, n8n, OpenWebUI
- Install Qdrant directly via binary
- Use snowflake-arctic-embed2 for embeddings
- Priority: Qdrant → Document processor → Slack bot → OpenWebUI
- All data in ~/persistent/ directory
Alternative: If Windsurf needs more context
/plan

Read and implement the deployment plan from: .specify/plans/runpod-deployment-plan.md

Summary: Deploy BMS Agent on Runpod.io without containerization, installing Qdrant directly and integrating with existing services (Ollama with snowflake-arctic-embed2, n8n, OpenWebUI). Focus on MVP with Slack bot first, then OpenWebUI integration.
Quick Setup Commands
bash# 1. Save the plan to file
cd ~/bms-agent
mkdir -p .specify/plans
cat > .specify/plans/runpod-deployment-plan.md << 'EOF'
[Paste your complete plan here]
EOF

# 2. Verify file was created
ls -la .specify/plans/
cat .specify/plans/runpod-deployment-plan.md | head -20

# 3. Go to Windsurf and use the short reference command
Git Integration
Don't forget to commit this plan to your repository:
bashcd ~/bms-agent
git add .specify/plans/runpod-deployment-plan.md
git commit -m "Add Runpod deployment plan for MVP"
git push origin main
This way, the plan is:

Version controlled in your repo
Available for Windsurf to reference
Documented for team members
Easy to update as needed

Would you like me to provide the exact commands to run first, or help you with creating the file?RetryClaude can make mistakes. Please double-check responses.
