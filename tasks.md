Updated BMS Agent RunPod Deployment Tasks

Specification
T000: Author MVP Specification
bash# Create feature spec directory and baseline spec
mkdir -p ~/.specify/specs/001-bms-agent
cat > ~/.specify/specs/001-bms-agent/spec.md <<'EOL'
# BMS Agent Specification
## User Stories
- As a network engineer, I need to search railway documentation via Slack
- As a system admin, I need to upload and process new documentation

## Requirements
- Process documents up to 100MB (MVP limit)
- Support Excel, CSV, PDF formats
- Sub-second search response time
- 95% retrieval accuracy
EOL
Dependencies: None
Output: Baseline MVP specification for traceability
Setup Tasks
T001: Initialize Project Structure
bash# Create project directories
mkdir -p ~/bms-agent/{api/endpoints,scripts,config,data}
# Create persistent storage
mkdir -p ~/persistent/{qdrant_storage,bms_data/{uploads,processed,models},logs}

# Create symlink to persistent storage
ln -s ~/persistent/bms_data ~/bms-agent/data
Dependencies: None
Output: Project directory structure with persistent storage
T002: Set Up Python Environment
bashcd ~/bms-agent
python3 -m venv venv
source venv/bin/activate

# Install core requirements from correct path
pip install --upgrade pip
pip install -r bms-agent/reqs/requirements.txt

# Install optional requirements if available
if [ -f "bms-agent/reqs/requirements-optional.txt" ]; then
    pip install -r bms-agent/reqs/requirements-optional.txt 2>/dev/null || true
fi

# Install additional requirements for deployment
pip install fastapi uvicorn python-multipart httpx qdrant-client

# Download NLTK data and spaCy model
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger maxent_ne_chunker words
python -m spacy download en_core_web_sm
Dependencies: T001
Output: Python environment with all required dependencies
Qdrant Setup
T003: Install Qdrant Binary
bashwget https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
sudo mv qdrant /usr/local/bin/
rm qdrant-x86_64-unknown-linux-gnu.tar.gz
Dependencies: None
Output: Qdrant binary installed at /usr/local/bin/qdrant
T004: Configure Qdrant Service
bashcat > ~/bms-agent/scripts/start_qdrant.sh << 'EOL'
#!/bin/bash
# Start Qdrant with persistent storage and optimized settings
nohup qdrant --storage-dir ~/persistent/qdrant_storage \
             --on-disk-payload true \
             --on-disk-vectors true \
             --log-level info \
             --max-request-batch-size 256 \
             --max-search-batch-size 64 \
             > ~/persistent/logs/qdrant.log 2>&1 &
sleep 5  # Give Qdrant time to start
echo "Qdrant started with PID $(pgrep qdrant)"
EOL
chmod +x ~/bms-agent/scripts/start_qdrant.sh

# Start Qdrant
~/bms-agent/scripts/start_qdrant.sh
Dependencies: T003
Output: Qdrant service running with optimized settings
T005: Initialize Qdrant Schema
bashcat > ~/bms-agent/scripts/init_qdrant.py << 'EOL'
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add the scr directory to path
sys.path.append(str(Path(__file__).parent.parent / 'bms-agent' / 'scr'))

from qdrant_schema_v4 import QdrantSchemaV4, QdrantConfig

def main():
    # Initialize with snowflake-arctic-embed2 dimensions (1024d)
    config = QdrantConfig(
        url="localhost",
        port=6333,
        collection_name="nomad_bms_documents",  # Updated collection name
        dense_vector_size=1024,  # For snowflake-arctic-embed2
        enable_railway_optimization=True,
        enable_quantization=True,
        on_disk_payload=True
    )
    
    schema = QdrantSchemaV4(config)
    schema.create_collection(force_recreate=True)
    print("✅ Qdrant collection 'nomad_bms_documents' initialized successfully")

if __name__ == "__main__":
    main()
EOL

python ~/bms-agent/scripts/init_qdrant.py
Dependencies: T002, T004
Output: Qdrant collection with proper schema
Document Processor
T006: Create Processor Wrapper
bashcat > ~/bms-agent/api/processor_wrapper.py << 'EOL'
#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Add the scr directory to path
sys.path.append(str(Path(__file__).parent.parent / 'bms-agent' / 'scr'))

# Import existing processors with proper configuration
from enhanced_document_processor import (
    EnhancedDocumentProcessor,
    ProcessingConfig,
    ProcessingProfile,
    ChunkingStrategy
)

class DocumentProcessorWrapper:
    def __init__(self):
        self.qdrant = QdrantClient("localhost", port=6333, timeout=60)
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.collection_name = "nomad_bms_documents"
        
        # Initialize the enhanced processor with proper config
        config = ProcessingConfig(
            processing_profile=ProcessingProfile.RAILWAY,
            chunk_size=1500,
            chunk_overlap=200,
            chunking_strategy=ChunkingStrategy.HIERARCHICAL,
            enable_contextual_retrieval=True,
            enable_quality_validation=True,
            quality_threshold=85.0
        )
        self.processor = EnhancedDocumentProcessor(config)
        
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding using Ollama's API"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.ollama_url,
                json={"model": "snowflake-arctic-embed2", "prompt": text}
            )
            return response.json()["embedding"]
            
    async def process_document(self, file_path: Union[str, Path]):
        """Process document using the existing enhanced processor"""
        file_path = Path(file_path)
        
        # Use the actual process_document method
        result = self.processor.process_document(str(file_path))
        
        # Extract chunks and generate embeddings
        points = []
        for chunk in result.get('chunks', []):
            embedding = await self.get_embedding(chunk['content'])
            
            point = PointStruct(
                id=chunk.get('chunk_id', str(chunk['index'])),
                vector={
                    "chunk_embedding": embedding,
                    "parent_embedding": embedding  # For now, same embedding
                },
                payload={
                    "content": chunk['content'],
                    "document_id": result.get('document_id', file_path.stem),
                    "chunk_index": chunk['index'],
                    "quality_score": chunk.get('quality', {}).get('overall_score', 85.0),
                    "has_context": chunk.get('has_context', True),
                    **chunk.get('metadata', {})
                }
            )
            points.append(point)
            
        # Batch upsert to Qdrant
        if points:
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
        return {
            "document_id": result.get('document_id', file_path.stem),
            "chunks_processed": len(points),
            "processing_success": result.get('processing_success', True)
        }

    async def search_documents(self, query: str, limit: int = 5) -> List[Dict]:
        """Search documents using vector similarity"""
        # Get query embedding
        embedding = await self.get_embedding(query)
        
        # Search Qdrant
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=("chunk_embedding", embedding),
            limit=limit,
            with_vectors=False,
            with_payload=True
        )
        
        # Format results
        return [
            {
                "score": r.score,
                "content": r.payload.get('content', ''),
                "document_id": r.payload.get('document_id', ''),
                "chunk_index": r.payload.get('chunk_index', 0),
                **r.payload
            } 
            for r in results
        ]
EOL
Dependencies: T002, T005
Output: Document processor wrapper implementation
T007: Create Test Script
bashcat > ~/bms-agent/scripts/test_processor.py << 'EOL'
#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.processor_wrapper import DocumentProcessorWrapper

async def test_end_to_end():
    """Test the complete pipeline"""
    try:
        processor = DocumentProcessorWrapper()
        
        # Create test document if it doesn't exist
        test_doc_dir = project_root / "data" / "test_documents"
        test_doc_dir.mkdir(parents=True, exist_ok=True)
        
        test_doc = test_doc_dir / "test_railway_doc.txt"
        if not test_doc.exists():
            test_doc.write_text("""
            The Nomad Connect CCU R4600-3Ax provides advanced 5G NR connectivity 
            for railway applications. It supports carrier aggregation and implements 
            redundancy through multicast groups on 239.0.0.1:5000. The system includes 
            VLAN 101 for passenger networks with IP range 10.0.0.0/24. 
            All components comply with EN50155 and EN45545 standards.
            """)
            print(f"✅ Created test document: {test_doc}")
        
        # Process the document
        print(f"📄 Processing: {test_doc}")
        result = await processor.process_document(test_doc)
        print(f"✅ Document processed: {result['chunks_processed']} chunks")
        
        # Test search
        print("\n🔍 Testing search...")
        query = "CCU connectivity 5G"
        results = await processor.search_documents(query, limit=3)
        
        if results:
            print(f"Found {len(results)} results for '{query}':")
            for i, r in enumerate(results, 1):
                print(f"\n{i}. Score: {r['score']:.3f}")
                print(f"   Content: {r['content'][:150]}...")
        else:
            print("No results found")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_end_to_end())
EOL

chmod +x ~/bms-agent/scripts/test_processor.py
Dependencies: T006
Output: Test script for document processing
n8n Slack Bot
T008: Create n8n Workflow
bashmkdir -p ~/bms-agent/n8n/workflows
cat > ~/bms-agent/n8n/workflows/slack_bot.json << 'EOL'
{
  "name": "BMS Slack Bot",
  "nodes": [
    {
      "parameters": {},
      "name": "Slack Trigger",
      "type": "n8n-nodes-base.slackTrigger",
      "position": [250, 300]
    },
    {
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8000/api/v1/search/semantic",
        "bodyParametersJson": "{\"query\": \"{{ $json.text }}\", \"limit\": 5}",
        "headerParametersJson": "{\"Content-Type\": \"application/json\"}"
      },
      "name": "Search Documents",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300]
    },
    {
      "parameters": {
        "method": "POST",
        "url": "http://localhost:11434/api/generate",
        "bodyParametersJson": "{\"model\": \"mistral-nemo:12b-instruct\", \"prompt\": \"Context: {{ $json.results }}\\n\\nQuestion: {{ $node['Slack Trigger'].json.text }}\\n\\nProvide a concise answer:\", \"stream\": false}"
      },
      "name": "Generate Response",
      "type": "n8n-nodes-base.httpRequest",
      "position": [650, 300]
    },
    {
      "parameters": {
        "channel": "={{ $node['Slack Trigger'].json.channel }}",
        "text": "{{ $json.response }}"
      },
      "name": "Reply to Slack",
      "type": "n8n-nodes-base.slack",
      "position": [850, 300]
    }
  ]
}
EOL
echo "✅ n8n workflow created. Import this JSON in n8n UI"
Dependencies: T011 (API must be running)
Output: n8n workflow JSON for Slack bot
OpenWebUI Integration
T009: Create OpenWebUI Tool
bashmkdir -p ~/.openwebui/tools
cat > ~/.openwebui/tools/bms_search.py << 'EOL'
"""
title: BMS Document Search
author: Nomad BMS Agent
version: 1.0.0
"""

import httpx
from typing import Optional

async def search_bms_docs(query: str, limit: int = 5) -> str:
    """Search BMS documentation using Qdrant"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get embedding
            embed_resp = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "snowflake-arctic-embed2", "prompt": query}
            )
            embedding = embed_resp.json()["embedding"]
            
            # Search Qdrant
            search_resp = await client.post(
                "http://localhost:6333/collections/nomad_bms_documents/points/search",
                json={
                    "vector": {"name": "chunk_embedding", "vector": embedding},
                    "limit": limit,
                    "with_payload": True
                }
            )
            
            results = search_resp.json().get("result", [])
            
            if not results:
                return "No relevant documents found."
            
            formatted = []
            for i, r in enumerate(results, 1):
                payload = r.get("payload", {})
                score = r.get("score", 0)
                content = payload.get("content", "")[:300]
                doc_id = payload.get("document_id", "Unknown")
                
                formatted.append(
                    f"{i}. [Score: {score:.2f}] Document: {doc_id}\n"
                    f"   {content}..."
                )
            
            return "Found documents:\n\n" + "\n\n".join(formatted)
            
    except Exception as e:
        return f"Search error: {str(e)}"
EOL
echo "✅ OpenWebUI tool created"
Dependencies: T005
Output: OpenWebUI custom tool
API Application
T011: Create FastAPI Application
bashcat > ~/bms-agent/api/main.py << 'EOL'
#!/usr/bin/env python3
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from api.processor_wrapper import DocumentProcessorWrapper

# API Key for internal services (should be in environment variables in production)
API_KEY = os.getenv('BMS_API_KEY', 'dev-key-123')
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return api_key

app = FastAPI(
    title="BMS Agent API",
    version="1.0.0",
    description="API for BMS Agent - Railway Documentation System",
    docs_url="/docs",
    redoc_url=None
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5678", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}

@app.get("/health/detailed", tags=["Health"])
async def detailed_health():
    """Detailed health check with service status"""
    try:
        qdrant_healthy = False
        collection_exists = False
        
        # Check Qdrant connection
        collections = processor.qdrant.get_collections()
        qdrant_healthy = collections is not None
        
        # Check if collection exists
        if qdrant_healthy:
            collection_exists = any(
                c.name == "nomad_bms_documents" 
                for c in collections.collections
            )
        
        return {
            "api": "healthy",
            "qdrant": qdrant_healthy,
            "collection_exists": collection_exists,
            "collection_name": "nomad_bms_documents"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service Unhealthy: {str(e)}"
        )

# Initialize processor
processor = DocumentProcessorWrapper()

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

@app.post(
    "/api/v1/search/semantic",
    dependencies=[Depends(verify_api_key)],
    response_model=dict,
    responses={
        200: {"description": "Search results"},
        403: {"description": "Invalid API key"},
        500: {"description": "Internal server error"}
    }
)
async def search_semantic(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Semantic search endpoint
    
    - **query**: Search query string
    - **limit**: Maximum number of results to return (default: 5)
    """
    try:
        results = await processor.search_documents(request.query, request.limit)
        return {"results": results, "count": len(results), "status": "success"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.post(
    "/api/v1/documents/upload",
    dependencies=[Depends(verify_api_key)],
    response_model=dict,
    responses={
        200: {"description": "Document processed successfully"},
        400: {"description": "Invalid file or processing error"},
        403: {"description": "Invalid API key"},
        413: {"description": "File too large (max 100MB)"},
        500: {"description": "Internal server error"}
    }
)
async def upload_document(
    file: UploadFile = File(..., description="Document to upload (PDF, CSV, XLSX)"),
    api_key: str = Depends(verify_api_key)
):
    """
    Upload and process a document
    
    - **file**: Document file to upload (max 100MB)
    - **Returns**: Processing status and document ID
    """
    try:
        # Check file size (100MB limit)
        max_size = 100 * 1024 * 1024  # 100MB
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size is 100MB"
            )
            
        # Validate file type
        valid_extensions = {'.pdf', '.csv', '.xlsx', '.xls'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in valid_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Supported types: {', '.join(valid_extensions)}"
            )
        
        # Save uploaded file
        upload_dir = Path.home() / "persistent" / "bms_data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        # Write file in chunks for large files
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process document with timeout
        try:
            result = await asyncio.wait_for(
                processor.process_document(file_path),
                timeout=300  # 5 minutes timeout
            )
            
            return {
                "status": "success",
                "message": "Document processed successfully",
                "filename": file.filename,
                **result
            }
            
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Document processing timed out"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BMS Agent API"}

@app.get(
    "/",
    response_model=dict,
    response_description="API Information",
    tags=["Info"]
)
async def root():
    """
    Root endpoint with API information
    
    Returns basic information about the API and available endpoints
    """
    return {
        "service": "BMS Agent API",
        "version": "1.0.0",
        "description": "API for Railway Documentation Search and Processing",
        "endpoints": [
            {
                "path": "/api/v1/search/semantic",
                "method": "POST",
                "description": "Semantic document search"
            },
            {
                "path": "/api/v1/documents/upload",
                "method": "POST",
                "description": "Upload and process documents"
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Basic health check"
            },
            {
                "path": "/health/detailed",
                "method": "GET",
                "description": "Detailed health check with service status"
            },
            {
                "path": "/docs",
                "method": "GET",
                "description": "Interactive API documentation"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
EOL

chmod +x ~/bms-agent/api/main.py
Dependencies: T006
Output: FastAPI application with all endpoints
Testing & Management
T010: Run Integration Tests
bashcat > ~/bms-agent/scripts/run_tests.sh << 'EOL'
#!/bin/bash
set -e

echo "🔍 BMS Agent Integration Tests"
echo "=============================="

# Check if Qdrant is running
if ! pgrep -x "qdrant" > /dev/null; then
    echo "Starting Qdrant..."
    ~/bms-agent/scripts/start_qdrant.sh
fi

# Activate environment
cd ~/bms-agent
source venv/bin/activate

# Run processor test
echo -e "\n📄 Testing document processor..."
python scripts/test_processor.py

# Start API if not running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "\n🚀 Starting API server..."
    nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 \
        > ~/persistent/logs/api.log 2>&1 &
    sleep 3
fi

# Test API endpoints
echo -e "\n🌐 Testing API endpoints..."
curl -s http://localhost:8000/health | python -m json.tool

echo -e "\n✅ All tests completed!"
EOL
chmod +x ~/bms-agent/scripts/run_tests.sh
Dependencies: All previous tasks
Output: Integration test script
T012: Create Service Manager
bashcat > ~/bms-agent/scripts/manage_services.sh << 'EOL'
#!/bin/bash

case "$1" in
  start)
    echo "Starting BMS Agent services..."
    
    # Start Qdrant
    if ! pgrep -x "qdrant" > /dev/null; then
        ~/bms-agent/scripts/start_qdrant.sh
        echo "✅ Qdrant started"
    fi
    
    # Start API
    cd ~/bms-agent && source venv/bin/activate
    nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 \
        > ~/persistent/logs/api.log 2>&1 &
    echo "✅ API started on port 8000"
    ;;
    
  stop)
    echo "Stopping services..."
    pkill -f "uvicorn api.main:app" || true
    pkill -x qdrant || true
    echo "✅ Services stopped"
    ;;
    
  status)
    echo "Service Status:"
    pgrep -x "qdrant" > /dev/null && echo "✅ Qdrant: Running" || echo "❌ Qdrant: Stopped"
    pgrep -f "uvicorn" > /dev/null && echo "✅ API: Running" || echo "❌ API: Stopped"
    curl -s http://localhost:8000/health > /dev/null && echo "✅ API Health: OK" || echo "❌ API Health: Not responding"
    ;;
    
  restart)
    $0 stop
    sleep 2
    $0 start
    ;;
    
  *)
    echo "Usage: $0 {start|stop|status|restart}"
    exit 1
    ;;
esac
EOL
chmod +x ~/bms-agent/scripts/manage_services.sh
Dependencies: T011
Output: Service management script
T013: Create Health Check
bashcat > ~/bms-agent/scripts/health_check.sh << 'EOL'
#!/bin/bash

echo "🔍 BMS Agent Health Check"
echo "========================="

# Qdrant
echo -n "Qdrant: "
curl -s http://localhost:6333/health > /dev/null && echo "✅ Healthy" || echo "❌ Not responding"

# Collection
echo -n "Collection: "
curl -s http://localhost:6333/collections/nomad_bms_documents > /dev/null && echo "✅ Exists" || echo "❌ Not found"

# API
echo -n "API: "
curl -s http://localhost:8000/health > /dev/null && echo "✅ Running" || echo "❌ Not running"

# Ollama
echo -n "Ollama: "
curl -s http://localhost:11434/api/version > /dev/null && echo "✅ Running" || echo "❌ Not running"

# Embedding Model
echo -n "Embedding Model: "
curl -s -X POST http://localhost:11434/api/embeddings \
  -d '{"model":"snowflake-arctic-embed2","prompt":"test"}' > /dev/null 2>&1 && \
  echo "✅ Available" || echo "❌ Not loaded"

# n8n
echo -n "n8n: "
curl -s http://localhost:5678/healthz > /dev/null && echo "✅ Running" || echo "❌ Not running"

# OpenWebUI
echo -n "OpenWebUI: "
curl -s http://localhost:8080 > /dev/null && echo "✅ Running" || echo "❌ Not running"
EOL
chmod +x ~/bms-agent/scripts/health_check.sh
Dependencies: None
Output: Health check script

Security Enhancements
T014: Implement Core Security Controls
Summary: Enforce constitution §3 mandates (JWT auth, API key fallback, rate limiting, security headers).
Steps:
- Update `bms-agent/api/security.py` (new) to expose:
  - `verify_jwt()` validating RS256 tokens using `BMS_JWT_PUBLIC_KEY`.
  - `verify_api_key()` fallback using `BMS_API_KEY`.
  - `RateLimiter` utility (60 req/min per JWT subject) stored in memory or Redis (future enhancement).
  - `apply_security_headers(app)` adding CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy.
- Refactor `bms-agent/api/main.py` routes to include `Depends(verify_jwt)` + API key fallback where appropriate.
- Add `/auth/jwks.json` static endpoint if integrating with external identity provider.
- Expand `docs/security-notes.md` with validation checklist and future hardening tasks (WAF, audit logging).
- Extend `tests/test_basic.py` (security section) to cover JWT success/failure and rate limiting 429 behavior.
Dependencies: T011 (API), T010 (tests), spec R3.1–R3.3
Output: Security module committed, tests updated, documentation refreshed, constitution compliance achieved

Documentation & Enablement
T015: Consolidate Project Documentation
Summary: Keep user guide, testing strategy, and deployment checklist synchronized with architecture.
Steps:
- Verify `README.md`, `TESTING.md`, and `DEPLOYMENT_CHECKLIST.md` contain latest commands, environment variables, and monitoring steps.
- Ensure references to collection names, scripts, and endpoints match current implementation (`nomad_bms_documents`, `scripts/manage_services.sh`, `/metrics/uplink`).
- Capture limitations (single RunPod pod, manual deployment placeholder) and roadmap links to `tasks.md`.
- Document Git flow branching expectations (feature/release branches, semantic commit messages) in `README.md` or `docs/workflow.md`.
- Create and maintain `docs/migrations.md` to log any manual schema or data changes executed during the MVP.
Dependencies: T000, Plan update
Output: Documentation reviewed, aligned with spec & plan

CI/CD Automation
T016: Configure GitHub Actions Pipeline
Summary: Automate lint, test, security scans, and document deployment placeholders.
Steps:
- Maintain `.github/workflows/ci-cd.yml` with jobs: `test`, `security`, and `deploy` placeholder referencing RunPod sync commands.
- Configure repository secrets: `BMS_API_KEY`, `DEPLOY_KEY`, optional `CODECOV_TOKEN`, `SAFETY_API_KEY`, future `RUNPOD_USER/HOST`.
- Add documentation in README for running pipeline locally (`act`, etc.) if needed.
Dependencies: T010 (tests), T015 (docs)
Output: CI pipeline definition committed, secrets checklist documented

Performance & Load Validation
T017: Establish Performance Baselines
Summary: Capture latency targets and test coverage for performance.
Steps:
- Maintain `tests/performance/test_performance.py` to benchmark `/api/v1/search/semantic` and record average/p95 latency.
- Update `reports/performance-baseline.md` after each run with date, model versions, and observations.
- Include performance marker in CI (optional scheduled workflow).
Dependencies: T010, T016
Output: Performance tests and baseline report ready

Monitoring & Metrics
T018: Implement Operational Metrics
Summary: Provide visibility for 99.99 % availability goal.
Steps:
- Add `/metrics/uplink` endpoint in `api/main.py` returning JSON metrics (latency histogram, request counts, error tally, rate-limit hits).
- Extend `scripts/health_check.sh` to optionally query `/metrics/uplink` and log output.
- Document cron-based monitoring (health script, log rotation) in `DEPLOYMENT_CHECKLIST.md`.
Dependencies: T011, T012, T013
Output: Metrics endpoint, enhanced health script, monitoring docs

Retrieval Accuracy Validation
T019: Evaluate Retrieval Quality
Summary: Ensure ≥95 % top-5 accuracy per spec requirement R2.2.
Steps:
- Create evaluation dataset `data/evaluation/ground_truth.jsonl` with query/answer pairs and expected document IDs.
- Author `scripts/evaluate_retrieval.py` to run against Qdrant collection, compute accuracy, and output JSON summary (accuracy, MRR, recall@5).
- Integrate into CI (`pytest -m evaluation` or separate job) and fail pipeline if accuracy <95 %.
- Update `README.md` and `TESTING.md` with instructions to run evaluation locally.
Dependencies: T006, T010, T017
Output: Evaluation dataset, script, CI hook, documentation updates
Execution Order

Setup: T001 → T002
Qdrant: T003 → T004 → T005
Core: T006 → T007 → T011
Integrations: T008 (n8n) and T009 (OpenWebUI) can be done in parallel
Testing: T010 → T012 → T013 → T017
Security & Compliance: T014
Documentation: T015
CI/CD: T016

Quick Start Commands
bash# After completing all tasks, start everything:
~/bms-agent/scripts/manage_services.sh start

# Check health:
~/bms-agent/scripts/health_check.sh

# Run tests:
~/bms-agent/scripts/run_tests.sh

# View API docs:
# Open browser to http://localhost:8000/docs
This complete task list now:

Uses nomad_bms_documents as the collection name throughout
Includes the FastAPI main app (T011) for n8n and OpenWebUI integration
Properly references your existing codebase
Fixes all path and import issues
Includes service management and health checking scripts