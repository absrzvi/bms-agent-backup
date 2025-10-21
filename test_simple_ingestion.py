#!/usr/bin/env python3
"""
Simplified test for document ingestion with minimal validation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ingest_documents_to_qdrant import DocumentIngestionBridge
from qdrant_schema_v4 import QdrantConfig
from enhanced_document_processor import ProcessingConfig, ProcessingProfile, ChunkingStrategy

print("="*60)
print("  Simple Document Ingestion Test")
print("="*60)

# Test with a simple markdown file
test_file = "/workspace/bms-agent/bms-agent/reqs/qdrant_integration_guide.md"

print(f"\nTest file: {test_file}")

# Initialize bridge with MINIMAL validation
print("\n1. Initializing ingestion bridge (minimal validation)...")
try:
    bridge = DocumentIngestionBridge(
        qdrant_config=QdrantConfig(
            collection_name="railway_documents_v4",
            enable_railway_optimization=False  # Disable railway-specific processing
        ),
        processor_config=ProcessingConfig(
            processing_profile=ProcessingProfile.GENERAL,  # Use GENERAL profile
            chunking_strategy=ChunkingStrategy.SLIDING_WINDOW,  # Use sliding window chunking
            enable_contextual_retrieval=False,  # Disable contextual retrieval
            enable_late_chunking=False,  # Disable late chunking
            enable_hybrid_search=False,  # Disable hybrid search
            enable_quality_validation=False,  # DISABLE quality validation
            min_quality_score=0.0,  # Accept all chunks
            chunk_size=1000,  # Larger chunks
            chunk_overlap=100
        )
    )
    print("   ✓ Bridge initialized")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test ingestion
print("\n2. Testing document ingestion...")
try:
    result = bridge.ingest_document(test_file, document_id="test_simple")

    print("\n" + "="*60)
    print("Ingestion Result:")
    print("="*60)

    if result.get('success'):
        print(f"✅ SUCCESS!")
        print(f"   Document ID: {result['document_id']}")
        print(f"   Chunks: {result['chunks_ingested']}")
        print(f"   Type: {result['document_type']}")
    else:
        print(f"❌ FAILED: {result.get('error')}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Verify
print("\n3. Verifying in Qdrant...")
try:
    from qdrant_client import QdrantClient

    client = QdrantClient(url="localhost", port=6333)
    info = client.get_collection("railway_documents_v4")

    print(f"   Total Points: {info.points_count}")

    if info.points_count > 0:
        sample = client.scroll(
            collection_name="railway_documents_v4",
            limit=1,
            with_payload=True
        )[0][0]

        print(f"   Sample:")
        print(f"   - ID: {sample.id}")
        print(f"   - Content: {sample.payload.get('content', '')[:100]}...")

except Exception as e:
    print(f"   ⚠️  {e}")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
