#!/usr/bin/env python3
"""
Test script for document ingestion pipeline
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from ingest_documents_to_qdrant import DocumentIngestionBridge
from qdrant_schema_v4 import QdrantConfig
from enhanced_document_processor import ProcessingConfig, ProcessingProfile, ChunkingStrategy

print("="*60)
print("  Testing Document Ingestion Pipeline")
print("="*60)

# Test with a simple markdown file
test_file = "/workspace/bms-agent/bms-agent/reqs/qdrant_integration_guide.md"

print(f"\nTest file: {test_file}")
print(f"File exists: {Path(test_file).exists()}")

if not Path(test_file).exists():
    print("❌ Test file not found!")
    sys.exit(1)

# Initialize bridge with configuration
print("\n1. Initializing ingestion bridge...")
try:
    bridge = DocumentIngestionBridge(
        qdrant_config=QdrantConfig(
            collection_name="railway_documents_v4",
            enable_railway_optimization=True
        ),
        processor_config=ProcessingConfig(
            processing_profile=ProcessingProfile.TECHNICAL,  # Use TECHNICAL for markdown
            chunking_strategy=ChunkingStrategy.HIERARCHICAL,
            enable_contextual_retrieval=True,
            enable_hybrid_search=True,
            enable_quality_validation=True
        )
    )
    print("   ✓ Bridge initialized")
except Exception as e:
    print(f"   ❌ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test ingestion
print("\n2. Testing document ingestion...")
try:
    result = bridge.ingest_document(test_file, document_id="test_qdrant_guide")

    print("\n" + "="*60)
    print("Ingestion Result:")
    print("="*60)

    if result.get('success'):
        print(f"✅ Status: SUCCESS")
        print(f"   Document ID: {result['document_id']}")
        print(f"   File: {result['file_name']}")
        print(f"   Type: {result['document_type']}")
        print(f"   Chunks Ingested: {result['chunks_ingested']}")
        print(f"   Processing Profile: {result['processing_profile']}")

        if result.get('quality_report'):
            qr = result['quality_report']
            print(f"\n   Quality Report:")
            print(f"   - Total Chunks: {qr.get('total_chunks', 'N/A')}")
            print(f"   - Passed: {qr.get('passed_chunks', 'N/A')}")
            print(f"   - Failed: {qr.get('failed_chunks', 'N/A')}")
            print(f"   - Avg Quality: {qr.get('average_quality', 'N/A'):.2f}")
    else:
        print(f"❌ Status: FAILED")
        print(f"   Error: {result.get('error')}")
        print(f"   Details: {result.get('details')}")

except Exception as e:
    print(f"❌ Ingestion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verify in Qdrant
print("\n3. Verifying in Qdrant...")
try:
    from qdrant_client import QdrantClient

    client = QdrantClient(url="localhost", port=6333)
    collection_info = client.get_collection("railway_documents_v4")

    print(f"   ✓ Collection Status: {collection_info.status}")
    print(f"   ✓ Total Points: {collection_info.points_count}")

    # Try to retrieve a point
    if collection_info.points_count > 0:
        scroll_result = client.scroll(
            collection_name="railway_documents_v4",
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        if scroll_result[0]:
            sample_point = scroll_result[0][0]
            print(f"\n   Sample Point Preview:")
            print(f"   - ID: {sample_point.id}")
            print(f"   - Document: {sample_point.payload.get('document_name', 'N/A')}")
            print(f"   - Content Length: {len(sample_point.payload.get('content', ''))}")
            print(f"   - Chunk Type: {sample_point.payload.get('chunk_type', 'N/A')}")
            print(f"   - Has Context: {sample_point.payload.get('has_context', False)}")

except Exception as e:
    print(f"   ⚠️  Could not verify: {e}")

print("\n" + "="*60)
print("✅ Test Complete!")
print("="*60)
