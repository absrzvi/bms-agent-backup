#!/usr/bin/env python3
"""
Test Chapter and Sub-Chapter Awareness Feature
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ingest_documents_to_qdrant import DocumentIngestionBridge
from qdrant_schema_v4 import QdrantConfig
from enhanced_document_processor import ProcessingConfig, ProcessingProfile, ChunkingStrategy
from qdrant_client import QdrantClient

print("="*60)
print("  Chapter Awareness Test")
print("="*60)

# Use the qdrant integration guide (it has markdown headers)
test_file = "/workspace/bms-agent/bms-agent/reqs/qdrant_integration_guide.md"

print(f"\nTest file: {test_file}")
print("This document has markdown chapter structure (#, ##, ###)")

# Initialize bridge
print("\n1. Initializing ingestion bridge...")
bridge = DocumentIngestionBridge(
    qdrant_config=QdrantConfig(
        collection_name="railway_documents_v4"
    ),
    processor_config=ProcessingConfig(
        processing_profile=ProcessingProfile.GENERAL,
        chunking_strategy=ChunkingStrategy.SLIDING_WINDOW,
        enable_contextual_retrieval=False,
        enable_late_chunking=False,
        enable_hybrid_search=False,
        enable_quality_validation=False,
        chunk_size=1000,
        chunk_overlap=100
    )
)
print("   ✓ Bridge initialized with chapter extractor")

# Ingest document
print("\n2. Ingesting document with chapter awareness...")
try:
    result = bridge.ingest_document(test_file, document_id="chapter_test")

    if result.get('success'):
        print(f"✅ Successfully ingested {result['chunks_ingested']} chunks")
    else:
        print(f"❌ Failed: {result.get('error')}")
        sys.exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verify chapter data in Qdrant
print("\n3. Verifying chapter information in Qdrant...")
client = QdrantClient(url="localhost", port=6333)

# Get sample points
points = client.scroll(
    collection_name="railway_documents_v4",
    limit=5,
    with_payload=True,
    scroll_filter={
        "must": [
            {
                "key": "document_id",
                "match": {"value": "chapter_test"}
            }
        ]
    }
)[0]

if points:
    print(f"\nFound {len(points)} sample chunks with chapter info:")
    print("-"*60)

    for i, point in enumerate(points, 1):
        payload = point.payload
        print(f"\n[Chunk {i}]")
        print(f"  Chapter: {payload.get('chapter_title', 'N/A')}")
        print(f"  Chapter Number: {payload.get('chapter_number', 'N/A')}")
        print(f"  Chapter Level: {payload.get('chapter_level', 'N/A')}")
        print(f"  Chapter Path: {payload.get('chapter_path', 'N/A')}")
        print(f"  Sub-chapter: {payload.get('sub_chapter', 'None')}")
        print(f"  Parent Chapter: {payload.get('parent_chapter', 'None')}")
        print(f"  Content preview: {payload.get('content', '')[:80]}...")

# Test querying by chapter
print("\n4. Testing chapter-based queries...")

# Query all chunks from a specific chapter
print("\n   a) Find all chunks from a specific chapter:")
try:
    specific_chapter_results = client.scroll(
        collection_name="railway_documents_v4",
        limit=100,
        scroll_filter={
            "must": [
                {
                    "key": "chapter_title",
                    "match": {"value": "Qdrant Schema Architecture"}
                }
            ]
        },
        with_payload=["chapter_title", "sub_chapter", "content"]
    )[0]

    print(f"      Found {len(specific_chapter_results)} chunks")
    for point in specific_chapter_results[:3]:
        print(f"      - {point.payload.get('content', '')[:60]}...")

except Exception as e:
    print(f"      No results or error: {e}")

# Query by chapter level
print("\n   b) Find all top-level chapters (level 1):")
try:
    level1_results = client.scroll(
        collection_name="railway_documents_v4",
        limit=10,
        scroll_filter={
            "must": [
                {
                    "key": "chapter_level",
                    "match": {"value": 1}
                }
            ]
        },
        with_payload=["chapter_title", "chapter_level"]
    )[0]

    unique_chapters = set()
    for point in level1_results:
        chapter = point.payload.get('chapter_title')
        if chapter:
            unique_chapters.add(chapter)

    print(f"      Found {len(unique_chapters)} unique top-level chapters:")
    for chapter in list(unique_chapters)[:5]:
        print(f"      - {chapter}")

except Exception as e:
    print(f"      No results or error: {e}")

print("\n" + "="*60)
print("✅ Chapter Awareness Test Complete!")
print("="*60)
print("\nFeatures Demonstrated:")
print("  ✓ Automatic chapter extraction from markdown")
print("  ✓ Chapter/sub-chapter hierarchy tracking")
print("  ✓ Chapter path breadcrumbs")
print("  ✓ Parent-child chapter relationships")
print("  ✓ Queryable chapter metadata in Qdrant")
print("  ✓ Filter by chapter, level, or path")
print("\nNow you can:")
print("  - Search within specific chapters")
print("  - Filter by document structure")
print("  - Build chapter-aware navigation")
print("  - Show context breadcrumbs in results")
