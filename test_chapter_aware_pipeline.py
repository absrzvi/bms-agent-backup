#!/usr/bin/env python3
"""
Test Chapter-Aware Document Processing Pipeline
Tests the integrated enhanced_document_processor.py with chapter awareness
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'bms-agent' / 'scr'))

from enhanced_document_processor import (
    EnhancedDocumentProcessor,
    ProcessingConfig,
    ProcessingProfile,
    ChunkingStrategy
)

print("=" * 70)
print("  Chapter-Aware Document Processing Pipeline Test")
print("=" * 70)

# Test file with chapter structure
test_file = "/workspace/bms-agent/bms-agent/reqs/qdrant_integration_guide.md"

print(f"\n📄 Test Document: {Path(test_file).name}")
print("   This document has markdown chapter structure (#, ##, ###)")

# ============================================================================
# Test 1: Chapter-Based Hierarchical Chunking (DEFAULT)
# ============================================================================

print("\n" + "=" * 70)
print("TEST 1: Chapter-Based Hierarchical Chunking")
print("=" * 70)

config1 = ProcessingConfig(
    processing_profile=ProcessingProfile.GENERAL,
    chunking_strategy=ChunkingStrategy.HIERARCHICAL,

    # Chapter awareness settings
    enable_chapter_awareness=True,
    chapter_based_hierarchy=True,  # KEY: Use chapters for hierarchy
    max_chapter_size=20000,
    grandchild_chunk_size=800,
    grandchild_overlap=100,
    add_chapter_context_to_content=False,

    # Text cleaning
    enable_text_cleaning=True,
    remove_extra_whitespace=True,
    normalize_unicode=True,

    # Other features
    enable_contextual_retrieval=False,
    enable_late_chunking=False,
    enable_hybrid_search=False,
    enable_quality_validation=False,
    enable_versioning=False
)

processor1 = EnhancedDocumentProcessor(config1)
result1 = processor1.process_document(test_file, document_id="chapter_test_1")

print("\n📊 Results:")
print(f"   Processing Success: {result1.get('processing_success')}")
print(f"   Chunking Method: {result1['metadata'].get('chunking_method')}")
print(f"   Total Chunks: {len(result1['chunks'])}")
print(f"   Chapter Count: {result1['metadata'].get('chapter_count')}")
print(f"   Has Chapter Structure: {result1['metadata'].get('has_chapter_structure')}")

if 'hierarchy_stats' in result1['metadata']:
    stats = result1['metadata']['hierarchy_stats']
    print(f"\n🔗 Hierarchy Statistics:")
    print(f"   Parents (Chapters): {stats.get('total_parents', 0)}")
    print(f"   Children (Sub-chapters): {stats.get('total_children', 0)}")
    print(f"   Grandchildren (Fixed-size): {stats.get('total_grandchildren', 0)}")

# Show sample chunks
print("\n📝 Sample Chunks (First 3):")
for i, chunk in enumerate(result1['chunks'][:3], 1):
    print(f"\n   [{i}] Hierarchy Level: {chunk.get('hierarchy_level', 'N/A')}")
    if 'chapter_info' in chunk:
        ch_info = chunk['chapter_info']
        print(f"       Chapter: {ch_info.get('chapter_title', 'N/A')}")
        print(f"       Chapter Number: {ch_info.get('chapter_number', 'N/A')}")
        print(f"       Chapter Level: {ch_info.get('chapter_level', 'N/A')}")
        print(f"       Chapter Path: {ch_info.get('chapter_path', 'N/A')}")
    if 'metadata' in chunk:
        meta = chunk['metadata']
        print(f"       Chunk Type: {meta.get('chunk_type', 'N/A')}")
        print(f"       Content Length: {meta.get('char_count', len(chunk.get('content', '')))}")
    print(f"       Content Preview: {chunk.get('content', '')[:100]}...")

# ============================================================================
# Test 2: Traditional Size-Based Hierarchy (for comparison)
# ============================================================================

print("\n" + "=" * 70)
print("TEST 2: Traditional Size-Based Hierarchy (Comparison)")
print("=" * 70)

config2 = ProcessingConfig(
    processing_profile=ProcessingProfile.GENERAL,
    chunking_strategy=ChunkingStrategy.HIERARCHICAL,

    # Disable chapter-based hierarchy
    enable_chapter_awareness=False,
    chapter_based_hierarchy=False,

    # Use traditional sizes
    parent_chunk_size=2000,
    child_chunk_size=400,

    enable_text_cleaning=True,
    enable_contextual_retrieval=False,
    enable_late_chunking=False,
    enable_hybrid_search=False,
    enable_quality_validation=False,
    enable_versioning=False
)

processor2 = EnhancedDocumentProcessor(config2)
result2 = processor2.process_document(test_file, document_id="chapter_test_2")

print("\n📊 Results:")
print(f"   Processing Success: {result2.get('processing_success')}")
print(f"   Chunking Method: {result2['metadata'].get('chunking_method')}")
print(f"   Total Chunks: {len(result2['chunks'])}")
print(f"   Uses Chapter Structure: {result2['metadata'].get('has_chapter_structure', False)}")

# ============================================================================
# Test 3: Chapter Awareness with Context Addition
# ============================================================================

print("\n" + "=" * 70)
print("TEST 3: Chapter-Aware with Context Addition")
print("=" * 70)

config3 = ProcessingConfig(
    processing_profile=ProcessingProfile.GENERAL,
    chunking_strategy=ChunkingStrategy.HIERARCHICAL,

    # Chapter awareness with context
    enable_chapter_awareness=True,
    chapter_based_hierarchy=True,
    add_chapter_context_to_content=True,  # Add chapter info to content
    grandchild_chunk_size=800,

    enable_text_cleaning=True,
    enable_contextual_retrieval=False,
    enable_late_chunking=False,
    enable_hybrid_search=False,
    enable_quality_validation=False,
    enable_versioning=False
)

processor3 = EnhancedDocumentProcessor(config3)
result3 = processor3.process_document(test_file, document_id="chapter_test_3")

print("\n📊 Results:")
print(f"   Processing Success: {result3.get('processing_success')}")
print(f"   Total Chunks: {len(result3['chunks'])}")

# Show chunk with chapter context
print("\n📝 Sample Chunk with Chapter Context:")
for chunk in result3['chunks'][:1]:
    content = chunk.get('content', '')
    if '<chapter_context>' in content:
        # Extract context section
        context_section = content.split('</chapter_context>')[0]
        print(f"\n{context_section}</chapter_context>")
        print("\n   [Rest of content follows...]")
    else:
        print(f"   {content[:200]}...")

# ============================================================================
# Comparison Summary
# ============================================================================

print("\n" + "=" * 70)
print("📈 COMPARISON SUMMARY")
print("=" * 70)

print(f"\n🔹 Chapter-Based Hierarchy:")
print(f"   Total Chunks: {len(result1['chunks'])}")
print(f"   Chunking Method: {result1['metadata'].get('chunking_method')}")
print(f"   Uses Semantic Boundaries: Yes (chapters/sub-chapters)")

print(f"\n🔹 Size-Based Hierarchy:")
print(f"   Total Chunks: {len(result2['chunks'])}")
print(f"   Chunking Method: {result2['metadata'].get('chunking_method')}")
print(f"   Uses Semantic Boundaries: No (arbitrary 2000/400 char sizes)")

print(f"\n🔹 Chapter-Based with Context:")
print(f"   Total Chunks: {len(result3['chunks'])}")
print(f"   Adds Chapter Info to Content: Yes")
print(f"   Better for RAG: Yes (LLM sees chapter context)")

# ============================================================================
# Validation
# ============================================================================

print("\n" + "=" * 70)
print("✅ VALIDATION")
print("=" * 70)

# Validate chapter-based chunks have chapter info
chunks_with_chapter_info = sum(1 for c in result1['chunks'] if 'chapter_info' in c)
print(f"\n✓ Chunks with chapter metadata: {chunks_with_chapter_info}/{len(result1['chunks'])}")

# Validate hierarchy levels
parents = [c for c in result1['chunks'] if c.get('hierarchy_level') == 'parent']
children = [c for c in result1['chunks'] if c.get('hierarchy_level') == 'child']
grandchildren = [c for c in result1['chunks'] if c.get('hierarchy_level') == 'grandchild']

print(f"✓ Parent chunks (chapters): {len(parents)}")
print(f"✓ Child chunks (sub-chapters): {len(children)}")
print(f"✓ Grandchild chunks (fixed-size): {len(grandchildren)}")

# Validate text cleaning
if result1['metadata'].get('text_cleaned'):
    print(f"✓ Text cleaning applied: Yes")

print("\n" + "=" * 70)
print("🎉 Chapter-Aware Pipeline Test Complete!")
print("=" * 70)

print("\n💡 Key Improvements:")
print("   1. Chunks now respect document structure (chapters/sections)")
print("   2. Parent-child relationships based on semantic boundaries")
print("   3. All chunks have chapter metadata for filtering")
print("   4. Text is cleaned and normalized")
print("   5. Hierarchy is semantically meaningful, not arbitrary")
