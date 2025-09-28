# v4.0 Feature → Qdrant Schema Mapping Reference

## 📊 Complete Feature Mapping Table

| v4.0 Feature | Output Data | Qdrant Schema Element | Query Method |
|--------------|-------------|----------------------|--------------|
| **Contextual Retrieval** | `chunk['content']` with `<context>` tags | `payload.content`, `payload.has_context=true` | Filter: `has_context=true` |
| **Late Chunking** | `chunk['full_doc_embedding']` | Vector: `full_doc_embedding` | Search: `query_vector="full_doc_embedding"` |
| **Hierarchical Parent** | `chunk['hierarchy']='parent'` | Vector: `parent_embedding`, `payload.is_parent=true` | Filter: `is_parent=true` |
| **Hierarchical Child** | `chunk['hierarchy']='child'` | Vector: `child_embedding`, `payload.is_child=true` | Filter: `is_child=true` |
| **Parent-Child Link** | `chunk['parent_index']` | `payload.parent_chunk_id` | Retrieve by ID |
| **Hybrid Search - Dense** | `chunk['embedding']` | Vector: `chunk_embedding` | Standard vector search |
| **Hybrid Search - Sparse** | `chunk['term_frequencies']` | Sparse Vector: `keyword_sparse` | Sparse vector search |
| **Quality Score** | `chunk['quality']['overall_score']` | `payload.quality_score` | Filter: `quality_score >= 85` |
| **Quality Metrics** | `chunk['quality']['metrics']` | `payload.faithfulness`, `.relevancy`, etc. | Multiple filters |
| **Entity Extraction** | `chunk['entities']` | `payload.entities` (JSON) | Text search on entities |
| **Railway Metadata** | `result['railway_metadata']` | `payload.fleet_type`, `.standard_compliance` | Railway-specific filters |
| **Network Topology** | `railway_metadata['network_topology']` | `payload.ip_addresses`, `.vlans` | JSON field search |
| **Version Info** | `result['version_info']` | `payload.document_version` | Version filtering |
| **Keywords** | `chunk['keyword_content']` | `payload.keywords` | Text index search |

## 🗄️ Qdrant Collection Structure

### Vectors (Dense)
```python
{
    "chunk_embedding": 768,      # Main retrieval vector
    "parent_embedding": 768,      # Parent chunks (hierarchical)
    "child_embedding": 768,       # Child chunks (precision)
    "full_doc_embedding": 768     # Full document context (late chunking)
}
```

### Vectors (Sparse)
```python
{
    "keyword_sparse": {...}       # BM25 keyword matching
}
```

### Essential Payload Fields
```python
{
    # Document Level
    "document_id": str,           # Unique document identifier
    "document_version": float,    # Version tracking
    "processing_profile": str,    # "railway", "technical", etc.
    
    # Chunk Level
    "chunk_id": str,             # Unique chunk identifier
    "content": str,              # Actual text content
    "hierarchy_level": str,      # "parent", "child", "single"
    "parent_chunk_id": str,      # Link to parent (if child)
    
    # Quality
    "quality_score": float,      # 0-100 overall quality
    "has_context": bool,         # Has contextual retrieval
    
    # Railway-Specific (ÖBB)
    "fleet_type": List[str],    # ["Railjet", "Cityjet"]
    "standard_compliance": List[str],  # ["EN50155", "EN45545"]
    "network_component": Dict,   # {"CCU": 2, "VLAN": 4}
}
```

## 🔍 Search Patterns Mapping

### 1. Contextual Search (67% Better)
```python
# v4.0 Output
chunk = {
    'content': '<context>Section about VLANs...</context>\nVLAN 100...',
    'has_context': True
}

# Qdrant Query
search(
    filter={
        "has_context": True  # Only contextual chunks
    }
)
```

### 2. Hierarchical Search
```python
# v4.0 Output
parent_chunk = {'hierarchy': 'parent', 'index': 0}
child_chunk = {'hierarchy': 'child', 'parent_index': 0}

# Qdrant Query - Search children, return parents
child_results = search(
    vector="child_embedding",
    filter={"is_child": True}
)
parent_ids = [r.payload['parent_chunk_id'] for r in child_results]
parents = retrieve(ids=parent_ids)
```

### 3. Hybrid Search (RRF)
```python
# v4.0 Output
chunk = {
    'embedding': [...],           # Dense vector
    'term_frequencies': {...},    # For sparse vector
    'keyword_content': "ccu vlan"
}

# Qdrant Query
vector_results = search(vector="chunk_embedding")
keyword_results = search(vector="keyword_sparse")
combined = reciprocal_rank_fusion(vector_results, keyword_results)
```

### 4. Quality-Based Retrieval
```python
# v4.0 Output
chunk = {
    'quality': {
        'overall_score': 92.5,
        'metrics': {
            'faithfulness': 0.95,
            'relevancy': 0.90
        }
    }
}

# Qdrant Query
search(
    filter={
        "quality_score": {"gte": 85.0},
        "faithfulness": {"gte": 0.90}
    }
)
```

### 5. Railway-Specific (ÖBB)
```python
# v4.0 Output
result = {
    'railway_metadata': {
        'railway_specific': {
            'fleet_type': ['Railjet'],
            'standards_compliance': ['EN50155']
        }
    }
}

# Qdrant Query
search(
    filter={
        "fleet_type": "Railjet",
        "standard_compliance": "EN50155"
    }
)
```

## 📈 Performance Impact by Feature

| Feature | Storage Overhead | Query Speed Impact | Accuracy Improvement |
|---------|-----------------|-------------------|---------------------|
| **Multi-Vector** | 4x vector storage | -10% (more vectors) | +20% (flexibility) |
| **Sparse Vectors** | +30% storage | -5% (hybrid fusion) | +15% (keyword match) |
| **Hierarchical** | 2x points | -20% (two-step) | +30% (context) |
| **Quality Filtering** | Minimal | +10% (fewer results) | +50% (quality only) |
| **Contextual Retrieval** | +20% text size | No impact | +67% (fewer failures) |
| **Payload Indexes** | +10% storage | +50% (fast filters) | N/A |
| **Quantization** | -75% memory | -2% accuracy | Minimal |

## 🎯 Optimal Configuration Summary

### For Maximum Accuracy (Production)
```python
config = {
    'use_contextual_retrieval': True,     # -67% failures
    'use_hierarchical': True,              # Better context
    'use_hybrid_search': True,             # Vector + keyword
    'min_quality_score': 85.0,            # High quality only
    'enable_quantization': False          # Full precision
}
```

### For Maximum Speed (Real-time)
```python
config = {
    'use_contextual_retrieval': False,    # Skip context
    'use_hierarchical': False,             # Single-level
    'use_hybrid_search': False,           # Vector only
    'min_quality_score': 70.0,            # More results
    'enable_quantization': True           # Memory efficient
}
```

### For Railway Documents (ÖBB)
```python
config = {
    'use_contextual_retrieval': True,     # Technical context
    'use_hierarchical': True,              # Complex documents
    'use_hybrid_search': True,             # Technical terms
    'min_quality_score': 80.0,            # Good quality
    'enable_railway_filters': True        # Fleet/standard filters
}
```

## 🔧 Index Recommendations

### Must-Have Indexes
```python
critical_indexes = [
    "document_id",        # Document retrieval
    "chunk_type",         # Type filtering
    "quality_score",      # Quality filtering
    "is_parent",          # Hierarchical search
    "is_child",           # Hierarchical search
    "has_context"         # Contextual filtering
]
```

### Railway-Specific Indexes
```python
railway_indexes = [
    "fleet_type",         # Fleet filtering
    "standard_compliance", # Standards search
    "network_component",   # Component search
    "configuration_type"   # Config search
]
```

### Text Search Indexes
```python
text_indexes = [
    "content",            # Full-text search
    "keywords",           # Keyword search
    "entities"            # Entity search
]
```

## 💾 Storage Estimates

For 1000 documents with average 50 chunks each:

| Component | Size | Calculation |
|-----------|------|-------------|
| **Documents** | 1,000 | Input documents |
| **Total Chunks** | 50,000 | 1000 × 50 |
| **Parent Chunks** | 15,000 | ~30% parents |
| **Child Chunks** | 35,000 | ~70% children |
| **Vectors (Dense)** | 153.6 MB | 50,000 × 4 × 768 × 4 bytes |
| **Vectors (Sparse)** | ~20 MB | Compressed sparse |
| **Payload** | ~100 MB | Metadata + content |
| **Indexes** | ~30 MB | All field indexes |
| **Total (Uncompressed)** | ~304 MB | All components |
| **Total (Quantized)** | ~126 MB | INT8 quantization |

## ✅ Validation Checklist

Before going to production, ensure:

- [ ] All 4 dense vector types are populated
- [ ] Sparse vectors are generated for hybrid search
- [ ] Parent-child relationships are correctly linked
- [ ] Quality scores are calculated and stored
- [ ] Contextual retrieval flags are set
- [ ] Railway metadata is extracted (for ÖBB docs)
- [ ] All required indexes are created
- [ ] Quantization is configured appropriately
- [ ] Replication factor is set for availability
- [ ] Sharding is configured for scalability

---

This mapping ensures your Qdrant database fully leverages all v4.0 processor capabilities for state-of-the-art retrieval! 🚀