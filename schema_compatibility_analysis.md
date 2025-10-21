# Schema Compatibility Analysis

## Processor Output vs Qdrant Schema

### ✅ Compatible Fields

| Processor Field | Qdrant Schema Field | Status |
|----------------|---------------------|--------|
| `document_id` | `document_id` | ✓ Match |
| `file_name` | `document_name` | ✓ Match |
| `processing_timestamp` | `processing_timestamp` | ✓ Match |
| `config.profile` | `processing_profile` | ✓ Match |
| `chunks[].content` | `content` | ✓ Match |
| `chunks[].index` | `chunk_index` | ✓ Match |
| `chunks[].hierarchy` | `hierarchy_level` | ✓ Match |
| `chunks[].parent_index` | `parent_chunk_id` | ✓ Match |
| `chunks[].has_context` | `has_context` | ✓ Match |
| `chunks[].quality` | `quality_score`, `faithfulness`, etc. | ✓ Match |
| `chunks[].term_frequencies` | `keyword_sparse` (sparse vector) | ✓ Match |
| `railway_metadata` | `railway_metadata` | ✓ Match |
| `configurations` | `configurations` | ✓ Match |

### ⚠️ Fields Requiring Transformation

| Issue | Processor | Qdrant Schema | Solution |
|-------|-----------|---------------|----------|
| **Embeddings not generated** | `embedding: None` | Requires dense vectors | Need to generate embeddings using SentenceTransformer |
| **Missing document_type** | Not in output | Required field | Derive from file extension |
| **Vector naming** | Single `embedding` field | Multiple vectors: `chunk_embedding`, `parent_embedding`, `child_embedding` | Copy embedding to appropriate vector based on hierarchy |
| **Chunk ID format** | Simple index | `doc_id_index_hash` format | Generate using schema's `_generate_chunk_id` method |

### ❌ Missing Fields

| Field | Needed By | Default/Solution |
|-------|-----------|------------------|
| `document_type` | Qdrant schema | Derive from file extension (pdf, docx, etc.) |
| `document_version` | Qdrant schema | Use `version_info.version` or default to 1.0 |
| Embeddings | All vector fields | Generate using sentence-transformers |

## Integration Requirements

### 1. Embedding Generation
**Status:** ❌ Not handled by processor
**Solution:** Generate embeddings after processing using SentenceTransformer

```python
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

for chunk in result['chunks']:
    chunk['embedding'] = embedder.encode(chunk['content']).tolist()
```

### 2. Vector Distribution
**Status:** ⚠️ Needs mapping
**Solution:** Map single embedding to appropriate vectors based on hierarchy

```python
if chunk['hierarchy'] == 'parent':
    vectors['chunk_embedding'] = chunk['embedding']
    vectors['parent_embedding'] = chunk['embedding']
elif chunk['hierarchy'] == 'child':
    vectors['chunk_embedding'] = chunk['embedding']
    vectors['child_embedding'] = chunk['embedding']
else:
    vectors['chunk_embedding'] = chunk['embedding']
```

### 3. Document Type Detection
**Status:** ❌ Missing
**Solution:** Add helper function

```python
def get_document_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    mapping = {
        '.pdf': 'pdf',
        '.docx': 'word',
        '.doc': 'word',
        '.txt': 'text',
        '.md': 'markdown'
    }
    return mapping.get(ext, 'unknown')
```

## Recommendation

**Create an integration bridge script** that:
1. Uses `enhanced_document_processor.py` to process documents
2. Generates embeddings for all chunks
3. Transforms the output to match Qdrant schema expectations
4. Uses `qdrant_schema_v4.py` methods to ingest into Qdrant

This approach keeps both components independent while ensuring compatibility.
