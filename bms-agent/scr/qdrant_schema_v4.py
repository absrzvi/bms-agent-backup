#!/usr/bin/env python3
"""
Qdrant Vector Database Schema for Enhanced Document Processor v4.0
Optimized for all v4.0 features including hierarchical chunking, hybrid search, and railway-specific data
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    CollectionInfo, OptimizersConfig, HnswConfigDiff,
    QuantizationConfig, ScalarQuantization, ScalarType,
    PayloadSchemaType, PayloadIndexParams, TextIndexParams,
    TokenizerType, Filter, FieldCondition, Range, MatchValue,
    SearchRequest, NamedVector, SparseVector, SparseVectorParams
)

# =============================
# Configuration Constants
# =============================

class VectorType(Enum):
    """Vector types in our multi-vector schema"""
    CHUNK_EMBEDDING = "chunk_embedding"  # Main chunk embedding
    PARENT_EMBEDDING = "parent_embedding"  # Parent chunk embedding
    CHILD_EMBEDDING = "child_embedding"  # Child chunk embedding
    FULL_DOC_EMBEDDING = "full_doc_embedding"  # Full document embedding (late chunking)
    KEYWORD_EMBEDDING = "keyword_embedding"  # Sparse vector for BM25

@dataclass
class QdrantConfig:
    """Qdrant configuration for v4.0 document processor"""
    
    # Connection
    url: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    
    # Collection settings
    collection_name: str = "railway_documents_v4"
    
    # Vector dimensions (for all-mpnet-base-v2)
    dense_vector_size: int = 768
    sparse_vector_size: int = 30000  # For BM25 vocabulary
    
    # Performance settings
    shard_count: int = 6  # Should be multiple of node count
    replication_factor: int = 2
    
    # HNSW parameters
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_full_scan_threshold: int = 10000
    
    # Quantization
    enable_quantization: bool = True
    quantization_type: str = "int8"
    
    # Railway-specific
    enable_railway_optimization: bool = True

# =============================
# Schema Definition
# =============================

class QdrantSchemaV4:
    """
    Qdrant schema optimized for Enhanced Document Processor v4.0
    Supports all features including hierarchical chunking and hybrid search
    """
    
    def __init__(self, config: QdrantConfig = None):
        self.config = config or QdrantConfig()
        self.client = QdrantClient(
            url=self.config.url,
            port=self.config.port,
            api_key=self.config.api_key
        )
    
    def create_collection(self, force_recreate: bool = False) -> None:
        """Create optimized collection with multi-vector support"""
        
        collection_name = self.config.collection_name
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists and not force_recreate:
            print(f"Collection '{collection_name}' already exists")
            return
        elif exists and force_recreate:
            self.client.delete_collection(collection_name)
            print(f"Deleted existing collection '{collection_name}'")
        
        # Create collection with multiple vector types
        self.client.create_collection(
            collection_name=collection_name,
            
            # Multi-vector configuration
            vectors_config={
                # Main chunk embedding (for standard retrieval)
                "chunk_embedding": VectorParams(
                    size=self.config.dense_vector_size,
                    distance=Distance.COSINE
                ),
                
                # Parent embedding (for hierarchical retrieval)
                "parent_embedding": VectorParams(
                    size=self.config.dense_vector_size,
                    distance=Distance.COSINE,
                    on_disk=False  # Keep in memory for fast access
                ),
                
                # Child embedding (for precise matching)
                "child_embedding": VectorParams(
                    size=self.config.dense_vector_size,
                    distance=Distance.COSINE
                ),
                
                # Full document embedding (from late chunking)
                "full_doc_embedding": VectorParams(
                    size=self.config.dense_vector_size,
                    distance=Distance.COSINE,
                    on_disk=True  # Can be on disk as accessed less frequently
                )
            },
            
            # Sparse vectors for hybrid search (BM25)
            sparse_vectors_config={
                "keyword_sparse": SparseVectorParams(
                    # No size needed for sparse vectors
                )
            },
            
            # Sharding for scalability
            shard_number=self.config.shard_count,
            replication_factor=self.config.replication_factor,
            
            # Optimization settings
            optimizers_config=OptimizersConfig(
                indexing_threshold=20000,
                max_segment_size=200000
            ),
            
            # HNSW configuration
            hnsw_config=HnswConfigDiff(
                m=self.config.hnsw_m,
                ef_construct=self.config.hnsw_ef_construct,
                full_scan_threshold=self.config.hnsw_full_scan_threshold
            ),
            
            # Quantization for memory efficiency
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True
                )
            ) if self.config.enable_quantization else None
        )
        
        print(f"✅ Created collection '{collection_name}' with multi-vector support")
        
        # Create payload indexes
        self._create_payload_indexes()
    
    def _create_payload_indexes(self) -> None:
        """Create optimized payload indexes for filtering"""
        
        collection_name = self.config.collection_name
        
        # Document-level indexes
        document_indexes = [
            ("document_id", PayloadSchemaType.KEYWORD),
            ("document_name", PayloadSchemaType.KEYWORD),
            ("document_type", PayloadSchemaType.KEYWORD),
            ("document_version", PayloadSchemaType.FLOAT),
            ("processing_profile", PayloadSchemaType.KEYWORD),
            ("processing_timestamp", PayloadSchemaType.DATETIME),
        ]
        
        # Chunk-level indexes
        chunk_indexes = [
            ("chunk_id", PayloadSchemaType.KEYWORD),
            ("chunk_type", PayloadSchemaType.KEYWORD),  # content, table, code, etc.
            ("hierarchy_level", PayloadSchemaType.KEYWORD),  # parent, child
            ("parent_chunk_id", PayloadSchemaType.KEYWORD),  # For hierarchical
            ("chunk_index", PayloadSchemaType.INTEGER),
            ("quality_score", PayloadSchemaType.FLOAT),
            ("chunk_size", PayloadSchemaType.INTEGER),
        ]
        
        # Railway-specific indexes (ÖBB)
        railway_indexes = [
            ("fleet_type", PayloadSchemaType.KEYWORD),
            ("train_id", PayloadSchemaType.KEYWORD),
            ("standard_compliance", PayloadSchemaType.KEYWORD),  # EN50155, etc.
            ("network_component", PayloadSchemaType.KEYWORD),  # CCU, VLAN, etc.
            ("configuration_type", PayloadSchemaType.KEYWORD),
        ]
        
        # Search optimization indexes
        search_indexes = [
            ("search_type", PayloadSchemaType.KEYWORD),  # vector, hybrid, keyword
            ("has_context", PayloadSchemaType.BOOL),  # Contextual retrieval
            ("is_parent", PayloadSchemaType.BOOL),
            ("is_child", PayloadSchemaType.BOOL),
        ]
        
        # Text search indexes for hybrid search
        text_indexes = [
            ("content", PayloadSchemaType.TEXT),  # Full-text search
            ("keywords", PayloadSchemaType.TEXT),  # Extracted keywords
            ("entities", PayloadSchemaType.TEXT),  # Named entities
        ]
        
        # Create all indexes
        all_indexes = (
            document_indexes + 
            chunk_indexes + 
            railway_indexes + 
            search_indexes
        )
        
        for field_name, field_type in all_indexes:
            self.client.create_field_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type
            )
            print(f"  ✓ Created index: {field_name} ({field_type})")
        
        # Create text indexes with special parameters
        for field_name, _ in text_indexes:
            self.client.create_field_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True
                )
            )
            print(f"  ✓ Created text index: {field_name}")
        
        print("✅ All payload indexes created")
    
    def prepare_point_from_v4_chunk(self, 
                                   chunk: Dict[str, Any], 
                                   document_metadata: Dict[str, Any]) -> PointStruct:
        """
        Convert v4.0 processor chunk to Qdrant point with all metadata
        """
        
        # Generate unique ID
        chunk_content = chunk.get('content', '')
        chunk_id = self._generate_chunk_id(
            document_metadata.get('document_id', 'unknown'),
            chunk.get('index', 0),
            chunk_content
        )
        
        # Prepare vectors dictionary
        vectors = {}
        
        # Main chunk embedding
        if 'embedding' in chunk:
            vectors['chunk_embedding'] = chunk['embedding']
        
        # Parent/child embeddings for hierarchical
        if chunk.get('hierarchy') == 'parent':
            vectors['parent_embedding'] = chunk.get('embedding')
        elif chunk.get('hierarchy') == 'child':
            vectors['child_embedding'] = chunk.get('embedding')
        
        # Full document embedding from late chunking
        if 'full_doc_embedding' in chunk:
            vectors['full_doc_embedding'] = chunk['full_doc_embedding']
        
        # Prepare sparse vector for hybrid search
        sparse_vectors = {}
        if 'term_frequencies' in chunk:
            sparse_vectors['keyword_sparse'] = self._create_sparse_vector(
                chunk['term_frequencies']
            )
        
        # Comprehensive payload
        payload = {
            # Document metadata
            "document_id": document_metadata.get('document_id'),
            "document_name": document_metadata.get('file_name'),
            "document_type": document_metadata.get('document_type'),
            "document_version": document_metadata.get('version', 1.0),
            "document_title": document_metadata.get('title', ''),
            "processing_profile": document_metadata.get('processing_profile'),
            "processing_timestamp": document_metadata.get('processing_timestamp'),
            
            # Chunk metadata
            "chunk_id": chunk_id,
            "chunk_index": chunk.get('index', 0),
            "chunk_type": chunk.get('metadata', {}).get('chunk_type', 'content'),
            "chunk_size": len(chunk_content),
            "chunking_method": chunk.get('metadata', {}).get('chunking_method', 'unknown'),
            
            # Content
            "content": chunk_content,
            "content_hash": hashlib.md5(chunk_content.encode()).hexdigest(),
            
            # Hierarchical chunking metadata
            "hierarchy_level": chunk.get('hierarchy', 'single'),
            "parent_chunk_id": chunk.get('parent_index'),
            "is_parent": chunk.get('hierarchy') == 'parent',
            "is_child": chunk.get('hierarchy') == 'child',
            "child_count": chunk.get('metadata', {}).get('child_count', 0),
            
            # Quality metrics
            "quality_score": chunk.get('quality', {}).get('overall_score', 0.0),
            "faithfulness": chunk.get('quality', {}).get('metrics', {}).get('faithfulness', 0.0),
            "relevancy": chunk.get('quality', {}).get('metrics', {}).get('answer_relevancy', 0.0),
            "precision": chunk.get('quality', {}).get('metrics', {}).get('context_precision', 0.0),
            "recall": chunk.get('quality', {}).get('metrics', {}).get('context_recall', 0.0),
            
            # Contextual retrieval
            "has_context": chunk.get('has_context', False),
            "context_similarity": chunk.get('context_similarity', 0.0),
            
            # Hybrid search metadata
            "search_type": "hybrid" if 'keyword_content' in chunk else "vector",
            "keywords": chunk.get('keyword_content', ''),
            "keyword_count": chunk.get('metadata', {}).get('keyword_count', 0),
            "vector_weight": chunk.get('metadata', {}).get('vector_weight', 0.5),
            "keyword_weight": chunk.get('metadata', {}).get('keyword_weight', 0.5),
            
            # Position and structure
            "section": chunk.get('metadata', {}).get('section', ''),
            "section_title": chunk.get('metadata', {}).get('section_title', ''),
            "section_level": chunk.get('metadata', {}).get('section_level', 0),
            "position": chunk.get('metadata', {}).get('position', 0),
            "position_in_parent": chunk.get('metadata', {}).get('position_in_parent', 0),
            
            # Entities and topics
            "entities": json.dumps(chunk.get('entities', [])),
            "topics": json.dumps(chunk.get('topics', [])),
            "relationships": json.dumps(chunk.get('relationships', [])),
        }
        
        # Add railway-specific metadata if present
        if 'railway_metadata' in document_metadata:
            railway = document_metadata['railway_metadata'].get('railway_specific', {})
            payload.update({
                "fleet_type": json.dumps(railway.get('fleet_type', [])),
                "network_component": json.dumps(railway.get('network_components', {})),
                "standard_compliance": json.dumps(railway.get('standards_compliance', [])),
                "connectivity_features": json.dumps(railway.get('connectivity_features', [])),
            })
        
        # Add configuration data if present
        if 'configurations' in document_metadata:
            payload['configurations'] = json.dumps(document_metadata['configurations'])
        
        # Add network topology if present
        if 'network_topology' in document_metadata.get('railway_metadata', {}):
            topology = document_metadata['railway_metadata']['network_topology']
            payload.update({
                "ip_addresses": json.dumps(topology.get('ip_addresses', [])),
                "vlans": json.dumps(topology.get('vlans', [])),
                "subnets": json.dumps(topology.get('subnets', [])),
            })
        
        # Create Qdrant point
        point = PointStruct(
            id=chunk_id,
            vector=vectors,
            payload=payload
        )
        
        # Add sparse vectors if available
        if sparse_vectors:
            point.sparse_vectors = sparse_vectors
        
        return point
    
    def _generate_chunk_id(self, 
                          document_id: str, 
                          chunk_index: int, 
                          content: str) -> str:
        """Generate unique chunk ID"""
        # Combine document ID, chunk index, and content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{document_id}_{chunk_index}_{content_hash}"
    
    def _create_sparse_vector(self, term_frequencies: Dict[str, float]) -> SparseVector:
        """Create sparse vector from term frequencies for BM25"""
        # Convert term frequencies to sparse vector format
        # In real implementation, you'd need a vocabulary mapping
        indices = []
        values = []
        
        # Simple hash-based index assignment (in production, use proper vocabulary)
        for term, freq in term_frequencies.items():
            # Hash term to get index (simplified - use proper vocabulary in production)
            index = abs(hash(term)) % self.config.sparse_vector_size
            indices.append(index)
            values.append(freq)
        
        return SparseVector(
            indices=indices,
            values=values
        )
    
    def batch_upsert_v4_results(self, 
                               processing_results: List[Dict[str, Any]], 
                               batch_size: int = 100) -> None:
        """
        Batch upsert all chunks from v4.0 processor results
        """
        
        all_points = []
        
        for result in processing_results:
            # Skip failed processing
            if not result.get('processing_success'):
                continue
            
            # Extract document metadata
            doc_metadata = {
                'document_id': result.get('document_id', result.get('file_name', 'unknown')),
                'file_name': result.get('file_name'),
                'document_type': result.get('document_type'),
                'processing_profile': result.get('config', {}).get('profile'),
                'processing_timestamp': result.get('processing_timestamp'),
                'title': result.get('front_matter', {}).get('title', ''),
                'version': result.get('version_info', {}).get('version', 1.0),
                'railway_metadata': result.get('railway_metadata', {}),
                'configurations': result.get('configurations', [])
            }
            
            # Convert each chunk to point
            for chunk in result.get('chunks', []):
                point = self.prepare_point_from_v4_chunk(chunk, doc_metadata)
                all_points.append(point)
        
        # Batch upsert
        total_points = len(all_points)
        print(f"Upserting {total_points} points in batches of {batch_size}...")
        
        for i in range(0, total_points, batch_size):
            batch = all_points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=batch
            )
            print(f"  ✓ Upserted batch {i//batch_size + 1}/{(total_points + batch_size - 1)//batch_size}")
        
        print(f"✅ Successfully upserted {total_points} points")
    
    def search_hierarchical(self, 
                           query_embedding: List[float],
                           limit: int = 10,
                           search_children_return_parents: bool = True) -> List[Dict]:
        """
        Hierarchical search: search children, return parents
        """
        
        if search_children_return_parents:
            # Step 1: Search child chunks
            child_results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=NamedVector(
                    name="child_embedding",
                    vector=query_embedding
                ),
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="is_child",
                            match=MatchValue(value=True)
                        )
                    ]
                ),
                limit=limit * 2,  # Get more children to find unique parents
                with_payload=True
            )
            
            # Step 2: Get unique parent IDs
            parent_ids = set()
            for result in child_results:
                parent_id = result.payload.get('parent_chunk_id')
                if parent_id:
                    parent_ids.add(parent_id)
            
            # Step 3: Retrieve parent chunks
            if parent_ids:
                parent_results = self.client.retrieve(
                    collection_name=self.config.collection_name,
                    ids=list(parent_ids)[:limit],
                    with_payload=True,
                    with_vectors=False
                )
                
                return [
                    {
                        'id': point.id,
                        'score': 0.0,  # Score from child search
                        'payload': point.payload
                    }
                    for point in parent_results
                ]
        else:
            # Direct parent search
            return self.client.search(
                collection_name=self.config.collection_name,
                query_vector=NamedVector(
                    name="parent_embedding",
                    vector=query_embedding
                ),
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="is_parent",
                            match=MatchValue(value=True)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True
            )
    
    def search_hybrid(self,
                     query_text: str,
                     query_embedding: List[float],
                     limit: int = 10,
                     alpha: float = 0.5) -> List[Dict]:
        """
        Hybrid search combining vector and keyword search
        
        Args:
            alpha: Weight for vector search (1-alpha for keyword search)
        """
        
        # Vector search
        vector_results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=NamedVector(
                name="chunk_embedding",
                vector=query_embedding
            ),
            limit=limit * 2,
            with_payload=True
        )
        
        # Keyword search using sparse vectors
        keyword_results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=NamedVector(
                name="keyword_sparse",
                vector=self._create_sparse_vector_from_query(query_text)
            ),
            limit=limit * 2,
            with_payload=True
        )
        
        # Combine results with Reciprocal Rank Fusion (RRF)
        combined_results = self._reciprocal_rank_fusion(
            vector_results, 
            keyword_results,
            alpha=alpha,
            k=60
        )
        
        return combined_results[:limit]
    
    def _create_sparse_vector_from_query(self, query: str) -> SparseVector:
        """Create sparse vector from query text"""
        # Tokenize query (simplified - use proper tokenization in production)
        tokens = query.lower().split()
        
        # Create term frequencies
        term_freq = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
        
        # Normalize
        total = sum(term_freq.values())
        term_freq = {k: v/total for k, v in term_freq.items()}
        
        return self._create_sparse_vector(term_freq)
    
    def _reciprocal_rank_fusion(self,
                               vector_results: List,
                               keyword_results: List,
                               alpha: float = 0.5,
                               k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion for combining vector and keyword results
        """
        
        # Calculate RRF scores
        scores = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_id = result.id
            scores[doc_id] = scores.get(doc_id, 0) + alpha / (k + rank + 1)
        
        # Process keyword results
        for rank, result in enumerate(keyword_results):
            doc_id = result.id
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) / (k + rank + 1)
        
        # Sort by combined score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Retrieve full documents for top results
        doc_ids = [doc_id for doc_id, _ in sorted_docs]
        
        results = self.client.retrieve(
            collection_name=self.config.collection_name,
            ids=doc_ids,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results with scores
        formatted_results = []
        for point in results:
            formatted_results.append({
                'id': point.id,
                'score': scores[point.id],
                'payload': point.payload
            })
        
        return formatted_results
    
    def search_railway_specific(self,
                               query_embedding: List[float],
                               fleet_type: Optional[str] = None,
                               standard: Optional[str] = None,
                               component: Optional[str] = None,
                               limit: int = 10) -> List[Dict]:
        """
        Railway-specific search with filtering
        """
        
        # Build filter conditions
        must_conditions = []
        
        if fleet_type:
            must_conditions.append(
                FieldCondition(
                    key="fleet_type",
                    match=MatchValue(value=fleet_type)
                )
            )
        
        if standard:
            must_conditions.append(
                FieldCondition(
                    key="standard_compliance",
                    match=MatchValue(value=standard)
                )
            )
        
        if component:
            must_conditions.append(
                FieldCondition(
                    key="network_component",
                    match=MatchValue(value=component)
                )
            )
        
        # Search with filters
        results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=NamedVector(
                name="chunk_embedding",
                vector=query_embedding
            ),
            query_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=limit,
            with_payload=True
        )
        
        return results
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get detailed collection statistics"""
        
        collection_info = self.client.get_collection(self.config.collection_name)
        
        # Count different chunk types
        chunk_type_counts = {}
        hierarchy_counts = {}
        quality_distribution = {}
        
        # Sample points for statistics
        sample = self.client.scroll(
            collection_name=self.config.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )[0]
        
        for point in sample:
            # Count chunk types
            chunk_type = point.payload.get('chunk_type', 'unknown')
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
            
            # Count hierarchy levels
            hierarchy = point.payload.get('hierarchy_level', 'single')
            hierarchy_counts[hierarchy] = hierarchy_counts.get(hierarchy, 0) + 1
            
            # Quality distribution
            quality = point.payload.get('quality_score', 0)
            quality_bucket = f"{int(quality // 10) * 10}-{int(quality // 10) * 10 + 10}"
            quality_distribution[quality_bucket] = quality_distribution.get(quality_bucket, 0) + 1
        
        return {
            'collection_name': self.config.collection_name,
            'total_points': collection_info.points_count,
            'vector_count': collection_info.vectors_count,
            'indexed_vectors': collection_info.indexed_vectors_count,
            'segments': collection_info.segments_count,
            'status': collection_info.status,
            'chunk_types': chunk_type_counts,
            'hierarchy_distribution': hierarchy_counts,
            'quality_distribution': quality_distribution,
            'config': {
                'shards': self.config.shard_count,
                'replication': self.config.replication_factor,
                'vector_size': self.config.dense_vector_size
            }
        }

# =============================
# Usage Examples
# =============================

def main():
    """Example usage of Qdrant schema with v4.0 processor"""
    
    print("="*60)
    print("  Qdrant Schema for Enhanced Document Processor v4.0")
    print("="*60)
    
    # Initialize schema
    config = QdrantConfig(
        collection_name="railway_documents_v4",
        enable_railway_optimization=True,
        enable_quantization=True
    )
    
    schema = QdrantSchemaV4(config)
    
    # Create collection
    print("\n1. Creating optimized collection...")
    schema.create_collection(force_recreate=True)
    
    # Example: Process and store documents
    print("\n2. Processing and storing documents...")
    
    # Simulate v4.0 processor results
    sample_results = [
        {
            'processing_success': True,
            'document_id': 'doc_001',
            'file_name': 'railway_spec.pdf',
            'document_type': 'pdf',
            'processing_timestamp': datetime.now().isoformat(),
            'config': {'profile': 'railway'},
            'chunks': [
                {
                    'index': 0,
                    'content': 'The CCU R4600-3Ax provides 5G connectivity...',
                    'embedding': [0.1] * 768,  # Placeholder embedding
                    'hierarchy': 'parent',
                    'has_context': True,
                    'quality': {
                        'overall_score': 92.5,
                        'metrics': {
                            'faithfulness': 0.95,
                            'answer_relevancy': 0.90
                        }
                    },
                    'keyword_content': 'ccu r4600 5g connectivity',
                    'term_frequencies': {'ccu': 0.2, 'r4600': 0.2, '5g': 0.3},
                    'metadata': {
                        'chunk_type': 'technical',
                        'section': 'Network Architecture',
                        'child_count': 3
                    }
                },
                {
                    'index': 1,
                    'content': 'VLAN 100 for management network...',
                    'embedding': [0.2] * 768,
                    'hierarchy': 'child',
                    'parent_index': 0,
                    'has_context': True,
                    'quality': {
                        'overall_score': 88.0,
                        'metrics': {
                            'faithfulness': 0.92,
                            'answer_relevancy': 0.88
                        }
                    }
                }
            ],
            'railway_metadata': {
                'railway_specific': {
                    'fleet_type': ['Railjet'],
                    'network_components': {'CCU': 2, 'VLAN': 4},
                    'standards_compliance': ['EN50155', 'EN45545']
                },
                'network_topology': {
                    'vlans': ['100', '200', '300'],
                    'ip_addresses': ['10.0.100.1', '10.0.100.2']
                }
            }
        }
    ]
    
    # Upsert to Qdrant
    schema.batch_upsert_v4_results(sample_results, batch_size=100)
    
    # Example searches
    print("\n3. Example searches...")
    
    # Hierarchical search
    print("\n  a) Hierarchical search (search children, return parents):")
    query_embedding = [0.15] * 768  # Placeholder
    hierarchical_results = schema.search_hierarchical(
        query_embedding=query_embedding,
        limit=5,
        search_children_return_parents=True
    )
    print(f"     Found {len(hierarchical_results)} parent chunks")
    
    # Hybrid search
    print("\n  b) Hybrid search (vector + keyword):")
    hybrid_results = schema.search_hybrid(
        query_text="CCU configuration VLAN",
        query_embedding=query_embedding,
        limit=5,
        alpha=0.6
    )
    print(f"     Found {len(hybrid_results)} results")
    
    # Railway-specific search
    print("\n  c) Railway-specific search:")
    railway_results = schema.search_railway_specific(
        query_embedding=query_embedding,
        standard="EN50155",
        component="CCU",
        limit=5
    )
    print(f"     Found {len(railway_results)} railway-specific results")
    
    # Get statistics
    print("\n4. Collection statistics:")
    stats = schema.get_collection_statistics()
    print(f"   Total points: {stats['total_points']}")
    print(f"   Chunk types: {stats['chunk_types']}")
    print(f"   Hierarchy distribution: {stats['hierarchy_distribution']}")
    print(f"   Quality distribution: {stats['quality_distribution']}")
    
    print("\n✅ Qdrant schema setup complete and tested!")

if __name__ == "__main__":
    main()
