#!/usr/bin/env python3
"""
Document Ingestion Bridge for Qdrant
Integrates enhanced_document_processor.py with qdrant_schema_v4.py
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add bms-agent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'bms-agent' / 'scr'))

from sentence_transformers import SentenceTransformer
from enhanced_document_processor import (
    EnhancedDocumentProcessor,
    ProcessingConfig,
    ProcessingProfile,
    ChunkingStrategy
)
from qdrant_schema_v4 import QdrantSchemaV4, QdrantConfig
from chapter_extractor import ChapterExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentIngestionBridge:
    """Bridge between document processor and Qdrant database"""

    def __init__(self,
                 qdrant_config: QdrantConfig = None,
                 processor_config: ProcessingConfig = None):
        """Initialize ingestion bridge"""

        # Initialize Qdrant schema
        self.qdrant_config = qdrant_config or QdrantConfig(
            collection_name="railway_documents_v4",
            enable_railway_optimization=True
        )
        self.qdrant = QdrantSchemaV4(self.qdrant_config)

        # Initialize document processor
        self.processor_config = processor_config or ProcessingConfig(
            processing_profile=ProcessingProfile.RAILWAY,
            chunking_strategy=ChunkingStrategy.HIERARCHICAL,
            enable_contextual_retrieval=True,
            enable_late_chunking=True,
            enable_hybrid_search=True,
            enable_quality_validation=True
        )
        self.processor = EnhancedDocumentProcessor(self.processor_config)

        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        logger.info("Embedding model loaded")

        # Initialize chapter extractor
        self.chapter_extractor = ChapterExtractor()
        logger.info("Chapter extractor initialized")

    def get_document_type(self, file_path: str) -> str:
        """Detect document type from file extension"""
        ext = Path(file_path).suffix.lower()
        mapping = {
            '.pdf': 'pdf',
            '.docx': 'word',
            '.doc': 'word',
            '.txt': 'text',
            '.md': 'markdown',
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.html': 'html'
        }
        return mapping.get(ext, 'unknown')

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for all chunks"""

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")

        # Extract content for batch encoding
        contents = [chunk.get('content', '') for chunk in chunks]

        # Generate embeddings in batch for efficiency
        embeddings = self.embedder.encode(contents,
                                         convert_to_tensor=False,
                                         show_progress_bar=True)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()

            # If late chunking provided full_doc_embedding, keep it
            # Otherwise, copy the chunk embedding
            if 'full_doc_embedding' not in chunk:
                chunk['full_doc_embedding'] = embedding.tolist()

        return chunks

    def add_chapter_awareness(self,
                             chunks: List[Dict[str, Any]],
                             full_text: str,
                             file_path: str) -> List[Dict[str, Any]]:
        """Add chapter and sub-chapter awareness to chunks"""

        logger.info("Extracting chapter structure...")

        # Determine document type for pattern matching
        ext = Path(file_path).suffix.lower()
        doc_type = 'markdown' if ext in ['.md', '.markdown'] else 'numbered'

        # Extract chapter structure
        chapters = self.chapter_extractor.extract_chapter_structure(
            full_text,
            document_type=doc_type
        )

        if chapters:
            logger.info(f"Found {len(chapters)} chapter headings")
        else:
            logger.info("No chapter structure detected")

        # Associate chunks with chapters
        chunks = self.chapter_extractor.associate_chunks_with_chapters(
            chunks,
            chapters,
            full_text
        )

        # Optionally add chapter context to content
        # (disabled by default to preserve original content)
        # for chunk in chunks:
        #     chunk['content'] = self.chapter_extractor.add_chapter_context_to_content(chunk)

        return chunks

    def transform_for_qdrant(self,
                            processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Transform processor output to match Qdrant schema expectations"""

        # Add missing document-level fields
        transformed = processing_result.copy()

        # Add document_type if missing
        if 'document_type' not in transformed:
            transformed['document_type'] = self.get_document_type(
                processing_result.get('file_path', '')
            )

        # Ensure version info exists
        if 'version_info' not in transformed:
            transformed['version_info'] = {'version': 1.0}

        # Add title if not present
        if 'title' not in transformed:
            transformed['title'] = processing_result.get('file_name', 'Untitled')

        return transformed

    def ingest_document(self,
                       file_path: str,
                       document_id: str = None) -> Dict[str, Any]:
        """
        Process and ingest a single document into Qdrant

        Args:
            file_path: Path to document file
            document_id: Optional document ID (defaults to filename)

        Returns:
            Ingestion result with statistics
        """

        logger.info(f"Starting ingestion for: {file_path}")

        # Step 1: Read document text (needed for chapter extraction)
        logger.info("Step 1/5: Reading document...")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                full_text = f.read()
        except Exception as e:
            logger.warning(f"Could not read full text: {e}")
            full_text = ""

        # Step 2: Process document
        logger.info("Step 2/5: Processing document...")
        result = self.processor.process_document(file_path, document_id)

        if not result.get('processing_success'):
            logger.error(f"Document processing failed: {result.get('errors')}")
            return {
                'success': False,
                'error': 'Processing failed',
                'details': result.get('errors')
            }

        # Step 3: Generate embeddings
        logger.info("Step 3/5: Generating embeddings...")
        chunks = result.get('chunks', [])
        if not chunks:
            logger.warning("No chunks generated from document")
            return {
                'success': False,
                'error': 'No chunks generated'
            }

        chunks_with_embeddings = self.generate_embeddings(chunks)
        result['chunks'] = chunks_with_embeddings

        # Step 4: Add chapter awareness
        logger.info("Step 4/5: Adding chapter awareness...")
        if full_text:
            chunks_with_chapters = self.add_chapter_awareness(
                chunks_with_embeddings,
                full_text,
                file_path
            )
            result['chunks'] = chunks_with_chapters
        else:
            logger.warning("Skipping chapter extraction (no full text)")

        # Step 5: Transform for Qdrant
        logger.info("Step 5/5: Transforming for Qdrant schema...")
        transformed_result = self.transform_for_qdrant(result)

        # Ingest to Qdrant
        logger.info("Ingesting to Qdrant...")
        self.qdrant.batch_upsert_v4_results([transformed_result], batch_size=100)

        # Return statistics
        ingestion_result = {
            'success': True,
            'document_id': transformed_result['document_id'],
            'file_name': transformed_result['file_name'],
            'chunks_ingested': len(chunks_with_embeddings),
            'document_type': transformed_result['document_type'],
            'processing_profile': transformed_result['config']['profile'],
            'quality_report': transformed_result.get('quality_report', {}),
            'has_railway_metadata': 'railway_metadata' in transformed_result
        }

        logger.info(f"✅ Successfully ingested {ingestion_result['chunks_ingested']} chunks")

        return ingestion_result

    def ingest_directory(self,
                        directory_path: str,
                        file_pattern: str = "*.pdf") -> List[Dict[str, Any]]:
        """
        Ingest all documents from a directory

        Args:
            directory_path: Path to directory containing documents
            file_pattern: Glob pattern for files (e.g., "*.pdf", "*.docx")

        Returns:
            List of ingestion results for each document
        """

        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")

        # Find all matching files
        files = list(directory.glob(file_pattern))
        logger.info(f"Found {len(files)} files matching '{file_pattern}'")

        results = []
        for file_path in files:
            try:
                result = self.ingest_document(str(file_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                results.append({
                    'success': False,
                    'file_name': file_path.name,
                    'error': str(e)
                })

        # Summary
        successful = sum(1 for r in results if r.get('success'))
        logger.info(f"\n{'='*60}")
        logger.info(f"Ingestion Complete: {successful}/{len(files)} successful")
        logger.info(f"{'='*60}")

        return results


def main():
    """Example usage"""

    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument('path', help="Path to document file or directory")
    parser.add_argument('--pattern', default='*.pdf',
                       help="File pattern for directory mode (default: *.pdf)")
    parser.add_argument('--document-id', help="Document ID (for single file)")

    args = parser.parse_args()

    # Initialize bridge
    bridge = DocumentIngestionBridge()

    path = Path(args.path)

    if path.is_file():
        # Ingest single file
        result = bridge.ingest_document(str(path), args.document_id)
        print(f"\nResult: {result}")
    elif path.is_dir():
        # Ingest directory
        results = bridge.ingest_directory(str(path), args.pattern)

        # Print summary
        print(f"\n{'='*60}")
        print("Ingestion Summary:")
        print(f"{'='*60}")
        for result in results:
            status = "✅" if result.get('success') else "❌"
            print(f"{status} {result.get('file_name', 'unknown')}")
            if result.get('success'):
                print(f"   Chunks: {result.get('chunks_ingested', 0)}")
                print(f"   Type: {result.get('document_type', 'unknown')}")
    else:
        print(f"Error: Path not found - {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
