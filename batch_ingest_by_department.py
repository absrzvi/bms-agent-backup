#!/usr/bin/env python3
"""
Batch Document Ingestion with Department Metadata
Uses enhanced_document_processor.py with chapter-based hierarchy
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent / 'bms-agent' / 'scr'))

from sentence_transformers import SentenceTransformer
from enhanced_document_processor import (
    EnhancedDocumentProcessor,
    ProcessingConfig,
    ProcessingProfile,
    ChunkingStrategy
)
from qdrant_schema_v4 import QdrantSchemaV4, QdrantConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DepartmentAwareBatchIngestion:
    """Batch ingestion with department/category awareness"""

    def __init__(self):
        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        logger.info("✅ Embedding model loaded")

        # Initialize Qdrant
        self.qdrant_config = QdrantConfig(
            collection_name="railway_documents_v4",
            enable_railway_optimization=True
        )
        self.qdrant = QdrantSchemaV4(self.qdrant_config)

        # Initialize enhanced document processor with chapter awareness
        self.processor_config = ProcessingConfig(
            processing_profile=ProcessingProfile.GENERAL,
            chunking_strategy=ChunkingStrategy.HIERARCHICAL,

            # Chapter-based hierarchy (DEFAULT)
            enable_chapter_awareness=True,
            chapter_based_hierarchy=True,
            max_chapter_size=20000,
            grandchild_chunk_size=800,
            grandchild_overlap=100,
            add_chapter_context_to_content=False,

            # Text cleaning
            enable_text_cleaning=True,
            remove_extra_whitespace=True,
            normalize_unicode=True,

            # Enable all features for production
            enable_contextual_retrieval=False,  # Disabled for speed
            enable_late_chunking=False,
            enable_hybrid_search=True,
            enable_quality_validation=True,
            min_quality_score=60.0,  # Lower threshold for diverse docs
            enable_versioning=True,

            # Chunk settings
            chunk_size=1500,
            chunk_overlap=200
        )
        self.processor = EnhancedDocumentProcessor(self.processor_config)

        logger.info("✅ Enhanced Document Processor initialized")
        logger.info(f"   Chapter-Based Hierarchy: {self.processor_config.chapter_based_hierarchy}")
        logger.info(f"   Text Cleaning: {self.processor_config.enable_text_cleaning}")

    def scan_upload_directory(self, base_path: str) -> Dict[str, List[Path]]:
        """
        Scan upload directory and organize files by department

        Returns:
            Dict mapping department name to list of file paths
        """
        base_path = Path(base_path)
        departments = {}

        for dept_dir in sorted(base_path.iterdir()):
            if dept_dir.is_dir():
                department = dept_dir.name
                files = []

                # Find all supported files
                for file_path in dept_dir.rglob('*'):
                    if file_path.is_file():
                        ext = file_path.suffix.lower()
                        # Supported file types
                        if ext in ['.pdf', '.docx', '.doc', '.txt', '.md',
                                   '.xlsx', '.xls', '.xlsm', '.pptx', '.ppt']:
                            files.append(file_path)

                if files:
                    departments[department] = files
                    logger.info(f"📁 {department}: {len(files)} files")

        return departments

    def process_and_ingest_document(self,
                                    file_path: Path,
                                    department: str,
                                    doc_index: int) -> Dict[str, Any]:
        """
        Process a single document with enhanced processor and ingest to Qdrant

        Args:
            file_path: Path to document
            department: Department/category name
            doc_index: Document index for unique ID

        Returns:
            Ingestion result dictionary
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {file_path.name}")
        logger.info(f"Department: {department}")
        logger.info(f"{'='*70}")

        # Generate unique document ID
        document_id = f"{department.lower().replace(' ', '_')}_{doc_index}_{file_path.stem}"

        try:
            # Step 1: Process document with enhanced processor
            logger.info("Step 1/3: Processing document with chapter awareness...")
            result = self.processor.process_document(
                str(file_path),
                document_id=document_id
            )

            if not result.get('processing_success'):
                logger.error(f"❌ Processing failed: {result.get('errors')}")
                return {
                    'success': False,
                    'file': file_path.name,
                    'department': department,
                    'error': 'Processing failed',
                    'details': result.get('errors')
                }

            chunks = result.get('chunks', [])
            if not chunks:
                logger.warning("⚠️  No chunks generated")
                return {
                    'success': False,
                    'file': file_path.name,
                    'department': department,
                    'error': 'No chunks generated'
                }

            logger.info(f"✅ Generated {len(chunks)} chunks")
            logger.info(f"   Chunking method: {result['metadata'].get('chunking_method')}")
            if 'chapter_count' in result['metadata']:
                logger.info(f"   Chapters found: {result['metadata']['chapter_count']}")

            # Step 2: Generate embeddings
            logger.info("Step 2/3: Generating embeddings...")
            chunks_with_embeddings = self.generate_embeddings(chunks)

            # Step 3: Add department metadata to all chunks
            logger.info("Step 3/3: Adding department metadata...")
            for chunk in chunks_with_embeddings:
                # Add department/category to chunk metadata
                if 'metadata' not in chunk:
                    chunk['metadata'] = {}

                chunk['metadata']['department'] = department
                chunk['metadata']['category'] = department  # Alias
                chunk['metadata']['source_file'] = file_path.name
                chunk['metadata']['source_path'] = str(file_path)

            # Update result
            result['chunks'] = chunks_with_embeddings
            result['metadata']['department'] = department
            result['metadata']['category'] = department

            # Step 4: Ingest to Qdrant
            logger.info("Step 4/4: Ingesting to Qdrant...")
            self.qdrant.batch_upsert_v4_results([result], batch_size=50)

            logger.info(f"✅ Successfully ingested {len(chunks_with_embeddings)} chunks")

            return {
                'success': True,
                'file': file_path.name,
                'department': department,
                'document_id': document_id,
                'chunks_ingested': len(chunks_with_embeddings),
                'chunking_method': result['metadata'].get('chunking_method'),
                'has_chapters': result['metadata'].get('has_chapter_structure', False),
                'chapter_count': result['metadata'].get('chapter_count', 0)
            }

        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}", exc_info=True)
            return {
                'success': False,
                'file': file_path.name,
                'department': department,
                'error': str(e)
            }

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for all chunks"""

        logger.info(f"   Generating embeddings for {len(chunks)} chunks...")

        # Extract content for batch encoding
        contents = [chunk.get('content', '') for chunk in chunks]

        # Generate embeddings in batch
        embeddings = self.embedder.encode(
            contents,
            convert_to_tensor=False,
            show_progress_bar=False,
            batch_size=32
        )

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()

            # Add to other vector fields if not present
            if 'full_doc_embedding' not in chunk:
                chunk['full_doc_embedding'] = embedding.tolist()

        return chunks

    def run_batch_ingestion(self, base_path: str, test_mode: bool = False):
        """
        Run batch ingestion for all documents

        Args:
            base_path: Path to upload-files directory
            test_mode: If True, only process first file per department
        """
        logger.info("="*70)
        logger.info("  Batch Document Ingestion with Department Metadata")
        logger.info("="*70)

        # Scan directory
        logger.info(f"\n📂 Scanning directory: {base_path}")
        departments = self.scan_upload_directory(base_path)

        total_files = sum(len(files) for files in departments.values())
        logger.info(f"\n📊 Found {total_files} documents across {len(departments)} departments")

        if test_mode:
            logger.info("⚠️  TEST MODE: Processing only first file per department")

        # Process documents by department
        results = []
        total_processed = 0
        total_chunks = 0

        for dept_idx, (department, files) in enumerate(departments.items(), 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Department {dept_idx}/{len(departments)}: {department}")
            logger.info(f"Files to process: {len(files)}")
            logger.info(f"{'='*70}")

            # In test mode, only process first file
            files_to_process = files[:1] if test_mode else files

            for file_idx, file_path in enumerate(files_to_process, 1):
                result = self.process_and_ingest_document(
                    file_path,
                    department,
                    file_idx
                )

                results.append(result)

                if result['success']:
                    total_processed += 1
                    total_chunks += result['chunks_ingested']

                # Progress update
                logger.info(f"\n📊 Progress: {total_processed}/{total_files if not test_mode else len(departments)} documents processed, {total_chunks} chunks ingested")

        # Final summary
        self.print_summary(results, total_files if not test_mode else len(departments))

        # Save detailed results
        self.save_results(results)

    def print_summary(self, results: List[Dict], total_files: int):
        """Print ingestion summary"""

        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        logger.info("\n" + "="*70)
        logger.info("  📊 INGESTION SUMMARY")
        logger.info("="*70)

        logger.info(f"\n✅ Successfully processed: {len(successful)}/{total_files}")
        logger.info(f"❌ Failed: {len(failed)}/{total_files}")

        total_chunks = sum(r.get('chunks_ingested', 0) for r in successful)
        logger.info(f"📦 Total chunks ingested: {total_chunks}")

        # Department breakdown
        logger.info(f"\n📁 By Department:")
        dept_stats = {}
        for r in successful:
            dept = r['department']
            if dept not in dept_stats:
                dept_stats[dept] = {'docs': 0, 'chunks': 0}
            dept_stats[dept]['docs'] += 1
            dept_stats[dept]['chunks'] += r.get('chunks_ingested', 0)

        for dept, stats in sorted(dept_stats.items()):
            logger.info(f"   {dept}: {stats['docs']} docs, {stats['chunks']} chunks")

        # Chapter awareness stats
        docs_with_chapters = sum(1 for r in successful if r.get('has_chapters'))
        logger.info(f"\n📖 Documents with chapter structure: {docs_with_chapters}/{len(successful)}")

        if failed:
            logger.info(f"\n❌ Failed Documents:")
            for r in failed:
                logger.info(f"   {r['department']}/{r['file']}: {r.get('error', 'Unknown error')}")

    def save_results(self, results: List[Dict]):
        """Save detailed results to JSON"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"ingestion_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_documents': len(results),
                'successful': sum(1 for r in results if r['success']),
                'failed': sum(1 for r in results if not r['success']),
                'results': results
            }, f, indent=2)

        logger.info(f"\n💾 Detailed results saved to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch ingest documents with department metadata"
    )
    parser.add_argument(
        '--path',
        default='/workspace/bms-agent/upload-files',
        help='Path to upload-files directory'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only first file per department'
    )

    args = parser.parse_args()

    # Initialize and run
    ingestion = DepartmentAwareBatchIngestion()
    ingestion.run_batch_ingestion(args.path, test_mode=args.test)


if __name__ == "__main__":
    main()
