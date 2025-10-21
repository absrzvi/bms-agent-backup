#!/usr/bin/env python3
"""
Chapter and Sub-Chapter Extractor for Document Chunks
Adds hierarchical chapter awareness to chunks for better contextual retrieval
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ChapterExtractor:
    """
    Extracts chapter and sub-chapter structure from documents
    and associates chunks with their chapter hierarchy
    """

    def __init__(self):
        self.chapter_patterns = {
            # Markdown headings
            'markdown': [
                (r'^# (.+)$', 1),           # # Chapter
                (r'^## (.+)$', 2),          # ## Sub-chapter
                (r'^### (.+)$', 3),         # ### Sub-sub-chapter
                (r'^#### (.+)$', 4),        # #### Level 4
            ],
            # Numbered chapters
            'numbered': [
                (r'^(\d+)\.\s+(.+)$', 1),         # 1. Chapter
                (r'^(\d+\.\d+)\s+(.+)$', 2),      # 1.1 Sub-chapter
                (r'^(\d+\.\d+\.\d+)\s+(.+)$', 3), # 1.1.1 Sub-sub
            ],
            # Word-style headings
            'word': [
                (r'^Chapter\s+(\d+):?\s*(.+)$', 1, 'i'),  # Chapter 1: Title
                (r'^Section\s+(\d+\.\d+):?\s*(.+)$', 2, 'i'),  # Section 1.1: Title
            ],
            # PDF bookmarks/outline style
            'pdf_outline': [
                (r'^([A-Z\s]+)$', 1),  # ALL CAPS = Chapter
            ]
        }

    def extract_chapter_structure(self,
                                  text: str,
                                  document_type: str = 'markdown') -> List[Dict[str, Any]]:
        """
        Extract chapter structure from document text

        Returns:
            List of chapter dictionaries with:
            - chapter_number: str (e.g., "1", "1.1", "1.1.1")
            - chapter_title: str
            - chapter_level: int (1, 2, 3, etc.)
            - start_position: int (char position in text)
            - end_position: int (estimated)
        """

        chapters = []
        lines = text.split('\n')

        current_position = 0

        for line_num, line in enumerate(lines):
            line = line.strip()

            # Try to match chapter patterns
            chapter_info = self._match_chapter_pattern(line, document_type)

            if chapter_info:
                chapter_info['start_position'] = current_position
                chapter_info['line_number'] = line_num

                # Estimate end position (will be updated)
                chapter_info['end_position'] = len(text)

                chapters.append(chapter_info)

            current_position += len(line) + 1  # +1 for newline

        # Update end positions
        for i in range(len(chapters) - 1):
            chapters[i]['end_position'] = chapters[i + 1]['start_position']

        # Add full path for each chapter
        chapters = self._build_chapter_paths(chapters)

        logger.info(f"Extracted {len(chapters)} chapter headings")

        return chapters

    def _match_chapter_pattern(self,
                               line: str,
                               document_type: str) -> Optional[Dict[str, Any]]:
        """Match line against chapter patterns"""

        patterns = self.chapter_patterns.get(document_type,
                                            self.chapter_patterns['markdown'])

        for pattern_info in patterns:
            if len(pattern_info) == 2:
                pattern, level = pattern_info
                flags = 0
            else:
                pattern, level, flag_str = pattern_info
                flags = re.IGNORECASE if 'i' in flag_str else 0

            match = re.match(pattern, line, flags)

            if match:
                if level == 1 and len(match.groups()) >= 2:
                    # Numbered chapter
                    return {
                        'chapter_number': match.group(1),
                        'chapter_title': match.group(2),
                        'chapter_level': level,
                        'chapter_full_title': line
                    }
                elif len(match.groups()) >= 1:
                    # Simple heading
                    return {
                        'chapter_number': '',
                        'chapter_title': match.group(1),
                        'chapter_level': level,
                        'chapter_full_title': line
                    }

        return None

    def _build_chapter_paths(self,
                            chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build full chapter paths (e.g., 'Chapter 1 > Section 1.1 > Sub 1.1.1')"""

        chapter_stack = []  # Stack to track hierarchy

        for chapter in chapters:
            level = chapter['chapter_level']

            # Pop chapters at same or deeper level
            while chapter_stack and chapter_stack[-1]['chapter_level'] >= level:
                chapter_stack.pop()

            # Build path
            path_parts = [c['chapter_title'] for c in chapter_stack]
            path_parts.append(chapter['chapter_title'])

            chapter['chapter_path'] = ' > '.join(path_parts)
            chapter['parent_chapter'] = chapter_stack[-1]['chapter_title'] if chapter_stack else None

            # Add to stack
            chapter_stack.append(chapter)

        return chapters

    def associate_chunks_with_chapters(self,
                                      chunks: List[Dict[str, Any]],
                                      chapters: List[Dict[str, Any]],
                                      text: str) -> List[Dict[str, Any]]:
        """
        Associate each chunk with its chapter/sub-chapter

        Args:
            chunks: List of chunks with 'index' and 'content'
            chapters: List of chapter structures
            text: Original document text

        Returns:
            Chunks with added chapter metadata
        """

        if not chapters:
            logger.warning("No chapters found - adding default chapter")
            # Add default chapter
            for chunk in chunks:
                chunk['chapter_info'] = {
                    'chapter_number': '',
                    'chapter_title': 'Document',
                    'chapter_level': 0,
                    'chapter_path': 'Document',
                    'sub_chapter': None,
                    'parent_chapter': None
                }
            return chunks

        # For each chunk, find which chapter it belongs to
        for chunk in chunks:
            # Estimate chunk position in document
            chunk_content = chunk.get('content', '')
            chunk_position = self._estimate_chunk_position(chunk_content, text)

            # Find the chapter that contains this position
            chapter = self._find_chapter_for_position(chunk_position, chapters)

            if chapter:
                # Find sub-chapter if exists
                sub_chapter = self._find_sub_chapter(chunk_position, chapters, chapter)

                chunk['chapter_info'] = {
                    'chapter_number': chapter.get('chapter_number', ''),
                    'chapter_title': chapter['chapter_title'],
                    'chapter_level': chapter['chapter_level'],
                    'chapter_path': chapter.get('chapter_path', chapter['chapter_title']),
                    'chapter_full_title': chapter.get('chapter_full_title', ''),
                    'sub_chapter': sub_chapter['chapter_title'] if sub_chapter else None,
                    'sub_chapter_number': sub_chapter.get('chapter_number', '') if sub_chapter else None,
                    'parent_chapter': chapter.get('parent_chapter'),
                    'position_in_chapter': chunk_position - chapter.get('start_position', 0)
                }
            else:
                # No chapter found - use first chapter or default
                chunk['chapter_info'] = {
                    'chapter_number': '',
                    'chapter_title': 'Unknown',
                    'chapter_level': 0,
                    'chapter_path': 'Unknown',
                    'sub_chapter': None,
                    'parent_chapter': None
                }

        logger.info(f"Associated {len(chunks)} chunks with chapters")

        return chunks

    def _estimate_chunk_position(self, chunk_content: str, full_text: str) -> int:
        """Estimate the position of chunk in full document"""
        # Find first occurrence of chunk start in document
        chunk_start = chunk_content[:100]  # Use first 100 chars

        try:
            position = full_text.index(chunk_start)
            return position
        except ValueError:
            # Chunk not found exactly - try fuzzy match
            return 0

    def _find_chapter_for_position(self,
                                   position: int,
                                   chapters: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the chapter that contains the given position"""

        # Find level 1 chapters first
        level_1_chapters = [c for c in chapters if c['chapter_level'] == 1]

        for chapter in level_1_chapters:
            if chapter['start_position'] <= position < chapter['end_position']:
                return chapter

        # If no level 1 found, try any chapter
        for chapter in chapters:
            if chapter['start_position'] <= position < chapter['end_position']:
                return chapter

        return None

    def _find_sub_chapter(self,
                         position: int,
                         chapters: List[Dict[str, Any]],
                         parent_chapter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find sub-chapter within a parent chapter"""

        parent_level = parent_chapter['chapter_level']

        # Look for chapters at next level within parent's range
        for chapter in chapters:
            if (chapter['chapter_level'] == parent_level + 1 and
                parent_chapter['start_position'] <= chapter['start_position'] and
                chapter['end_position'] <= parent_chapter['end_position'] and
                chapter['start_position'] <= position < chapter['end_position']):
                return chapter

        return None

    def add_chapter_context_to_content(self, chunk: Dict[str, Any]) -> str:
        """
        Add chapter context to chunk content for better retrieval

        Example output:
        <chapter_context>
        Chapter: 3. Network Configuration
        Sub-chapter: 3.2 VLAN Setup
        Path: Network Configuration > VLAN Setup
        </chapter_context>

        [original chunk content]
        """

        chapter_info = chunk.get('chapter_info', {})

        if not chapter_info or chapter_info.get('chapter_title') == 'Unknown':
            return chunk.get('content', '')

        context_parts = []

        # Main chapter
        chapter_num = chapter_info.get('chapter_number', '')
        chapter_title = chapter_info.get('chapter_title', '')
        if chapter_num:
            context_parts.append(f"Chapter: {chapter_num} {chapter_title}")
        else:
            context_parts.append(f"Chapter: {chapter_title}")

        # Sub-chapter
        if chapter_info.get('sub_chapter'):
            sub_num = chapter_info.get('sub_chapter_number', '')
            sub_title = chapter_info.get('sub_chapter')
            if sub_num:
                context_parts.append(f"Sub-chapter: {sub_num} {sub_title}")
            else:
                context_parts.append(f"Sub-chapter: {sub_title}")

        # Path
        if chapter_info.get('chapter_path'):
            context_parts.append(f"Path: {chapter_info['chapter_path']}")

        context = '\n'.join(context_parts)

        enhanced_content = f"""<chapter_context>
{context}
</chapter_context>

{chunk.get('content', '')}"""

        return enhanced_content


# Example usage
if __name__ == "__main__":
    # Test with sample markdown
    sample_doc = """
# Chapter 1: Introduction

This is the introduction text.

## 1.1 Background

Some background information.

### 1.1.1 Historical Context

Historical details here.

## 1.2 Objectives

The objectives are...

# Chapter 2: Methodology

Methodology content here.

## 2.1 Data Collection

Data collection details.
"""

    extractor = ChapterExtractor()

    # Extract chapters
    chapters = extractor.extract_chapter_structure(sample_doc)

    print("Extracted Chapters:")
    for ch in chapters:
        print(f"  Level {ch['chapter_level']}: {ch['chapter_title']}")
        print(f"    Path: {ch.get('chapter_path', 'N/A')}")
        print(f"    Parent: {ch.get('parent_chapter', 'None')}")

    # Sample chunks
    chunks = [
        {'index': 0, 'content': 'This is the introduction text.'},
        {'index': 1, 'content': 'Some background information.'},
        {'index': 2, 'content': 'Historical details here.'},
    ]

    # Associate chunks with chapters
    chunks = extractor.associate_chunks_with_chapters(chunks, chapters, sample_doc)

    print("\nChunks with Chapter Info:")
    for chunk in chunks:
        print(f"  Chunk {chunk['index']}: {chunk['chapter_info']['chapter_path']}")
