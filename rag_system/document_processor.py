"""
Document Processor Module
Handles PDF extraction and text chunking for the RAG system
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import pdfplumber
from tqdm import tqdm

from config import (
    ARTICLES_DIR, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    MIN_CHUNK_SIZE,
    CHUNKS_METADATA_PATH
)


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    chunk_id: str
    document_name: str
    page_number: int
    content: str
    word_count: int
    section_title: str = ""
    

class PDFExtractor:
    """Extracts text content from PDF files"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: Path) -> List[Dict]:
        """
        Extract text from PDF file with page information
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dicts with page number and text content
        """
        pages_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Clean the extracted text
                        text = PDFExtractor._clean_text(text)
                        pages_content.append({
                            "page_number": page_num,
                            "content": text
                        })
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            
        return pages_content
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text by removing artifacts"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers patterns
        text = re.sub(r'\b\d+\s*$', '', text)
        # Remove common PDF artifacts
        text = re.sub(r'[•●■▪]', '-', text)
        # Normalize line breaks
        text = text.replace('\n', ' ').strip()
        return text
    
    @staticmethod
    def extract_tables_from_pdf(pdf_path: Path) -> List[Dict]:
        """Extract tables from PDF file"""
        tables_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            # Convert table to readable text
                            table_text = PDFExtractor._table_to_text(table)
                            tables_content.append({
                                "page_number": page_num,
                                "table_index": table_idx,
                                "content": table_text
                            })
        except Exception as e:
            print(f"Error extracting tables from {pdf_path}: {e}")
            
        return tables_content
    
    @staticmethod
    def _table_to_text(table: List[List]) -> str:
        """Convert table data to readable text format"""
        rows = []
        for row in table:
            if row:
                # Filter None values and convert to strings
                cleaned_row = [str(cell) if cell else "" for cell in row]
                rows.append(" | ".join(cleaned_row))
        return "\n".join(rows)


class TextChunker:
    """Splits text into overlapping chunks for processing"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            return [text] if len(words) >= MIN_CHUNK_SIZE else []
            
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            
            if len(chunk_words) >= MIN_CHUNK_SIZE:
                chunks.append(' '.join(chunk_words))
                
            start += self.chunk_size - self.overlap
            
        return chunks
    
    def detect_section_title(self, text: str) -> str:
        """Try to detect section title from beginning of text"""
        # Common section patterns in medical papers
        patterns = [
            r'^(Introduction|Background|Methods|Results|Discussion|Conclusion|Abstract|Summary)',
            r'^(\d+\.?\s*[A-Z][a-z]+)',
            r'^([A-Z][A-Z\s]+)(?=\s)',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text[:100], re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""


class DocumentProcessor:
    """Main class for processing documents into chunks"""
    
    def __init__(self):
        self.extractor = PDFExtractor()
        self.chunker = TextChunker()
        self.chunks: List[DocumentChunk] = []
        
    def process_all_documents(self) -> List[DocumentChunk]:
        """
        Process all PDF documents in the Articles directory
        
        Returns:
            List of DocumentChunk objects
        """
        pdf_files = list(ARTICLES_DIR.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to process")
        
        all_chunks = []
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            doc_chunks = self.process_document(pdf_path)
            all_chunks.extend(doc_chunks)
            print(f"  - {pdf_path.name}: {len(doc_chunks)} chunks")
            
        self.chunks = all_chunks
        print(f"\nTotal chunks created: {len(all_chunks)}")
        
        return all_chunks
    
    def process_document(self, pdf_path: Path) -> List[DocumentChunk]:
        """
        Process a single PDF document into chunks
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of DocumentChunk objects
        """
        doc_name = pdf_path.stem
        pages = self.extractor.extract_text_from_pdf(pdf_path)
        
        # Also extract tables
        tables = self.extractor.extract_tables_from_pdf(pdf_path)
        
        chunks = []
        chunk_counter = 0
        
        # Process regular text
        for page_data in pages:
            page_num = page_data["page_number"]
            content = page_data["content"]
            
            text_chunks = self.chunker.chunk_text(content)
            
            for chunk_text in text_chunks:
                section_title = self.chunker.detect_section_title(chunk_text)
                
                chunk = DocumentChunk(
                    chunk_id=f"{doc_name}_chunk_{chunk_counter}",
                    document_name=doc_name,
                    page_number=page_num,
                    content=chunk_text,
                    word_count=len(chunk_text.split()),
                    section_title=section_title
                )
                chunks.append(chunk)
                chunk_counter += 1
                
        # Process tables as separate chunks
        for table_data in tables:
            chunk = DocumentChunk(
                chunk_id=f"{doc_name}_table_{table_data['table_index']}_p{table_data['page_number']}",
                document_name=doc_name,
                page_number=table_data["page_number"],
                content=table_data["content"],
                word_count=len(table_data["content"].split()),
                section_title="Table"
            )
            chunks.append(chunk)
            
        return chunks
    
    def save_chunks_metadata(self):
        """Save chunks metadata to JSON file"""
        metadata = [asdict(chunk) for chunk in self.chunks]
        
        with open(CHUNKS_METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        print(f"Saved chunks metadata to {CHUNKS_METADATA_PATH}")
        
    def load_chunks_metadata(self) -> List[DocumentChunk]:
        """Load chunks metadata from JSON file"""
        if not CHUNKS_METADATA_PATH.exists():
            return []
            
        with open(CHUNKS_METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        self.chunks = [DocumentChunk(**data) for data in metadata]
        return self.chunks


if __name__ == "__main__":
    # Test the document processor
    processor = DocumentProcessor()
    chunks = processor.process_all_documents()
    processor.save_chunks_metadata()
    
    # Print sample chunks
    print("\n--- Sample Chunks ---")
    for chunk in chunks[:3]:
        print(f"\nChunk: {chunk.chunk_id}")
        print(f"Document: {chunk.document_name}")
        print(f"Page: {chunk.page_number}")
        print(f"Words: {chunk.word_count}")
        print(f"Content preview: {chunk.content[:200]}...")
