"""
Document Processor Module - Medical Optimized
Handles PDF extraction and text chunking for the RAG system
Preserves medical metadata: evidence levels, recommendations, risk groups
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field
import pdfplumber
from tqdm import tqdm

from config import (
    ARTICLES_DIR, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    MIN_CHUNK_SIZE,
    CHUNKS_METADATA_PATH,
    MEDICAL_PATTERNS,
)


@dataclass
class MedicalMetadata:
    """Medical-specific metadata extracted from text"""
    evidence_levels: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_groups: List[str] = field(default_factory=list)
    molecular_markers: List[str] = field(default_factory=list)
    has_table: bool = False
    has_statistics: bool = False


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with rich metadata"""
    chunk_id: str
    document_name: str
    document_title: str
    page_start: int
    page_end: int
    content: str
    word_count: int
    section_title: str = ""
    medical_metadata: Dict = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class PDFExtractor:
    """Extracts text content from PDF files with medical-aware processing"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: Path) -> List[Dict]:
        """Extract text from PDF file with page information"""
        pages_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text = PDFExtractor._clean_text(text)
                        tables = page.extract_tables()
                        has_tables = len(tables) > 0
                        
                        pages_content.append({
                            "page_number": page_num,
                            "content": text,
                            "has_tables": has_tables,
                        })
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            
        return pages_content
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text while preserving medical information"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'\s+\d+\s*$', '', line)
            line = re.sub(r'^[•●■▪◦‣⁃]\s*', '- ', line)
            cleaned_lines.append(line)
        
        text = ' '.join(cleaned_lines)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def extract_tables_from_pdf(pdf_path: Path) -> List[Dict]:
        """Extract tables from PDF file and convert to structured text"""
        tables_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            table_text = PDFExtractor._table_to_markdown(table)
                            if table_text:
                                tables_content.append({
                                    "page_number": page_num,
                                    "table_index": table_idx,
                                    "content": table_text,
                                    "is_table": True,
                                })
        except Exception as e:
            print(f"Error extracting tables from {pdf_path}: {e}")
            
        return tables_content
    
    @staticmethod
    def _table_to_markdown(table: List[List]) -> str:
        """Convert table data to markdown format"""
        if not table or len(table) < 2:
            return ""
            
        rows = []
        header = table[0]
        
        header_clean = [str(cell).strip() if cell else "" for cell in header]
        rows.append("| " + " | ".join(header_clean) + " |")
        rows.append("|" + "|".join(["---"] * len(header_clean)) + "|")
        
        for row in table[1:]:
            if row:
                row_clean = [str(cell).strip() if cell else "" for cell in row]
                while len(row_clean) < len(header_clean):
                    row_clean.append("")
                row_clean = row_clean[:len(header_clean)]
                rows.append("| " + " | ".join(row_clean) + " |")
                
        return "\n".join(rows)
    
    @staticmethod
    def extract_document_title(pdf_path: Path) -> str:
        """Try to extract document title from first page"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if pdf.pages:
                    first_page = pdf.pages[0].extract_text()
                    if first_page:
                        lines = first_page.split('\n')
                        for line in lines[:5]:
                            line = line.strip()
                            if len(line) > 10 and len(line) < 200:
                                return line
        except Exception:
            pass
        return pdf_path.stem


class MedicalMetadataExtractor:
    """Extracts medical-specific metadata from text"""
    
    @staticmethod
    def extract_metadata(text: str) -> MedicalMetadata:
        """Extract medical metadata from text using patterns"""
        metadata = MedicalMetadata()
        
        evidence_pattern = MEDICAL_PATTERNS.get("evidence_levels", "")
        if evidence_pattern:
            matches = re.findall(evidence_pattern, text, re.IGNORECASE)
            metadata.evidence_levels = list(set(matches))
        
        rec_pattern = MEDICAL_PATTERNS.get("recommendations", "")
        if rec_pattern:
            matches = re.findall(rec_pattern, text, re.IGNORECASE)
            metadata.recommendations = list(set(matches))
        
        risk_pattern = MEDICAL_PATTERNS.get("risk_groups", "")
        if risk_pattern:
            matches = re.findall(risk_pattern, text, re.IGNORECASE)
            metadata.risk_groups = list(set(m.lower() for m in matches))
        
        marker_pattern = MEDICAL_PATTERNS.get("molecular_markers", "")
        if marker_pattern:
            matches = re.findall(marker_pattern, text, re.IGNORECASE)
            metadata.molecular_markers = list(set(m.upper() for m in matches))
        
        if re.search(r'\d+(?:\.\d+)?%|\d+\s*-?\s*year\s+survival|hazard\s+ratio', text, re.IGNORECASE):
            metadata.has_statistics = True
            
        return metadata


class TextChunker:
    """Splits text into overlapping chunks with medical-aware boundaries"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.metadata_extractor = MedicalMetadataExtractor()
        
    def chunk_text(self, text: str, preserve_structure: bool = True) -> List[Tuple[str, MedicalMetadata]]:
        """Split text into overlapping chunks, trying to preserve structure"""
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            if len(words) >= MIN_CHUNK_SIZE:
                metadata = self.metadata_extractor.extract_metadata(text)
                return [(text, metadata)]
            return []
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                metadata = self.metadata_extractor.extract_metadata(chunk_text)
                chunks.append((chunk_text, metadata))
                
                overlap_start = max(0, len(current_chunk) - self.overlap // 10)
                current_chunk = current_chunk[overlap_start:]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        if current_chunk and current_word_count >= MIN_CHUNK_SIZE:
            chunk_text = ' '.join(current_chunk)
            metadata = self.metadata_extractor.extract_metadata(chunk_text)
            chunks.append((chunk_text, metadata))
            
        return chunks
    
    def detect_section_title(self, text: str) -> str:
        """Try to detect section title from beginning of text"""
        patterns = [
            r'^(Introduction|Background|Methods|Results|Discussion|Conclusion|Abstract|Summary|Recommendations?)',
            r'^(\d+\.?\s+[A-Z][a-zA-Z\s]+)',
            r'^([A-Z][A-Z\s]{3,30})(?=\s)',
            r'^(Tratamiento|Diagnóstico|Seguimiento|Recomendaciones|Estadificación)',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text[:150], re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        return ""


class DocumentProcessor:
    """Main class for processing documents into chunks with medical metadata"""
    
    def __init__(self):
        self.extractor = PDFExtractor()
        self.chunker = TextChunker()
        self.chunks: List[DocumentChunk] = []
        
    def process_all_documents(self) -> List[DocumentChunk]:
        """Process all PDF documents in the Articles directory"""
        pdf_files = list(ARTICLES_DIR.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to process")
        
        all_chunks = []
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            doc_chunks = self.process_document(pdf_path)
            all_chunks.extend(doc_chunks)
            print(f"  - {pdf_path.name}: {len(doc_chunks)} chunks")
            
        self.chunks = all_chunks
        print(f"\nTotal chunks created: {len(all_chunks)}")
        self._print_metadata_stats()
        
        return all_chunks
    
    def process_document(self, pdf_path: Path) -> List[DocumentChunk]:
        """Process a single PDF document into chunks with metadata"""
        doc_name = pdf_path.stem
        doc_title = self.extractor.extract_document_title(pdf_path)
        pages = self.extractor.extract_text_from_pdf(pdf_path)
        tables = self.extractor.extract_tables_from_pdf(pdf_path)
        
        chunks = []
        chunk_counter = 0
        
        for page_data in pages:
            page_num = page_data["page_number"]
            content = page_data["content"]
            has_tables = page_data.get("has_tables", False)
            
            text_chunks = self.chunker.chunk_text(content)
            
            for chunk_text, metadata in text_chunks:
                section_title = self.chunker.detect_section_title(chunk_text)
                metadata_dict = asdict(metadata)
                metadata_dict["has_table"] = has_tables
                
                chunk = DocumentChunk(
                    chunk_id=f"{doc_name}_chunk_{chunk_counter:04d}",
                    document_name=doc_name,
                    document_title=doc_title,
                    page_start=page_num,
                    page_end=page_num,
                    content=chunk_text,
                    word_count=len(chunk_text.split()),
                    section_title=section_title,
                    medical_metadata=metadata_dict,
                )
                chunks.append(chunk)
                chunk_counter += 1
                
        for table_data in tables:
            metadata = MedicalMetadataExtractor.extract_metadata(table_data["content"])
            metadata_dict = asdict(metadata)
            metadata_dict["has_table"] = True
            
            chunk = DocumentChunk(
                chunk_id=f"{doc_name}_table_{table_data['table_index']:02d}_p{table_data['page_number']:03d}",
                document_name=doc_name,
                document_title=doc_title,
                page_start=table_data["page_number"],
                page_end=table_data["page_number"],
                content=table_data["content"],
                word_count=len(table_data["content"].split()),
                section_title="Table",
                medical_metadata=metadata_dict,
            )
            chunks.append(chunk)
            
        return chunks
    
    def _print_metadata_stats(self):
        """Print statistics about extracted medical metadata"""
        total_evidence = 0
        total_recommendations = 0
        total_risk_groups = 0
        total_markers = 0
        total_tables = 0
        total_stats = 0
        
        for chunk in self.chunks:
            meta = chunk.medical_metadata
            total_evidence += len(meta.get("evidence_levels", []))
            total_recommendations += len(meta.get("recommendations", []))
            total_risk_groups += len(meta.get("risk_groups", []))
            total_markers += len(meta.get("molecular_markers", []))
            total_tables += 1 if meta.get("has_table") else 0
            total_stats += 1 if meta.get("has_statistics") else 0
            
        print("\n--- Medical Metadata Statistics ---")
        print(f"  Evidence levels found: {total_evidence}")
        print(f"  Recommendations found: {total_recommendations}")
        print(f"  Risk groups mentioned: {total_risk_groups}")
        print(f"  Molecular markers: {total_markers}")
        print(f"  Chunks with tables: {total_tables}")
        print(f"  Chunks with statistics: {total_stats}")
    
    def save_chunks_metadata(self):
        """Save chunks metadata to JSON file"""
        metadata = [chunk.to_dict() for chunk in self.chunks]
        
        with open(CHUNKS_METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        print(f"Saved chunks metadata to {CHUNKS_METADATA_PATH}")
        
    def load_chunks_metadata(self) -> List[DocumentChunk]:
        """Load chunks metadata from JSON file"""
        if not CHUNKS_METADATA_PATH.exists():
            return []
            
        with open(CHUNKS_METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        self.chunks = [DocumentChunk.from_dict(data) for data in metadata]
        return self.chunks


if __name__ == "__main__":
    processor = DocumentProcessor()
    chunks = processor.process_all_documents()
    processor.save_chunks_metadata()
    
    print("\n--- Sample Chunks with Medical Metadata ---")
    for chunk in chunks[:3]:
        print(f"\nChunk: {chunk.chunk_id}")
        print(f"Document: {chunk.document_name}")
        print(f"Pages: {chunk.page_start}-{chunk.page_end}")
        print(f"Section: {chunk.section_title}")
        print(f"Words: {chunk.word_count}")
        print(f"Medical markers: {chunk.medical_metadata.get('molecular_markers', [])}")
        print(f"Risk groups: {chunk.medical_metadata.get('risk_groups', [])}")
        print(f"Content preview: {chunk.content[:200]}...")
