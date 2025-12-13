"""
Initialization script for the RAG system
Processes documents and builds the vector index
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from document_processor import DocumentProcessor
from embeddings import VectorStore
from question_generator import QuestionGenerator
from config import DATA_DIR, ARTICLES_DIR


def initialize_system():
    """Initialize the complete RAG system"""
    
    print("=" * 60)
    print("NSMP Cancer RAG System - Initialization")
    print("=" * 60)
    
    # Check if Articles directory exists
    if not ARTICLES_DIR.exists():
        print(f"Error: Articles directory not found at {ARTICLES_DIR}")
        return False
        
    pdf_files = list(ARTICLES_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in {ARTICLES_DIR}")
        return False
        
    print(f"\nFound {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
        
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    
    # Step 1: Process documents
    print("\n" + "-" * 40)
    print("Step 1: Processing documents")
    print("-" * 40)
    
    processor = DocumentProcessor()
    chunks = processor.process_all_documents()
    processor.save_chunks_metadata()
    
    # Step 2: Build vector index
    print("\n" + "-" * 40)
    print("Step 2: Building vector index")
    print("-" * 40)
    
    vector_store = VectorStore()
    vector_store.build_from_documents()
    
    # Step 3: Generate suggested questions
    print("\n" + "-" * 40)
    print("Step 3: Generating suggested questions")
    print("-" * 40)
    
    question_gen = QuestionGenerator()
    question_gen.load_chunks()
    question_gen.extract_topics()
    question_gen.generate_questions()
    question_gen.save_questions()
    
    print("\n" + "=" * 60)
    print("Initialization Complete!")
    print("=" * 60)
    print(f"\nStats:")
    print(f"  - Documents processed: {len(pdf_files)}")
    print(f"  - Chunks created: {len(chunks)}")
    print(f"  - Questions generated: {len(question_gen.questions)}")
    print(f"\nData saved to: {DATA_DIR}")
    
    return True


if __name__ == "__main__":
    success = initialize_system()
    sys.exit(0 if success else 1)
