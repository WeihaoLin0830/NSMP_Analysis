"""
Embedding and Indexing Module
Handles vector embeddings and FAISS index for semantic search
Uses local sentence-transformers (BGE model) for embeddings

Architecture:
- Embeddings: sentence-transformers (BAAI/bge-base-en-v1.5)
- Vector DB: FAISS with IndexFlatIP (cosine similarity via normalized vectors)
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_QUERY_PREFIX,
    FAISS_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    DATA_DIR
)
from document_processor import DocumentChunk, DocumentProcessor


@dataclass
class SearchResult:
    """Represents a search result with chunk and score"""
    chunk: DocumentChunk
    similarity_score: float
    rank: int


class EmbeddingModel:
    """
    Handles text embedding using local sentence-transformers
    
    Uses BGE (BAAI General Embedding) model which is optimized for RAG:
    - Supports query/document distinction
    - Good multilingual performance
    - 768-dimensional embeddings
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model
        
        Args:
            model_name: Model to use (default from config)
        """
        model_name = model_name or EMBEDDING_MODEL
        print(f"Loading embedding model: {model_name}")
        print("(This may take a moment on first run...)")
        
        self.model = SentenceTransformer(model_name)
        self.dimension = EMBEDDING_DIMENSION
        self.query_prefix = EMBEDDING_QUERY_PREFIX
        
        print(f"Embedding model ready (dimension: {self.dimension})")
        
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: List of texts to encode (documents/chunks)
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (normalized for cosine similarity)
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            batch_size=32,
        )
        return embeddings
    
    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode a single text into embedding
        
        Args:
            text: Text to encode
            is_query: If True, add query prefix for better retrieval
            
        Returns:
            Normalized embedding vector
        """
        # BGE models benefit from query instruction prefix
        if is_query and self.query_prefix:
            text = self.query_prefix + text
            
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        
        return embedding
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encode multiple queries with query prefix
        
        Args:
            queries: List of query strings
            
        Returns:
            Numpy array of query embeddings
        """
        # Add query prefix to all queries
        if self.query_prefix:
            queries = [self.query_prefix + q for q in queries]
            
        return self.encode(queries, show_progress=False)


class FAISSIndex:
    """Manages FAISS index for efficient similarity search"""
    
    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []
        
    def build_index(self, embeddings: np.ndarray, chunks: List[DocumentChunk]):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings
            chunks: List of corresponding DocumentChunks
        """
        self.chunks = chunks
        
        # Use IndexFlatIP for cosine similarity (since embeddings are normalized)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        print(f"Built FAISS index with {self.index.ntotal} vectors")
        
    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K_RESULTS) -> List[SearchResult]:
        """
        Search for most similar chunks
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
            
        # Reshape query for FAISS
        query = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        scores, indices = self.index.search(query, top_k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and score >= SIMILARITY_THRESHOLD:
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    similarity_score=float(score),
                    rank=rank + 1
                ))
                
        return results
    
    def save(self, index_path: Path = FAISS_INDEX_PATH):
        """Save FAISS index to disk"""
        if self.index is None:
            raise ValueError("No index to save")
            
        faiss.write_index(self.index, str(index_path))
        print(f"Saved FAISS index to {index_path}")
        
    def load(self, index_path: Path = FAISS_INDEX_PATH):
        """Load FAISS index from disk"""
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
            
        self.index = faiss.read_index(str(index_path))
        
        # Load chunks metadata
        processor = DocumentProcessor()
        self.chunks = processor.load_chunks_metadata()
        
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")


class VectorStore:
    """Main class for managing embeddings and search"""
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.faiss_index = FAISSIndex()
        self._embeddings: Optional[np.ndarray] = None
        
    def build_from_documents(self):
        """Build vector store from all documents"""
        # Process documents
        processor = DocumentProcessor()
        chunks = processor.process_all_documents()
        processor.save_chunks_metadata()
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        self._embeddings = embeddings
        
        # Save embeddings
        embeddings_path = DATA_DIR / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to {embeddings_path}")
        
        # Build FAISS index
        self.faiss_index.build_index(embeddings, chunks)
        self.faiss_index.save()
        
        return chunks, embeddings
        
    def load(self):
        """Load existing vector store"""
        self.faiss_index.load()
        
        embeddings_path = DATA_DIR / "embeddings.npy"
        if embeddings_path.exists():
            self._embeddings = np.load(embeddings_path)
            
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[SearchResult]:
        """
        Search for relevant chunks using query-optimized embedding
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        query_embedding = self.embedding_model.encode_single(query, is_query=True)
        results = self.faiss_index.search(query_embedding, top_k)
        return results
    
    def get_context_for_query(self, query: str, top_k: int = TOP_K_RESULTS) -> str:
        """
        Get concatenated context from top search results
        
        Args:
            query: Search query
            top_k: Number of chunks to include
            
        Returns:
            Concatenated context string
        """
        results = self.search(query, top_k)
        
        context_parts = []
        for result in results:
            source = f"[Fuente: {result.chunk.document_name}, Página {result.chunk.page_start}]"
            context_parts.append(f"{source}\n{result.chunk.content}")
            
        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    # Build vector store
    vector_store = VectorStore()
    chunks, embeddings = vector_store.build_from_documents()
    
    # Test search
    print("\n--- Testing Search ---")
    test_queries = [
        "¿Cuál es el tratamiento para pacientes NSMP?",
        "factores de riesgo cáncer de endometrio",
        "tasa de supervivencia grupos de riesgo"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.search(query, top_k=3)
        for result in results:
            print(f"  [{result.rank}] Score: {result.similarity_score:.3f}")
            print(f"      Source: {result.chunk.document_name} (p.{result.chunk.page_start})")
            print(f"      Preview: {result.chunk.content[:100]}...")
