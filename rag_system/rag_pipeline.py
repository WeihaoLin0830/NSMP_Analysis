"""
RAG Pipeline Module
Main pipeline orchestrating document retrieval and response generation
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from embeddings import VectorStore, SearchResult
from generator import get_response_generator, SimpleResponseGenerator


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    sources: List[Dict] = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class RAGResponse:
    """Complete RAG response with metadata"""
    answer: str
    sources: List[Dict]
    query: str
    confidence: float
    
    def to_dict(self):
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "confidence": self.confidence
        }


class RAGPipeline:
    """
    Main RAG Pipeline for question answering
    
    Orchestrates:
    1. Query embedding
    2. Document retrieval
    3. Response generation
    """
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize RAG pipeline
        
        Args:
            use_llm: Whether to use full LLM for generation
        """
        self.vector_store = VectorStore()
        self.generator = get_response_generator(use_llm)
        self._is_initialized = False
        
    def initialize(self):
        """Load vector store and models"""
        if self._is_initialized:
            return
            
        print("Initializing RAG pipeline...")
        
        try:
            # Try to load existing index
            self.vector_store.load()
            print("Loaded existing vector store")
        except FileNotFoundError:
            # Build from scratch
            print("Building vector store from documents...")
            self.vector_store.build_from_documents()
            
        self._is_initialized = True
        print("RAG pipeline ready")
        
    def query(
        self,
        question: str,
        role: str = "patient",
        top_k: int = 5
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User's question
            role: User role ('patient' or 'doctor')
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResponse with answer and sources
        """
        if not self._is_initialized:
            self.initialize()
            
        # Step 1: Retrieve relevant documents
        search_results = self.vector_store.search(question, top_k)
        
        if not search_results:
            return RAGResponse(
                answer="Lo siento, no encontré información relevante para tu pregunta en los documentos disponibles. Por favor, intenta reformular tu consulta.",
                sources=[],
                query=question,
                confidence=0.0
            )
            
        # Step 2: Build context from retrieved documents
        context = self._build_context(search_results)
        
        # Step 3: Generate response
        answer = self.generator.generate_response(
            query=question,
            context=context,
            role=role
        )
        
        # Step 4: Format sources
        sources = self._format_sources(search_results)
        
        # Calculate confidence based on similarity scores
        avg_score = sum(r.similarity_score for r in search_results) / len(search_results)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            confidence=avg_score
        )
        
    def _build_context(self, results: List[SearchResult]) -> str:
        """Build context string from search results"""
        context_parts = []
        
        for result in results:
            source_ref = f"[Fuente: {result.chunk.document_name}, Página {result.chunk.page_number}]"
            context_parts.append(f"{source_ref}\n{result.chunk.content}")
            
        return "\n\n---\n\n".join(context_parts)
        
    def _format_sources(self, results: List[SearchResult]) -> List[Dict]:
        """Format sources for response"""
        sources = []
        seen = set()
        
        for result in results:
            key = (result.chunk.document_name, result.chunk.page_number)
            if key not in seen:
                sources.append({
                    "document": result.chunk.document_name,
                    "page": result.chunk.page_number,
                    "relevance": round(result.similarity_score, 3),
                    "preview": result.chunk.content[:150] + "..."
                })
                seen.add(key)
                
        return sources
    
    def get_suggested_questions(self, role: str = "patient") -> List[str]:
        """Get suggested questions for the user role"""
        from question_generator import QuestionGenerator
        
        gen = QuestionGenerator()
        try:
            gen.load_chunks()
            gen.extract_topics()
            questions = gen.generate_questions()
            role_questions = gen.get_questions_for_role(role)
            return [q["question"] for q in role_questions[:10]]
        except Exception as e:
            print(f"Error getting suggestions: {e}")
            return [
                "¿Qué es el perfil molecular NSMP?",
                "¿Cuál es el tratamiento estándar?",
                "¿Cuáles son los factores de pronóstico?",
                "¿Qué seguimiento se recomienda?",
            ]


class ConversationManager:
    """Manages conversation history and context"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.pipeline = rag_pipeline
        self.conversations: Dict[str, List[ChatMessage]] = {}
        
    def create_session(self, session_id: str):
        """Create a new conversation session"""
        self.conversations[session_id] = []
        
    def add_message(self, session_id: str, message: ChatMessage):
        """Add message to conversation history"""
        if session_id not in self.conversations:
            self.create_session(session_id)
        self.conversations[session_id].append(message)
        
    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Get conversation history"""
        return self.conversations.get(session_id, [])
        
    def process_message(
        self,
        session_id: str,
        message: str,
        role: str = "patient"
    ) -> RAGResponse:
        """Process a user message and generate response"""
        
        # Add user message
        user_msg = ChatMessage(role="user", content=message)
        self.add_message(session_id, user_msg)
        
        # Get RAG response
        response = self.pipeline.query(message, role)
        
        # Add assistant message
        assistant_msg = ChatMessage(
            role="assistant",
            content=response.answer,
            sources=response.sources
        )
        self.add_message(session_id, assistant_msg)
        
        return response
        
    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            self.conversations[session_id] = []


if __name__ == "__main__":
    # Test the pipeline
    pipeline = RAGPipeline(use_llm=False)
    pipeline.initialize()
    
    # Test queries
    test_queries = [
        ("¿Qué es el perfil NSMP?", "patient"),
        ("¿Cuáles son las indicaciones de radioterapia adyuvante?", "doctor"),
    ]
    
    for query, role in test_queries:
        print(f"\n{'='*60}")
        print(f"Query ({role}): {query}")
        print("="*60)
        
        response = pipeline.query(query, role)
        
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nConfidence: {response.confidence:.2f}")
        print(f"\nSources:")
        for source in response.sources:
            print(f"  - {source['document']} (p.{source['page']})")
