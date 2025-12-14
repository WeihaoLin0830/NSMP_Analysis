"""
RAG Pipeline Module
Main pipeline orchestrating document retrieval and response generation

Architecture:
- Embeddings: Local sentence-transformers (BGE model)
- Retrieval: FAISS vector search
- Generation: Groq API (Llama models)
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from embeddings import VectorStore, SearchResult
from groq_client import GroqClient
from config import TOP_K_RESULTS, RERANK_TOP_K


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
    latency_ms: Optional[float] = None
    
    def to_dict(self):
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
        }


class RAGPipeline:
    """
    Main RAG Pipeline for question answering
    
    Orchestrates:
    1. Query embedding with local sentence-transformers
    2. Document retrieval with FAISS
    3. Response generation with Groq API (Llama models)
    
    Note: Reranking disabled since Groq doesn't offer it natively.
    Future: Could implement with cross-encoder model locally.
    """
    
    def __init__(self):
        """Initialize RAG pipeline"""
        self.vector_store = VectorStore()
        self.groq_client: Optional[GroqClient] = None
        self._is_initialized = False
        self._last_search_results = []  # Store last search results for contextual suggestions
        
    def initialize(self):
        """Load vector store and initialize clients"""
        if self._is_initialized:
            return
            
        print("Initializing RAG pipeline...")
        print("  - Loading embedding model (local)")
        print("  - Connecting to Groq API")
        
        # Initialize Groq client
        self.groq_client = GroqClient()
        
        try:
            # Try to load existing index
            self.vector_store.load()
            print("  - Loaded existing vector store")
        except FileNotFoundError:
            # Build from scratch
            print("  - Building vector store from documents...")
            self.vector_store.build_from_documents()
            
        self._is_initialized = True
        print("RAG pipeline ready!")
        
    def query(
        self,
        question: str,
        role: str = "patient",
        top_k: int = None,
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User's question
            role: User role ('patient' or 'doctor')
            top_k: Number of documents to retrieve (default from config)
            
        Returns:
            RAGResponse with answer and sources
        """
        if not self._is_initialized:
            self.initialize()
            
        top_k = top_k or TOP_K_RESULTS
            
        # Step 1: Retrieve relevant documents with FAISS
        search_results = self.vector_store.search(question, top_k)
        
        if not search_results:
            return RAGResponse(
                answer="Lo siento, no encontré información relevante para tu pregunta en los documentos disponibles. Por favor, intenta reformular tu consulta.",
                sources=[],
                query=question,
                confidence=0.0
            )
        
        # Step 2: Limit to top results for context (adaptive k)
        # Take top RERANK_TOP_K results to keep context manageable
        search_results = search_results[:RERANK_TOP_K]
        
        # Store search results for contextual suggestions
        self._last_search_results = search_results
        
        # Step 3: Build context from retrieved documents
        context = self._build_context(search_results)
        
        # Step 4: Generate response with Groq
        groq_response = self.groq_client.generate(
            query=question,
            context=context,
            role=role
        )
        
        # Step 5: Format sources
        sources = self._format_sources(search_results)
        
        # Calculate confidence based on similarity scores
        avg_score = sum(r.similarity_score for r in search_results) / len(search_results)
        
        return RAGResponse(
            answer=groq_response.text,
            sources=sources,
            query=question,
            confidence=avg_score,
            latency_ms=groq_response.latency_ms,
        )
        
    def _build_context(self, results: List[SearchResult]) -> str:
        """Build context string from search results with medical metadata"""
        context_parts = []
        
        for idx, result in enumerate(results, 1):
            chunk = result.chunk
            
            # Build source reference with metadata
            source_ref = f"[FUENTE {idx}: {chunk.document_name}, Página {chunk.page_start}]"
            
            # Add section title if available
            if chunk.section_title:
                source_ref += f" - {chunk.section_title}"
            
            # Add medical metadata if relevant
            metadata_parts = []
            if chunk.medical_metadata:
                if chunk.medical_metadata.get("evidence_levels"):
                    metadata_parts.append(f"Niveles de evidencia: {', '.join(chunk.medical_metadata['evidence_levels'][:2])}")
                if chunk.medical_metadata.get("risk_groups"):
                    metadata_parts.append(f"Grupos de riesgo: {', '.join(chunk.medical_metadata['risk_groups'][:3])}")
                if chunk.medical_metadata.get("molecular_markers"):
                    metadata_parts.append(f"Marcadores: {', '.join(chunk.medical_metadata['molecular_markers'][:3])}")
            
            metadata_str = f" ({'; '.join(metadata_parts)})" if metadata_parts else ""
            
            context_parts.append(f"{source_ref}{metadata_str}\n{chunk.content}")
            
        return "\n\n---\n\n".join(context_parts)
        
    def _format_sources(self, results: List[SearchResult]) -> List[Dict]:
        """Format sources for response"""
        sources = []
        seen = set()
        
        for result in results:
            key = (result.chunk.document_name, result.chunk.page_start)
            if key not in seen:
                sources.append({
                    "document": result.chunk.document_name,
                    "page": result.chunk.page_start,
                    "section": result.chunk.section_title or "",
                    "relevance": round(result.similarity_score, 3),
                    "preview": result.chunk.content[:150] + "..."
                })
                seen.add(key)
                
        return sources
    
    def get_suggested_questions(
        self, 
        role: str = "patient",
        context_chunks: List[SearchResult] = None,
        previous_query: str = None,
    ) -> List[str]:
        """
        Get suggested questions for the user role
        
        Args:
            role: User role ('patient' or 'doctor')
            context_chunks: Recent search results to base suggestions on
            previous_query: The user's previous query for context
            
        Returns:
            List of suggested question strings
        """
        import random
        
        # Base questions by role
        base_questions = {
            "patient": [
                "¿Qué es el perfil molecular NSMP?",
                "¿Cuál es el tratamiento estándar para mi caso?",
                "¿Cuáles son los factores de pronóstico?",
                "¿Qué seguimiento médico necesitaré?",
                "¿Cuáles son los posibles efectos secundarios del tratamiento?",
                "¿Qué síntomas debo vigilar después del tratamiento?",
                "¿Cuál es la tasa de supervivencia?",
                "¿Es necesaria la quimioterapia en mi caso?",
                "¿Qué es la radioterapia adyuvante?",
                "¿Cuándo debo consultar con urgencia?",
            ],
            "doctor": [
                "¿Cuáles son los criterios de clasificación molecular del cáncer de endometrio?",
                "¿Cuál es el manejo de pacientes NSMP de alto riesgo?",
                "¿Cuáles son las indicaciones de radioterapia adyuvante?",
                "¿Qué protocolo de seguimiento se recomienda según las guías ESGO?",
                "¿Cuáles son los factores pronósticos en carcinoma endometrioide?",
                "¿Cuándo está indicada la linfadenectomía?",
                "¿Qué evidencia existe sobre quimioterapia adyuvante en NSMP?",
                "¿Cómo se define la invasión miometrial profunda?",
                "¿Cuáles son las tasas de recurrencia por grupo de riesgo?",
                "¿Qué rol tiene la braquiterapia vaginal?",
            ]
        }
        
        role_questions = base_questions.get(role, base_questions["patient"])
        
        # If we have context from recent search, generate contextual suggestions
        if context_chunks and len(context_chunks) > 0:
            contextual_questions = self._generate_contextual_questions(
                context_chunks, role, previous_query
            )
            if contextual_questions:
                # Mix contextual with base questions
                selected_contextual = contextual_questions[:3]
                remaining_base = [q for q in role_questions if q not in selected_contextual]
                random.shuffle(remaining_base)
                return selected_contextual + remaining_base[:2]
        
        # Otherwise return shuffled base questions
        random.shuffle(role_questions)
        return role_questions[:5]
    
    def _generate_contextual_questions(
        self,
        chunks: List[SearchResult],
        role: str,
        previous_query: str = None,
    ) -> List[str]:
        """
        Generate questions based on the retrieved context
        
        Args:
            chunks: Search results with relevant content
            role: User role
            previous_query: Previous user query
            
        Returns:
            List of contextual question suggestions
        """
        contextual_questions = []
        
        # Analyze chunks for topics
        topics_found = set()
        
        for result in chunks[:3]:  # Look at top 3 chunks
            content_lower = result.chunk.content.lower()
            metadata = result.chunk.medical_metadata or {}
            
            # Check for specific topics in content
            if 'tratamiento' in content_lower or 'treatment' in content_lower:
                topics_found.add('treatment')
            if 'radioterapia' in content_lower or 'radiotherapy' in content_lower:
                topics_found.add('radiotherapy')
            if 'quimioterapia' in content_lower or 'chemotherapy' in content_lower:
                topics_found.add('chemotherapy')
            if 'supervivencia' in content_lower or 'survival' in content_lower:
                topics_found.add('survival')
            if 'riesgo' in content_lower or 'risk' in content_lower:
                topics_found.add('risk')
            if 'seguimiento' in content_lower or 'follow-up' in content_lower:
                topics_found.add('followup')
            if 'cirugía' in content_lower or 'surgery' in content_lower or 'histerectomía' in content_lower:
                topics_found.add('surgery')
            if 'estadio' in content_lower or 'stage' in content_lower:
                topics_found.add('staging')
            
            # Check medical metadata
            if metadata.get('risk_groups'):
                topics_found.add('risk')
            if metadata.get('molecular_markers'):
                topics_found.add('molecular')
        
        # Generate questions based on found topics
        topic_questions = {
            "treatment": {
                "patient": "¿Cuáles son las opciones de tratamiento disponibles?",
                "doctor": "¿Cuál es el algoritmo de tratamiento según grupo de riesgo?",
            },
            "radiotherapy": {
                "patient": "¿En qué consiste la radioterapia y cuándo es necesaria?",
                "doctor": "¿Cuáles son las indicaciones específicas de radioterapia adyuvante?",
            },
            "chemotherapy": {
                "patient": "¿Necesitaré quimioterapia? ¿Cuáles son los efectos secundarios?",
                "doctor": "¿Cuál es la evidencia sobre quimioterapia adyuvante en NSMP?",
            },
            "survival": {
                "patient": "¿Cuál es el pronóstico general para mi perfil?",
                "doctor": "¿Cuáles son las tasas de supervivencia por estadio y grupo de riesgo?",
            },
            "risk": {
                "patient": "¿En qué grupo de riesgo me encuentro?",
                "doctor": "¿Cómo se estratifica el riesgo en pacientes NSMP?",
            },
            "followup": {
                "patient": "¿Con qué frecuencia tendré que hacer revisiones?",
                "doctor": "¿Cuál es el protocolo de seguimiento recomendado?",
            },
            "surgery": {
                "patient": "¿Qué tipo de cirugía se recomienda?",
                "doctor": "¿Cuáles son las indicaciones de linfadenectomía?",
            },
            "staging": {
                "patient": "¿Qué significa el estadio de mi cáncer?",
                "doctor": "¿Cómo afecta el estadio a las decisiones terapéuticas?",
            },
            "molecular": {
                "patient": "¿Qué significa tener un perfil molecular NSMP?",
                "doctor": "¿Cómo influye el perfil molecular en el pronóstico?",
            },
        }
        
        for topic in topics_found:
            if topic in topic_questions:
                question = topic_questions[topic].get(role, topic_questions[topic]["patient"])
                if question not in contextual_questions:
                    contextual_questions.append(question)
        
        return contextual_questions[:5]


class ConversationManager:
    """Manages conversation history and context"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.pipeline = rag_pipeline
        self.conversations: Dict[str, List[ChatMessage]] = {}
        self._last_search_results: Dict[str, List[SearchResult]] = {}
        
    def create_session(self, session_id: str):
        """Create a new conversation session"""
        self.conversations[session_id] = []
        self._last_search_results[session_id] = []
        
    def add_message(self, session_id: str, message: ChatMessage):
        """Add message to conversation history"""
        if session_id not in self.conversations:
            self.create_session(session_id)
        self.conversations[session_id].append(message)
        
    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Get conversation history"""
        return self.conversations.get(session_id, [])
    
    def get_last_search_results(self, session_id: str) -> List[SearchResult]:
        """Get last search results for a session"""
        return self._last_search_results.get(session_id, [])
        
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
        
        # Get RAG response (this internally does the search)
        response = self.pipeline.query(message, role)
        
        # Store search results for contextual suggestions
        # We need to get them from the last query
        self._last_search_results[session_id] = getattr(
            self.pipeline, '_last_search_results', []
        )
        
        # Add assistant message
        assistant_msg = ChatMessage(
            role="assistant",
            content=response.answer,
            sources=response.sources
        )
        self.add_message(session_id, assistant_msg)
        
        return response
    
    def get_contextual_suggestions(
        self, 
        session_id: str, 
        role: str = "patient"
    ) -> List[str]:
        """
        Get suggested questions based on conversation context
        
        Args:
            session_id: Session ID
            role: User role
            
        Returns:
            List of contextual question suggestions
        """
        # Get last search results for this session
        last_results = self.get_last_search_results(session_id)
        
        # Get last user message
        history = self.get_history(session_id)
        last_query = None
        for msg in reversed(history):
            if msg.role == "user":
                last_query = msg.content
                break
        
        return self.pipeline.get_suggested_questions(
            role=role,
            context_chunks=last_results if last_results else None,
            previous_query=last_query,
        )
        
    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            self.conversations[session_id] = []
        if session_id in self._last_search_results:
            self._last_search_results[session_id] = []


if __name__ == "__main__":
    # Test the RAG pipeline
    pipeline = RAGPipeline()
    pipeline.initialize()
    
    # Test queries for both roles
    test_queries = [
        ("Hola, ¿cómo estás?", "patient"),
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
        
        # Show suggestions
        suggestions = pipeline.get_suggested_questions(role)
        print(f"\nSuggested questions:")
        for q in suggestions[:3]:
            print(f"  • {q}")
