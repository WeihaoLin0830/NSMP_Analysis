"""
FastAPI Application for RAG Chatbot
Provides REST API endpoints for the chatbot system
"""
import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from config import API_HOST, API_PORT, CORS_ORIGINS
from rag_pipeline import RAGPipeline, ConversationManager, RAGResponse


# =============================================================================
# Pydantic Models
# =============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, description="User's message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    role: str = Field("patient", description="User role: 'patient' or 'doctor'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "¿Qué es el perfil molecular NSMP?",
                "session_id": "abc123",
                "role": "patient"
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str
    sources: List[dict]
    session_id: str
    confidence: float
    suggested_questions: Optional[List[str]] = None


class SourceDocument(BaseModel):
    """Model for source document information"""
    document: str
    page: int
    relevance: float
    preview: str


class IndexStatus(BaseModel):
    """Model for index status"""
    status: str
    total_chunks: int
    documents_indexed: List[str]
    last_updated: Optional[str] = None


class SuggestedQuestionsResponse(BaseModel):
    """Response model for suggested questions"""
    questions: List[str]
    role: str


# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="NSMP Cancer RAG Chatbot API",
    description="API para el chatbot de asistencia sobre cáncer de endometrio NSMP",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_pipeline: Optional[RAGPipeline] = None
conversation_manager: Optional[ConversationManager] = None


# =============================================================================
# Startup & Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline, conversation_manager
    
    print("=" * 50)
    print("Initializing RAG system...")
    print("  - Embeddings: Local (sentence-transformers)")
    print("  - Generation: Groq API (Llama)")
    print("=" * 50)
    
    rag_pipeline = RAGPipeline()
    
    try:
        rag_pipeline.initialize()
        conversation_manager = ConversationManager(rag_pipeline)
        print("RAG system initialized successfully!")
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        print("System will attempt to build index on first query")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down RAG system...")


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "NSMP Cancer RAG Chatbot",
        "version": "2.0.0",
        "backend": "Groq + Local Embeddings",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "rag_initialized": rag_pipeline is not None and rag_pipeline._is_initialized,
        "conversation_manager": conversation_manager is not None
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Process a chat message and return AI response
    
    - **message**: The user's question or message
    - **session_id**: Optional session ID for conversation tracking
    - **role**: User role ('patient' or 'doctor') affects response style
    """
    global rag_pipeline, conversation_manager
    
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
        
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Process through RAG pipeline
        if conversation_manager:
            response = conversation_manager.process_message(
                session_id=session_id,
                message=request.message,
                role=request.role
            )
            # Get contextual suggested questions based on last query context
            suggestions = conversation_manager.get_contextual_suggestions(
                session_id=session_id,
                role=request.role
            )[:5]
        else:
            response = rag_pipeline.query(
                question=request.message,
                role=request.role
            )
            # Fallback to static suggestions if no conversation manager
            suggestions = rag_pipeline.get_suggested_questions(request.role)[:5]
        
        return ChatResponse(
            answer=response.answer,
            sources=response.sources,
            session_id=session_id,
            confidence=response.confidence,
            suggested_questions=suggestions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/suggestions/{role}", response_model=SuggestedQuestionsResponse, tags=["Chat"])
async def get_suggestions(role: str = "patient"):
    """
    Get suggested questions for a specific role
    
    - **role**: User role ('patient' or 'doctor')
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
        
    if role not in ["patient", "doctor"]:
        raise HTTPException(status_code=400, detail="Invalid role. Use 'patient' or 'doctor'")
        
    questions = rag_pipeline.get_suggested_questions(role)
    
    return SuggestedQuestionsResponse(
        questions=questions,
        role=role
    )


@app.post("/index/rebuild", tags=["Admin"])
async def rebuild_index(background_tasks: BackgroundTasks):
    """
    Rebuild the document index from scratch
    
    This will reprocess all PDFs in the Articles folder
    """
    global rag_pipeline
    
    def rebuild():
        global rag_pipeline
        rag_pipeline = RAGPipeline()
        rag_pipeline.vector_store.build_from_documents()
        rag_pipeline._is_initialized = True
        
        # Also regenerate questions
        try:
            from question_generator import QuestionGenerator
            gen = QuestionGenerator()
            gen.load_chunks()
            gen.extract_topics()
            gen.generate_questions()
            gen.save_questions()
        except Exception as e:
            print(f"Error regenerating questions: {e}")
        
    background_tasks.add_task(rebuild)
    
    return {
        "status": "rebuilding",
        "message": "Index rebuild started in background"
    }


@app.get("/index/status", response_model=IndexStatus, tags=["Admin"])
async def get_index_status():
    """Get current status of the document index"""
    if rag_pipeline is None or not rag_pipeline._is_initialized:
        return IndexStatus(
            status="not_initialized",
            total_chunks=0,
            documents_indexed=[]
        )
        
    chunks = rag_pipeline.vector_store.faiss_index.chunks
    documents = list(set(c.document_name for c in chunks))
    
    return IndexStatus(
        status="ready",
        total_chunks=len(chunks),
        documents_indexed=documents
    )


@app.delete("/session/{session_id}", tags=["Chat"])
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    if conversation_manager:
        conversation_manager.clear_session(session_id)
        return {"status": "cleared", "session_id": session_id}
    return {"status": "no_session_manager"}


@app.get("/session/{session_id}/history", tags=["Chat"])
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    if conversation_manager:
        history = conversation_manager.get_history(session_id)
        return {
            "session_id": session_id,
            "messages": [msg.to_dict() for msg in history]
        }
    return {"session_id": session_id, "messages": []}


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print(f"Starting RAG API server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
