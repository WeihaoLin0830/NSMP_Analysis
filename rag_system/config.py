"""
Configuration settings for the RAG system with Groq API + Local Embeddings
Optimized for medical domain (Uterine Cancer NSMP)

Architecture:
- Embeddings: Local sentence-transformers (bge-base-en-v1.5)
- Generation: Groq API (OpenAI-compatible) with Llama models
- Vector DB: FAISS
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Base Paths
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
ARTICLES_DIR = BASE_DIR / "Articles"
DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Groq API Configuration
# =============================================================================
GROQ_API_KEY = os.getenv("API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Available Groq models for RAG (choose based on speed vs quality tradeoff)
GROQ_MODELS = {
    "fast": "llama-3.1-8b-instant",      # Very fast, good for simple queries
    "balanced": "llama-3.3-70b-versatile", # Best quality-speed balance
    "quality": "llama-3.3-70b-versatile",  # Highest quality
}

# Default model for generation
GROQ_MODEL = GROQ_MODELS["fast"]  # Using fast model for lower latency

# Generation parameters (conservative for medical domain)
GENERATION_CONFIG = {
    "temperature": 0.3,      # Low for medical accuracy
    "top_p": 0.85,
    "max_tokens": 1024,      # Reasonable response length
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1,
}

# Rate limiting configuration (Groq has TPM/RPM limits)
GROQ_RATE_LIMITS = {
    "requests_per_minute": 30,
    "tokens_per_minute": 14400,  # Conservative estimate
    "retry_delay": 2.0,
    "max_retries": 3,
    "backoff_multiplier": 2.0,
}

# =============================================================================
# Local Embedding Configuration (sentence-transformers)
# =============================================================================
# Recommended models for RAG (in order of preference):
# - BAAI/bge-base-en-v1.5: Best quality, 768 dims
# - BAAI/bge-small-en-v1.5: Faster, 384 dims
# - sentence-transformers/all-MiniLM-L6-v2: Lightweight, 384 dims

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Good balance of quality and speed
EMBEDDING_DIMENSION = 768  # Matches bge-base model

# Query instruction prefix for BGE models (improves retrieval)
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# =============================================================================
# Document Processing Settings (Medical-optimized)
# =============================================================================
CHUNK_SIZE = 500  # tokens (~300-800 recommended for medical)
CHUNK_OVERLAP = 75  # ~15% overlap
MIN_CHUNK_SIZE = 100

# Medical metadata patterns to preserve
MEDICAL_PATTERNS = {
    "evidence_levels": r"(Level\s+[IVA-D]+|Grade\s+[A-D]|[IVX]+[A-D]?)",
    "recommendations": r"(Recommendation\s*\d+|Recomendación\s*\d+)",
    "risk_groups": r"(low[-\s]?risk|intermediate[-\s]?risk|high[-\s]?risk|alto\s+riesgo|bajo\s+riesgo|riesgo\s+intermedio)",
    "molecular_markers": r"(p53|MMRd|POLE|NSMP|ER|PR|HER2|Ki-?67)",
}

# =============================================================================
# Vector Index Settings
# =============================================================================
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
CHUNKS_METADATA_PATH = DATA_DIR / "chunks_metadata.json"
SUGGESTED_QUESTIONS_PATH = DATA_DIR / "preguntas_sugeridas.txt"

# =============================================================================
# Retrieval Settings
# =============================================================================
TOP_K_RESULTS = 8  # Initial retrieval (before reranking)
SIMILARITY_THRESHOLD = 0.3
RERANK_ENABLED = False  # Disabled - Groq doesn't have native reranking
RERANK_TOP_K = 5  # Final chunks after any reranking

# Adaptive context: reduce chunks if prompt too large
MAX_CONTEXT_TOKENS = 6000  # Leave room for system prompt + response
ADAPTIVE_K_ENABLED = True  # Reduce k if context exceeds limit

# =============================================================================
# Role-based Response Configuration
# =============================================================================
ROLE_CONFIGS = {
    "patient": {
        "language_level": "simple",
        "include_technical_terms": False,
        "emotional_support": True,
        "always_recommend_doctor": True,
        "max_response_length": 600,
        "system_prompt_addon": """
Comunícate de forma cálida y empática. Usa un lenguaje sencillo evitando 
términos médicos complejos. Si mencionas un término técnico, explícalo 
brevemente. Siempre recomienda consultar con el equipo médico para 
decisiones importantes.
"""
    },
    "doctor": {
        "language_level": "technical",
        "include_technical_terms": True,
        "emotional_support": False,
        "include_evidence_levels": True,
        "include_references": True,
        "max_response_length": 1200,
        "system_prompt_addon": """
Usa terminología médica precisa. Incluye niveles de evidencia cuando estén 
disponibles (ej: Level I, Grade A). Referencia las guías clínicas y fuentes.
Estructura la respuesta de forma clínica: hallazgos, evidencia, recomendaciones.
"""
    }
}

# Base system prompt for medical RAG (más conversacional)
SYSTEM_PROMPT_BASE = """Eres un asistente médico amigable y profesional, especializado en cáncer de endometrio 
y el perfil molecular NSMP (No Specific Molecular Profile).

TU PERSONALIDAD:
- Eres cálido, empático y accesible
- Saludas cordialmente cuando el usuario te saluda
- Puedes mantener conversaciones naturales además de responder preguntas médicas

INSTRUCCIONES PARA RESPONDER:

1. **Si el usuario te saluda o hace una pregunta conversacional** (ej: "Hola", "¿Cómo estás?", "Gracias"):
   - Responde de forma natural y amigable
   - Preséntate brevemente si es apropiado
   - Ofrece tu ayuda para preguntas sobre cáncer de endometrio NSMP

2. **Si el usuario hace una pregunta médica relacionada con el contexto**:
   - Responde basándote principalmente en el CONTEXTO proporcionado
   - Cita las fuentes usando [Documento, Página X]
   - Si el contexto es limitado, puedes complementar con conocimiento general pero indica claramente qué viene del contexto y qué es información general

3. **Si el usuario hace una pregunta médica NO relacionada con el contexto**:
   - Indica amablemente que tu especialidad es cáncer de endometrio NSMP
   - Ofrece responder preguntas dentro de tu área de conocimiento
   - Sugiere que consulte a un especialista para otras áreas

4. **NUNCA inventes datos estadísticos, estudios o tratamientos específicos**

FORMATO DE RESPUESTA:
- Sé conciso pero completo
- Para preguntas médicas, incluye fuentes cuando estén disponibles
- Siempre mantén un tono profesional pero cercano
"""

# Patrones para detectar mensajes conversacionales (no médicos)
CONVERSATIONAL_PATTERNS = [
    r'^(hola|buenos?\s*d[ií]as?|buenas?\s*tardes?|buenas?\s*noches?|hey|hi|hello)',
    r'^(gracias|muchas\s*gracias|te\s*agradezco|thank)',
    r'^(adi[oó]s|hasta\s*luego|chao|bye|nos\s*vemos)',
    r'^(c[oó]mo\s*est[aá]s?|qu[eé]\s*tal|c[oó]mo\s*te\s*va)',
    r'^(qui[eé]n\s*eres|qu[eé]\s*eres|cu[aá]l\s*es\s*tu\s*nombre)',
    r'^(ok|vale|entendido|de\s*acuerdo|perfecto|genial)',
    r'^(ayuda|help|necesito\s*ayuda)$',
]

# =============================================================================
# API Settings
# =============================================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
