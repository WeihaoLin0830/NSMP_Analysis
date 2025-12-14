"""
Gemini API Client Module
Handles all interactions with Google's Gemini API for embeddings and generation
Optimized for medical domain (Uterine Cancer NSMP)
"""
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import google.generativeai as genai

from config import (
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_GENERATION_MODEL,
    GENERATION_CONFIG,
    SAFETY_SETTINGS,
    ROLE_CONFIGS,
)


@dataclass
class GeminiResponse:
    """Structured response from Gemini"""
    text: str
    finish_reason: str
    safety_ratings: List[Dict] = None
    usage: Dict = None


class GeminiClient:
    """
    Client for Google Gemini API
    Handles embeddings and text generation with medical-optimized settings
    """
    
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=GEMINI_API_KEY)
        self._generation_model = None
        self._is_initialized = False
        
    def initialize(self):
        """Initialize the generation model"""
        if self._is_initialized:
            return
            
        print("Initializing Gemini client...")
        self._generation_model = genai.GenerativeModel(
            model_name=GEMINI_GENERATION_MODEL,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
        )
        self._is_initialized = True
        print(f"Gemini client ready (model: {GEMINI_GENERATION_MODEL})")
        
    # =========================================================================
    # Embedding Methods
    # =========================================================================
    
    def get_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """
        Get embedding for a single text
        
        Args:
            text: Text to embed
            task_type: RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY for queries
            
        Returns:
            List of floats representing the embedding
        """
        try:
            result = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=text,
                task_type=task_type,
            )
            return result['embedding']
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise
            
    def get_embeddings_batch(
        self, 
        texts: List[str], 
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 100,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Get embeddings for multiple texts with batching and rate limiting
        
        Args:
            texts: List of texts to embed
            task_type: RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY for queries
            batch_size: Number of texts per batch
            show_progress: Whether to print progress
            
        Returns:
            List of embeddings
        """
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            if show_progress:
                print(f"  Embedding batch {batch_num}/{total_batches}...")
            
            try:
                result = genai.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    content=batch,
                    task_type=task_type,
                )
                embeddings.extend(result['embedding'])
                
                # Rate limiting (be nice to the API)
                if i + batch_size < len(texts):
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error in batch {batch_num}: {e}")
                # Retry with smaller batch
                for text in batch:
                    try:
                        emb = self.get_embedding(text, task_type)
                        embeddings.append(emb)
                        time.sleep(0.2)
                    except Exception as e2:
                        print(f"Failed to embed text: {e2}")
                        embeddings.append([0.0] * 768)  # Zero vector as fallback
                        
        return embeddings
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding optimized for query retrieval"""
        return self.get_embedding(query, task_type="RETRIEVAL_QUERY")
    
    # =========================================================================
    # Generation Methods
    # =========================================================================
    
    def generate(
        self,
        prompt: str,
        role: str = "patient",
        context: str = "",
        max_retries: int = 3,
    ) -> GeminiResponse:
        """
        Generate a response using Gemini
        
        Args:
            prompt: User's question
            role: User role ('patient' or 'doctor')
            context: Retrieved context from documents
            max_retries: Number of retries on failure
            
        Returns:
            GeminiResponse with generated text
        """
        if not self._is_initialized:
            self.initialize()
            
        # Build the full prompt with system instructions
        full_prompt = self._build_medical_prompt(prompt, role, context)
        
        for attempt in range(max_retries):
            try:
                response = self._generation_model.generate_content(full_prompt)
                
                return GeminiResponse(
                    text=response.text,
                    finish_reason=str(response.candidates[0].finish_reason) if response.candidates else "UNKNOWN",
                    safety_ratings=[
                        {"category": str(r.category), "probability": str(r.probability)}
                        for r in response.candidates[0].safety_ratings
                    ] if response.candidates else None,
                )
                
            except Exception as e:
                print(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return GeminiResponse(
                        text="Lo siento, hubo un error al generar la respuesta. Por favor, intenta de nuevo.",
                        finish_reason="ERROR",
                    )
    
    def _build_medical_prompt(self, question: str, role: str, context: str) -> str:
        """
        Build a grounded medical prompt based on user role
        
        Ensures responses are:
        - Grounded in the provided context
        - Appropriate for the user's role
        - Anti-hallucination compliant
        """
        role_config = ROLE_CONFIGS.get(role, ROLE_CONFIGS["patient"])
        
        if role == "doctor":
            system_prompt = """Eres un asistente clínico especializado en cáncer de endometrio y clasificación molecular NSMP (No Specific Molecular Profile).

INSTRUCCIONES CRÍTICAS:
1. SOLO responde basándote en la información del CONTEXTO proporcionado
2. Si la información no está en el contexto, di explícitamente: "Esta información no está disponible en los documentos consultados"
3. Cita siempre las fuentes: [Doc: nombre_documento, Pág: X]
4. Incluye niveles de evidencia cuando estén disponibles
5. Usa terminología médica apropiada

FORMATO DE RESPUESTA:
- **Respuesta clínica**: [respuesta técnica basada en evidencia]
- **Evidencia**: [citas de documentos con página]
- **Nivel de evidencia**: [si está disponible]
- **Limitaciones**: [qué información falta o no está clara]
- **Consideraciones adicionales**: [si aplica]"""

        else:  # patient
            system_prompt = """Eres un asistente amable que ayuda a pacientes a entender información sobre cáncer de endometrio.

INSTRUCCIONES CRÍTICAS:
1. SOLO responde basándote en la información del CONTEXTO proporcionado
2. Si la información no está en el contexto, di: "No tengo esa información en los documentos disponibles. Te recomiendo consultar con tu médico"
3. Usa lenguaje sencillo y evita términos técnicos (o explícalos si los usas)
4. Sé empático y tranquilizador, pero honesto
5. SIEMPRE recomienda consultar con el médico para decisiones específicas

FORMATO DE RESPUESTA:
- Explicación clara y comprensible
- Puntos clave resumidos
- Recordatorio de consultar con el profesional médico
- Fuentes consultadas (de forma simple)"""

        # Build the full prompt
        prompt = f"""{system_prompt}

═══════════════════════════════════════════════════════════════
CONTEXTO MÉDICO (Documentos de referencia):
═══════════════════════════════════════════════════════════════
{context if context else "No hay contexto disponible para esta consulta."}

═══════════════════════════════════════════════════════════════
PREGUNTA DEL {'PROFESIONAL MÉDICO' if role == 'doctor' else 'PACIENTE'}:
═══════════════════════════════════════════════════════════════
{question}

═══════════════════════════════════════════════════════════════
RESPUESTA:
═══════════════════════════════════════════════════════════════"""

        return prompt
    
    # =========================================================================
    # Reranking Methods
    # =========================================================================
    
    def rerank_chunks(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Use Gemini to rerank retrieved chunks by relevance
        
        Args:
            query: User's question
            chunks: List of chunk dictionaries with 'content' key
            top_k: Number of chunks to return after reranking
            
        Returns:
            Reranked list of chunks
        """
        if not self._is_initialized:
            self.initialize()
            
        if len(chunks) <= top_k:
            return chunks
            
        # Build reranking prompt
        chunks_text = "\n\n".join([
            f"[CHUNK {i+1}]: {c.get('content', '')[:500]}"
            for i, c in enumerate(chunks[:12])  # Limit to 12 for token efficiency
        ])
        
        rerank_prompt = f"""Eres un experto en oncología ginecológica. 
Ordena los siguientes fragmentos de documentos médicos por relevancia para responder la pregunta.

PREGUNTA: {query}

FRAGMENTOS:
{chunks_text}

Devuelve SOLO los números de los {top_k} fragmentos más relevantes, ordenados de mayor a menor relevancia.
Formato: 1, 3, 5, 2, 4 (solo números separados por comas)"""

        try:
            response = self._generation_model.generate_content(rerank_prompt)
            
            # Parse the response to get indices
            indices_text = response.text.strip()
            indices = []
            for part in indices_text.replace(",", " ").split():
                try:
                    idx = int(part.strip()) - 1  # Convert to 0-indexed
                    if 0 <= idx < len(chunks):
                        indices.append(idx)
                except ValueError:
                    continue
                    
            # Return reranked chunks
            if indices:
                reranked = [chunks[i] for i in indices[:top_k]]
                return reranked
            else:
                return chunks[:top_k]
                
        except Exception as e:
            print(f"Reranking failed: {e}")
            return chunks[:top_k]


# Singleton instance
_gemini_client: Optional[GeminiClient] = None

def get_gemini_client() -> GeminiClient:
    """Get or create the Gemini client singleton"""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


if __name__ == "__main__":
    # Test the client
    client = get_gemini_client()
    client.initialize()
    
    # Test embedding
    print("\n--- Testing Embedding ---")
    test_text = "El carcinoma de endometrio NSMP representa el 30% de los casos"
    embedding = client.get_embedding(test_text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test generation
    print("\n--- Testing Generation (Patient) ---")
    response = client.generate(
        prompt="¿Qué es el perfil molecular NSMP?",
        role="patient",
        context="[Doc: ESGO 2021, Pág: 5] NSMP (No Specific Molecular Profile) representa aproximadamente el 30% de los carcinomas de endometrio. Se caracteriza por la ausencia de mutaciones específicas en p53, POLE o deficiencia de MMR."
    )
    print(f"Response: {response.text[:500]}...")
    
    print("\n--- Testing Generation (Doctor) ---")
    response = client.generate(
        prompt="¿Cuáles son las indicaciones de radioterapia adyuvante en NSMP de riesgo intermedio?",
        role="doctor",
        context="[Doc: ESGO-ESTRO 2021, Pág: 12] Para pacientes con carcinoma endometrioide NSMP de riesgo intermedio (estadio I, grado 1-2, invasión miometrial <50%), se recomienda braquiterapia vaginal adyuvante (Nivel de evidencia IIA)."
    )
    print(f"Response: {response.text[:500]}...")
