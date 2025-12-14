"""
Groq API Client Module
Handles text generation via Groq's OpenAI-compatible API
Optimized for medical RAG (Uterine Cancer NSMP)

Note: Groq does not provide embeddings - use sentence-transformers locally
"""
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from openai import OpenAI

from config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    GROQ_MODELS,
    GENERATION_CONFIG,
    GROQ_RATE_LIMITS,
    ROLE_CONFIGS,
    SYSTEM_PROMPT_BASE,
    MAX_CONTEXT_TOKENS,
)


@dataclass
class GroqResponse:
    """Structured response from Groq API"""
    text: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    latency_ms: float


class GroqClient:
    """
    Client for Groq API (OpenAI-compatible)
    Handles text generation with rate limiting and retry logic
    
    Architecture:
    - Uses OpenAI SDK with Groq base URL
    - Implements exponential backoff for rate limits
    - Supports multiple Llama models (fast/balanced/quality)
    """
    
    def __init__(self, model: str = None):
        """
        Initialize Groq client
        
        Args:
            model: Model to use (default from config)
        """
        self.model = model or GROQ_MODEL
        self.client: Optional[OpenAI] = None
        self._last_request_time = 0
        self._request_count = 0
        self._minute_start = time.time()
        
        self.initialize()
        
    def initialize(self):
        """Initialize the OpenAI client with Groq configuration"""
        if not GROQ_API_KEY:
            raise ValueError(
                "Groq API key not found. Set API_KEY in .env file.\n"
                "Get your key at: https://console.groq.com/keys"
            )
            
        self.client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
        )
        
        print(f"Groq client initialized (model: {self.model})")
        
    def switch_model(self, model_tier: str):
        """
        Switch to a different model tier
        
        Args:
            model_tier: 'fast', 'balanced', or 'quality'
        """
        if model_tier in GROQ_MODELS:
            self.model = GROQ_MODELS[model_tier]
            print(f"Switched to model: {self.model}")
        else:
            raise ValueError(f"Unknown model tier: {model_tier}. Use 'fast', 'balanced', or 'quality'")
    
    # =========================================================================
    # Rate Limiting
    # =========================================================================
    
    def _check_rate_limit(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self._minute_start >= 60:
            self._request_count = 0
            self._minute_start = current_time
            
        # Check if we're at the limit
        if self._request_count >= GROQ_RATE_LIMITS["requests_per_minute"]:
            wait_time = 60 - (current_time - self._minute_start)
            if wait_time > 0:
                print(f"Rate limit approaching, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self._request_count = 0
                self._minute_start = time.time()
                
        self._request_count += 1
        
    def _handle_rate_limit_error(self, attempt: int) -> float:
        """
        Calculate wait time for rate limit errors
        
        Args:
            attempt: Current retry attempt number
            
        Returns:
            Wait time in seconds
        """
        base_delay = GROQ_RATE_LIMITS["retry_delay"]
        multiplier = GROQ_RATE_LIMITS["backoff_multiplier"]
        return base_delay * (multiplier ** attempt)
    
    # =========================================================================
    # Generation Methods
    # =========================================================================
    
    def generate(
        self,
        query: str,
        context: str,
        role: str = "patient",
        max_retries: int = None,
    ) -> GroqResponse:
        """
        Generate a response using Groq API
        
        Args:
            query: User's question
            context: Retrieved context from RAG
            role: User role ('patient' or 'doctor')
            max_retries: Override default max retries
            
        Returns:
            GroqResponse with generated text and metadata
        """
        max_retries = max_retries or GROQ_RATE_LIMITS["max_retries"]
        
        # Build the prompt
        messages = self._build_messages(query, context, role)
        
        # Estimate tokens and potentially reduce context
        messages = self._ensure_context_fits(messages, query, context, role)
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                self._check_rate_limit()
                
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=GENERATION_CONFIG["temperature"],
                    top_p=GENERATION_CONFIG["top_p"],
                    max_tokens=GENERATION_CONFIG["max_tokens"],
                    frequency_penalty=GENERATION_CONFIG.get("frequency_penalty", 0),
                    presence_penalty=GENERATION_CONFIG.get("presence_penalty", 0),
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                return GroqResponse(
                    text=response.choices[0].message.content,
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    finish_reason=response.choices[0].finish_reason,
                    latency_ms=latency_ms,
                )
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check for rate limit errors
                if "rate" in error_str or "429" in str(e) or "limit" in error_str:
                    wait_time = self._handle_rate_limit_error(attempt)
                    print(f"Rate limit hit (attempt {attempt + 1}), waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                    
                # For other errors, still retry but with shorter delay
                print(f"Generation error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
        # All retries failed
        raise RuntimeError(
            f"Failed to generate response after {max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
    def _is_conversational(self, query: str) -> bool:
        """
        Detect if the query is conversational (greeting, thanks, etc.)
        rather than a medical question
        
        Args:
            query: User's message
            
        Returns:
            True if conversational, False if likely medical question
        """
        import re
        from config import CONVERSATIONAL_PATTERNS
        
        query_lower = query.lower().strip()
        
        # Check against conversational patterns
        for pattern in CONVERSATIONAL_PATTERNS:
            if re.match(pattern, query_lower, re.IGNORECASE):
                return True
                
        # Very short messages are often conversational
        if len(query_lower.split()) <= 2 and not any(
            term in query_lower for term in 
            ['nsmp', 'cáncer', 'cancer', 'tratamiento', 'quimio', 'radio', 
             'cirugía', 'pronóstico', 'supervivencia', 'riesgo', 'estadio']
        ):
            return True
            
        return False
    
    def _build_messages(
        self, 
        query: str, 
        context: str, 
        role: str,
        is_conversational: bool = None,
    ) -> List[Dict[str, str]]:
        """
        Build the messages array for the API call
        
        Args:
            query: User's question
            context: Retrieved context
            role: User role
            is_conversational: Override conversational detection
            
        Returns:
            List of message dicts for OpenAI API format
        """
        role_config = ROLE_CONFIGS.get(role, ROLE_CONFIGS["patient"])
        
        # Detect if conversational
        if is_conversational is None:
            is_conversational = self._is_conversational(query)
        
        # Build system prompt
        system_prompt = SYSTEM_PROMPT_BASE
        if "system_prompt_addon" in role_config:
            system_prompt += "\n" + role_config["system_prompt_addon"]
            
        # Add role-specific instructions
        if role == "patient":
            system_prompt += "\n\nRecuerda: estás hablando con un/a paciente. Sé claro, empático y reconfortante."
        elif role == "doctor":
            system_prompt += "\n\nRecuerda: estás hablando con un profesional médico. Usa terminología técnica apropiada."
        
        # Different message format based on query type
        if is_conversational:
            # For greetings and conversational messages, no need for strict context
            user_message = f"""{query}

(Nota: Tienes acceso a documentación sobre cáncer de endometrio NSMP si el usuario tiene preguntas médicas.)"""
        else:
            # For medical questions, include context but with more flexibility
            if context and context.strip():
                user_message = f"""CONTEXTO MÉDICO DISPONIBLE:
---
{context}
---

PREGUNTA DEL USUARIO:
{query}

INSTRUCCIONES:
1. Responde de forma útil y profesional
2. Usa el contexto proporcionado cuando sea relevante
3. Si el contexto no cubre completamente la pregunta, puedes:
   - Responder con la información disponible
   - Indicar qué aspectos no están cubiertos en los documentos
   - Ofrecer información general si es apropiado (indicando que es información general)
4. Para datos específicos (estadísticas, dosis, protocolos), cita siempre la fuente
5. Mantén un tono profesional pero accesible"""
            else:
                user_message = f"""{query}

(No se encontró contexto específico en los documentos. Responde con tu conocimiento general sobre cáncer de endometrio NSMP si es relevante, o indica amablemente que la pregunta está fuera de tu área de especialización.)"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    
    def _ensure_context_fits(
        self,
        messages: List[Dict[str, str]],
        query: str,
        context: str,
        role: str,
    ) -> List[Dict[str, str]]:
        """
        Ensure the context fits within token limits
        Uses adaptive k strategy to reduce context if needed
        
        Args:
            messages: Current message list
            query: Original query
            context: Full context
            role: User role
            
        Returns:
            Potentially truncated messages
        """
        # Rough token estimation (4 chars ≈ 1 token)
        total_chars = sum(len(m["content"]) for m in messages)
        estimated_tokens = total_chars // 4
        
        if estimated_tokens <= MAX_CONTEXT_TOKENS:
            return messages
            
        # Context too large, need to truncate
        print(f"Context too large (~{estimated_tokens} tokens), reducing...")
        
        # Split context into chunks and take first N
        context_chunks = context.split("\n\n---\n\n")
        
        # Reduce chunks until we fit
        while len(context_chunks) > 1:
            context_chunks = context_chunks[:-1]  # Remove last chunk
            reduced_context = "\n\n---\n\n".join(context_chunks)
            
            new_messages = self._build_messages(query, reduced_context, role)
            total_chars = sum(len(m["content"]) for m in new_messages)
            estimated_tokens = total_chars // 4
            
            if estimated_tokens <= MAX_CONTEXT_TOKENS:
                print(f"Reduced to {len(context_chunks)} chunks (~{estimated_tokens} tokens)")
                return new_messages
                
        return messages  # Return original if can't reduce enough
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def test_connection(self) -> bool:
        """
        Test the API connection with a simple query
        
        Returns:
            True if connection successful
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Responde solo con: OK"}],
                max_tokens=10,
            )
            return response.choices[0].message.content.strip().upper() == "OK"
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model configuration"""
        return {
            "current_model": self.model,
            "available_models": GROQ_MODELS,
            "generation_config": GENERATION_CONFIG,
            "rate_limits": GROQ_RATE_LIMITS,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
        }


# Singleton instance
_groq_client: Optional[GroqClient] = None


def get_groq_client() -> GroqClient:
    """Get or create singleton Groq client"""
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client


if __name__ == "__main__":
    # Test the client
    client = GroqClient()
    
    print("\n--- Testing Groq Connection ---")
    if client.test_connection():
        print("✓ Connection successful!")
    else:
        print("✗ Connection failed")
        
    print("\n--- Model Info ---")
    info = client.get_model_info()
    print(f"Model: {info['current_model']}")
    print(f"Available: {list(info['available_models'].keys())}")
