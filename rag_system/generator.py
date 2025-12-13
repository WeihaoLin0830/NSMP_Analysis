"""
Text Generation Module
Handles response generation using language models
"""
import torch
from typing import Optional, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    set_seed
)

from config import (
    GENERATION_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_K,
    TOP_P
)


class ResponseGenerator:
    """Generates responses using a language model"""
    
    def __init__(self, model_name: str = GENERATION_MODEL):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.generator = None
        self._is_loaded = False
        
    def load_model(self):
        """Load the language model"""
        if self._is_loaded:
            return
            
        print(f"Loading generation model: {self.model_name}")
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with appropriate settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
            
        # Create pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1
        )
        
        self._is_loaded = True
        print("Model loaded successfully")
        
    def generate_response(
        self,
        query: str,
        context: str,
        role: str = "patient",
        max_tokens: int = MAX_NEW_TOKENS
    ) -> str:
        """
        Generate a response based on query and context
        
        Args:
            query: User's question
            context: Retrieved context from documents
            role: User role ('patient' or 'doctor')
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        if not self._is_loaded:
            self.load_model()
            
        # Create prompt based on role
        prompt = self._create_prompt(query, context, role)
        
        # Generate response
        set_seed(42)
        outputs = self.generator(
            prompt,
            max_new_tokens=max_tokens,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        response = outputs[0]["generated_text"]
        
        # Clean up response
        response = self._clean_response(response)
        
        return response
    
    def _create_prompt(self, query: str, context: str, role: str) -> str:
        """Create appropriate prompt based on user role"""
        
        if role == "doctor":
            system_instruction = """Eres un asistente médico especializado en cáncer de endometrio y el perfil molecular NSMP.
Proporciona información técnica y detallada basada en las guías clínicas y evidencia científica.
Incluye referencias a los estudios cuando sea relevante.
Responde de manera profesional y precisa."""
        else:
            system_instruction = """Eres un asistente amable que ayuda a pacientes a entender información sobre cáncer de endometrio.
Explica los conceptos de forma sencilla y comprensible.
Evita términos muy técnicos o explícalos cuando los uses.
Sé empático y tranquilizador, pero siempre recomienda consultar con su médico para decisiones específicas."""

        prompt = f"""{system_instruction}

Información relevante de documentos médicos:
{context}

Pregunta del usuario: {query}

Respuesta:"""

        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean up the generated response"""
        # Remove any repeated content
        lines = response.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and line not in unique_lines:
                unique_lines.append(line)
                
        response = '\n'.join(unique_lines)
        
        # Truncate at common end markers
        end_markers = ['Pregunta:', 'Usuario:', 'Human:', '###']
        for marker in end_markers:
            if marker in response:
                response = response.split(marker)[0]
                
        return response.strip()


class SimpleResponseGenerator:
    """
    Simplified response generator that uses context directly without heavy LLM
    Good for resource-constrained environments
    """
    
    def generate_response(
        self,
        query: str,
        context: str,
        role: str = "patient"
    ) -> str:
        """
        Generate a response by formatting the retrieved context
        
        This is a fallback for when GPU/heavy models are not available
        """
        if not context:
            return "Lo siento, no encontré información relevante en los documentos disponibles. Por favor, reformula tu pregunta o consulta con un profesional médico."
        
        if role == "doctor":
            intro = "Basándome en la documentación científica disponible, he encontrado la siguiente información relevante:\n\n"
        else:
            intro = "He encontrado información que puede ayudarte a entender mejor tu consulta:\n\n"
            
        # Format the context nicely
        formatted_context = context.replace("---", "\n")
        
        conclusion = "\n\n⚠️ Nota: Esta información proviene de documentos médicos de referencia. "
        if role == "patient":
            conclusion += "Por favor, consulta siempre con tu médico antes de tomar cualquier decisión sobre tu tratamiento."
        else:
            conclusion += "Se recomienda verificar con las guías clínicas más actualizadas."
            
        return intro + formatted_context + conclusion


# Factory function to get appropriate generator
def get_response_generator(use_llm: bool = False) -> ResponseGenerator:
    """
    Get appropriate response generator based on configuration
    
    Args:
        use_llm: Whether to use full LLM generation
        
    Returns:
        Response generator instance
    """
    if use_llm:
        return ResponseGenerator()
    else:
        return SimpleResponseGenerator()


if __name__ == "__main__":
    # Test the generator
    generator = SimpleResponseGenerator()
    
    test_context = """[Fuente: ESGO ESTRO ENDOMETRIAL CANCER 2021, Página 5]
    El grupo NSMP (No Specific Molecular Profile) representa aproximadamente el 30% de los casos de cáncer de endometrio.
    
    [Fuente: Oncoguía cancer endometrio 2023, Página 12]
    El tratamiento estándar incluye histerectomía total con salpingo-ooforectomía bilateral."""
    
    response = generator.generate_response(
        query="¿Qué es el perfil NSMP?",
        context=test_context,
        role="patient"
    )
    
    print("Generated Response:")
    print(response)
