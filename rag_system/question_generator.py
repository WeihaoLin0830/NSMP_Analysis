"""
Question Generator Module
Generates suggested questions based on document content
"""
import json
import re
from typing import List, Dict, Set
from collections import Counter
from pathlib import Path

from config import (
    SUGGESTED_QUESTIONS_PATH,
    CHUNKS_METADATA_PATH,
    DATA_DIR
)
from document_processor import DocumentChunk, DocumentProcessor


class QuestionGenerator:
    """Generates suggested questions from document content"""
    
    # Medical topic patterns for question generation
    TOPIC_PATTERNS = {
        "tratamiento": [
            "¿Cuál es el tratamiento recomendado para {topic}?",
            "¿Qué opciones de tratamiento existen para {topic}?",
            "¿Cuándo está indicado el tratamiento con {topic}?",
        ],
        "diagnóstico": [
            "¿Cómo se diagnostica {topic}?",
            "¿Qué pruebas se utilizan para detectar {topic}?",
            "¿Cuáles son los criterios diagnósticos para {topic}?",
        ],
        "pronóstico": [
            "¿Cuál es el pronóstico para pacientes con {topic}?",
            "¿Qué factores afectan la supervivencia en {topic}?",
            "¿Cuáles son las tasas de recurrencia en {topic}?",
        ],
        "riesgo": [
            "¿Cuáles son los factores de riesgo para {topic}?",
            "¿Cómo se clasifican los grupos de riesgo en {topic}?",
            "¿Qué aumenta el riesgo de {topic}?",
        ],
        "síntomas": [
            "¿Cuáles son los síntomas de {topic}?",
            "¿Cómo se presenta clínicamente {topic}?",
            "¿Qué signos indican {topic}?",
        ],
        "seguimiento": [
            "¿Cuál es el seguimiento recomendado para {topic}?",
            "¿Con qué frecuencia deben hacerse controles en {topic}?",
            "¿Qué pruebas de seguimiento se recomiendan para {topic}?",
        ]
    }
    
    # Key medical terms to look for
    MEDICAL_TERMS = [
        "NSMP", "cáncer de endometrio", "carcinoma endometrial",
        "histerectomía", "radioterapia", "quimioterapia", "braquiterapia",
        "estadio", "grado", "invasión miometrial", "ganglios linfáticos",
        "supervivencia", "recurrencia", "remisión",
        "perfil molecular", "p53", "MMRd", "POLE",
        "bajo riesgo", "riesgo intermedio", "alto riesgo",
        "endometrioide", "seroso", "células claras",
        "CA-125", "PET-CT", "resonancia magnética",
        "linfadenectomía", "omentectomía", "citorreducción"
    ]
    
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.extracted_topics: Set[str] = set()
        self.questions: List[Dict] = []
        
    def load_chunks(self):
        """Load processed document chunks"""
        processor = DocumentProcessor()
        self.chunks = processor.load_chunks_metadata()
        print(f"Loaded {len(self.chunks)} chunks")
        
    def extract_topics(self) -> Set[str]:
        """Extract key topics from document content"""
        all_text = " ".join([chunk.content for chunk in self.chunks])
        
        # Find medical terms in content
        found_terms = set()
        for term in self.MEDICAL_TERMS:
            if term.lower() in all_text.lower():
                found_terms.add(term)
                
        # Extract capitalized phrases (potential medical terms)
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(cap_pattern, all_text)
        
        # Filter and add relevant matches
        term_counts = Counter(matches)
        for term, count in term_counts.most_common(50):
            if count >= 3 and len(term) > 3:
                found_terms.add(term)
                
        self.extracted_topics = found_terms
        print(f"Extracted {len(found_terms)} topics")
        
        return found_terms
    
    def generate_questions(self) -> List[Dict]:
        """Generate questions based on extracted topics and patterns"""
        questions = []
        question_id = 1
        
        # General questions about endometrial cancer
        general_questions = [
            {
                "id": question_id,
                "category": "general",
                "question": "¿Qué es el cáncer de endometrio?",
                "target_audience": "both"
            },
            {
                "id": question_id + 1,
                "category": "general", 
                "question": "¿Qué significa tener un perfil molecular NSMP?",
                "target_audience": "both"
            },
            {
                "id": question_id + 2,
                "category": "general",
                "question": "¿Cuáles son los diferentes tipos de cáncer de endometrio?",
                "target_audience": "both"
            },
            {
                "id": question_id + 3,
                "category": "general",
                "question": "¿Cómo se clasifica el cáncer de endometrio por estadios?",
                "target_audience": "both"
            },
        ]
        questions.extend(general_questions)
        question_id += len(general_questions)
        
        # Patient-specific questions
        patient_questions = [
            {
                "id": question_id,
                "category": "paciente",
                "question": "¿Qué puedo esperar después de una histerectomía?",
                "target_audience": "patient"
            },
            {
                "id": question_id + 1,
                "category": "paciente",
                "question": "¿Cuáles son los efectos secundarios del tratamiento?",
                "target_audience": "patient"
            },
            {
                "id": question_id + 2,
                "category": "paciente",
                "question": "¿Con qué frecuencia necesito hacerme revisiones?",
                "target_audience": "patient"
            },
            {
                "id": question_id + 3,
                "category": "paciente",
                "question": "¿Qué síntomas debo vigilar después del tratamiento?",
                "target_audience": "patient"
            },
            {
                "id": question_id + 4,
                "category": "paciente",
                "question": "¿Cuál es mi pronóstico con perfil NSMP?",
                "target_audience": "patient"
            },
        ]
        questions.extend(patient_questions)
        question_id += len(patient_questions)
        
        # Doctor-specific questions
        doctor_questions = [
            {
                "id": question_id,
                "category": "médico",
                "question": "¿Cuáles son las guías ESGO-ESTRO actuales para cáncer de endometrio?",
                "target_audience": "doctor"
            },
            {
                "id": question_id + 1,
                "category": "médico",
                "question": "¿Cuándo está indicada la linfadenectomía en pacientes NSMP?",
                "target_audience": "doctor"
            },
            {
                "id": question_id + 2,
                "category": "médico",
                "question": "¿Cuál es el rol de la radioterapia adyuvante en riesgo intermedio?",
                "target_audience": "doctor"
            },
            {
                "id": question_id + 3,
                "category": "médico",
                "question": "¿Cómo se define la invasión miometrial profunda?",
                "target_audience": "doctor"
            },
            {
                "id": question_id + 4,
                "category": "médico",
                "question": "¿Qué evidencia existe sobre quimioterapia en NSMP de alto riesgo?",
                "target_audience": "doctor"
            },
            {
                "id": question_id + 5,
                "category": "médico",
                "question": "¿Cuáles son los factores pronósticos en carcinoma endometrioide?",
                "target_audience": "doctor"
            },
        ]
        questions.extend(doctor_questions)
        question_id += len(doctor_questions)
        
        # Generate topic-based questions
        for topic in list(self.extracted_topics)[:20]:
            for category, patterns in self.TOPIC_PATTERNS.items():
                question_text = patterns[0].format(topic=topic)
                questions.append({
                    "id": question_id,
                    "category": category,
                    "question": question_text,
                    "target_audience": "both",
                    "related_topic": topic
                })
                question_id += 1
                
        self.questions = questions
        print(f"Generated {len(questions)} questions")
        
        return questions
    
    def save_questions(self, filepath: Path = SUGGESTED_QUESTIONS_PATH):
        """Save generated questions to file"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PREGUNTAS SUGERIDAS - Sistema RAG Cáncer de Endometrio NSMP\n")
            f.write("=" * 80 + "\n\n")
            f.write("Estas preguntas han sido generadas automáticamente basándose en el\n")
            f.write("contenido de los documentos médicos indexados en el sistema.\n\n")
            
            # Group by category
            categories = {}
            for q in self.questions:
                cat = q["category"]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(q)
                
            for category, cat_questions in categories.items():
                f.write("-" * 60 + "\n")
                f.write(f"CATEGORÍA: {category.upper()}\n")
                f.write("-" * 60 + "\n\n")
                
                for q in cat_questions:
                    audience = q.get("target_audience", "both")
                    audience_tag = ""
                    if audience == "patient":
                        audience_tag = " [Para pacientes]"
                    elif audience == "doctor":
                        audience_tag = " [Para profesionales]"
                        
                    f.write(f"• {q['question']}{audience_tag}\n")
                    
                f.write("\n")
                
            f.write("=" * 80 + "\n")
            f.write(f"Total de preguntas generadas: {len(self.questions)}\n")
            f.write("=" * 80 + "\n")
            
        print(f"Saved questions to {filepath}")
        
        # Also save as JSON for API use
        json_path = DATA_DIR / "suggested_questions.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.questions, f, ensure_ascii=False, indent=2)
        print(f"Saved questions JSON to {json_path}")
        
    def get_questions_for_role(self, role: str) -> List[Dict]:
        """Get questions appropriate for a specific role"""
        if role == "patient":
            return [q for q in self.questions 
                    if q["target_audience"] in ["patient", "both"]]
        elif role == "doctor":
            return [q for q in self.questions 
                    if q["target_audience"] in ["doctor", "both"]]
        return self.questions


if __name__ == "__main__":
    generator = QuestionGenerator()
    generator.load_chunks()
    generator.extract_topics()
    questions = generator.generate_questions()
    generator.save_questions()
    
    print("\n--- Sample Questions ---")
    for q in questions[:10]:
        print(f"[{q['category']}] {q['question']}")
