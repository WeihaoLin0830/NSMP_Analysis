# NSMP Cancer RAG System

Sistema de Recuperación Aumentada por Generación (RAG) para asistencia en cáncer de endometrio con perfil molecular NSMP.

## Estructura del Proyecto

```
rag_system/
├── config.py              # Configuración del sistema
├── document_processor.py  # Extracción y segmentación de PDFs
├── embeddings.py          # Generación de embeddings y índice FAISS
├── generator.py           # Generación de respuestas
├── question_generator.py  # Generación de preguntas sugeridas
├── rag_pipeline.py        # Pipeline principal RAG
├── api.py                 # API FastAPI
├── initialize.py          # Script de inicialización
├── requirements.txt       # Dependencias Python
└── data/                  # Datos generados (índice, embeddings, etc.)
```

## Requisitos

- Python 3.9+
- Node.js 16+
- 4GB+ RAM (para modelos de embeddings)

## Instalación

### 1. Instalar dependencias Python

```bash
cd rag_system
pip install -r requirements.txt
```

### 2. Inicializar el sistema RAG

Este paso procesa los PDFs, genera embeddings y construye el índice:

```bash
python initialize.py
```

Esto creará:
- `data/chunks_metadata.json` - Metadatos de fragmentos
- `data/faiss_index.bin` - Índice vectorial
- `data/embeddings.npy` - Embeddings generados
- `data/preguntas_sugeridas.txt` - Preguntas sugeridas

### 3. Iniciar la API RAG

```bash
python api.py
```

La API estará disponible en `http://localhost:8000`

### 4. Iniciar el servidor web (en otra terminal)

```bash
cd web
npm install
node server.js
```

La aplicación estará en `http://localhost:3000`

## Uso

### API Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/chat` | POST | Enviar mensaje al chatbot |
| `/suggestions/{role}` | GET | Obtener preguntas sugeridas |
| `/index/status` | GET | Estado del índice |
| `/index/rebuild` | POST | Reconstruir índice |
| `/health` | GET | Estado del sistema |

### Ejemplo de uso de la API

```python
import requests

response = requests.post('http://localhost:8000/chat', json={
    'message': '¿Qué es el perfil molecular NSMP?',
    'role': 'patient',
    'session_id': 'test123'
})

print(response.json())
```

## Pipeline RAG

1. **Preprocesamiento**: Los PDFs se procesan con `pdfplumber` extrayendo texto y tablas
2. **Segmentación**: El texto se divide en chunks de ~500 palabras con overlap
3. **Indexación**: Se generan embeddings con `sentence-transformers` y se indexan con FAISS
4. **Recuperación**: Las consultas se vectorizan y se buscan los chunks más similares
5. **Generación**: Se construye un contexto y se genera una respuesta adaptada al rol

## Configuración

Editar `config.py` para ajustar:

- `CHUNK_SIZE`: Palabras por fragmento (default: 500)
- `TOP_K_RESULTS`: Documentos a recuperar (default: 5)
- `EMBEDDING_MODEL`: Modelo de embeddings
- `API_PORT`: Puerto de la API (default: 8000)

## Documentos Soportados

El sistema procesa automáticamente todos los PDFs en la carpeta `Articles/`:

- Early endometrial cancer recurrence risk prediction model.pdf
- EJC.pdf
- ESGO ESTRO ENDOMETRIAL CANCER 2021.pdf
- ESGO ESTRO ENDOMETRIAL GUIDELINES 2025.pdf
- Oncoguía cancer endometrio 2023.pdf

## Notas

- El sistema usa un generador simplificado por defecto (sin LLM pesado)
- Para usar GPT-Neo/GPT-J, cambiar `use_llm=True` en la inicialización
- El chatbox adapta las respuestas según el rol (paciente/médico)
