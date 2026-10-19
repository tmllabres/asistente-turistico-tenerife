"""Asistente turístico de Tenerife basado en un LLM comercial.

Tiene tres partes:

- RAG sobre la guía turística en PDF (ingest.py y retrieval.py).
- Diálogo multiturno con memoria limitada por tokens (memory.py).
- Llamadas a funciones externas como get_weather (tools.py).

La clase que lo une todo es TourismAssistant, en assistant.py.
"""

__version__ = "1.0.0"
