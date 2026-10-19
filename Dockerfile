# Imagen para desplegar la interfaz web (Streamlit) del asistente.
#   docker build -t asistente-tenerife .
#   docker run -p 8501:8501 --env-file .env asistente-tenerife
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# PyTorch en versión CPU (evita descargar ~2 GB de CUDA).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
RUN pip install -r requirements.txt

COPY data ./data
COPY app ./app
COPY .streamlit ./.streamlit

# Descarga el modelo de embeddings y construye el índice FAISS dentro de la imagen,
# para que el contenedor arranque rápido y sin depender de Hugging Face.
RUN python -c "from tenerife_assistant.config import Settings; \
from tenerife_assistant.assistant import build_default_retriever; \
build_default_retriever(Settings.from_env(allow_missing_key=True))"

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
