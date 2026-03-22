# ============================================
# VERSÃO ATUALIZADA DO DOCKERFILE
# ============================================
# Melhorias implementadas (Março 2026):
# - Python 3.12 (era 3.9)
# - Multi-stage build para otimizar tamanho
# - Usuário não-root por segurança
# - Health check automático
# - Imagem slim para reduzir footprint
# - Variáveis de ambiente otimizadas
# - Labels descritivos adicionados
# - Suporte a MLflow e TensorFlow
# ============================================

# Stage 1: Builder
FROM python:3.12-slim as builder

WORKDIR /tmp

# Instalar dependências de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar requirements
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

# Metadados
LABEL maintainer="Renan Santos Mendes"
LABEL description="MLOps API para predição de saúde fetal com TensorFlow/Keras"
LABEL version="1.0"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/appuser/.local/bin:$PATH

WORKDIR /app

# Criar usuário não-root por segurança
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copiar pacotes Python do builder
COPY --from=builder /root/.local /home/appuser/.local

# Copiar arquivos da aplicação
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser train.py .
COPY --chown=appuser:appuser requirements.txt .

# Trocar para usuário não-root
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/', timeout=5)" || exit 1

# Expor porta
EXPOSE 8000

# Executar aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]