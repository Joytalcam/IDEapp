# ============================================================
# Imagen base: Python 3.12 versión ligera (Debian slim)
# ============================================================
FROM python:3.12-slim

# ============================================================
# Variables de entorno para Python
# - No escribir archivos .pyc en el contenedor
# - Salida sin buffer (logs en tiempo real)
# ============================================================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ============================================================
# Directorio de trabajo dentro del contenedor
# ============================================================
WORKDIR /app

# ============================================================
# Copiar SOLO requirements.txt primero
# Esto aprovecha la caché de Docker: si no cambian las
# dependencias, no se reinstalan en cada build
# ============================================================
COPY requirements.txt .

# ============================================================
# Instalar dependencias (versiones compatibles con Python 3.12)
# - --no-cache-dir: no guarda caché de pip (imagen más pequeña)
# ============================================================
RUN pip install --no-cache-dir \
    Flask==3.0.3 \
    Werkzeug==3.0.4 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \ 
    joblib==1.4.2

# ============================================================
# Copiar el resto del código del proyecto
# (.dockerignore excluye .venv, __pycache__, etc.)
# ============================================================
COPY . .

# ============================================================
# Crear carpeta para el volumen de datos (persistencia)
# ============================================================
RUN mkdir -p /app/data

# ============================================================
# Exponer el puerto 5000 (informativo)
# ============================================================
EXPOSE 5000

# ============================================================
# Variables de entorno para la app
# ============================================================
ENV FLASK_DEBUG=false \
    DATA_DIR=/app/data

# ============================================================
# Comando de arranque
# ============================================================
CMD ["python", "app.py"]
