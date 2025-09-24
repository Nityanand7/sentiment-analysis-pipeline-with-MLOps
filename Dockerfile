FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY flask_app/ ./flask_app/
# If your model/vectorizer are needed at runtime, include them:
# COPY lgbm_model.pkl tfidf_vectorizer.pkl ./

EXPOSE 5002
CMD ["python", "flask_app/app.py"]