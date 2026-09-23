FROM python:3.11-slim

WORKDIR /code

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, source code, and the fine-tuned model
COPY app/ ./app/
COPY src/ ./src/
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
