FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/service
WORKDIR /workspace
COPY service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY service /workspace/service
COPY artifacts /workspace/artifacts
CMD ["uvicorn", "app.main:app", "--app-dir", "/workspace/service", "--host", "0.0.0.0", "--port", "8000"]
