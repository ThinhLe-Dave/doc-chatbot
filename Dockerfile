FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 --prefer-binary -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY . .

RUN chmod +x docker-entrypoint.sh

EXPOSE 7860

RUN mkdir -p database

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "web_frontend.fastapi_app:app", "--host", "0.0.0.0", "--port", "7860"]
