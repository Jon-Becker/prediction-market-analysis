# Mock service image: intentionally independent of the repository's
# dependency lock so production dependencies never leak into the mock.
FROM python:3.9-slim

WORKDIR /srv

RUN pip install --no-cache-dir "fastapi>=0.110,<1" "uvicorn>=0.30,<1" "websockets>=12,<16"

COPY app.py fixtures.py ./

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "warning"]
