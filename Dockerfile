FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY docs ./docs

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir .[ui]

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3).read(1)"

CMD ["auralock", "webui", "--host", "0.0.0.0", "--port", "7860"]
