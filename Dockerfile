# --- Build stage: build the wheel in isolation -------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir build && python -m build --wheel

# --- Runtime stage: minimal image with a non-root user -----------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EVALPIPE_DB=/data/evalpipe.db

RUN useradd --create-home --shell /usr/sbin/nologin evalpipe \
    && mkdir -p /data && chown evalpipe:evalpipe /data

COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

COPY --chown=evalpipe:evalpipe examples /home/evalpipe/examples

USER evalpipe
WORKDIR /home/evalpipe
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["evalpipe", "serve", "--host", "0.0.0.0", "--port", "8000"]
