# Runs Citra's TEST SUITE, not Citra. The assistant itself is Windows-only
# (pycaw, pywin32, a real microphone and real relay boards); every test
# fakes those boundaries, so the suite runs anywhere - including here.
#
#   docker build -t citra-tests .
#   docker run --rm citra-tests
FROM python:3.12-slim

WORKDIR /app
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .
CMD ["sh", "-c", "ruff check . && pytest -q"]
