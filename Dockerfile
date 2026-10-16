# Use a small official Python image to keep the runtime image lean.
FROM python:3.12-slim

# All following paths are relative to this folder inside the container.
WORKDIR /app

# Avoid writing .pyc files and send Python logs directly to container output.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Copy dependency metadata and source before installing the project.
COPY pyproject.toml ./
COPY src ./src
COPY artifacts ./artifacts

# Install the app and its declared runtime dependencies into the image.
RUN pip install --no-cache-dir .

# Document the port the API listens on. Publishing it to the host is configured
# separately in docker-compose.yml or by the deployment platform.
EXPOSE 8000

# Start the ASGI server and import the FastAPI app from src/api.py.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
