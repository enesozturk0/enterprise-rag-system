# Enterprise RAG System

A production-oriented RAG system that indexes PDF documents and answers
questions with source citations.

The system consists of:

- A FastAPI backend
- A Streamlit user interface
- Ollama for LLM and embedding inference
- ChromaDB for vector storage
- A RAG evaluation suite and quality gate

## Requirements

- Docker Desktop or Docker Engine
- Docker Compose v2
- At least 8 GB of RAM
- At least 10 GB of free disk space for model files

The first startup automatically downloads the `llama3.1:8b` and `bge-m3`
models, which require approximately 6.1 GB in total.

## Quick Start

### 1. Create the environment file

PowerShell:

```powershell
Copy-Item .env.production.example .env
```

Linux or macOS:

```bash
cp .env.production.example .env
```

Replace `API_KEY` and `RAG_API_KEY` in `.env` with the same strong random
value.

Generate a secure key with PowerShell:

```powershell
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
[Convert]::ToBase64String($bytes)
```

Assign the generated value to both variables:

```env
API_KEY=GENERATED_SECRET
RAG_API_KEY=GENERATED_SECRET
```

Never commit the `.env` file.

### 2. Start the services

```powershell
docker compose up --detach --build
```

Follow the initial model download:

```powershell
docker compose logs --follow ollama-models
```

The initialization service prints the following message when the models are
ready:

```text
Required Ollama models are ready.
```

### 3. Verify the services

```powershell
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8501/_stcore/health
```

Expected service states:

- `ollama`: healthy
- `ollama-models`: exited (0)
- `api`: healthy
- `ui`: healthy

`ollama-models` is a one-shot initialization service. An `Exited (0)` status
after the models have been downloaded is expected.

## Service Endpoints

- Streamlit UI: http://127.0.0.1:8501
- FastAPI documentation: http://127.0.0.1:8000/docs
- API health check: http://127.0.0.1:8000/health
- Prometheus metrics: http://127.0.0.1:8000/metrics

The Ollama port is intentionally not published to the host. It is accessible
only by the API service through the internal Compose network.

## First Use

1. Open the Streamlit interface.
2. Upload a PDF from the sidebar.
3. Ask a question about the document after indexing is complete.
4. Review the source, page, and score details below the answer.

## Common Docker Commands

```powershell
# Show service status
docker compose ps

# Follow all logs
docker compose logs --follow

# Follow API logs
docker compose logs --follow api

# Restart the services
docker compose restart

# Stop services without deleting persistent data
docker compose down

# Rebuild the application images
docker compose build api ui
```

Named volumes preserve Chroma data and Ollama models after
`docker compose down`.

## Developer Commands

Windows:

```powershell
.\scripts\dev.ps1 help
.\scripts\dev.ps1 quality
.\scripts\dev.ps1 docker-config
```

Systems with GNU Make:

```bash
make help
make quality
make docker-config
```

Integration tests require running Ollama and Chroma services:

```powershell
.\scripts\dev.ps1 test-all
```

## Production Notes

The API and UI bind to `127.0.0.1` by default. For remote access, expose them
through a TLS-enabled reverse proxy.

In production:

- Never commit the `.env` file.
- Use a strong and unique API key.
- Change `BIND_ADDRESS` only when required by the network architecture.
- Back up the Chroma and Ollama volumes regularly.
