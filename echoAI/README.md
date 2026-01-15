
# Echo Mermaid Platform – Interface-First, Microservices-Ready (FastAPI)

Implements the architecture from the provided Mermaid class diagram with **interfaces**, **adapters**, and **APIs**.
Runs locally via **Uvicorn** (no Docker) and can be split into microservices for Kubernetes later.

## Quick Start (Monolith via Gateway)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn apps.gateway.main:app --reload --port 8000
```

Open http://localhost:8000/docs.

## Microservices Mode
Run each app on its own port (examples below).

## Notes
- External systems (Kafka, Memcached, Azure AD, Key Vault, OTel) are provided as **adapter classes** with local dev stubs.
- Swap implementations by rebinding providers in each `container.py`.
