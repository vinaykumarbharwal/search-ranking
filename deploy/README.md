# Deployment Structure

This folder provides production deployment options for RankSmart.

## Tree

```text
deploy/
  env/
    .env.prod.example
  systemd/
    ranksmart.service
```

## Quick Start (Without Docker)

1. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train model artifacts if needed:

```powershell
python src\train_native.py
```

3. Run the API:

```powershell
uvicorn api.main_native:app --host 0.0.0.0 --port 8000
```

4. Verify health:

```powershell
curl http://localhost:8000/health
```

## Notes

- If model files are missing and `STRICT_MODEL_LOADING=true`, startup fails.
- Use `deploy/systemd/ranksmart.service` on Linux VMs for process supervision.
