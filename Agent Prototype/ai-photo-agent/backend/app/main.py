"""
AI Photo Selection Agent — FastAPI Application
=============================================

Access points after `docker-compose up`:
  Swagger UI  →  http://localhost:8000/api/docs
  ReDoc       →  http://localhost:8000/api/redoc
  OpenAPI     →  http://localhost:8000/openapi.json
  Metrics     →  http://localhost:8000/metrics
  Flower      →  http://localhost:5555
"""
from contextlib import asynccontextmanager
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.utils import get_openapi
from prometheus_client import make_asgi_app

from app.core.config import settings
from app.core.database import engine, Base
from app.core.logging import configure_logging
from app.api.v1 import jobs, images, feedback, health, policies, upload

log = structlog.get_logger()

# ─── OpenAPI Metadata ─────────────────────────────────────────────────────

DESCRIPTION = """
## AI Photo Selection Agent API

Enterprise-grade system for **automated photo quality evaluation and classification** using
a hybrid of rule-based filters and multimodal AI models.

---

### 🔄 Typical Workflow

1. **POST** `/api/v1/jobs/` — Submit a Google Drive folder URL → returns a `job_id`
2. **GET** `/api/v1/jobs/{job_id}` — Poll progress (`progress_pct`, `status`)
3. **GET** `/api/v1/images/?job_id=...` — Browse evaluated images with scores and decisions
4. **POST** `/api/v1/images/{image_id}/override` — Human override for any decision
5. **GET** `/api/v1/jobs/{job_id}/report` — Download structured JSON report
6. **GET** `/api/v1/feedback/threshold-suggestion` — AI-driven threshold tuning

---

### 🧠 AI Pipeline Summary

| Layer | Model | Purpose |
|---|---|---|
| Preprocessing | OpenCV | Blur, brightness, resolution, aspect ratio |
| Semantic | CLIP ViT-B/32 | Professional photo scoring via text-image similarity |
| Detection | YOLOv8n | People count, phone detection, object policy |
| Safety | NudeNet | NSFW / inappropriate content detection |
| **Policy Engine** | **Rule-based + CLIP** | **Company-specific hard/soft constraints** |

---

### 📊 Scoring System

```
Final Score = CLIP(0.35) + Quality(0.25) + Resolution(0.15) + Object(0.15) + Aesthetic(0.10)

≥ 0.75  →  ✅ Approved
0.60–0.74 →  🔍 Review (optional human validation)
< 0.60  →  ❌ Rejected
```

Hard rules (immediate rejection regardless of score):
- Blurry image (Laplacian variance < threshold)
- Extreme brightness (too dark or overexposed)
- Low resolution (< 400×400 px)
- NSFW content detected
- Phone detected (if policy enabled)

---

### 🔐 Authentication
Currently open. For production, add an API key via `X-API-Key` header (configurable).

---

### 📦 Object Storage
Processed images are stored in MinIO (S3-compatible):
- `approved-images/` bucket → approved photos
- `rejected-images/` bucket → rejected photos
- `reports/` bucket → JSON evaluation reports
"""

TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Liveness and readiness probes for Kubernetes / container orchestration.",
    },
    {
        "name": "Jobs",
        "description": """
Manage batch photo processing jobs.

Each job corresponds to a **Google Drive folder**. Upon submission, the system:
1. Extracts the folder ID and validates access
2. Fetches all supported image files (JPEG, PNG, WEBP, BMP, GIF, TIFF)
3. Queues parallel Celery tasks for preprocessing + AI inference
4. Persists results and generates a downloadable JSON report
        """,
    },
    {
        "name": "Images",
        "description": """
Browse, filter, and inspect individual image evaluation results.

Supports:
- Filtering by job, decision (`approved` / `rejected` / `review`), and score range
- Full score breakdown per image (CLIP, quality, resolution, object compliance, aesthetic)
- YOLO detection details (people count, phone/laptop detected)
- Human decision override with reason tracking (feeds the feedback loop)
        """,
    },
    {
        "name": "Feedback",
        "description": """
Human feedback and threshold tuning.

Every override recorded via `/images/{id}/override` is stored here.
The `/feedback/threshold-suggestion` endpoint analyses override patterns to 
recommend adjusted approval/review thresholds for future runs.
        """,
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    configure_logging()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("photo_agent.startup", version=settings.APP_VERSION)
    yield
    await engine.dispose()
    log.info("photo_agent.shutdown")


app = FastAPI(
    title="AI Photo Selection Agent",
    description=DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/openapi.json",
    openapi_tags=TAGS_METADATA,
    contact={
        "name": "Platform Team",
        "email": "platform@yourcompany.com",
    },
    license_info={
        "name": "Proprietary — Internal Use Only",
    },
    lifespan=lifespan,
)

# ─── Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ─── Prometheus metrics endpoint ──────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ─── Routers ──────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(upload.router, prefix="/api/v1/upload", tags=["Upload (Test)"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])
app.include_router(images.router, prefix="/api/v1/images", tags=["Images"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["Feedback"])
app.include_router(policies.router, prefix="/api/v1/policies", tags=["Policies"])
