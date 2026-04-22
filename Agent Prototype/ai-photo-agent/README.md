# 📸 AI Photo Selection Agent

A scalable, enterprise-grade system for automated photo quality evaluation, safety filtering, and company policy enforcement.

> [!IMPORTANT]  
> **Testing Priority:** For quick evaluation without setting up Google Cloud, use the **`Direct Upload`** endpoints (`/api/v1/upload/evaluate-batch`) and the **`Policy Engine`** endpoints. The Google Drive functionality is fully implemented but requires the service account JSON to be placed in `backend/credentials/` and mounted into the container at `/app/credentials/`.

---

## 🚀 Two Running Modes

This system is optimized for both rapid local development and high-performance production.

### 1. Development Mode (CPU-only)
**Use this for coding, debugging, and testing individual photos.** It uses a lightweight `python:slim` base and skips the massive 3GB+ CUDA libraries.
```bash
# Start the lightweight dev stack
docker compose -f docker-compose.dev.yml up -d --build
```

### 2. Production Mode (GPU-accelerated)
**Use this for processing thousands of photos.** Requires an NVIDIA GPU and takes longer to build, but runs inference ~10x faster.
```bash
# Start the full-speed production stack
docker compose up -d --build
```

---

## 🧠 The AI Pipeline
Every photo passes through a 4-stage evaluation pipeline:

1.  **Preprocessing (CPU):** Validates raw quality (Blur variance, Brightness, Resolution, Aspect Ratio). Rejects low-quality images early to save cost.
2.  **Object Detection (YOLOv8s):** Detects people, counts them, and identifies prohibited items (like mobile phones) or required equipment (laptops/office settings).
3.  **Semantic Evaluation (CLIP ViT-L/14@336px):** Understands the "vibe" of the photo. Checks if it matches "Professional corporate portraits" or "Casual selfies."
4.  **Safety Check (NudeNet):** Ensures 100% compliance with NSFW/Safety standards.

---

## 🏢 Company Policy Engine
The system supports strict enforcement of configurable company rules.
*   **Hard Constraints:** e.g., "Maximum 3 people," "No mobile phones," "Minimum 1080p resolution."
*   **Soft Scoring:** Give bonuses for "Office settings" or "Professional attire" detected via semantic scoring.
*   **Explainability:** Every rejection includes human-readable reasons (e.g., *"Rejected: No people detected in frame"*).

---

## 🛠️ Quick Access Links

| Service | Address | Description |
| :--- | :--- | :--- |
| **Backend API** | [http://localhost:8000/api/docs](http://localhost:8000/api/docs) | **Interactive Swagger UI** — Test uploads here! |
| **Worker Monitor** | [http://localhost:5555](http://localhost:5555) | **Flower** — See AI tasks processing in real-time. |
| **Object Storage** | [http://localhost:9001](http://localhost:9001) | **MinIO Console** — View the Approved/Rejected image buckets. |
| **Database** | `localhost:5432` | **PostgreSQL** — Stores metadata and scores. |

---

## 📥 Testing without Google Drive
I have integrated a **Direct Upload API** specifically for testing:
1.  Go to the [Swagger Docs](http://localhost:8000/api/docs#/Upload%20(Test)/evaluate_image_api_v1_upload_evaluate_post).
2.  Use the `POST /api/v1/upload/evaluate` endpoint.
3.  Upload any image from your computer.
4.  The system will return the **Score**, **Decision**, and **Detailed Breakdown** instantly.

---

## ⚙️ Configuration
Configuration is managed via the `.env` file. Key settings include:
- `SCORE_APPROVED_THRESHOLD`: (Default: 0.75)
- `MIN_RESOLUTION_HEIGHT`: (Lowered to 200 for screenshot testing)
- `CLIP_MODEL_NAME`: Set to `ViT-L/14@336px` for better high-resolution semantic matching.
- `YOLO_MODEL_PATH`: Set to `yolov8s.pt` for consistent accuracy across environments.
- `YOLO_AUGMENT`: `false` for API requests, `true` only for worker-side warm inference if needed.

---

## 📦 Tech Stack
- **Framework:** FastAPI (Python 3.11)
- **Task Queue:** Celery + Redis
- **AI Models:** Ultralytics YOLOv8, OpenAI CLIP, NudeNet
- **Database:** PostgreSQL (SQLAlchemy + AsyncPG)
- **Storage:** MinIO (S3 Compatible)
- **Infra:** Docker Compose / Multi-stage Dockerfiles
