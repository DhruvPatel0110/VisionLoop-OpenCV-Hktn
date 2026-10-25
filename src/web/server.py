"""FastAPI Backend Server for VisionLoop Autonomous Developer Agent.

Serves the REST API for task selection, autonomous agent optimization runs,
audit traces, report downloads, and static frontend dashboard.
"""

from typing import Optional, Dict, Any, List
import os
import sys
import json
import base64
import time
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.vision.edge_detector import EdgeDetector, EdgeDetectorConfig
from src.agent.coordinator import VisionLoopCoordinator, OptimizationSummary
from src.aws.cloud_executor import CloudExecutor

app = FastAPI(
    title="VisionLoop API",
    description="Developer agent for autonomous OpenCV 5 workload tuning and benchmarking",
    version="1.0.0"
)

# Enable CORS for local development and browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MANIFEST_PATH = "data/dataset_manifest.json"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class RunRequest(BaseModel):
    preset_id: Optional[str] = Field(default="sample_04_low_contrast", description="ID of preset image")
    image_base64: Optional[str] = Field(default=None, description="Custom uploaded image encoded in base64")
    execution_mode: str = Field(default="local", description="'local' CPU or 'aws' Lambda container")
    max_iterations: int = Field(default=5, ge=2, le=10, description="Max candidate experiment iterations")
    target_min_quality: float = Field(default=0.70, ge=0.30, le=0.95, description="Quality guardrail threshold")


def image_to_base64_data_uri(img: np.ndarray) -> str:
    """Helper to convert OpenCV image to web-ready base64 data URI."""
    if img is None:
        return ""
    success, buf = cv2.imencode(".png", img)
    if not success:
        return ""
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.get("/api/tasks")
def get_tasks_and_presets():
    """Lists supported CV tasks and curated benchmark test presets."""
    manifest = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    return {
        "status": "SUCCESS",
        "task_name": "Edge Detection (OpenCV 5 CPU Canny)",
        "supported_tasks": [
            {
                "id": "edge_detection",
                "name": "Canny Edge Detection",
                "description": "Adaptive tuning of hysteresis thresholds, Sobel apertures, and Gaussian pre-filters",
                "is_active": True
            }
        ],
        "execution_modes": [
            {
                "id": "local",
                "name": "Local CPU",
                "description": "Local OpenCV 5.0.0 engine running on host CPU"
            },
            {
                "id": "aws",
                "name": "AWS Lambda Container",
                "description": "Cloud-backed serverless CPU compute container with zero idle cost"
            }
        ],
        "presets": manifest
    }


@app.get("/api/preset-image/{preset_id}/{kind}")
def get_preset_image(preset_id: str, kind: str):
    """Returns preset original or reference image as PNG file."""
    if not os.path.exists(MANIFEST_PATH):
        raise HTTPException(status_code=404, detail="Manifest not found")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    item = next((p for p in manifest if p["id"] == preset_id), None)
    if not item:
        raise HTTPException(status_code=404, detail=f"Preset {preset_id} not found")

    file_path = item["image_path"] if kind == "original" else item.get("reference_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Image file not found: {file_path}")

    return FileResponse(file_path, media_type="image/png")


@app.post("/api/run")
def run_optimization(req: RunRequest):
    """Launches the autonomous agent loop on an input image."""
    img = None
    ref_img = None
    image_id = "custom_upload"

    # 1. Load preset or custom upload
    if req.image_base64:
        try:
            # Strip data URI header if present
            raw_b64 = req.image_base64.split(",")[-1]
            img_bytes = base64.b64decode(raw_b64)
            img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image")
            image_id = f"upload_{int(time.time())}"

            # Auto-scale large uploads to safe CPU limit while preserving aspect ratio
            h, w = img.shape[:2]
            max_dim = EdgeDetector.MAX_DIMENSION
            if h > max_dim or w > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image_base64: {str(e)}")
    elif req.preset_id:
        if not os.path.exists(MANIFEST_PATH):
            raise HTTPException(status_code=404, detail="Dataset manifest not found")
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        item = next((p for p in manifest if p["id"] == req.preset_id), None)
        if not item:
            raise HTTPException(status_code=404, detail=f"Preset '{req.preset_id}' not found")
        img = cv2.imread(item["image_path"])
        if item.get("reference_path") and os.path.exists(item["reference_path"]):
            ref_img = cv2.imread(item["reference_path"], cv2.IMREAD_GRAYSCALE)
        image_id = item["id"]
    else:
        raise HTTPException(status_code=400, detail="Must provide either preset_id or image_base64")

    # 2. Configure Coordinator
    coordinator = VisionLoopCoordinator(
        max_iterations=req.max_iterations,
        max_seconds=30.0,
        target_min_quality=req.target_min_quality,
        output_dir=RUNS_DIR
    )

    # 3. Execute Loop
    try:
        summary: OptimizationSummary = coordinator.run(
            image=img,
            reference_edges=ref_img,
            image_id=image_id,
            save_candidate_images=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent loop failed: {str(e)}")

    # 4. Enhance response with base64 previews for UI display
    response_data = summary.model_dump()
    session_dir = os.path.join(RUNS_DIR, summary.session_id).replace("\\", "/")

    # Original Image preview
    response_data["preview_original_b64"] = image_to_base64_data_uri(img)

    # Baseline Image preview
    base_img_path = os.path.join(session_dir, "iter_00_baseline.png")
    if os.path.exists(base_img_path):
        b_img = cv2.imread(base_img_path, cv2.IMREAD_GRAYSCALE)
        response_data["preview_baseline_b64"] = image_to_base64_data_uri(b_img)

    # Ground truth reference preview
    if ref_img is not None:
        response_data["preview_reference_b64"] = image_to_base64_data_uri(ref_img)
    else:
        response_data["preview_reference_b64"] = None

    # Best Image preview (only if a valid incumbent winner exists)
    winner_trace = next((t for t in summary.traces if getattr(t, "is_final_winner", False) and t.output_image_path), None)
    if not winner_trace and summary.best_config and summary.traces:
        winner_trace = next((t for t in summary.traces if t.config == summary.best_config and t.output_image_path), None)

    if winner_trace and winner_trace.output_image_path and os.path.exists(winner_trace.output_image_path):
        w_img = cv2.imread(winner_trace.output_image_path, cv2.IMREAD_GRAYSCALE)
        response_data["preview_best_b64"] = image_to_base64_data_uri(w_img)
    else:
        response_data["preview_best_b64"] = None

    # Attach base64 thumbnails to each trace for live timeline
    for idx, trace in enumerate(response_data.get("traces", [])):
        img_p = trace.get("output_image_path")
        if img_p and os.path.exists(img_p):
            t_img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
            trace["thumbnail_b64"] = image_to_base64_data_uri(t_img)

    response_data["execution_mode_used"] = req.execution_mode
    return response_data


@app.get("/api/trace/{session_id}")
def get_trace(session_id: str):
    """Returns the full JSON trace for a session."""
    trace_path = os.path.join(RUNS_DIR, session_id, "experiment_trace.json")
    if not os.path.exists(trace_path):
        raise HTTPException(status_code=404, detail="Trace not found for session")
    with open(trace_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/report/{session_id}")
def download_report(session_id: str, format: str = Query("markdown", enum=["markdown", "json"])):
    """Downloads the reproducibility report in Markdown or JSON format."""
    session_dir = os.path.join(RUNS_DIR, session_id)
    if format == "json":
        file_path = os.path.join(session_dir, "experiment_trace.json")
        media = "application/json"
        dl_name = f"visionloop_{session_id}_trace.json"
    else:
        file_path = os.path.join(session_dir, "experiment_report.md")
        media = "text/markdown"
        dl_name = f"visionloop_{session_id}_report.md"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(file_path, media_type=media, filename=dl_name)


@app.get("/api/system/status")
def get_system_status():
    """Returns actual hardware, OpenCV 5 environment, and AWS connection status."""
    import platform

    has_endpoint = bool(os.environ.get("AWS_LAMBDA_ENDPOINT_URL"))
    has_creds = CloudExecutor._has_aws_credentials()

    if has_endpoint:
        aws_status = "ONLINE"
        aws_desc = "AWS Lambda Function URL Active"
    elif has_creds:
        aws_status = "CONNECTED"
        aws_desc = "AWS Credentials Active (Boto3)"
    else:
        aws_status = "LOCAL_SIMULATION"
        aws_desc = "Local CPU / Container Simulation Mode"

    # Count historical runs
    session_count = 0
    if os.path.exists(RUNS_DIR):
        session_count = len([d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d)) and d.startswith("opt_")])

    return {
        "status": "HEALTHY",
        "opencv_version": cv2.__version__,
        "python_version": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "cpu_cores": os.cpu_count() or 1,
        "aws_status": aws_status,
        "aws_description": aws_desc,
        "total_sessions_recorded": session_count,
        "agent_status": "READY"
    }


@app.get("/api/history")
def get_experiment_history(limit: int = 10):
    """Returns past optimization sessions and their recorded traces."""
    sessions = []
    if os.path.exists(RUNS_DIR):
        subdirs = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d)) and d.startswith("opt_")]
        # Sort by creation time descending
        subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(RUNS_DIR, d)), reverse=True)

        for d in subdirs[:limit]:
            trace_p = os.path.join(RUNS_DIR, d, "experiment_trace.json")
            if os.path.exists(trace_p):
                try:
                    with open(trace_p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        sessions.append(data)
                except Exception:
                    pass

    return {
        "status": "SUCCESS",
        "count": len(sessions),
        "history": sessions
    }


# Mount static web UI files if directory exists
WEB_DIR = "web"
os.makedirs(WEB_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def run_server(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    print(f"Starting VisionLoop Dashboard at http://{host}:{port} ...")
    uvicorn.run("src.web.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
