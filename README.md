# VisionLoop: Autonomous OpenCV 5 Workload Tuning Agent

<p align="center">
  <b>Vision intelligence, optimized.</b><br>
  <i>Adaptive computer-vision experiments with evidence-backed decisions for OpenCV 5.0.</i>
</p>

---

## ⚡ Overview

Computer vision engineers spend countless hours manually tuning hyperparameters (Canny thresholds, Sobel apertures, Gaussian kernels, morphological operations) across varying contrast, lighting, and noise conditions.

**VisionLoop** is an autonomous developer agent designed for the **OpenCV AI Competition 2026 (powered by AWS)**. It iteratively runs computer-vision experiments on CPU and AWS cloud compute, evaluates perceptual quality using objective ground-truth F1 scoring or continuity proxies, enforces strict quality guardrails, and arrives at optimal parameter configurations backed by an immutable decision audit trail.

---

## 🖥️ Dashboard Architecture & Visual Design

VisionLoop features a **premium, dark cinematic AI Engineering Workstation** styled after modern high-performance developer tools:

- **Top Status Navigation**: Real-time status indicators for OpenCV 5.0 CPU engine, AWS Graviton/Lambda connectivity, and Agent state.
- **KPI Metrics Grid**:
  - **Best Quality**: Validated F1 score with delta indicator and sparkline.
  - **Latency**: High-precision 5-pass median processing time (ms) with percentage reduction.
  - **Experiments**: Total iterations with accepted vs. rejected counts.
  - **Improvement**: Composite performance improvement metric.
- **Live Experiment Stepper**: 4-stage pipeline visualization (`BASELINE` ➔ `EVALUATE` ➔ `ADJUST PARAMETERS` ➔ `VALIDATE`) tracking the active agent iteration in real-time.
- **Agent Reasoning Stream**: Timestamped chronological decision feed detailing diagnosis, rationale, parameter mutations, and quality verdicts.
- **4-Panel Visual Comparison Workspace**: Signature visual centerpiece showing:
  1. *Input Original Image* (Dimensions, color space)
  2. *Baseline Output* (Default parameters, latency)
  3. *Best Validated Candidate* (Selected crown badge, glowing cyan highlight, speedup metrics)
  4. *Ground Truth Reference* (Annotated edge map with 1px morphological tolerance)
- **Quality vs. Latency Log-Scale Scatter Plot**: Interactive trade-off chart categorizing Validated, Rejected, and Baseline experiments with hover tooltips.
- **Experiment History Table**: Detailed tabular log with parameter strings, latencies, F1 scores, delta vs. baseline, and status pills.
- **Audit Trail & Export**: One-click download of full reproducibility reports in Markdown and JSON formats.

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup
Clone the repository and ensure Python 3.10+ is installed:

```bash
git clone https://github.com/your-username/OpenCV-Hktn.git
cd OpenCV-Hktn
```

Install pinned dependencies (including verified OpenCV 5.0.0 headless build):

```bash
pip install -r requirements.txt
```

Verify OpenCV 5 environment sanity:

```bash
python scripts/verify_opencv5.py
```

### 2. Run Test Suite
Run the comprehensive automated pytest suite (30+ tests covering vision pipelines, evaluators, agent state machine, AWS bridge, and web endpoints):

```bash
python -m pytest tests/
```

### 3. Launch the Dashboard
Start the FastAPI server:

```bash
python src/web/server.py
```

Open your browser and navigate to:
```
http://localhost:8000
```

---

## 🧠 Autonomous Agent Optimization Loop

The agent follows an iterative **Perception ➔ Diagnosis ➔ Action** cycle:

1. **Perception**: Runs the candidate pipeline using OpenCV 5. Computes precision, recall, and F1 score against ground truth (using a 1-pixel dilation tolerance window per BSDS benchmark standards). On unlabelled images, computes edge continuity, noise ratio, and density fitness.
2. **Diagnosis**: Detects failure conditions:
   - *Edge Silence*: Detected zero or too few edges; recall dropped below guardrail (`F1 < 0.70`).
   - *Noise Explosion*: Excessive spurious edge fragments detected; precision collapsed.
   - *Over-Smoothing*: Details washed out by excessive Gaussian pre-filtering.
3. **Action**: Selects targeted parameter adjustments (lowering/raising hysteresis thresholds, modifying aperture size, adjusting kernel sizes) and logs the explicit decision rationale.
4. **Guardrail Evaluation**: A candidate is only accepted if it satisfies the minimum quality guardrail (`F1 >= threshold`). If quality degrades, the candidate is rejected and rolled back.

---

## ☁️ AWS Cloud Architecture (Single Focused Component)

VisionLoop incorporates a serverless containerized execution target on **AWS Lambda**:

- **Containerized OpenCV 5 Workload**: Defined in `Dockerfile.aws` with the official AWS Lambda Python runtime and OpenCV 5 dependencies.
- **Cost Safeguards & Limits**:
  - Image payloads restricted to `< 1 MB`.
  - 15-second execution timeout.
  - 1024 MB RAM allocation.
  - Zero idle cost serverless execution.
- **Unified Client Bridge (`src/aws/cloud_executor.py`)**: Seamlessly routes execution between:
  1. *AWS Lambda Function URL* (Direct HTTPS endpoint)
  2. *Boto3 Invocation* (AWS credentials)
  3. *Local Simulation Mode* (Graceful fallback when AWS credentials are not configured)

### Deployment & Teardown Commands
To deploy the container to Amazon ECR and AWS Lambda:
```bash
python src/aws/deploy_lambda.py
```

To tear down all deployed AWS resources and avoid lingering charges:
```bash
python scripts/teardown_aws.py
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/system/status` | `GET` | Hardware specs, OpenCV 5 version, AWS connectivity status, and agent health |
| `/api/tasks` | `GET` | Supported vision tasks, execution modes, and curated test image presets |
| `/api/preset-image/{id}/{kind}` | `GET` | Serves original benchmark image or reference ground-truth map |
| `/api/run` | `POST` | Launches the autonomous optimization loop on a preset or uploaded image |
| `/api/trace/{session_id}` | `GET` | Retrieves full JSON decision trace and telemetry for a session |
| `/api/report/{session_id}` | `GET` | Downloads reproducibility report (`?format=markdown` or `?format=json`) |
| `/api/history` | `GET` | Lists recent optimization sessions with summary statistics |

---

## 📊 Benchmark Datasets Included

Located in `data/test_images/` with ground-truth reference edge maps in `data/reference_edges/`:

1. `sample_01`: Geometric high-contrast calibration image (Circle & Rectangle).
2. `sample_02`: Multi-feature shapes with varied edge orientations.
3. `sample_03`: Urban architecture with fine structural line details.
4. `sample_04_low_contrast`: Challenging low-contrast gradient boundaries.
5. `sample_05_noisy`: Image with synthetic additive Gaussian noise.

---

## 📜 Reproducibility & Audit Trail

Every experiment produces an immutable run directory in `runs/<session_id>/`:
- `experiment_trace.json`: Machine-readable structured audit trace of every iteration, parameter, timing sample, and rationale.
- `experiment_report.md`: Human-readable markdown report with before/after comparisons and executive summary.
- `iter_*.png`: Processed edge output images for all evaluated candidates.

---

## 🏆 OpenCV AI Competition 2026 Compliance

- **OpenCV Version**: Uses verified OpenCV 5.0.0 CPU execution throughout.
- **AWS Requirement**: Includes focused, containerized AWS Lambda component with verified deployment and teardown automation.
- **Agentic Vision Track**: Full perception-diagnosis-action cycle with explainable decision rationale and quality guardrails.
- **CPU-First**: Benchmarked with 3-warmup + 5-timed-pass median harness eliminating scheduler jitter.
