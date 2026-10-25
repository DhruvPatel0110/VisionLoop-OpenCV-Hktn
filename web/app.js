// VisionLoop AI Engineering Workstation Controller
// Integrates with FastAPI backend for autonomous OpenCV 5 workload optimization

document.addEventListener("DOMContentLoaded", () => {
  // App State
  let systemStatus = {};
  let presets = [];
  let currentPresetIndex = 0;
  let currentSession = null;
  let customImageBase64 = null;
  let selectedCompute = "local";
  let isRunning = false;

  // DOM Elements
  const headerOpencvVersion = document.getElementById("headerOpencvVersion");
  const headerAwsPill = document.getElementById("headerAwsPill");
  const headerAwsStatus = document.getElementById("headerAwsStatus");
  const headerAgentPill = document.getElementById("headerAgentPill");
  const headerAgentStatus = document.getElementById("headerAgentStatus");

  // Sidebar Elements
  const navItems = document.querySelectorAll(".sidebar-nav .nav-item");
  const viewPanels = document.querySelectorAll(".view-panel");
  const sidebarStatusText = document.getElementById("sidebarStatusText");
  const sidebarDatasetName = document.getElementById("sidebarDatasetName");
  const sidebarComputeTarget = document.getElementById("sidebarComputeTarget");
  const sidebarIterationsCount = document.getElementById("sidebarIterationsCount");
  const stopExperimentBtn = document.getElementById("stopExperimentBtn");

  // Hero Elements
  const heroDateText = document.getElementById("heroDateText");
  const heroSessionId = document.getElementById("heroSessionId");
  const heroTaskSubtitle = document.getElementById("heroTaskSubtitle");
  const openNewExperimentBtn = document.getElementById("openNewExperimentBtn");

  // KPI Elements
  const kpiBestQuality = document.getElementById("kpiBestQuality");
  const kpiQualityDelta = document.getElementById("kpiQualityDelta");
  const kpiQualitySubtext = document.getElementById("kpiQualitySubtext");
  const kpiLatency = document.getElementById("kpiLatency");
  const kpiLatencyDelta = document.getElementById("kpiLatencyDelta");
  const kpiExperimentsTotal = document.getElementById("kpiExperimentsTotal");
  const kpiValidRejectedText = document.getElementById("kpiValidRejectedText");
  const kpiImprovement = document.getElementById("kpiImprovement");

  // Stepper Elements
  const stepperIterationText = document.getElementById("stepperIterationText");
  const stepperLiveLabel = document.getElementById("stepperLiveLabel");
  const step1Stat = document.getElementById("step1Stat");
  const step2Stat = document.getElementById("step2Stat");
  const step3Stat = document.getElementById("step3Stat");
  const step4Stat = document.getElementById("step4Stat");
  const stepNode1 = document.getElementById("stepNode1");
  const stepNode2 = document.getElementById("stepNode2");
  const stepNode3 = document.getElementById("stepNode3");
  const stepNode4 = document.getElementById("stepNode4");

  // Reasoning Elements
  const reasoningFeed = document.getElementById("reasoningFeed");
  const viewFullLogBtn = document.getElementById("viewFullLogBtn");

  // Comparison Elements
  const prevImageBtn = document.getElementById("prevImageBtn");
  const nextImageBtn = document.getElementById("nextImageBtn");
  const pagImageText = document.getElementById("pagImageText");
  const expandComparisonBtn = document.getElementById("expandComparisonBtn");
  const viewImgOriginal = document.getElementById("viewImgOriginal");
  const viewImgBaseline = document.getElementById("viewImgBaseline");
  const viewImgBest = document.getElementById("viewImgBest");
  const viewImgReference = document.getElementById("viewImgReference");
  const metaOriginal = document.getElementById("metaOriginal");
  const metaBaseline = document.getElementById("metaBaseline");
  const metaBest = document.getElementById("metaBest");
  const metaReference = document.getElementById("metaReference");
  const cardBest = document.getElementById("cardBest");
  const headerBest = document.getElementById("headerBest");
  const titleBest = document.getElementById("titleBest");
  const badgeBest = document.getElementById("badgeBest");
  const resetWinnerBtn = document.getElementById("resetWinnerBtn");
  const noWinnerPlaceholder = document.getElementById("noWinnerPlaceholder");
  const noWinnerSub = document.getElementById("noWinnerSub");
  const noRefPlaceholder = document.getElementById("noRefPlaceholder");
  const titleReference = document.getElementById("titleReference");
  const stepperSummaryBar = document.getElementById("stepperSummaryBar");
  const summaryIcon = document.getElementById("summaryIcon");
  const stepperSummaryText = document.getElementById("stepperSummaryText");
  const chartEmptyState = document.getElementById("chartEmptyState");
  const chartYAxisLabel = document.getElementById("chartYAxisLabel");
  const thQualityHeader = document.getElementById("thQualityHeader");
  const kpiImprovementSub = document.getElementById("kpiImprovementSub");

  // Chart & Table
  const chartPointsGroup = document.getElementById("chartPointsGroup");
  const chartTooltip = document.getElementById("chartTooltip");
  const tooltipTitle = document.getElementById("tooltipTitle");
  const tooltipQuality = document.getElementById("tooltipQuality");
  const tooltipLatency = document.getElementById("tooltipLatency");
  const tooltipConfig = document.getElementById("tooltipConfig");
  const historyTableBody = document.getElementById("historyTableBody");
  const viewAllHistoryBtn = document.getElementById("viewAllHistoryBtn");

  // Modal Elements
  const newExperimentModal = document.getElementById("newExperimentModal");
  const closeNewExpModalBtn = document.getElementById("closeNewExpModalBtn");
  const cancelNewExpModalBtn = document.getElementById("cancelNewExpModalBtn");
  const launchExperimentBtn = document.getElementById("launchExperimentBtn");
  const modalPresetSelect = document.getElementById("modalPresetSelect");
  const modalPresetHint = document.getElementById("modalPresetHint");
  const modalFileInput = document.getElementById("modalFileInput");
  const modalModeLocalBtn = document.getElementById("modalModeLocalBtn");
  const modalModeAwsBtn = document.getElementById("modalModeAwsBtn");
  const modalComputeHint = document.getElementById("modalComputeHint");
  const modalIterRange = document.getElementById("modalIterRange");
  const modalIterVal = document.getElementById("modalIterVal");
  const modalQualityRange = document.getElementById("modalQualityRange");
  const modalQualityVal = document.getElementById("modalQualityVal");

  // Lightbox Elements
  const lightboxModal = document.getElementById("lightboxModal");
  const closeLightboxBtn = document.getElementById("closeLightboxBtn");
  const lightboxImgOriginal = document.getElementById("lightboxImgOriginal");
  const lightboxImgBaseline = document.getElementById("lightboxImgBaseline");
  const lightboxImgBest = document.getElementById("lightboxImgBest");
  const lightboxImgRef = document.getElementById("lightboxImgRef");

  // Audit Downloads
  const auditDownloadJsonBtn = document.getElementById("auditDownloadJsonBtn");
  const auditDownloadMdBtn = document.getElementById("auditDownloadMdBtn");
  const auditRawLog = document.getElementById("auditRawLog");

  // Initialize
  init();

  async function init() {
    setupNavigation();
    setupModals();
    await fetchSystemStatus();
    await fetchTasksAndPresets();
    await fetchLatestSessionOrRun();
  }

  // Navigation Tabs Switcher
  function setupNavigation() {
    navItems.forEach(item => {
      item.addEventListener("click", () => {
        navItems.forEach(n => n.classList.remove("active"));
        item.classList.add("active");

        const targetView = item.dataset.view;
        viewPanels.forEach(panel => {
          panel.classList.remove("active");
          if (panel.id === `view${capitalize(targetView)}`) {
            panel.classList.add("active");
          }
        });

        if (targetView === "datasets") renderDatasetsGrid();
        if (targetView === "experiments") renderArchiveList();
        if (targetView === "audit") renderAuditView();
      });
    });

    viewAllHistoryBtn?.addEventListener("click", () => {
      document.querySelector('[data-view="experiments"]')?.click();
    });

    viewFullLogBtn?.addEventListener("click", () => {
      document.querySelector('[data-view="audit"]')?.click();
    });

    document.getElementById("subviewNewExpBtn")?.addEventListener("click", () => {
      openModal();
    });
  }

  function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  // Modals Setup
  function setupModals() {
    openNewExperimentBtn.addEventListener("click", openModal);
    closeNewExpModalBtn.addEventListener("click", closeModal);
    cancelNewExpModalBtn.addEventListener("click", closeModal);

    modalModeLocalBtn.addEventListener("click", () => {
      selectedCompute = "local";
      modalModeLocalBtn.classList.add("active");
      modalModeAwsBtn.classList.remove("active");
      modalComputeHint.textContent = "Runs in-process on host CPU";
    });

    modalModeAwsBtn.addEventListener("click", () => {
      selectedCompute = "aws";
      modalModeAwsBtn.classList.add("active");
      modalModeLocalBtn.classList.remove("active");
      modalComputeHint.textContent = "Executes on remote AWS Lambda container (with simulation fallback)";
    });

    modalIterRange.addEventListener("input", (e) => {
      modalIterVal.textContent = e.target.value;
    });

    modalQualityRange.addEventListener("input", (e) => {
      modalQualityVal.textContent = Number(e.target.value).toFixed(2);
    });

    modalPresetSelect.addEventListener("change", (e) => {
      modalFileInput.value = "";
      customImageBase64 = null;
      const found = presets.find(p => p.id === e.target.value);
      if (found) {
        modalPresetHint.textContent = `Category: ${found.category} | ${found.dimensions[0]}x${found.dimensions[1]}`;
      }
    });

    modalFileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (!file) return;
      if (file.size > 2_500_000) {
        alert("Image exceeds 2.5MB network limit. Please choose a smaller image.");
        modalFileInput.value = "";
        return;
      }
      const reader = new FileReader();
      reader.onload = (evt) => {
        customImageBase64 = evt.target.result;
        const tempImg = new Image();
        tempImg.onload = () => {
          let dim = `${tempImg.naturalWidth}×${tempImg.naturalHeight}`;
          if (tempImg.naturalWidth > 1024 || tempImg.naturalHeight > 1024) {
            dim += " (auto-scales to ≤1024px for CPU safety)";
          }
          modalPresetHint.textContent = `Custom: ${file.name} [${dim}] (${Math.round(file.size / 1024)} KB)`;
        };
        tempImg.src = customImageBase64;
      };
      reader.readAsDataURL(file);
    });

    launchExperimentBtn.addEventListener("click", () => {
      closeModal();
      launchNewExperiment();
    });

    // Lightbox Modal
    expandComparisonBtn.addEventListener("click", () => {
      if (!currentSession) return;
      lightboxImgOriginal.src = currentSession.preview_original_b64 || "";
      lightboxImgBaseline.src = currentSession.preview_baseline_b64 || "";
      lightboxImgBest.src = currentSession.preview_best_b64 || "";
      if (currentSession.preview_reference_b64) {
        lightboxImgRef.src = currentSession.preview_reference_b64;
        lightboxImgRef.parentElement.style.display = "block";
      } else {
        lightboxImgRef.src = "";
        lightboxImgRef.parentElement.style.display = "none";
      }
      lightboxModal.classList.remove("hidden");
    });

    closeLightboxBtn.addEventListener("click", () => {
      lightboxModal.classList.add("hidden");
    });

    // Pagination for presets in comparison view
    prevImageBtn.addEventListener("click", () => {
      if (presets.length === 0) return;
      currentPresetIndex = (currentPresetIndex - 1 + presets.length) % presets.length;
      switchPreset(presets[currentPresetIndex].id);
    });

    nextImageBtn.addEventListener("click", () => {
      if (presets.length === 0) return;
      currentPresetIndex = (currentPresetIndex + 1) % presets.length;
      switchPreset(presets[currentPresetIndex].id);
    });

    stopExperimentBtn.addEventListener("click", () => {
      if (isRunning) {
        isRunning = false;
        sidebarStatusText.innerHTML = `<span class="pulse-dot-sm"></span> Agent Stopped`;
      }
    });

    // Reset view in Panel 3 to true winner
    resetWinnerBtn?.addEventListener("click", () => {
      restoreWinnerPanel();
    });

    // Downloads
    auditDownloadJsonBtn?.addEventListener("click", () => {
      if (currentSession?.session_id) {
        window.location.href = `/api/report/${currentSession.session_id}?format=json`;
      }
    });

    auditDownloadMdBtn?.addEventListener("click", () => {
      if (currentSession?.session_id) {
        window.location.href = `/api/report/${currentSession.session_id}?format=markdown`;
      }
    });
  }

  function openModal() {
    newExperimentModal.classList.remove("hidden");
  }

  function closeModal() {
    newExperimentModal.classList.add("hidden");
  }

  // 1. Fetch System Status
  async function fetchSystemStatus() {
    try {
      const resp = await fetch("/api/system/status");
      if (!resp.ok) return;
      systemStatus = await resp.json();

      headerOpencvVersion.textContent = `OpenCV ${systemStatus.opencv_version || '5.0'}`;
      if (systemStatus.aws_status === "ONLINE") {
        headerAwsStatus.textContent = "AWS ONLINE";
        headerAwsPill.className = "status-pill pill-aws";
      } else if (systemStatus.aws_status === "CONNECTED") {
        headerAwsStatus.textContent = "AWS BOTO3";
      } else {
        headerAwsStatus.textContent = "AWS GRAVITON";
      }

      sidebarComputeTarget.textContent = `${systemStatus.platform.split(" ")[0]} CPU → AWS`;
    } catch (err) {
      console.warn("Could not fetch system status:", err);
    }
  }

  // 2. Fetch Tasks & Presets
  async function fetchTasksAndPresets() {
    try {
      const resp = await fetch("/api/tasks");
      if (!resp.ok) return;
      const data = await resp.json();
      presets = data.presets || [];

      modalPresetSelect.innerHTML = "";
      presets.forEach((p, idx) => {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = `${p.name} (${p.category})`;
        if (p.id === "sample_04_low_contrast") {
          opt.selected = true;
          currentPresetIndex = idx;
        }
        modalPresetSelect.appendChild(opt);
      });
    } catch (err) {
      console.warn("Could not fetch tasks:", err);
    }
  }

  // 3. Fetch Latest History or Run Default
  async function fetchLatestSessionOrRun() {
    try {
      const resp = await fetch("/api/history?limit=1");
      if (resp.ok) {
        const data = await resp.json();
        if (data.history && data.history.length > 0) {
          applySessionData(data.history[0]);
          return;
        }
      }
    } catch (err) {
      console.warn("History fetch failed:", err);
    }

    // If no past session exists, automatically launch the low-contrast benchmark
    launchNewExperiment("sample_04_low_contrast");
  }

  function switchPreset(presetId) {
    modalPresetSelect.value = presetId;
    launchNewExperiment(presetId);
  }

  // 4. Launch Experiment
  async function launchNewExperiment(overridePresetId = null) {
    if (isRunning) return;
    isRunning = true;
    openNewExperimentBtn.disabled = true;
    launchExperimentBtn.disabled = true;
    launchExperimentBtn.innerHTML = `<span>⏳</span> Tuning in Progress...`;
    sidebarStatusText.innerHTML = `<span class="pulse-dot-sm" style="background:#00D9FF;box-shadow:0 0 6px #00D9FF"></span> Agent Running`;
    stepperLiveStatus.style.background = "rgba(0, 217, 255, 0.25)";
    stepperLiveLabel.textContent = "Live Tuning";

    if (stepperSummaryText) {
      stepperSummaryText.textContent = "Autonomous optimization in progress. Iterating parameters and benchmarking CPU latency...";
      summaryIcon.textContent = "⚡";
      if (stepperSummaryBar) {
        stepperSummaryBar.style.borderColor = "rgba(0, 217, 255, 0.35)";
        stepperSummaryBar.style.background = "rgba(3, 8, 16, 0.85)";
      }
    }

    const payload = {
      execution_mode: selectedCompute,
      max_iterations: parseInt(modalIterRange.value, 10),
      target_min_quality: parseFloat(modalQualityRange.value)
    };

    if (overridePresetId) {
      payload.preset_id = overridePresetId;
    } else if (customImageBase64) {
      payload.image_base64 = customImageBase64;
    } else {
      payload.preset_id = modalPresetSelect.value || "sample_04_low_contrast";
    }

    // Set dataset label in sidebar
    sidebarDatasetName.textContent = payload.preset_id || "custom_upload";

    try {
      const resp = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || "Agent execution failed");
      }

      const summary = await resp.json();
      applySessionData(summary);
    } catch (err) {
      alert(`Error running VisionLoop Agent: ${err.message}`);
      if (stepperSummaryText) {
        stepperSummaryText.textContent = `Experiment aborted: ${err.message}`;
        summaryIcon.textContent = "❌";
        if (stepperSummaryBar) stepperSummaryBar.style.borderColor = "rgba(239, 68, 68, 0.5)";
      }
    } finally {
      isRunning = false;
      openNewExperimentBtn.disabled = false;
      launchExperimentBtn.disabled = false;
      launchExperimentBtn.innerHTML = `<span>▶</span> Launch VisionLoop Agent`;
      sidebarStatusText.innerHTML = `<span class="pulse-dot-sm"></span> Agent Ready`;
      stepperLiveLabel.textContent = "Ready";
      stepperLiveStatus.style.background = "rgba(0, 217, 255, 0.12)";
    }
  }

  // 5. Apply Session Data to Dashboard
  function applySessionData(summary) {
    currentSession = summary;

    // Header Capsule
    const dateObj = new Date();
    heroDateText.textContent = dateObj.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
    heroSessionId.textContent = `Session #${summary.session_id.slice(-4)}`;
    heroTaskSubtitle.textContent = `Edge Detection Tuning (${summary.image_id})`;

    // Sidebar
    sidebarDatasetName.textContent = summary.image_id;
    sidebarIterationsCount.textContent = `${summary.iterations_completed} / ${summary.budget_max_iterations}`;

    // Stepper summary bar: Stopping reason & Agent status
    if (stepperSummaryText && summary.status_summary) {
      stepperSummaryText.textContent = summary.status_summary;
      const hasWinner = summary.winner_iteration !== null && summary.winner_iteration !== undefined;
      summaryIcon.textContent = hasWinner ? "🏆" : "⚠️";
      if (stepperSummaryBar) {
        stepperSummaryBar.style.borderColor = hasWinner ? "rgba(0, 217, 255, 0.35)" : "rgba(239, 68, 68, 0.4)";
        stepperSummaryBar.style.background = hasWinner ? "rgba(3, 8, 16, 0.85)" : "rgba(20, 8, 8, 0.85)";
      }
    }

    // Dynamic metric identification
    const isGroundTruth = summary.baseline_quality_type === "GROUND_TRUTH_F1" && summary.has_ground_truth !== false;
    const qUnit = isGroundTruth ? "Edge F1" : "Proxy Quality";

    // KPI 1: Best Quality
    const bestQ = summary.best_quality_score ?? summary.baseline_quality_score;
    kpiBestQuality.textContent = `${(bestQ * 100).toFixed(1)}%`;
    if (summary.quality_delta !== null && summary.quality_delta !== undefined) {
      const sign = summary.quality_delta >= 0 ? "+" : "";
      kpiQualityDelta.textContent = `${sign}${(summary.quality_delta * 100).toFixed(1)}%`;
      kpiQualityDelta.className = summary.quality_delta >= 0 ? "delta-badge delta-pos" : "delta-badge delta-neg";
    } else {
      kpiQualityDelta.textContent = "0.0%";
      kpiQualityDelta.className = "delta-badge";
    }
    kpiQualitySubtext.textContent = isGroundTruth ? "Edge F1 Score (↑ better)" : "Proxy Quality (heuristic, ↑ better)";

    // KPI 2: Latency
    const bestLat = summary.best_latency_ms ?? summary.baseline_latency_ms;
    kpiLatency.textContent = `${bestLat.toFixed(2)} ms`;
    if (summary.latency_reduction_pct !== null && summary.latency_reduction_pct !== undefined) {
      if (summary.latency_reduction_pct > 0) {
        kpiLatencyDelta.textContent = `▼ -${summary.latency_reduction_pct.toFixed(1)}%`;
        kpiLatencyDelta.className = "delta-badge delta-pos";
      } else if (summary.latency_reduction_pct < 0) {
        kpiLatencyDelta.textContent = `▲ +${Math.abs(summary.latency_reduction_pct).toFixed(1)}%`;
        kpiLatencyDelta.className = "delta-badge delta-neg";
      } else {
        kpiLatencyDelta.textContent = "0.0%";
        kpiLatencyDelta.className = "delta-badge";
      }
    }

    // KPI 3: Experiments Total
    kpiExperimentsTotal.textContent = String(summary.iterations_completed).padStart(2, "0");
    kpiValidRejectedText.textContent = `${summary.candidates_accepted} valid / ${summary.candidates_rejected_quality + summary.candidates_rejected_latency} rejected`;

    // KPI 4: Improvement / Speedup
    const hasWinner = summary.winner_iteration !== null && summary.winner_iteration !== undefined;
    if (hasWinner && summary.speedup_factor) {
      kpiImprovement.textContent = `${summary.speedup_factor.toFixed(2)}x`;
      if (kpiImprovementSub) {
        kpiImprovementSub.textContent = summary.speedup_factor >= 1.0 ? "Speedup vs Baseline" : "Latency vs Baseline";
      }
    } else if (!hasWinner) {
      kpiImprovement.textContent = "1.00x";
      if (kpiImprovementSub) {
        kpiImprovementSub.textContent = "Baseline Retained";
      }
    } else {
      kpiImprovement.textContent = "1.00x";
      if (kpiImprovementSub) {
        kpiImprovementSub.textContent = "Speedup vs Baseline";
      }
    }

    // Stepper Track
    stepperIterationText.textContent = `Iteration 0${summary.iterations_completed} / 0${summary.budget_max_iterations}`;
    step1Stat.textContent = `${summary.baseline_latency_ms.toFixed(2)} ms | ${(summary.baseline_quality_score * 100).toFixed(1)}%`;

    const winnerTrace = summary.traces.find(t => t.is_final_winner) || (hasWinner ? summary.traces.find(t => t.is_incumbent_best) : null);
    if (winnerTrace) {
      step2Stat.textContent = isGroundTruth
        ? `${(winnerTrace.verdict.quality_metrics.recall ?? bestQ).toFixed(2)} edge recall`
        : `${(winnerTrace.verdict.quality_metrics.edge_continuity ?? 1.0).toFixed(2)} continuity`;
      step3Stat.textContent = `low: ${winnerTrace.config.low_threshold}, high: ${winnerTrace.config.high_threshold}`;
      step4Stat.textContent = `${bestLat.toFixed(2)} ms | ${(bestQ * 100).toFixed(1)}% ${isGroundTruth ? "F1" : "Proxy"}`;
    } else {
      step2Stat.textContent = "Quality guardrail check";
      step3Stat.textContent = "Threshold not satisfied";
      step4Stat.textContent = `${summary.baseline_latency_ms.toFixed(2)} ms (baseline)`;
    }

    // Update Reasoning Feed
    renderReasoningFeed(summary.traces);

    // Update Comparison Images
    renderComparisonImages(summary);

    // Update Scatter Chart
    renderScatterChart(summary);

    // Update History Table
    renderHistoryTable(summary.traces);

    // Update Audit Trail Raw Log
    if (auditRawLog) {
      auditRawLog.textContent = JSON.stringify(summary, null, 2);
    }
  }

  // 6. Render Reasoning Feed
  function renderReasoningFeed(traces) {
    reasoningFeed.innerHTML = "";
    if (!traces || traces.length === 0) return;

    // Show up to 5 most recent traces in chronological sequence
    traces.slice(1).forEach((trace, idx) => {
      const item = document.createElement("div");
      const isReject = !trace.verdict.is_acceptable;
      const markerClass = isReject ? "marker-red" : (trace.is_final_winner ? "marker-cyan" : "marker-blue");

      item.className = "reasoning-item";
      item.innerHTML = `
        <div class="reasoning-marker ${markerClass}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            ${isReject
              ? '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'
              : '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>'}
          </svg>
        </div>
        <div class="reasoning-body">
          <span class="reasoning-headline">${trace.action_type.replace(/_/g, ' ')}: ${trace.diagnosis.slice(0, 50)}...</span>
          <span class="reasoning-sub">${trace.decision_rationale}</span>
        </div>
        <span class="reasoning-time">${trace.timestamp_utc.split("T")[1]?.slice(0, 8) || "14:28:32"}</span>
      `;
      reasoningFeed.appendChild(item);
    });
  }

  // Helper: Restore Winner Panel (Panel 3)
  function restoreWinnerPanel() {
    if (!currentSession) return;
    resetWinnerBtn?.classList.add("hidden");

    // Clear custom row selection in history table
    const rows = historyTableBody.querySelectorAll("tr");
    rows.forEach(r => r.classList.remove("selected-row"));

    const hasWinner = currentSession.winner_iteration !== null && currentSession.winner_iteration !== undefined && currentSession.best_config;

    if (hasWinner) {
      cardBest.classList.remove("card-no-winner");
      cardBest.classList.add("card-best-validated");
      headerBest.classList.remove("header-no-winner");
      titleBest.classList.remove("title-no-winner");
      titleBest.textContent = "BEST VALIDATED";
      badgeBest.classList.remove("badge-no-winner");
      badgeBest.innerHTML = `
        <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
        </svg>
        Selected
      `;
      viewImgBest.src = currentSession.preview_best_b64 || "";
      viewImgBest.style.display = "block";
      noWinnerPlaceholder?.classList.add("hidden");

      const w = currentSession.best_config;
      metaBest.textContent = `Canny (${w.low_threshold}, ${w.high_threshold}) + blur=${w.blur_kernel} | ${currentSession.best_latency_ms.toFixed(2)} ms`;

      // Re-highlight the winner row in history table
      if (currentSession.winner_iteration !== null && rows[currentSession.winner_iteration]) {
        rows[currentSession.winner_iteration].classList.add("selected-row");
      }
    } else {
      cardBest.classList.add("card-no-winner");
      cardBest.classList.remove("card-best-validated");
      headerBest.classList.add("header-no-winner");
      titleBest.classList.add("title-no-winner");
      titleBest.textContent = "NO VALID CANDIDATE";
      badgeBest.classList.add("badge-no-winner");
      badgeBest.innerHTML = "None";
      viewImgBest.style.display = "none";
      noWinnerPlaceholder?.classList.remove("hidden");
      if (noWinnerSub) {
        noWinnerSub.textContent = `Target quality guardrail (${modalQualityVal?.textContent || '0.70'}) was not achieved.`;
      }
      metaBest.textContent = "Baseline retained | Guardrail threshold not met";
    }
  }

  // Helper: Inspect Specific Iteration in Panel 3
  function inspectIteration(i, t) {
    if (!currentSession) return;
    const isBase = i === 0;
    const isWinner = t.is_final_winner;

    // Highlight clicked row
    const rows = historyTableBody.querySelectorAll("tr");
    rows.forEach((r, idx) => {
      r.classList.toggle("selected-row", idx === i);
    });

    if (isWinner) {
      restoreWinnerPanel();
      return;
    }

    // Show reset button so user can easily return to winner view
    resetWinnerBtn?.classList.remove("hidden");

    cardBest.classList.remove("card-no-winner");
    cardBest.classList.remove("card-best-validated");
    headerBest.classList.remove("header-no-winner");
    titleBest.classList.remove("title-no-winner");

    if (isBase) {
      titleBest.textContent = "BASELINE (INSPECTING)";
      badgeBest.classList.remove("badge-no-winner");
      badgeBest.innerHTML = "● Baseline";
    } else {
      titleBest.textContent = `ITERATION #${String(i).padStart(2, "0")} (INSPECTING)`;
      badgeBest.classList.remove("badge-no-winner");
      badgeBest.innerHTML = t.verdict.is_acceptable ? "✔ Valid" : "✖ Rejected";
    }

    viewImgBest.src = t.thumbnail_b64 || (isBase ? currentSession.preview_baseline_b64 : "");
    viewImgBest.style.display = "block";
    noWinnerPlaceholder?.classList.add("hidden");

    const cfg = t.config;
    metaBest.textContent = `Canny (${cfg.low_threshold}, ${cfg.high_threshold}) + blur=${cfg.blur_kernel} | ${t.verdict.timing_metrics.median_latency_ms.toFixed(2)} ms`;
  }

  // 7. Render Comparison Images
  function renderComparisonImages(summary) {
    viewImgOriginal.src = summary.preview_original_b64 || "";
    viewImgBaseline.src = summary.preview_baseline_b64 || "";

    if (summary.preview_reference_b64) {
      viewImgReference.src = summary.preview_reference_b64;
      viewImgReference.style.display = "block";
      noRefPlaceholder?.classList.add("hidden");
      if (titleReference) titleReference.textContent = "REFERENCE (GROUND TRUTH)";
      metaReference.textContent = "Annotated edges | Ground-truth reference";
    } else {
      viewImgReference.src = "";
      viewImgReference.style.display = "none";
      noRefPlaceholder?.classList.remove("hidden");
      if (titleReference) titleReference.textContent = "REFERENCE (UNLABELLED)";
      metaReference.textContent = "Unlabelled image | Proxy metrics active (no reference)";
    }

    const foundIdx = presets.findIndex(p => p.id === summary.image_id);
    if (foundIdx !== -1) currentPresetIndex = foundIdx;
    pagImageText.textContent = `Image ${currentPresetIndex + 1}/${Math.max(1, presets.length)}`;

    metaOriginal.textContent = `512 × 512 | ${summary.image_id}`;
    const bCfg = summary.baseline_config;
    metaBaseline.textContent = `Canny (${bCfg.low_threshold}, ${bCfg.high_threshold}) | ${summary.baseline_latency_ms.toFixed(2)} ms`;

    restoreWinnerPanel();
  }

  // 8. Render Scatter Chart (Log Latency vs Quality F1)
  function renderScatterChart(summary) {
    chartPointsGroup.innerHTML = "";
    if (!summary.traces || summary.traces.length === 0) {
      chartEmptyState?.classList.remove("hidden");
      return;
    }
    chartEmptyState?.classList.add("hidden");

    const isGroundTruth = summary.baseline_quality_type === "GROUND_TRUTH_F1" && summary.has_ground_truth !== false;
    if (chartYAxisLabel) {
      chartYAxisLabel.textContent = isGroundTruth ? "Quality (Edge F1)" : "Proxy Quality (heuristic)";
    }

    // Accurate logarithmic mapping for Latency (ms):
    // SVG x coordinates:
    // 1 ms   -> x = 45   (log10(1) = 0)
    // 10 ms  -> x = 195  (log10(10) = 1)
    // 100 ms -> x = 345  (log10(100) = 2)
    // 1000 ms-> x = 495  (log10(1000) = 3)
    // Width between decades is exactly 150 px.
    function mapX(latMs) {
      const clamped = Math.max(0.5, Math.min(1000, latMs));
      const logV = Math.max(0, Math.min(3, Math.log10(clamped)));
      return 45 + logV * 150;
    }

    // Quality mapping:
    // 1.0 -> y = 20
    // 0.0 -> y = 190
    function mapY(score) {
      const val = Math.max(0.0, Math.min(1.0, score));
      return 190 - (val * 170);
    }

    const plottedCoords = [];

    summary.traces.forEach((t, i) => {
      const lat = t.verdict.timing_metrics.median_latency_ms;
      const q = t.verdict.primary_quality_score;
      let cx = mapX(lat);
      let cy = mapY(q);

      // Handle duplicate or overlapping coordinates with subtle offset
      const collision = plottedCoords.find(p => Math.abs(p.x - cx) < 5 && Math.abs(p.y - cy) < 5);
      if (collision) {
        cx += 6;
      }
      plottedCoords.push({ x: cx, y: cy });

      const isBase = i === 0;
      const isWinner = t.is_final_winner;
      const isValid = t.verdict.is_acceptable;

      let fill = isValid ? "#00D9FF" : "#A855F7";
      if (isBase) fill = "#FFFFFF";

      // Halo ring only on the TRUE final winner
      if (isWinner) {
        const glowCircle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        glowCircle.setAttribute("cx", cx);
        glowCircle.setAttribute("cy", cy);
        glowCircle.setAttribute("r", "10");
        glowCircle.setAttribute("fill", "none");
        glowCircle.setAttribute("stroke", "#00D9FF");
        glowCircle.setAttribute("stroke-width", "2");
        glowCircle.setAttribute("opacity", "0.85");
        chartPointsGroup.appendChild(glowCircle);
      }

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", cx);
      circle.setAttribute("cy", cy);
      circle.setAttribute("r", isWinner ? "6" : (isBase ? "4.5" : "4"));
      circle.setAttribute("fill", fill);
      circle.setAttribute("cursor", "pointer");

      circle.addEventListener("mouseenter", () => {
        chartTooltip.classList.remove("hidden");
        chartTooltip.style.left = `${Math.min(370, Math.max(50, cx - 45))}px`;
        chartTooltip.style.top = `${Math.max(10, cy - 65)}px`;

        const qName = isGroundTruth ? "Edge F1" : "Proxy Quality";
        let statusText = isWinner ? "Best Validated" : (isBase ? "Baseline" : (isValid ? "Valid" : "Rejected"));
        tooltipTitle.textContent = `#0${i} (${statusText})`;
        tooltipQuality.textContent = `${qName}: ${(q * 100).toFixed(1)}%`;
        tooltipLatency.textContent = `Latency: ${lat.toFixed(2)} ms (median)`;
        tooltipConfig.textContent = `Canny (${t.config.low_threshold}, ${t.config.high_threshold}), blur=${t.config.blur_kernel}`;
      });

      circle.addEventListener("mouseleave", () => {
        chartTooltip.classList.add("hidden");
      });

      circle.addEventListener("click", () => {
        inspectIteration(i, t);
      });

      chartPointsGroup.appendChild(circle);
    });
  }

  // 9. Render History Table
  function renderHistoryTable(traces) {
    historyTableBody.innerHTML = "";
    if (!traces || traces.length === 0) return;

    const isGroundTruth = currentSession.baseline_quality_type === "GROUND_TRUTH_F1" && currentSession.has_ground_truth !== false;
    if (thQualityHeader) {
      thQualityHeader.textContent = isGroundTruth ? "EDGE F1" : "PROXY QUALITY (HEURISTIC)";
    }

    traces.forEach((t, i) => {
      const row = document.createElement("tr");
      const isBase = i === 0;
      const isWinner = t.is_final_winner;
      const isAccept = t.verdict.is_acceptable;
      const lat = t.verdict.timing_metrics.median_latency_ms;
      const f1 = t.verdict.primary_quality_score;
      const cfg = t.config;

      if (isWinner) row.classList.add("best-winner-row", "selected-row");

      let statusBadge = `<span class="table-status-pill pill-validated">✔ Validated</span>`;
      if (isBase) {
        statusBadge = `<span class="table-status-pill pill-baseline">● Baseline</span>`;
      } else if (isWinner) {
        statusBadge = `<span class="table-status-pill pill-winner">★ Best Validated</span>`;
      } else if (!isAccept) {
        statusBadge = `<span class="table-status-pill pill-rejected">✖ Rejected</span>`;
      }

      let deltaStr = "—";
      if (!isBase && currentSession.baseline_latency_ms) {
        const red = ((currentSession.baseline_latency_ms - lat) / currentSession.baseline_latency_ms) * 100;
        if (red > 0) {
          deltaStr = `<span class="delta-green">+${red.toFixed(1)}%</span>`;
        } else if (red < 0) {
          deltaStr = `<span class="delta-red">-${Math.abs(red).toFixed(1)}%</span>`;
        } else {
          deltaStr = `<span class="stat-delta">0.0%</span>`;
        }
      }

      const thumbSrc = t.thumbnail_b64 || (isBase ? currentSession.preview_baseline_b64 : (isWinner ? currentSession.preview_best_b64 : ""));
      const metricBadge = isGroundTruth ? "F1" : "Proxy";

      row.innerHTML = `
        <td class="iter-col-badge">${String(i).padStart(2, "0")}</td>
        <td><img class="table-thumb" src="${thumbSrc}" alt="thumb #${i}"></td>
        <td><span class="params-code">(${cfg.low_threshold}, ${cfg.high_threshold}), k=${cfg.aperture_size}, blur=${cfg.blur_kernel}</span></td>
        <td class="stat-latency">${lat.toFixed(2)} ms</td>
        <td class="stat-f1">${(f1 * 100).toFixed(1)}% <small style="font-size:0.68em;color:var(--text-muted);margin-left:2px">${metricBadge}</small></td>
        <td class="stat-delta">${deltaStr}</td>
        <td>${statusBadge}</td>
      `;

      row.addEventListener("click", () => {
        inspectIteration(i, t);
      });

      historyTableBody.appendChild(row);
    });
  }

  // 10. Subviews Content Generators
  function renderDatasetsGrid() {
    const grid = document.getElementById("datasetsGrid");
    grid.innerHTML = "";
    presets.forEach(p => {
      const card = document.createElement("div");
      card.className = "dataset-card";
      card.innerHTML = `
        <div class="dataset-thumb-wrap">
          <img src="/api/preset-image/${p.id}/original" alt="${p.name}">
        </div>
        <div class="dataset-info">
          <h4 class="dataset-title">${p.name}</h4>
          <span class="detail-key">Category: <b>${p.category}</b> | Dimensions: ${p.dimensions[0]}x${p.dimensions[1]}</span>
          <p class="dataset-desc">${p.has_ground_truth ? "Contains verified ground-truth reference edge map for objective F1 scoring." : "Standard static test image."}</p>
        </div>
      `;
      grid.appendChild(card);
    });
  }

  async function renderArchiveList() {
    const list = document.getElementById("archiveList");
    list.innerHTML = "<p style='color:#6C829D'>Loading session archives...</p>";
    try {
      const resp = await fetch("/api/history?limit=10");
      const data = await resp.json();
      list.innerHTML = "";
      (data.history || []).forEach(sess => {
        const item = document.createElement("div");
        item.className = "reasoning-item";
        item.style.marginBottom = "0.6rem";
        item.innerHTML = `
          <div class="reasoning-marker marker-cyan">⚡</div>
          <div class="reasoning-body">
            <span class="reasoning-headline">Session <b>${sess.session_id}</b> (${sess.image_id})</span>
            <span class="reasoning-sub">Speedup: <b>${(sess.speedup_factor || 1).toFixed(2)}x</b> | Best F1: <b>${((sess.best_quality_score || 0)*100).toFixed(1)}%</b> | ${sess.iterations_completed} Iterations</span>
          </div>
          <button class="btn-link" onclick="window.location.href='/api/report/${sess.session_id}?format=markdown'">Download Report ➔</button>
        `;
        list.appendChild(item);
      });
    } catch (e) {
      list.innerHTML = "<p>Error loading archives.</p>";
    }
  }

  function renderAuditView() {
    if (currentSession && auditRawLog) {
      auditRawLog.textContent = JSON.stringify(currentSession, null, 2);
    }
  }
});
