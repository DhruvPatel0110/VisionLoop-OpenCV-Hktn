# VisionLoop --- Project Idea, Architecture & Execution Plan

**Competition:** OpenCV AI Competition 2026, powered by AWS\
**Proposed direction:** Developer agents that work with OpenCV 5
workloads, with an optional Agentic Vision special-prize attempt\
**Working title:** VisionLoop\
**Status:** Project proposal and implementation blueprint; this document
does not claim that any component has already been built.

------------------------------------------------------------------------

## 1. Executive Summary

VisionLoop is a lightweight developer agent that autonomously tests and
improves computer-vision workloads. A user selects a supported OpenCV 5
task and supplies a sample image or uses a bundled test set. VisionLoop
runs a baseline, measures execution time and output quality, chooses a
new configuration, executes another experiment, evaluates the result,
and retains the best configuration that satisfies quality constraints.

The core value is not simply that the system runs benchmarks. It is that
**measured visual output and runtime results change the agent's
subsequent action**. If an experiment is too slow, produces poor output,
or fails validation, the agent changes its next test rather than blindly
following a fixed script.

The project is designed for CPU-only development on a low-end laptop. It
uses small static images and classical computer-vision operations; it
does not require a local GPU or cloud GPU. A meaningful component must
run on AWS to satisfy the competition's baseline requirement. AWS will
host a bounded CPU workload or experiment-execution component, not a GPU
inference service.

Two optional architecture extensions are planned: - **RAG:** Retrieve
relevant documentation, previous experiment results, and troubleshooting
notes to inform experiment selection. - **A2A and Agent Cards:** Allow a
coordinator to discover and delegate work to small specialist agents,
such as a vision-experiment agent and a result-evaluation agent. - **MCP
(optional):** Expose selected vision and experiment operations as tools
if it can be completed without putting the core deliverable at risk.

These extensions are secondary. The priority order is: working OpenCV 5
workload → genuine feedback loop → meaningful AWS component →
reproducible evaluation and demo → optional RAG/A2A/MCP polish.

## 2. Competition Fit and Non-Negotiable Requirements

### Official competition requirements

According to the official Devpost competition page and rules:

1.  Every entry must use **OpenCV 5 for substantive image or video
    analysis**.
2.  Every entry must run a **meaningful component on AWS**.
3.  Final submission must include a technical report describing the
    problem, users, architecture, OpenCV 5 implementation, AWS
    deployment, evaluation, limitations, and responsible-use
    considerations.
4.  Code must be available in a public repository or a private
    repository/archive that judges can access.
5.  Dependencies must be pinned, with clear build, deployment, and test
    instructions.
6.  An architecture diagram must show OpenCV 5 and AWS components, plus
    agent/COOL components where relevant.
7.  The submission needs a working web endpoint or an arranged live
    screen-share demonstration.
8.  A judge-accessible public or unlisted demo video must be no longer
    than five minutes.
9.  Evaluation evidence should include failure cases and limitations.

Official sources: - Competition page: https://opencv26.devpost.com/ -
Official rules: https://opencv26.devpost.com/rules

### Relevant suggested project area

The closest suggested area is **developer agents that integrate COOL
with tools such as Codex, Claude Code, Kiro, or MCP-compatible
systems**. The proposal therefore treats the agent as a developer tool
that iteratively runs, evaluates, and adjusts a real OpenCV workload.

The competition also describes an Agentic Vision special award. To make
a credible attempt at that award, visual output must affect a subsequent
plan, tool call, action, or request for human approval. A chatbot that
only describes a fixed result is insufficient.

### COOL clarification

COOL means Cloud-Optimized OpenCV Library. COOL is not automatically
required for every overall-award entry, but the named developer-agent
suggested area mentions integrating COOL, and the Best Use of COOL
special award has stricter requirements. If time and compatibility
permit, VisionLoop should test COOL on a qualifying AWS Graviton/Arm
path. The project must not claim the COOL award unless the core workload
demonstrably executes with COOL on the qualifying Arm environment and a
reproducible baseline comparison is supplied.

If COOL integration proves impractical before the deadline, preserve the
valid core project: OpenCV 5 + iterative developer agent + meaningful
AWS execution. Do not falsely describe a standard OpenCV run as a COOL
run.

## 3. Problem Statement

Computer-vision developers often manually test different parameter
values, compare outputs, measure runtime, search documentation, and
repeat the process until they find an acceptable configuration. This
process can be repetitive, difficult to reproduce, and vulnerable to
choosing a faster configuration that damages output quality.

VisionLoop aims to automate a bounded version of this workflow: -
establish a baseline; - propose a candidate configuration; - run a real
OpenCV 5 workload; - inspect visual and runtime results; - accept or
reject the candidate based on measurable criteria; - decide what to test
next; - preserve experiment history and evidence.

The project is an experimental developer-assistance tool. It is not
intended to guarantee globally optimal parameters or replace expert
review.

## 4. Target Users and Use Case

**Primary users:** Students, developers, and computer-vision
practitioners who need to compare configurations for a small
image-processing task.

**Primary use case:** A developer wants to find a faster configuration
for a supported OpenCV operation without violating a defined
output-quality requirement.

**Example:** A user chooses edge detection and supplies a sample image
set. The baseline is measured. VisionLoop tests alternative parameters.
A candidate that is faster but loses too many relevant edges is
rejected. The agent uses that evidence to choose another candidate. At
the end, the user receives the best validated configuration found within
the experiment budget, together with the measurements and failed
candidates.

## 5. Product Functionality

### 5.1 Task and image selection

-   Let the user select from a small, predefined list of supported
    OpenCV 5 tasks.
-   Allow upload of a static image or use of a bundled test image set.
-   Display the input image and selected task.
-   Enforce image-size and file-type limits to keep execution bounded
    and safe.

### 5.2 Baseline execution

-   Run the selected OpenCV operation using a documented baseline
    configuration.
-   Save the output image.
-   Record elapsed time, input dimensions, parameter values, and
    relevant runtime metadata.
-   Establish a baseline quality measurement using a reference output or
    an appropriate task-specific metric.

### 5.3 Autonomous experiment loop

-   Generate a candidate configuration from a bounded, preapproved
    parameter space.
-   Execute the OpenCV 5 operation.
-   Inspect output and measurements.
-   Compare the candidate with the baseline and prior experiments.
-   Decide whether to accept, reject, retry, or stop.
-   Change the next configuration based on evidence from the latest
    experiment.
-   Stop after a configured experiment budget or when no useful
    candidate remains.

This feedback loop is the project's central technical contribution. The
sequence must not merely be a fixed list of parameter values:
observations from an experiment must influence a later action.

### 5.4 Quality guardrails

-   Define a minimum acceptable quality threshold before testing.
-   Do not select a candidate solely because it is faster.
-   Reject candidates that violate quality constraints or produce
    invalid outputs.
-   Display the trade-off between speed and quality.
-   Report when no candidate improves on the baseline.

### 5.5 Experiment history and audit trail

For every run, preserve: - experiment identifier and timestamp; - input
dataset identifier and image dimensions; - operation and parameter
configuration; - runtime measurement and measurement conditions; -
output-quality metric; - validation status; - reason for
accepting/rejecting the candidate; - the next action selected by the
agent.

### 5.6 Results dashboard

The interface should show: - baseline and selected result; -
before/after output images; - experiment timeline; - candidate
configurations and status; - runtime and quality comparison; - failure
reasons; - final validated configuration; - a downloadable report (JSON
or Markdown is sufficient for the MVP).

### 5.7 AWS-backed execution

-   Send a bounded experiment request to a CPU-based AWS component.
-   Execute the meaningful OpenCV 5 workload on AWS.
-   Return output metadata and results to the UI.
-   Log errors and execution status.
-   Set limits for image size, number of experiments, runtime, and
    request frequency.
-   Monitor AWS billing and stop unused resources.

The precise AWS service should be selected after verifying OpenCV 5 and
COOL compatibility, deployment constraints, available credits, and
current pricing. Do not assume that every service is covered by
promotional credits.

## 6. Example End-to-End User Journey

1.  The user opens VisionLoop.
2.  The user selects a task such as edge detection and uploads a small
    image set.
3.  VisionLoop runs a baseline configuration.
4.  The system records the baseline runtime and quality score.
5.  The coordinator asks the experiment agent for a candidate
    configuration.
6.  The experiment agent runs the OpenCV 5 operation.
7.  The evaluation component compares the candidate output and
    measurements against the acceptance criteria.
8.  If the candidate is faster and remains within the quality threshold,
    it may be retained as the best result.
9.  If it is faster but damages output quality, it is rejected; the next
    experiment is changed accordingly.
10. The system records the decision and the reason for it.
11. The loop repeats within a strict experiment budget.
12. The user sees the baseline, candidate history, final selected
    configuration, and limitations in the dashboard.
13. The final report allows the result to be reproduced.

## 7. Proposed OpenCV Workload

Start with **one** well-understood, CPU-friendly task. Do not implement
many tasks before the feedback loop works.

Possible first task: **edge detection on static images**, using a
supported OpenCV 5 implementation and a small set of tunable parameters.

Potential additional tasks, only if time permits: - thresholding; -
blur/filter parameters; - contour detection; - image resizing or color
conversion for performance-only experiments.

For each task, define a meaningful quality measurement. Where a
labelled/reference output is available, use a suitable comparison
metric. If no reference exists, use a carefully explained proxy and
include human inspection. Do not claim a quality metric proves
correctness if it does not.

Use a small, fixed, legally usable dataset. Keep test images consistent
between candidate runs so comparisons are fair. Include different image
characteristics if possible, such as lighting, contrast, and noise.

## 8. Agent Design

### 8.1 Coordinator / Orchestrator

Responsibilities: - accept the user task and constraints; - determine
which capability is needed; - maintain the experiment budget and current
best candidate; - delegate execution and evaluation; - choose whether to
continue, retry, stop, or ask for human review; - produce the final
experiment summary.

### 8.2 Vision Experiment Agent

Responsibilities: - run an approved OpenCV 5 operation; - accept only
supported parameter configurations; - return the output image and
measurements; - report execution errors in a structured form.

### 8.3 Evaluation Agent / Evaluator

Responsibilities: - compare output against reference or acceptance
criteria; - check whether the result is valid; - evaluate speed-quality
trade-offs; - return pass/fail and a reason; - flag cases that need
human review.

For the deadline MVP, the evaluator may be deterministic Python logic
rather than an LLM. This improves reproducibility and avoids paid
inference.

### 8.4 Optional Research/RAG component

Responsibilities: - retrieve relevant documentation or past
experiments; - provide source references with retrieved guidance; -
suggest candidate configurations to test; - avoid presenting retrieved
suggestions as proven until a real experiment validates them.

Do not allow retrieved text to execute arbitrary commands or override
safety limits.

## 9. RAG Plan

RAG is an enhancement, not a prerequisite for the core competition
submission.

### Knowledge sources

-   relevant OpenCV 5 documentation and API notes;
-   locally stored troubleshooting notes;
-   prior experiment records and benchmark summaries;
-   validated configuration examples;
-   known failure cases and explanations.

### Retrieval approach

Start with lightweight keyword/BM25 retrieval over local Markdown/JSON
files. This avoids a mandatory embedding API, GPU, or large local model.
If time permits, a small local vector index can be evaluated, but it is
not necessary for the MVP.

### RAG workflow

1.  Receive the current task and latest experiment outcome.
2.  Retrieve documentation and previous experiments relevant to the
    operation, parameters, or failure.
3.  Pass retrieved context to the decision component.
4.  Propose a candidate configuration.
5.  Run the actual OpenCV workload.
6.  Evaluate the measured result.
7.  Save the new evidence to experiment history.

### RAG evaluation

-   Use a small set of known questions or failure scenarios.
-   Check whether retrieval returns relevant documentation/history.
-   Check that advice is traceable to retrieved sources.
-   Compare experiment outcomes with and without retrieval where
    feasible.
-   Never claim that RAG improves optimization unless results support
    that claim.

## 10. A2A and Agent Cards Plan

A2A is an optional extension. It should be added only after a reliable
local experiment loop exists.

### Proposed agents

-   Coordinator Agent;
-   Vision Experiment Agent;
-   Evaluation Agent.

### Agent Cards

Each independently addressable agent should advertise its identity,
description, endpoint, skills/capabilities, and applicable interaction
requirements according to the chosen A2A implementation and protocol
version.

Example capability descriptions: - Vision Experiment Agent: run
supported OpenCV tasks, test bounded parameter configurations, return
images and measurements. - Evaluation Agent: compare outputs with
reference metrics, validate quality thresholds, and return a structured
evaluation. - Coordinator: plan experiments, delegate work, manage stop
conditions, and produce reports.

### A2A workflow

1.  Coordinator discovers or reads available agent capabilities.
2.  Coordinator assigns a bounded experiment task to the vision agent.
3.  Vision agent returns structured output and runtime evidence.
4.  Coordinator sends the result to the evaluator.
5.  Evaluator returns a verdict and reason.
6.  Coordinator uses that verdict to choose the next action.
7.  Task and experiment status are recorded.

The A2A integration must be genuine agent-to-agent task exchange, not
just a diagram or renamed local function call. If protocol setup
threatens the core deliverable, implement the agents as internal modules
first and document A2A as a future extension rather than making
unsupported claims.

## 11. MCP Plan

MCP is optional. If implemented, expose a small set of useful tools,
such as: - `run_vision_experiment`; - `get_experiment_result`; -
`compare_experiments`; - `retrieve_experiment_knowledge`; -
`get_runtime_status`.

The agent should invoke these tools as part of the loop. MCP is for
standardized tool/resource access; A2A is for communication and task
delegation between agents. Avoid implementing both protocols merely for
appearance.

## 12. AWS Architecture and Cost Controls

### Minimum architecture

-   Web UI: upload task, progress, results, and report.
-   Local development/test environment: small CPU-based OpenCV 5
    workloads.
-   AWS execution component: a meaningful CPU-based image-processing
    workload.
-   Result interface: structured response containing status,
    configuration, metrics, and output reference.
-   Logging: record experiment identifiers, execution status, runtime,
    and errors.

### AWS service selection

Choose a deployment only after confirming: - OpenCV 5 package/build
compatibility; - runtime and memory limits; - whether the selected
service can run the workload appropriately; - whether COOL can be
supported if pursuing that path; - current regional availability and
pricing; - whether promotional credits apply to the selected services.

A small container/service or suitable serverless component may be
considered, but do not assume either is automatically the cheapest or
compatible option.

### Budget safeguards

-   Prefer CPU-only execution.
-   Use small images and short runs.
-   Limit experiment count and execution time.
-   Do not leave unused instances running.
-   Set AWS budgets/alerts where available, while remembering that
    alerts are not necessarily hard spending caps.
-   Verify promotional-credit eligibility and expiry.
-   Do not enter a paid plan or provision paid resources without
    understanding the possible charge.
-   Keep a local fallback for development, but ensure the final
    submission still demonstrates meaningful AWS execution.

## 13. Evaluation and Evidence

The submission should prove the system works, rather than relying on
claims.

### Core functional tests

-   Baseline workload runs successfully.
-   Agent can execute more than one candidate configuration.
-   A measured result changes a later decision.
-   Candidate rejection works when quality falls below threshold.
-   The system handles invalid inputs and execution errors.
-   The loop stops at its experiment budget.
-   Final selected configuration can be rerun.
-   AWS component returns a valid result.
-   Dashboard results match recorded experiment data.

### Quantitative evaluation

Record: - baseline runtime; - candidate runtime; - speed ratio or
percentage change; - quality metric for baseline and candidate; - number
of candidates tested; - proportion accepted/rejected; - successful
completion rate; - errors and recovery behaviour; - cloud execution time
and cost estimate, where measurable.

Use repeated runs and consistent input data. Report measurement
conditions and variability. Avoid claiming a speedup if differences are
within measurement noise.

### Agentic Vision evidence

If pursuing the Agentic Vision special award, include: - a diagram of
perception → decision/orchestration → action; - a trace showing visual
output or runtime measurements changing a later tool
call/configuration; - task-success and failure-handling evaluation; -
observability and human-control safeguards; - a running image-processing
workload in the demo.

### COOL evidence

If pursuing Best Use of COOL, additionally include: - COOL version; -
AWS Graviton instance/deployment configuration; - proof the claimed core
workload executes through COOL on the qualifying Arm path; - baseline
methodology and inputs; - reproducible latency/throughput/resource/cost
measurements; - x86/Arm/hybrid architecture details if applicable.

Do not claim special-award eligibility without meeting its specific
requirements.

## 14. Responsible Use and Security

-   Process only user-provided or appropriately licensed images.
-   Avoid collecting personal images unless necessary; do not retain
    uploads longer than needed.
-   Limit file types and image dimensions.
-   Do not execute arbitrary code supplied by users or retrieved from
    RAG.
-   Restrict agent actions to approved operations and bounded
    parameters.
-   Use least-privilege AWS credentials and never expose secrets in
    frontend code or logs.
-   Add execution timeouts and limits on image size, experiment count,
    and resource use.
-   Keep human review available for ambiguous or failed evaluations.
-   Clearly label proxy metrics and limitations.
-   Report negative results honestly; never fabricate performance
    improvements.

## 15. Main Risks and Mitigations

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  OpenCV 5 installation or package    Verify the exact version early;
  compatibility problems              record environment and build
                                      instructions.

  AWS deployment takes too long       Prove one minimal CPU workload on
                                      AWS before expanding the UI or
                                      agent features.

  COOL cannot be integrated in time   Keep the core OpenCV 5 + AWS
                                      project valid; do not misrepresent
                                      standard OpenCV as COOL.

  Agent is merely a fixed script      Ensure experiment results alter
                                      subsequent configurations/actions
                                      and show the trace.

  Faster output is lower quality      Enforce explicit quality
                                      constraints and reject degraded
                                      candidates.

  RAG adds complexity without value   Start with simple local retrieval;
                                      include only if it helps select or
                                      explain experiments.

  A2A integration consumes the        Build the loop first; add
  deadline                            protocol-based delegation only
                                      after the core is proven.

  Cloud charges                       Use bounded CPU runs, verify
                                      service pricing/credits, and shut
                                      down unused resources.

  Results are not reproducible        Pin dependencies, fix
                                      datasets/configurations, log
                                      metrics, and provide a rerun
                                      command.

  Deadline pressure                   Prioritize a working end-to-end
                                      demo over extra protocols and
                                      visual polish.
  -----------------------------------------------------------------------

## 16. Implementation Procedure and Priorities

### Phase 0 --- Verify the foundation

1.  Confirm the official submission deadline and account/registration
    status.
2.  Create a clean repository and environment.
3.  Verify OpenCV 5 is actually available and the selected operation
    runs.
4.  Record the exact package/build/version.
5.  Run one sample image end to end.

**Exit condition:** A reproducible OpenCV 5 output exists.

### Phase 1 --- Build the baseline and evaluator

1.  Select one operation and a small fixed test set.
2.  Define the baseline parameters.
3.  Define runtime and quality measurements.
4.  Save output and metadata.
5.  Test invalid input and error handling.

**Exit condition:** Baseline and evaluation results can be reproduced.

### Phase 2 --- Implement the autonomous loop

1.  Define a bounded set of candidate parameters.
2.  Run a candidate.
3.  evaluate runtime and output quality;
4.  accept or reject it;
5.  select the next experiment based on the result;
6.  enforce a maximum experiment budget;
7.  preserve a full decision trace.

**Exit condition:** At least one demonstrated case where a result
changes the next experiment.

### Phase 3 --- Prove AWS execution

1.  Choose a suitable CPU-based AWS deployment.
2.  Deploy the smallest meaningful OpenCV 5 component.
3.  send a bounded test request;
4.  return a result to the application;
5.  log execution and errors;
6.  verify cost and shutdown procedures.

**Exit condition:** The demo proves a meaningful component actually runs
on AWS.

### Phase 4 --- Build the minimum dashboard

1.  Add task/image selection.
2.  Display progress and experiment timeline.
3.  Display baseline/candidate outputs.
4.  Show runtime and quality comparisons.
5.  Show final result and limitations.
6.  Connect the UI to actual backend data.

**Exit condition:** A judge can understand the system without reading
the source code first.

### Phase 5 --- Optional RAG/A2A/MCP

1.  Add local keyword retrieval over documentation and history if it
    helps the agent.
2.  Add two specialist agents with real task/result exchange if time
    permits.
3.  Publish Agent Cards and verify discovery/communication if
    implementing A2A.
4.  Add MCP tools only if they are genuinely used.
5.  Re-run the end-to-end tests.

**Exit condition:** Optional components improve the workflow without
destabilizing the core.

### Phase 6 --- Submission readiness

1.  Pin dependencies.
2.  Write setup, deployment, and test instructions.
3.  Add the architecture diagram.
4.  Capture benchmark tables and failed cases.
5.  Write the technical report.
6.  Record a demo video of no more than five minutes.
7.  Verify repository access for judges.
8.  Verify every required link and artifact.
9.  Submit before the official deadline.

## 17. Realistic Scope Under a Tight Deadline

A rough focused-work estimate for a minimal end-to-end MVP is **14--30
hours**, but this is not guaranteed. OpenCV 5 compatibility, AWS
permissions, and deployment problems can increase it.

With a very short remaining window, prioritize: 1. OpenCV 5 workload; 2.
real autonomous experiment loop; 3. meaningful AWS execution; 4.
measurable evaluation and trace; 5. minimal dashboard and documentation.

RAG should be a small local retrieval feature if time permits. A2A and
MCP should be stretch goals, not blockers. A polished multi-agent
architecture without a working vision feedback loop is a weaker
submission than a smaller system with reproducible evidence.

## 18. Suggested Five-Minute Demo

1.  **0:00--0:30 --- Problem and product:** Explain the manual tuning
    problem and VisionLoop's purpose.
2.  **0:30--1:10 --- Baseline:** Select an image and run the initial
    OpenCV 5 workload.
3.  **1:10--2:30 --- Agentic loop:** Show a candidate result, its
    evaluation, and how it changes the next configuration.
4.  **2:30--3:20 --- Comparison:** Show runtime, output quality,
    rejected candidate(s), and the final validated result.
5.  **3:20--4:00 --- AWS proof:** Show the meaningful AWS component
    executing and returning results.
6.  **4:00--4:35 --- RAG/A2A (only if implemented):** Show retrieval
    evidence or agent task exchange.
7.  **4:35--5:00 --- Results and limitations:** State measured outcomes
    honestly and acknowledge what is not yet supported.

The video must show actual working behaviour; do not use fabricated
metrics or imply that planned features already exist.

## 19. Submission Checklist

### Core

-   [ ] OpenCV 5 performs substantive image analysis.
-   [ ] Agent runs and evaluates multiple configurations.
-   [ ] Results alter subsequent decisions/actions.
-   [ ] Quality guardrails prevent selecting a faster but unacceptable
    output.
-   [ ] Meaningful component runs on AWS.
-   [ ] End-to-end demonstration works.
-   [ ] Failure cases are documented.
-   [ ] Measurements are reproducible.

### Documentation and packaging

-   [ ] Technical report completed.
-   [ ] Architecture diagram completed.
-   [ ] Dependencies pinned.
-   [ ] Build/deployment/test instructions included.
-   [ ] Repository or archive accessible to judges.
-   [ ] Demo video is five minutes or less.
-   [ ] Final submission completed before the official cutoff.

### Optional claims

-   [ ] RAG is backed by an actual retrieval demonstration.
-   [ ] A2A uses real agent communication and Agent Cards.
-   [ ] MCP tools are actually invoked by the workflow.
-   [ ] COOL award is claimed only with qualifying COOL-on-Arm evidence.
-   [ ] Agentic Vision award is pursued only with qualifying
    perception--decision--action evidence.

## 20. Official References

-   OpenCV AI Competition 2026: https://opencv26.devpost.com/
-   Official rules and judging rubrics:
    https://opencv26.devpost.com/rules
-   Competition overview and suggested project areas:
    https://opencv26.devpost.com/?ref_medium=homepage-recommended-hackathons

Always re-check the official Devpost page for the current cutoff time,
required fields, and rule updates before submitting.

------------------------------------------------------------------------

## Final Recommendation

Build **VisionLoop as a small, evidence-driven autonomous developer
agent**. The core is a real OpenCV 5 workload whose measured outputs
change the next experiment. AWS execution, reproducible evaluation, and
clear evidence are mandatory priorities. RAG can add knowledge
retrieval; A2A and Agent Cards can add interoperable specialist agents;
MCP can expose tools. Keep all of those extensions subordinate to a
reliable end-to-end demonstration, and do not claim capabilities or
award eligibility that have not been verified.
