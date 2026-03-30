---
title: SevZero
emoji: 🔥
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
---

# SevZero - SRE Incident Response Environment

**An RL environment where your agent is the on-call engineer: diagnose and fix cascading cloud failures before the system collapses.**

A reinforcement learning benchmark where AI agents must act as autonomous Site Reliability Engineers (SREs) -- the people responsible for keeping production systems running. The agent observes a live microservice cluster, reads alerts and logs, and issues remediation commands to restore service health before the incident escalates.

Built with [OpenEnv](https://github.com/meta-pytorch/OpenEnv) for the **OpenEnv AI Hackathon 2026**.

---

## Why This Exists

Most RL benchmarks (Atari, MuJoCo, MiniGrid) train agents on perception and motor control. They do not train agents to reason causally under partial observability, manage multi-step diagnostic workflows, or handle high-stakes irreversible actions with delayed consequences.

SRE incident response requires all of these things simultaneously:

- **Partial observability**: The agent cannot see root causes directly -- only noisy symptoms (error rates, latency spikes, memory graphs, log lines)
- **Causal reasoning**: Fixing the wrong service first can cascade into a wider outage
- **Delayed consequences**: A service restart takes multiple simulation ticks to take effect
- **Time pressure**: The scoring penalty for slow resolution compounds every step

This directly mirrors the skills needed to deploy autonomous agents in real infrastructure. SRE agents are already being built and deployed in production by major cloud providers -- SevZero provides a safe, reproducible simulation environment to develop and benchmark them.

| Property | Games (Atari, MuJoCo) | SevZero |
|---|---|---|
| Observability | Full or structured | Genuinely partial (logs, metrics, alerts only) |
| Action consequences | Immediate (next frame) | Delayed (restarts take multiple ticks) |
| Causal structure | Fixed physics | Dynamic service dependency graphs |
| Reward signal | Synthetic game score | SLO compliance, MTTR, action efficiency |
| Procedural diversity | Fixed levels | Infinite graph topologies from seed |
| Domain transfer | Game-only | Maps directly to production systems |

---

## Quick Start

```bash
git clone https://github.com/mist-ic/SevZero.git
cd SevZero
uv sync  # or: pip install -e .
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Then interact via the HTTP API:

```python
import httpx

# Start a new episode
resp = httpx.post("http://localhost:7860/reset", json={"task_id": "easy", "seed": 42})
obs = resp.json()["observation"]
print(f"SLO: {obs['global_slo_score']:.0%} | Summary: {obs['observation_summary']}")

# Take an action
resp = httpx.post("http://localhost:7860/step", json={
    "action": {"action_type": "inspect_logs", "params": {"service_id": "order-service"}}
})
print(resp.json()["observation"]["logs"])
```

Or connect using the Python client:

```python
from client import SevZeroEnv
from models import SevZeroAction

with SevZeroEnv(base_url="http://localhost:7860") as env:
    result = env.reset(task_id="medium", seed=123)
    action = SevZeroAction(action_type="inspect_logs", params={"service_id": "auth-service"})
    result = env.step(action)
    print(result.observation.logs)
```

---

## Tasks

| Task | Services | Steps | Simultaneous Failures | Description |
|------|----------|-------|-----------------------|-------------|
| **easy** | 3-5 | 10 | 1 | Single service outage in a linear chain. One root cause, straightforward diagnosis. |
| **medium** | 8-15 | 20 | 2-3 | Cascading failure through a branching dependency graph. Fixing the wrong service first makes things worse. |
| **hard** | 15-30 | 50 | 4-6 | Multi-region incident with simultaneous independent root causes. Requires correctly prioritizing across regions while managing propagation. |

All scenarios are procedurally generated from a seed. Same seed always produces the same incident. Different seeds produce structurally distinct topologies.

---

## Episode Trace

A concrete walkthrough of one medium-difficulty episode (seed=123):

```
EPISODE START  |  Task: medium  |  Max Steps: 20  |  Seed: 123
SLO: 50%  |  Services: auth-service [CRITICAL], api-gateway [DEGRADED], 6 healthy

Step 0 - Observation
  Alerts:
    [CRITICAL] auth-service: error_rate=94%, p99=4800ms
    [WARNING]  api-gateway: error_rate=28%, p99=1200ms (downstream effect)
  Recent deploys: auth-service -> v2.1.3 (2 ticks ago)

Step 1 - Action: inspect_logs(service_id="auth-service")
  Logs reveal: "NullPointerException in UserSessionManager.validate()"
               "Caused by: incompatible schema in v2.1.3 migration"
  Reward: 0.0 (diagnostic step, free)

Step 2 - Action: rollback_service(service_id="auth-service")
  Effect queued: auth-service rollback to v2.1.2 (resolves in 2 ticks)
  Reward: +0.35 (correct root cause action)

Step 4 - auth-service rollback completes
  auth-service: CRITICAL -> HEALTHY (error_rate: 94% -> 1%)
  api-gateway: DEGRADED -> HEALTHY (upstream dependency restored)
  SLO: 50% -> 100%

Step 4 - Action: inspect_logs(service_id="api-gateway")
  Logs: nominal, confirming recovery
  Reward: +0.10 (verification)

EPISODE END  |  Score: 0.9325  |  SLO Recovery: 100%  |  Steps: 4/20
Termination: resolved
```

The key dynamic: `api-gateway` was degraded not because of its own failure, but because `auth-service` (a dependency) was down. A naive agent that restarts `api-gateway` first wastes steps, gets penalized on action efficiency, and delays resolution.

---

## Action Space

The agent issues one action per step as `{"action_type": "...", "params": {...}}`:

| Action | Parameters | Description |
|--------|-----------|-------------|
| `inspect_logs` | `service_id` | Read recent log lines for a service (diagnostic, no tick cost) |
| `inspect_metrics` | `service_id` | View metric history: CPU, memory, error rate, latency over last 10 ticks |
| `inspect_traces` | `service_id` | View distributed traces showing call paths and error spans |
| `restart_service` | `service_id` | Restart a crashed or leaking service (resolves in 1-2 ticks) |
| `rollback_service` | `service_id` | Roll back to previous version (resolves in 2-3 ticks) |
| `scale_service` | `service_id`, `replicas` | Scale replicas horizontally (helps with overload, resolves in 1 tick) |
| `tune_config` | `service_id`, `key`, `value` | Fix a misconfigured key (resolves in 1 tick) |
| `clear_cache` | `cache_name` | Flush a cache (resolves in 1 tick) |
| `rebalance_traffic` | `from_region`, `to_region`, `pct` | Shift traffic between regions |
| `pause_job` | `job_name` | Pause a background job consuming resources |
| `noop` | (none) | Advance one tick without acting |

Inspect actions are free (they add information without consuming a remediation step). Remediation actions have 1-4 tick delays before effects appear.

---

## Observation Space

Each step returns a structured observation ordered by SRE triage priority:

```python
{
  # Episode context
  "tick": 4,
  "episode_id": "3a4b...",
  "task_id": "medium",
  "status": "playing",          # "playing" | "resolved" | "timeout"
  "max_steps": 20,

  # Health summary (read this first)
  "global_slo_score": 0.72,     # 0.0 (all down) to 1.0 (all healthy)
  "observation_summary": "Tick 4/20: SLO compliance 72% (1 CRITICAL, 2 DEGRADED, 5 healthy)",

  # Per-service state
  "services": [{
    "id": "auth-service",
    "layer": "identity",        # edge | identity | business | data | infrastructure
    "status": "critical",       # healthy | degraded | critical | down
    "error_rate": 0.94,
    "latency_p50_ms": 320.0,
    "latency_p95_ms": 1800.0,
    "latency_p99_ms": 4800.0,
    "throughput_rps": 45.2,
    "cpu_pct": 88.0,
    "memory_pct": 76.0,
    "connection_pool_usage_pct": 95.0,
    "replicas": 2,
    "version": "v2.1.3",
    "depends_on": ["postgres-primary"],
    "circuit_breakers": {"api-gateway": "OPEN"}
  }],

  # Active alerts (sorted by severity)
  "alerts": [{"severity": "critical", "message": "auth-service error_rate=94%", ...}],

  # Context
  "recent_deploys": [{"service": "auth-service", "version": "v2.1.3", "ticks_ago": 2}],
  "actions_taken": [{"tick": 1, "action": "inspect_logs", "target": "auth-service", "success": true}],

  # What actions are currently valid
  "legal_actions": [{"action_type": "rollback_service", "valid_targets": ["auth-service"]}, ...],

  # Populated after inspect_* actions
  "logs": "ERROR auth-service NullPointerException in UserSessionManager...",
  "metric_history": {...},
  "traces": {"spans": [...]}
}
```

---

## Reward Function

```
score = slo_recovery x 0.70 + action_efficiency x 0.15 + time_efficiency x 0.15
```

- **SLO Recovery (70%)**: Final global SLO score (fraction of services meeting their error rate / latency targets). A +10% bonus applies when the episode ends with full resolution (SLO = 1.0, all failures cleared).
- **Action Efficiency (15%)**: Penalizes excessive actions. Ratio of minimum required actions to actual actions taken.
- **Time Efficiency (15%)**: Penalizes slow resolution. Based on how many steps were used relative to the step budget.

The reward is dense across the episode (delta-SLO shaping at each tick), not just binary at the end. An agent that partially fixes one of three failures gets partial credit proportional to the SLO improvement.

---

## Failure Types

Eight failure types, each with a distinct diagnostic signature:

| Failure Type | Log Pattern | Correct Fix |
|---|---|---|
| Bad deploy | NullPointerException / TypeError after recent deploy | `rollback_service` |
| Config error | "Configuration diagnostic: key 'X' has invalid value" | `tune_config` with the exact key |
| OOM / crash | OOMKilled, CrashLoopBackOff | `restart_service` |
| Resource leak | Memory climbing linearly over 10+ ticks | `restart_service` |
| DB degradation | HikariPool exhaustion, slow queries (CPU paradoxically low) | `scale_service` on the DB or `restart_service` |
| Cache failure | CLUSTERDOWN, "cache miss rate 100%" | `clear_cache` |
| Cascade | High latency on upstream causes downstream error spikes | Fix the upstream root cause first |
| Network | DNS resolution failures, connection timeouts | `rebalance_traffic` |

Failures propagate through the service dependency graph using queueing theory (Little's Law, M/M/c approximation). Circuit breakers (CLOSED -> OPEN -> HALF_OPEN -> CLOSED) dampen propagation with 1-2 tick delay.

---

## Baseline Scores

Baseline agent: `llama-3.3-70b-versatile` via Groq (zero-shot, greedy, no fine-tuning).

| Task | Score | SLO Recovery | Action Efficiency | Time Efficiency | Steps | Outcome |
|------|-------|-------------|-------------------|-----------------|-------|---------|
| easy | 0.9300 | 1.0000 | 0.8333 | 0.7000 | 3/10 | resolved |
| medium | 0.9325 | 1.0000 | 0.7500 | 0.8000 | 4/20 | resolved |
| hard | 0.7906 | 0.8800 | 0.9000 | 0.2640 | 50/50 | timeout |
| **avg** | **0.8844** | **0.9600** | **0.8278** | **0.5880** | | |

Full results: `outputs/baseline_latest.json`

The easy and medium tasks are consistently resolved (SLO = 100%). The hard task requires correctly diagnosing and resolving 4-6 simultaneous failures across a multi-region topology within 50 steps -- a 70B model reaches 88% SLO recovery but runs out of steps before full resolution.

---

## Setup

### Install

```bash
git clone https://github.com/mist-ic/SevZero.git
cd SevZero
uv sync
```

### Run the Server

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Tests

```bash
uv run pytest tests/ -v
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your-groq-api-key"
export ENV_URL="http://localhost:7860"

uv run python inference.py
# Results saved to outputs/baseline_latest.json
```

### Docker

```bash
docker build -t sevzero .
docker run -p 7860:7860 sevzero
```

### Validate OpenEnv Compliance

```bash
uv run openenv validate
uv run openenv validate --url http://localhost:7860
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws` | WebSocket | Primary evaluation protocol (used by OpenEnv framework) |
| `/reset` | POST | Start episode: `{"task_id": "easy", "seed": 42}` |
| `/step` | POST | Execute action: `{"action": {"action_type": "...", "params": {...}}}` |
| `/state` | GET | Current episode state (task_id, seed, SLO score, step count) |
| `/tasks` | GET | List available tasks with metadata |
| `/grader` | POST | Score a completed episode |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

---

## Architecture

```
inference.py          -- Baseline LLM agent (OpenAI-compatible client)
client.py             -- SevZeroEnv(EnvClient) for programmatic access
models.py             -- Pydantic API contract: SevZeroAction, SevZeroObservation, SevZeroState
server/
  app.py              -- FastAPI app wired via OpenEnv create_app() + custom routes
  environment.py      -- SevZeroEnvironment(Environment): reset/step/state
  simulator.py        -- Tick-based discrete-event engine
  propagation.py      -- Queueing theory cascade engine + circuit breakers
  failures.py         -- 8 failure types with temporal metric evolution curves
  scenarios.py        -- Procedural scenario generation (3 difficulty tiers)
  graph.py            -- Service topology: layered DAG with typed service roles
  logs.py             -- Framework-specific log templates (Spring, Node, FastAPI, Redis, gRPC)
  traces.py           -- Distributed trace generation
  grader.py           -- Deterministic SLO-based scoring
tests/                -- 37 tests: simulator determinism, grader bounds, propagation, actions
```

The simulator runs a tick-based loop: at each step, active failures evolve their metric signatures, propagation cascades through the dependency graph via queueing theory, pending remediation effects resolve after their delay, and the agent receives an updated observation.

---

## Design Decisions

**Determinism**: All randomness uses `random.Random(seed)` exclusively. Same seed always produces the same incident topology, failure sequence, and metric evolution. No numpy, no OS entropy.

**Queueing theory cascades**: Propagation uses Little's Law (L = lambda x W), utilization rho = L/T, and latency multiplier 1/(1-rho). This means near-saturated services (rho > 0.9) experience nonlinear latency explosion -- realistic behavior seen in production systems.

**Circuit breakers**: Services implement the standard CLOSED -> OPEN -> HALF_OPEN state machine. When a dependency fails, circuit breakers trip after a threshold, dampening further propagation. This prevents instant full-cluster collapse and gives agents meaningful time windows to diagnose and act.

**Distinctive failure signatures**: Each failure type has a unique temporal metric pattern designed to require log inspection to diagnose correctly (for example: cascading latency spikes p99 before errors appear; resource leaks show linear memory growth over 10+ ticks; DB degradation shows CPU paradoxically low due to I/O wait).

**What this environment does not model**: Actual network I/O, real containerized services, or multi-agent coordination. Service graphs are simulated, not real. The environment is designed for benchmarking agent decision-making, not as a digital twin.
