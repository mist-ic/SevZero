# SevZero — SRE Incident Response Environment

A reinforcement learning environment where AI agents act as autonomous on-call Site Reliability Engineers managing microservice clusters undergoing cascading failures.

Built with [OpenEnv](https://github.com/meta-pytorch/OpenEnv) for the **OpenEnv AI Hackathon 2026**.

## Why SRE Incident Response?

Incident response is one of the most expensive and error-prone aspects of running production systems. Engineers must rapidly diagnose root causes from noisy signals, contain blast radius, and restore service health — often under 3 AM pressure. SevZero provides a realistic simulation environment for training and evaluating AI agents on this critical task.

The environment models:
- **Realistic microservice topologies** with typed service layers (edge, identity, business, infrastructure)
- **Cascading failures** driven by queueing theory (Little's Law, M/M/c approximation, retry amplification)
- **Circuit breaker state machines** (CLOSED → OPEN → HALF_OPEN → CLOSED)
- **8 failure types** weighted by real-world incident data (config errors 32%, bad deploys 25%, cascading latency 15%, crashes 10%, resource leaks 8%, DB degradation 5%, cache failures 3%, network errors 2%)
- **Framework-specific log patterns** from Spring Boot, Node.js, FastAPI, Kubernetes, HikariCP, Redis, and gRPC

## Tasks

| Task | Services | Steps | Failures | Description |
|------|----------|-------|----------|-------------|
| **Easy** | 3–5 | 10 | 1 | Single service outage in a linear chain. Diagnose and fix within 10 steps. |
| **Medium** | 8–15 | 20 | 2–3 | Cascading failure from shared infrastructure through a branching dependency graph. |
| **Hard** | 15–30 | 50 | 4–6 | Multiple simultaneous root causes with conflicting mitigations across a complex mesh topology. |

All scenarios are procedurally generated from a seed for full determinism.

## Action Space

The agent can issue 11 action types via `{"action_type": "...", "params": {...}}`:

| Action | Parameters | Effect |
|--------|-----------|--------|
| `inspect_logs` | `service_id` | View recent logs for a service (free action) |
| `inspect_metrics` | `service_id` | View metric history for a service (free action) |
| `inspect_traces` | `service_id` | View distributed traces through a service (free action) |
| `restart_service` | `service_id` | Restart a service (fixes crashes, resource leaks) |
| `rollback_service` | `service_id` | Roll back to previous version (fixes bad deploys) |
| `scale_service` | `service_id`, `replicas` | Scale horizontally (helps with load) |
| `tune_config` | `service_id`, `key`, `value` | Update configuration (fixes config errors) |
| `clear_cache` | `cache_name` | Flush a cache service |
| `rebalance_traffic` | `from_region`, `to_region`, `pct` | Shift traffic between regions |
| `pause_job` | `job_name` | Pause a background job |
| `noop` | — | Do nothing, advance one tick |

Remediation actions have 1–4 tick delays before taking effect. Inspect actions are free (no tick cost beyond the step).

## Observation Space

Observations are ordered by SRE triage priority:

- **Episode context**: `tick`, `episode_id`, `task_id`, `status`, `max_steps`
- **Health summary**: `global_slo_score` (0.0–1.0), `observation_summary`
- **Per-service state**: `services[]` — each with `id`, `layer`, `status`, `error_rate`, `latency_p50/p95/p99_ms`, `throughput_rps`, `cpu_pct`, `memory_pct`, `connection_pool_usage_pct`, `replicas`, `version`, `depends_on`, `circuit_breakers`
- **Active alerts**: sorted by severity (`critical` > `warning` > `info`)
- **Context**: `recent_deploys`, `actions_taken` (history of agent's actions and outcomes)
- **Action space**: `legal_actions` with valid targets for each action type
- **Diagnostic output**: `logs`, `metric_history`, `traces` (populated after `inspect_*` actions)

## Grading

Episodes are scored deterministically on a 0.0–1.0 scale:

```
score = slo_recovery × 0.70 + action_efficiency × 0.15 + time_efficiency × 0.15
```

- **SLO Recovery (70%)**: Final global SLO score across all services
- **Action Efficiency (15%)**: Ratio of effective actions to total actions (penalizes excessive inspection without remediation)
- **Time Efficiency (15%)**: How quickly the agent resolves the incident relative to the step budget

A +10% bonus is applied when the episode terminates with full resolution (all failures remediated, SLO = 1.0).

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

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

Requires an LLM API endpoint:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token-here"
export ENV_URL="http://localhost:7860"

uv run python inference.py
```

### Validate OpenEnv Compliance

```bash
uv run openenv validate
```

### Docker

```bash
docker build -t sevzero .
docker run -p 7860:7860 sevzero
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws` | WebSocket | OpenEnv evaluation protocol (primary) |
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment with `{"task_id": "easy", "seed": 42}` |
| `/step` | POST | Execute action with `{"action": {"action_type": "...", "params": {...}}}` |
| `/state` | GET | Current environment state |
| `/tasks` | GET | List available tasks |
| `/grader` | POST | Score an episode |
| `/docs` | GET | Interactive API documentation |

## Architecture

```
inference.py          ← Baseline LLM agent (OpenAI client)
server/
  app.py              ← FastAPI app + stateful HTTP routes
  environment.py      ← OpenEnv Environment subclass (reset/step/state)
  simulator.py        ← Discrete-event simulation engine
  propagation.py      ← Queueing theory cascade engine + circuit breakers
  failures.py         ← 8 failure types with temporal metric signatures
  scenarios.py        ← Procedural scenario generation (3 difficulty tiers)
  graph.py            ← Service topology generation
  logs.py             ← Framework-specific log templates
  traces.py           ← Distributed trace generation
models.py             ← Pydantic API contract (Action, Observation, State)
```

The simulator runs a tick-based loop: each step, failures evolve their metric signatures, propagation cascades through the dependency graph via queueing theory, pending remediation effects resolve after their delay, and the agent receives an updated observation.

## License

MIT
