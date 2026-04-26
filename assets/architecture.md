# Architecture diagram (Mermaid)

Use this as the editable source. GitHub and Hugging Face render the same Mermaid subset as `README.md`.

```mermaid
flowchart TB
  subgraph LLM[Agent]
    P[Llama-3.1-8B + LoRA]
  end
  API[HTTP / OpenEnv API]
  subgraph Core[SevZero core]
    SIM[Simulator + propagation + grader]
  end
  subgraph R2[Round 2 modules]
    SD[Schema drift\nmiddleware on inspect_*]
    GOV[Oversight\nhigh-impact action gate]
    CUR[Adversarial curriculum\ndifficulty / budget / topology]
  end
  P <--> API
  API <--> SIM
  API <--> SD
  API <--> GOV
  API <--> CUR
  SD -.-> SIM
  GOV -.-> SIM
  CUR -.-> SIM
```

**Narration line:** the agent only sees HTTP; the simulator is the world model; R2 injects non-stationarity (drift), safety (oversight), and harder scenarios (curriculum) without breaking determinism of a fixed seed for the same code version.
