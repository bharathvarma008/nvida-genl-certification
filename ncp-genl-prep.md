# NCP-GENL Preparation – Domains, Resources, and 10-Day Plan

## 0. Core official / anchor links

- NVIDIA NCP-GENL (Professional) official page  
  - [https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/)
- NVIDIA NCA-GENL (Associate) official page  
  - [https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/)
- Coursera specialization – Exam Prep (NCA-GENL)  
  - [https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate](https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate)
- Preporato – NCP-GENL complete guide  
  - [https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide](https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide)
- NCP-GENL YouTube – practice tests / full guide  
  - Practice tests: [https://www.youtube.com/watch?v=8OCYDMuT9QA](https://www.youtube.com/watch?v=8OCYDMuT9QA)  
  - Passing strategy: [https://www.youtube.com/watch?v=FyhVbF5sNsU](https://www.youtube.com/watch?v=FyhVbF5sNsU)  
  - 2026 overview: [https://www.youtube.com/watch?v=5R3sRkf6P7M](https://www.youtube.com/watch?v=5R3sRkf6P7M)

---

## 1. NCP-GENL Domains Table


| Domain                                  | % of exam | Official-style description                                                                                                                                 |
| --------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM Architecture                        | 6%        | Understanding and applying foundational LLM structures and mechanisms (transformer blocks, attention, positional encodings, context length, scaling laws). |
| Prompt Engineering                      | 13%       | Adapting LLMs to new domains, tasks, or distributions via prompts: zero/one/few-shot, chain-of-thought, tool use, system prompts, and output control.      |
| Data Preparation                        | 9%        | Data sourcing, cleaning, de-duplication, tokenization, vocab management, corpus curation, filtering for quality and safety.                                |
| Model Optimization                      | 17%       | Improving performance and efficiency via quantization, pruning, distillation, KV-cache tuning, sampling strategies, and TensorRT-LLM level optimizations.  |
| Fine-Tuning                             | 13%       | Full-parameter and parameter-efficient fine-tuning (LoRA/QLoRA), domain/task adaptation, alignment techniques, and evaluation of tuned models.             |
| GPU Acceleration & Distributed Training | 14%       | Single- and multi-GPU training/inference, tensor/data/pipeline parallelism, memory planning, profiling, and performance tuning on NVIDIA GPUs/DGX.         |
| Model Deployment                        | 9%        | Serving LLMs using Triton, NIM, and related services, including containers, routing, dynamic batching, versioning, and scaling.                            |
| Evaluation                              | 7%        | Offline and online evaluation: perplexity, task metrics (ROUGE, BLEU, BERTScore), human eval, A/B tests, regression tests.                                 |
| Production Monitoring & Reliability     | 7%        | Telemetry, logging, SLOs, drift detection, rollback, capacity planning, incident response for LLM services.                                                |
| Safety, Ethics & Compliance             | 5%        | Guardrails, content filtering, PII/redaction, bias and toxicity mitigation, regulatory/organizational policy compliance.                                   |


You can optionally add columns: `Planned day(s)`, `Status`, `Notes`.

---

## 2. Domains, Subtopics, and Resources

### 2.1 LLM Architecture (6%)

**Subtopics**

- Transformer mechanics: multi-head attention, feed-forward blocks, residuals, layer norm, KV cache.
- Positional encodings: absolute vs rotary (RoPE), impact on context length.
- Scaling: depth vs width, parameter count vs context length vs throughput.
- Decoder-only vs encoder-decoder; why LLMs are mostly decoder-only for text generation.

**Key resources**

- NVIDIA NCP-GENL high-level description (includes architecture expectations):  
  - [https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/)
- Preporato NCP-GENL guide – domain table & architecture content:  
  - [https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide](https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide)
- Coursera NCA-GENL specialization – LLM architecture modules (transferable concepts):  
  - [https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate](https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate)
- NCP-GENL YouTube practice / overview including architecture questions:  
  - [https://www.youtube.com/watch?v=8OCYDMuT9QA](https://www.youtube.com/watch?v=8OCYDMuT9QA)

---

### 2.2 Prompt Engineering (13%)

**Subtopics**

- Prompt patterns: zero/one/few-shot, chain-of-thought, self-consistency.
- System vs user vs tool messages; instruction-following vs completion.
- Output control: JSON-only, schemas, delimiters, content filters via prompts.
- Domain adaptation via prompts vs via fine-tuning.
- Prompting in RAG: retrieval prompts, context windows, citations.

**Key resources**

- Coursera NCA-GENL – prompt engineering content:  
  - [https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate](https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate)
- Preporato NCP-GENL guide – Prompt Engineering domain notes:  
  - [https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide](https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide)
- NCA-GENL prompt-engineering practice test (good warm-up):  
  - [https://www.youtube.com/watch?v=MMIRBvarP-U](https://www.youtube.com/watch?v=MMIRBvarP-U)
- NCP-GENL passing strategy (includes prompt examples):  
  - [https://www.youtube.com/watch?v=FyhVbF5sNsU](https://www.youtube.com/watch?v=FyhVbF5sNsU)

---

### 2.3 Data Preparation (9%)

**Subtopics**

- Data pipelines: collection, labeling, cleaning, de-duplication, filtering for offensive/PII content.
- Tokenization: BPE/WordPiece, vocab size, special tokens.
- Dataset splits: pre-train vs fine-tune vs eval; leakage & contamination.
- RAG data prep: chunking, overlap, metadata.

**Key resources**

- NVIDIA Generative AI & LLMs study guide (Associate) – data prep & trustworthy AI:  
  - [https://studylib.net/doc/28069466/nvidia-genai-and-llms](https://studylib.net/doc/28069466/nvidia-genai-and-llms)
- Coursera specialization – data preparation sections:  
  - [https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate](https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate)
- Preporato NCP-GENL – Data Preparation content:  
  - [https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide](https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide)

---

### 2.4 Model Optimization (17%)

**Subtopics**

- Quantization: FP32 → FP16/bfloat16 → INT8/INT4, activation vs weight, trade-offs.
- Pruning and distillation for smaller, faster LLMs.
- KV cache & decoding: batch size, max tokens, streaming; beam search vs sampling (temperature, top-k, top-p).
- TensorRT-LLM: graph fusion, kernel auto-tuning, quantization flows, typical throughput gains.

**Key resources**

- TensorRT-LLM product & docs (entry point):  
  - [https://developer.nvidia.com/tensorrt-llm](https://developer.nvidia.com/tensorrt-llm)
- Preporato NCP-GENL – Model Optimization + TensorRT-LLM sections:  
  - [https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide](https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide)
- NCP-GENL practice tests (optimization-heavy scenarios):  
  - [https://www.youtube.com/watch?v=8OCYDMuT9QA](https://www.youtube.com/watch?v=8OCYDMuT9QA)

---

### 2.5 Fine-Tuning (13%)

**Subtopics**

- Full fine-tune vs PEFT (LoRA/QLoRA); cost and VRAM implications.
- Core hyperparameters: LR, warmup, batch size, early stopping.
- Instruction tuning vs domain adaptation vs safety tuning.
- Avoiding catastrophic forgetting via data mixing.

**Key resources**

- NVIDIA NeMo (LLM fine-tuning tutorials):  
  - [https://developer.nvidia.com/nvidia-nemo](https://developer.nvidia.com/nvidia-nemo)  
  - (navigate to “LLM fine-tuning” docs)
- Preporato NCP-GENL – Fine-Tuning & PEFT domain:  
  - [https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide](https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide)
- Coursera specialization – fine-tuning modules:  
  - [https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate](https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate)

---

### 2.6 GPU Acceleration & Distributed Training (14%)

**Subtopics**

- Single-GPU performance: tensor cores, mixed precision, batch size vs VRAM.
- Parallelism: data, tensor, pipeline; sharding strategies.
- Profiling: identifying bottlenecks, under-utilization, memory fragmentation.
- Multi-node issues: communication overhead, NCCL, all-reduce.

**Key resources**

- NVIDIA Gen AI LLM Recommended Training (Dell PDF summarizing NVIDIA recommendations):  
  - [https://learning.dell.com/content/dam/dell-emc/documents/en-english/NVIDIA%20Gen%20AI%20LLM%20Recommended%20Training%20-%20Ex.pdf](https://learning.dell.com/content/dam/dell-emc/documents/en-english/NVIDIA%20Gen%20AI%20LLM%20Recommended%20Training%20-%20Ex.pdf)
- Preporato NCP-GENL – GPU Acceleration / Distributed Training:  
  - [https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide](https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide)
- Coursera – NVIDIA GPUs & scaling content (associate level):  
  - [https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate](https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate)

---

### 2.7 Model Deployment (9%)

**Subtopics**

- Triton Inference Server: model repo, configs, dynamic batching, concurrent models, HTTP/gRPC.
- NIM microservices: packaging models, routing, scaling, integration with apps.
- Containerization & CI/CD: Docker, GPU runtime, rollout strategies.
- Latency/throughput budgets, autoscaling.

**Key resources**

- Triton Inference Server GitHub docs:  
  - [https://github.com/triton-inference-server/server](https://github.com/triton-inference-server/server)
- NVIDIA NIM overview:  
  - [https://www.nvidia.com/en-us/ai/nim/](https://www.nvidia.com/en-us/ai/nim/)
- Preporato NCP-GENL – Deployment domain (Triton & NIM):  
  - [https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide](https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide)
- Coursera – deployment & serving modules:  
  - [https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate](https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate)

---

### 2.8 Evaluation (7%)

**Subtopics**

- Intrinsic metrics: perplexity, log-loss.
- Task metrics: ROUGE, BLEU, BERTScore, accuracy/F1.
- Human evaluation: rubrics, pairwise comparisons.
- Regression suites and test harnesses.

**Key resources**

- NVIDIA Associate study guide – Evaluation & Trustworthy AI:  
  - [https://studylib.net/doc/28069466/nvidia-genai-and-llms](https://studylib.net/doc/28069466/nvidia-genai-and-llms)
- Preporato NCP-GENL – Evaluation domain write-up:  
  - [https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide](https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide)

---

### 2.9 Production Monitoring & Reliability (7%)

**Subtopics**

- Metrics: latency, tokens/s, error rates, timeouts, cache hit rate.
- Drift & anomaly detection.
- SLOs, SLIs; alerting; incident response.
- Safe rollout: canary, blue–green, shadow.

**Key resources**

- NVIDIA NCP-GENL description (production / ops emphasis):  
  - [https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/)
- Study guide (Associate) – ops & monitoring bits:  
  - [https://studylib.net/doc/28069466/nvidia-genai-and-llms](https://studylib.net/doc/28069466/nvidia-genai-and-llms)
- Preporato NCP-GENL – Monitoring & reliability sections:  
  - [https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide](https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide)

---

### 2.10 Safety, Ethics & Compliance (5%)

**Subtopics**

- Guardrails: pre-prompt, post-generation filters, blocklists/allowlists.
- Sensitive content categories; safety policies.
- PII detection & redaction; logging policies.
- Regulatory context: GDPR-like rules, auditability.

**Key resources**

- NVIDIA Associate study guide – safety & Trustworthy AI:  
  - [https://studylib.net/doc/28069466/nvidia-genai-and-llms](https://studylib.net/doc/28069466/nvidia-genai-and-llms)
- Whizlabs NCA-GENL guide – safety/ethics overview:  
  - [https://www.whizlabs.com/blog/nvidia-certified-associate-generative-ai-llms/](https://www.whizlabs.com/blog/nvidia-certified-associate-generative-ai-llms/)
- Preporato NCP-GENL – safety-related content in Data/Eval/Safety week:  
  - [https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide](https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide)

---

