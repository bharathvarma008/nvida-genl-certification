# NCP-GENL Intensive 2-Week Study Plan

## Overview
- **Exam**: NCP-GENL (NVIDIA Certified Professional - Generative AI LLMs)
- **Format**: 60-70 questions, 120 minutes, online proctored
- **Passing Score**: ~70% (approximately 46/65 questions)
- **Study Duration**: 14 days (3.5-5 hours/day = 50+ total hours)
- **Target**: Pass with flying colors (>80% on practice exams)

## Daily Schedule Structure

### Morning Session (1.5-2 hours)
- Concept review and reading
- Watch video resources
- Study official documentation

### Afternoon Session (1.5-2 hours)
- Practice questions (30-50 questions)
- Hands-on exercises where applicable
- Review incorrect answers

### Evening Session (30min-1 hour)
- Quick review of day's topics
- Flashcards
- Plan next day's focus

---

## Week 1: Foundation Building (Days 1-7)

### Day 1: Blueprint + Model Optimization (17%)
**Focus**: Exam overview + Highest weight domain

**Morning (2h)**:
- Read NCP-GENL official page and exam blueprint
- Review domain breakdown and weightings
- Study Model Optimization domain: Quantization (FP32→FP16→INT8→INT4)
- Review TensorRT-LLM documentation

**Afternoon (2h)**:
- Practice 30-40 questions on Model Optimization
- Review TensorRT-LLM features: graph fusion, kernel auto-tuning
- Study KV cache optimization and decoding strategies

**Evening (1h)**:
- Create flashcards for quantization trade-offs
- Review incorrect answers
- Quick reference: Model Optimization cheat sheet

**Resources**:
- [NCP-GENL Official Page](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/)
- [TensorRT-LLM](https://developer.nvidia.com/tensorrt-llm)
- [Preporato NCP-GENL Guide](https://preporato.com/certifications/nvidia/generative-ai-llm-professional/articles/nvidia-ncp-genl-certification-complete-guide)
- [NCP-GENL Practice Tests](https://www.youtube.com/watch?v=8OCYDMuT9QA)

---

### Day 2: GPU Acceleration & Distributed Training (14%)
**Focus**: Second highest weight domain

**Morning (2h)**:
- Study single-GPU performance: tensor cores, mixed precision
- Review parallelism types: data, tensor, pipeline
- Study memory planning and VRAM optimization
- Review NVIDIA GPU architecture (A100, H100, DGX systems)

**Afternoon (2h)**:
- Practice 30-40 questions on GPU acceleration
- Study profiling techniques: identifying bottlenecks
- Review multi-node training: NCCL, all-reduce, communication overhead
- Draw diagrams for different parallelism strategies

**Evening (1h)**:
- Flashcards: Parallelism types and use cases
- Review incorrect answers
- Quick reference: GPU acceleration patterns

**Resources**:
- [NVIDIA Gen AI LLM Recommended Training PDF](https://learning.dell.com/content/dam/dell-emc/documents/en-english/NVIDIA%20Gen%20AI%20LLM%20Recommended%20Training%20-%20Ex.pdf)
- [Coursera NCA-GENL Specialization](https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate)

---

### Day 3: Model Optimization Deep Dive + Review
**Focus**: Complete Model Optimization domain

**Morning (2h)**:
- Review pruning and distillation techniques
- Study sampling strategies: beam search vs sampling (temperature, top-k, top-p)
- Review TensorRT-LLM quantization flows and throughput gains
- Study activation vs weight quantization trade-offs

**Afternoon (2h)**:
- Practice 40-50 questions on Model Optimization
- Review Day 1-2 weak areas
- Complete any hands-on exercises if possible

**Evening (1h)**:
- Review all Model Optimization flashcards
- Update quick reference guide
- Plan Day 4 focus

---

### Day 4: Prompt Engineering (13%)
**Focus**: High-weight domain, practical application

**Morning (2h)**:
- Study prompt patterns: zero/one/few-shot, chain-of-thought
- Review system vs user vs tool messages
- Study output control: JSON-only, schemas, delimiters
- Review domain adaptation via prompts vs fine-tuning

**Afternoon (2h)**:
- Practice 35-45 questions on Prompt Engineering
- Design prompts for 5 enterprise scenarios
- Study prompting in RAG: retrieval prompts, context windows
- Review self-consistency and advanced prompting techniques

**Evening (1h)**:
- Flashcards: Prompt patterns and use cases
- Review incorrect answers
- Quick reference: Prompt Engineering patterns

**Resources**:
- [Coursera Prompt Engineering Modules](https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate)
- [Prompt Engineering Practice Test](https://www.youtube.com/watch?v=MMIRBvarP-U)
- [NCP-GENL Passing Strategy](https://www.youtube.com/watch?v=FyhVbF5sNsU)

---

### Day 5: Fine-Tuning (13%)
**Focus**: High-weight domain, practical techniques

**Morning (2h)**:
- Study full fine-tuning vs PEFT (LoRA/QLoRA)
- Review cost and VRAM implications
- Study core hyperparameters: LR, warmup, batch size, early stopping
- Review instruction tuning vs domain adaptation vs safety tuning

**Afternoon (2h)**:
- Practice 35-45 questions on Fine-Tuning
- Study NeMo fine-tuning tutorials
- Review avoiding catastrophic forgetting via data mixing
- Sketch full FT vs LoRA vs QLoRA recipes

**Evening (1h)**:
- Flashcards: Fine-tuning techniques and hyperparameters
- Review incorrect answers
- Quick reference: Fine-tuning comparison table

**Resources**:
- [NVIDIA NeMo](https://developer.nvidia.com/nvidia-nemo)
- [Coursera Fine-Tuning Modules](https://www.coursera.org/specializations/exam-prep-nca-genl-nvidia-certified-generative-ai-llms-associate)

---

### Day 6: Data Preparation (9%) + Model Deployment (9%)
**Focus**: Two medium-weight domains

**Morning (2h)**:
- Study Data Preparation: pipelines, tokenization (BPE/WordPiece), vocab management
- Review data cleaning, de-duplication, filtering for quality/safety
- Study RAG data prep: chunking, overlap, metadata
- Review dataset splits: pre-train vs fine-tune vs eval

**Afternoon (2h)**:
- Study Model Deployment: Triton Inference Server
- Review Triton model repo, configs, dynamic batching
- Study NIM microservices: packaging, routing, scaling
- Review containerization, CI/CD, rollout strategies

**Evening (1h)**:
- Practice 30-40 questions covering both domains
- Flashcards: Deployment patterns and data prep steps
- Review incorrect answers

**Resources**:
- [NVIDIA Study Guide](https://studylib.net/doc/28069466/nvidia-genai-and-llms)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [NVIDIA NIM Overview](https://www.nvidia.com/en-us/ai/nim/)

---

### Day 7: Mock Exam 1 + Analysis
**Focus**: Baseline assessment

**Morning (2h)**:
- Take full mock exam (60-70 questions, 120 minutes)
- Simulate exam conditions: quiet space, timed, no distractions
- Use practice question banks or YouTube practice tests

**Afternoon (2h)**:
- Analyze mock exam results by domain
- Identify weak areas (<70% accuracy)
- Create remediation plan for weak domains
- Review all incorrect answers with explanations

**Evening (1h)**:
- Update progress dashboard
- Plan Week 2 focus based on mock results
- Review exam strategy guide

**Resources**:
- [NCP-GENL Practice Test Video](https://www.youtube.com/watch?v=8OCYDMuT9QA)
- ExamTopics, Reddit 300-Q bank, FlashGenius quizzes

---

## Week 2: Intensive Practice + Remediation (Days 8-14)

### Day 8: Evaluation (7%) + Production Monitoring (7%)
**Focus**: Two lower-weight but important domains

**Morning (2h)**:
- Study Evaluation metrics: perplexity, ROUGE, BLEU, BERTScore
- Review offline vs online evaluation
- Study human evaluation: rubrics, pairwise comparisons
- Review regression suites and test harnesses

**Afternoon (2h)**:
- Study Production Monitoring: latency, tokens/s, error rates
- Review SLOs, SLIs, alerting, incident response
- Study drift detection and anomaly detection
- Review safe rollout: canary, blue-green, shadow

**Evening (1h)**:
- Practice 30-40 questions on both domains
- Flashcards: Evaluation metrics and monitoring metrics
- Review incorrect answers

**Resources**:
- [NVIDIA Study Guide - Evaluation](https://studylib.net/doc/28069466/nvidia-genai-and-llms)

---

### Day 9: LLM Architecture (6%) + Safety/Ethics (5%)
**Focus**: Complete remaining domains

**Morning (2h)**:
- Study LLM Architecture: transformer blocks, attention mechanisms
- Review positional encodings: absolute vs rotary (RoPE)
- Study scaling: depth vs width, parameter count vs context length
- Review decoder-only vs encoder-decoder architectures

**Afternoon (2h)**:
- Study Safety & Ethics: guardrails, content filtering
- Review PII detection & redaction, logging policies
- Study bias and toxicity mitigation
- Review regulatory compliance (GDPR-like rules)

**Evening (1h)**:
- Practice 30-40 questions on both domains
- Flashcards: Architecture components and safety measures
- Review incorrect answers

**Resources**:
- [Whizlabs NCA-GENL Guide - Safety](https://www.whizlabs.com/blog/nvidia-certified-associate-generative-ai-llms/)

---

### Day 10: Weak Area Remediation Day 1
**Focus**: Target weakest domains from Mock Exam 1

**Morning (2h)**:
- Deep dive into weakest domain #1
- Review all resources for that domain
- Study concepts in detail
- Create targeted practice plan

**Afternoon (2h)**:
- Practice 40-50 questions on weakest domain #1
- Review all incorrect answers with detailed explanations
- Study related concepts and edge cases

**Evening (1h)**:
- Review flashcards for weak domain
- Update quick reference guide
- Plan Day 11 remediation

---

### Day 11: Weak Area Remediation Day 2
**Focus**: Target weakest domains #2 and #3

**Morning (2h)**:
- Deep dive into weakest domain #2
- Review resources and concepts
- Study domain #3 if time permits

**Afternoon (2h)**:
- Practice 40-50 questions on weak domains #2 and #3
- Review incorrect answers
- Cross-reference with other domains

**Evening (1h)**:
- Review all weak domain flashcards
- Update progress dashboard
- Plan final review days

---

### Day 12: Full Review + Practice Marathon
**Focus**: Comprehensive review of all domains

**Morning (2h)**:
- Review quick reference guide for all domains
- Review all flashcards
- Study exam strategy and time management

**Afternoon (3h)**:
- Practice question marathon: 60-70 questions
- Mix all domains
- Practice under timed conditions (2 min/question)
- Flag difficult questions for review

**Evening (1h)**:
- Review all incorrect answers
- Update weak area list
- Final review of exam strategy

---

### Day 13: Final Review + Strategy
**Focus**: Exam readiness

**Morning (2h)**:
- Review all domain quick references
- Review exam strategy guide
- Study common NVIDIA-specific terminology
- Review answer patterns and best practices

**Afternoon (2h)**:
- Practice 50-60 questions (mixed domains)
- Focus on question analysis and elimination techniques
- Practice time management

**Evening (1h)**:
- Review pre-exam checklist
- Prepare exam environment
- Review stress management techniques
- Get good rest

---

### Day 14: Final Mock Exam + Exam Day Prep
**Focus**: Final assessment and exam readiness

**Morning (2h)**:
- Take Final Mock Exam 2 (60-70 questions, 120 minutes)
- Full exam simulation
- Strict time management

**Afternoon (2h)**:
- Analyze Final Mock Exam results
- Quick review of any remaining weak areas
- Final review of exam strategy
- Confidence building

**Evening (1h)**:
- Review exam registration and logistics
- Prepare exam environment checklist
- Review exam day schedule
- Relax and prepare mentally

**Exam Day**:
- Arrive early/check tech setup
- Review quick reference one last time
- Stay calm and confident
- Apply exam strategy throughout

---

## Success Checklist

By Day 14, you should have:
- ✅ Completed 400+ practice questions
- ✅ Achieved >75% accuracy in all domains
- ✅ Scored >80% on final mock exam
- ✅ Logged 50+ study hours
- ✅ Reviewed all official resources
- ✅ Created comprehensive quick reference guide
- ✅ Identified and remediated weak areas
- ✅ Mastered exam strategy and time management

## Daily Tracking

Use the `daily-study-log-template.md` to track:
- Domains covered each day
- Resources used
- Questions attempted and accuracy
- Key learnings
- Tomorrow's focus

## Progress Monitoring

Update `progress-dashboard.md` daily to track:
- Domain mastery levels
- Practice question statistics
- Study hours logged
- Mock exam scores
- Weak areas identified

Good luck! You've got this! 🚀
