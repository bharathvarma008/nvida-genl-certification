# NCP-GENL Flashcard Topics

## How to Use This Guide
Create flashcards (physical or digital) for each topic below. Review flashcards daily during evening sessions (30min-1h). Focus on topics you're weak on.

**Flashcard Format Suggestion**:
- **Front**: Question or concept name
- **Back**: Definition, explanation, key points, formulas

---

## 1. LLM Architecture (6%)

### Transformer Components
- **Multi-Head Attention**: What is it? How does it work?
- **Feed-Forward Blocks**: Structure and purpose
- **Residual Connections**: Why are they important?
- **Layer Normalization**: Pre-norm vs post-norm
- **Self-Attention**: Q, K, V matrices and scaled dot-product
- **KV Cache**: What is it? Why is it used?

### Positional Encodings
- **Absolute Positional Encoding**: How does it work?
- **Rotary Positional Encoding (RoPE)**: Advantages over absolute
- **Impact on Context Length**: How RoPE enables longer contexts

### Architecture Types
- **Decoder-Only Architecture**: Characteristics, examples (GPT)
- **Encoder-Decoder Architecture**: Characteristics, examples (BART, T5)
- **Why Decoder-Only for LLMs**: Reasons for popularity

### Scaling Laws
- **Depth vs Width**: Trade-offs
- **Parameter Count**: Impact on performance
- **Context Length**: Memory and speed implications
- **Throughput**: Factors affecting it

---

## 2. Prompt Engineering (13%)

### Prompt Patterns
- **Zero-Shot**: Definition and use cases
- **One-Shot**: Definition and use cases
- **Few-Shot**: Definition and typical number of examples
- **Chain-of-Thought (CoT)**: What is it? When to use?
- **Self-Consistency**: How does it work?

### Message Types
- **System Message**: Purpose and characteristics
- **User Message**: Purpose and characteristics
- **Tool/Function Messages**: When are they used?

### Output Control
- **JSON-Only Output**: How to enforce it
- **Delimiters**: Purpose and examples
- **Content Filters**: Pre vs post-generation
- **Temperature**: Effect on output (low vs high)

### Domain Adaptation
- **Via Prompts**: Pros and cons
- **Via Fine-Tuning**: Pros and cons
- **When to Use Each**: Decision criteria

### RAG Prompting
- **Retrieval Prompts**: Purpose and design
- **Context Windows**: How much context to include
- **Citations**: How to include source references

---

## 3. Data Preparation (9%)

### Data Pipeline Steps
- **Collection**: Methods and sources
- **Cleaning**: Common techniques
- **De-duplication**: Why and how
- **Filtering**: What to filter (offensive, PII, low-quality)
- **Labeling**: When needed

### Tokenization
- **BPE (Byte Pair Encoding)**: How it works, used by GPT
- **WordPiece**: How it works, used by BERT
- **Vocab Size**: Typical ranges (30K-50K)
- **Special Tokens**: Common ones (`<BOS>`, `<EOS>`, `<PAD>`, `<UNK>`)

### Dataset Splits
- **Pre-train Split**: Purpose and size
- **Fine-tune Split**: Purpose and size
- **Eval/Test Split**: Purpose and importance
- **Leakage**: What is it? How to prevent?

### RAG Data Prep
- **Chunking**: Typical sizes (256-512 tokens)
- **Overlap**: Typical percentage (10-20%)
- **Metadata**: What to store (source, position, embeddings)

---

## 4. Model Optimization (17%)

### Quantization Types
- **FP32**: Baseline precision
- **FP16**: Memory reduction (2x), speed gain (1.5-2x)
- **INT8**: Memory reduction (4x), speed gain (2-3x)
- **INT4**: Memory reduction (8x), speed gain (3-4x)
- **Weight Quantization**: What is quantized?
- **Activation Quantization**: More complex, when used?
- **PTQ (Post-Training Quantization)**: When used?
- **QAT (Quantization-Aware Training)**: When used?

### TensorRT-LLM Features
- **Graph Fusion**: What is it?
- **Kernel Auto-Tuning**: Purpose
- **Quantization Flows**: Automated pipelines
- **Typical Throughput Gains**: 2-5x speedup

### Pruning & Distillation
- **Pruning**: What is removed? Benefits?
- **Distillation**: How does it work? Benefits?

### Decoding Strategies
- **Beam Search**: How it works, pros/cons
- **Sampling**: How it works, pros/cons
- **Temperature**: Effect (low = deterministic, high = creative)
- **Top-k Sampling**: How it works
- **Top-p (Nucleus) Sampling**: How it works

### KV Cache Optimization
- **Batch Size**: Impact on memory and utilization
- **Max Tokens**: Impact on memory
- **Streaming**: How it works

---

## 5. Fine-Tuning (13%)

### Fine-Tuning Types Comparison
- **Full Fine-Tune**: Parameters updated, VRAM, speed, use case
- **LoRA**: Parameters updated, VRAM, speed, use case
- **QLoRA**: Parameters updated, VRAM, speed, use case

### LoRA Details
- **Rank (r)**: Typical values (4-16)
- **Alpha**: Scaling factor (typically 2× rank)
- **Target Modules**: Usually attention layers (Q, K, V, O)

### QLoRA Details
- **4-bit Quantization**: What is quantized?
- **LoRA on Top**: Trained in what precision?
- **Memory Reduction**: ~75% vs full fine-tune

### Hyperparameters
- **Learning Rate**: Typical ranges (1e-5 to 5e-4)
- **Warmup**: Typical percentage (3-10% of steps)
- **Batch Size**: Typical ranges (1-8 for LoRA)
- **Epochs**: Typically 1-5 sufficient
- **Early Stopping**: When to use?

### Fine-Tuning Goals
- **Instruction Tuning**: Purpose
- **Domain Adaptation**: Purpose
- **Safety Tuning**: Purpose

### Avoiding Catastrophic Forgetting
- **Data Mixing**: How to do it?
- **Lower Learning Rate**: Why?
- **Regularization**: What constraints?

---

## 6. GPU Acceleration & Distributed Training (14%)

### Single-GPU Optimization
- **Tensor Cores**: What are they? Which GPUs have them?
- **Mixed Precision**: FP16 training with FP32 master weights
- **Batch Size vs VRAM**: Trade-offs
- **Gradient Accumulation**: How to simulate larger batches?

### Parallelism Types
- **Data Parallelism**: How it works, use case, communication
- **Tensor Parallelism**: How it works, use case, communication
- **Pipeline Parallelism**: How it works, use case, communication

### Memory Optimization
- **Gradient Checkpointing**: Trade-off (compute vs memory)
- **Offloading**: What is moved to CPU?
- **ZeRO**: What does it partition?

### Profiling
- **Bottlenecks**: What to look for?
- **Tools**: NVIDIA Nsight, PyTorch Profiler
- **Metrics**: GPU utilization, memory usage, communication overhead

### Multi-Node Training
- **NCCL**: What is it?
- **All-Reduce**: What does it do?
- **Communication Overhead**: When is it a bottleneck?
- **Scaling Efficiency**: Typical ranges (70-90%)

---

## 7. Model Deployment (9%)

### Triton Inference Server
- **Model Repository**: Structure
- **Model Config**: `config.pbtxt` purpose
- **Dynamic Batching**: `max_batch_size`, `preferred_batch_size`, `max_queue_delay`
- **Concurrent Models**: How to run multiple?
- **Protocols**: HTTP REST, gRPC, C API

### NIM (NVIDIA Inference Microservices)
- **Packaging**: How are models packaged?
- **Routing**: How does load balancing work?
- **Scaling**: Auto-scaling based on what?
- **Integration**: How easy is it?

### Containerization
- **Docker**: Standard format
- **GPU Runtime**: `nvidia-docker` or `--gpus` flag
- **Base Images**: `nvcr.io/nvidia` official images

### Rollout Strategies
- **Blue-Green**: How does it work?
- **Canary**: How does gradual rollout work?
- **Shadow**: How does testing work without user impact?

### Performance Targets
- **Latency**: P50, P95, P99 percentiles
- **Throughput**: Requests/second, tokens/second
- **Autoscaling**: Based on queue depth, latency

---

## 8. Evaluation (7%)

### Intrinsic Metrics
- **Perplexity**: Formula, interpretation, typical values (10-50)
- **Log-Loss**: What is it? Lower is better

### Task Metrics
- **ROUGE-L**: Longest common subsequence
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **BLEU**: N-gram precision, range (0-1)
- **BERTScore**: Semantic similarity using BERT
- **Accuracy/F1**: For classification tasks

### Human Evaluation
- **Rubrics**: Structured criteria
- **Pairwise Comparison**: How it works
- **Likert Scales**: Rating scales

### Regression Testing
- **Test Harnesses**: Automated suites
- **Baseline Comparison**: Compare against what?
- **A/B Testing**: How to compare models in production?

---

## 9. Production Monitoring & Reliability (7%)

### Key Metrics
- **Latency**: P50, P95, P99 percentiles
- **Throughput**: Tokens/second, requests/second
- **Error Rate**: Failed requests / total requests
- **Timeout Rate**: Requests exceeding timeout
- **Cache Hit Rate**: Percentage of cached responses

### SLOs & SLIs
- **SLO (Service Level Objective)**: Target for reliability (e.g., 99.9% uptime)
- **SLI (Service Level Indicator)**: Measured metric (e.g., actual uptime)
- **Error Budget**: Allowable failures before SLO violation

### Drift Detection
- **Data Drift**: Input distribution changes
- **Concept Drift**: Relationship between input/output changes
- **Model Drift**: Model performance degrades over time

### Incident Response
- **Alerting**: When to set up alerts?
- **Rollback**: How to revert to previous version?
- **Capacity Planning**: Plan for what?

### Safe Rollout Strategies
- **Canary**: Gradual rollout percentages (5% → 25% → 50% → 100%)
- **Blue-Green**: Instant switch
- **Shadow**: Test without user impact

---

## 10. Safety, Ethics & Compliance (5%)

### Guardrails
- **Pre-Prompt Filters**: Check what? When?
- **Post-Generation Filters**: Check what? When?
- **Blocklists**: What to block?
- **Allowlists**: What to allow?

### Content Categories
- **Violence**: Physical harm, threats
- **Hate Speech**: Discriminatory content
- **Sexual Content**: Explicit material
- **Self-Harm**: Suicide, self-injury content

### PII (Personally Identifiable Information)
- **Detection**: How to identify PII?
- **Redaction**: How to remove/mask PII?
- **Logging Policies**: Don't log PII, anonymize data

### Bias & Toxicity
- **Bias Detection**: Identify what?
- **Toxicity Mitigation**: Reduce what?
- **Fairness**: Ensure equal treatment across groups

### Regulatory Compliance
- **GDPR**: EU data protection regulation
- **Auditability**: Log decisions, maintain records
- **Right to Explanation**: Explain model decisions
- **Data Retention**: Policies for storage/deletion

---

## NVIDIA-Specific Terminology

### Hardware
- **DGX**: NVIDIA's AI supercomputer systems
- **A100**: GPU architecture
- **H100**: Latest GPU architecture
- **NVLink**: High-speed GPU interconnect

### Software/Tools
- **CUDA**: Parallel computing platform
- **TensorRT**: Inference optimization SDK
- **TensorRT-LLM**: LLM-specific TensorRT optimizations
- **NeMo**: Framework for building conversational AI
- **Triton**: Inference server for production
- **NIM**: NVIDIA Inference Microservices
- **NCCL**: NVIDIA Collective Communications Library

---

## Quick Decision Trees (Flashcard Format)

### When to Use Quantization?
- **INT8**: When? (Good balance of speed/quality)
- **INT4**: When? (Maximum compression, acceptable quality loss)
- **FP16**: When? (Minimal quality loss, good speedup)

### When to Use Fine-Tuning?
- **Full FT**: When? (Major domain shift, sufficient resources)
- **LoRA**: When? (Task adaptation, limited resources)
- **QLoRA**: When? (Very limited resources, still need adaptation)

### When to Use Parallelism?
- **Data Parallelism**: When? (Model fits on single GPU, need larger batches)
- **Tensor Parallelism**: When? (Model too large for single GPU)
- **Pipeline Parallelism**: When? (Very long models, sequential processing)

### When to Use Deployment Strategy?
- **Triton**: When? (Need flexible batching, multiple models)
- **NIM**: When? (Need easy integration, microservices architecture)
- **Direct**: When? (Simple use case, single model)

---

## Formula Reference Cards

### Perplexity
```
Perplexity = exp(cross_entropy_loss)
Lower is better, typical values: 10-50
```

### ROUGE-L
```
ROUGE-L = LCS(reference, candidate) / length(reference)
LCS = Longest Common Subsequence
```

### BLEU
```
BLEU = BP × exp(Σ log(p_n))
BP = brevity penalty
p_n = n-gram precision
Range: 0-1, higher is better
```

### Memory Estimation (Approximate)
```
Model Memory ≈ (parameters × bytes_per_param) + activations + optimizer_states
```

---

## Study Tips for Flashcards

### Daily Review
- **Evening Session**: Review 20-30 flashcards (30min-1h)
- **Focus on Weak Areas**: Prioritize domains you're struggling with
- **Active Recall**: Try to recall answer before flipping card

### Spaced Repetition
- **New Cards**: Review daily until mastered
- **Mastered Cards**: Review every 2-3 days
- **Weak Cards**: Review daily until improved

### Organization
- **By Domain**: Group cards by the 10 domains
- **By Difficulty**: Separate easy/medium/hard cards
- **By Status**: New/Mastered/Need Review

### Review Schedule
- **Week 1**: Focus on creating cards for all domains
- **Week 2**: Focus on reviewing weak area cards
- **Days 12-13**: Review all cards
- **Day 14**: Quick review before exam

---

## Flashcard Creation Checklist

### For Each Domain:
- [ ] Architecture/Components cards created
- [ ] Key concepts cards created
- [ ] Comparison cards created (e.g., LoRA vs Full FT)
- [ ] Formula cards created (if applicable)
- [ ] Decision tree cards created
- [ ] NVIDIA-specific terminology cards created

### Total Target:
- **Minimum**: 50-60 flashcards (5-6 per domain)
- **Comprehensive**: 100+ flashcards (10+ per domain)
- **Focus**: Quality over quantity - ensure you understand each card

---

**Start Creating Flashcards**: Day 1  
**Daily Review**: Days 1-14 (evening sessions)  
**Final Review**: Days 12-14

*Use flashcards to reinforce key concepts and improve retention*
