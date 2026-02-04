# NCP-GENL Quick Reference Guide

## Exam Overview
- **Questions**: 60-70
- **Duration**: 120 minutes
- **Passing Score**: ~70% (46/65 questions)
- **Format**: Multiple choice, online proctored

---

## 1. LLM Architecture (6%)

### Transformer Components
- **Multi-Head Attention**: Parallel attention mechanisms, allows model to focus on different aspects
- **Feed-Forward Blocks**: Two linear transformations with activation (ReLU/GELU)
- **Residual Connections**: Skip connections prevent vanishing gradients
- **Layer Normalization**: Normalizes inputs to each layer (pre-norm vs post-norm)

### Attention Mechanisms
- **Self-Attention**: Q (query), K (key), V (value) matrices
- **Scaled Dot-Product**: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- **KV Cache**: Stores K,V for previous tokens to avoid recomputation during generation

### Positional Encodings
- **Absolute**: Fixed sinusoidal patterns (original Transformer)
- **Rotary (RoPE)**: Relative positional encoding, better for longer contexts
- **Impact**: RoPE enables longer context windows (e.g., 32K+ tokens)

### Architecture Types
- **Decoder-Only**: GPT-style, autoregressive generation (most LLMs)
- **Encoder-Decoder**: BART/T5-style, good for tasks requiring bidirectional understanding
- **Why Decoder-Only**: Simpler, better for text generation, sufficient for most tasks

### Scaling Laws
- **Depth vs Width**: Deeper networks vs wider layers
- **Parameter Count**: More parameters = better performance (up to a point)
- **Context Length**: Longer context = more memory, slower inference
- **Throughput**: Batch size × tokens/second

---

## 2. Prompt Engineering (13%)

### Prompt Patterns
- **Zero-Shot**: No examples, direct instruction
- **One-Shot**: Single example provided
- **Few-Shot**: Multiple examples (typically 2-5)
- **Chain-of-Thought (CoT)**: Step-by-step reasoning, improves complex reasoning
- **Self-Consistency**: Multiple CoT paths, majority vote

### Message Types
- **System Message**: Sets behavior, role, constraints (persistent)
- **User Message**: Actual query/task
- **Tool/Function Messages**: For function calling, tool use

### Output Control
- **JSON-Only**: Force structured output with schema
- **Delimiters**: Use markers to separate sections
- **Content Filters**: Pre/post-generation filtering via prompts
- **Temperature**: Lower = more deterministic, Higher = more creative

### Domain Adaptation
- **Via Prompts**: Quick adaptation, no training, limited effectiveness
- **Via Fine-Tuning**: Better adaptation, requires training data, more cost

### RAG Prompting
- **Retrieval Prompts**: Query formulation for retrieval
- **Context Windows**: How much retrieved context to include
- **Citations**: Including source references in output

---

## 3. Data Preparation (9%)

### Data Pipeline
1. **Collection**: Sourcing data from various sources
2. **Cleaning**: Remove noise, fix formatting
3. **De-duplication**: Remove duplicate content
4. **Filtering**: Remove offensive/PII content, low-quality data
5. **Labeling**: For supervised tasks (if needed)

### Tokenization
- **BPE (Byte Pair Encoding)**: Subword tokenization, used by GPT models
- **WordPiece**: Similar to BPE, used by BERT
- **Vocab Size**: Typically 30K-50K tokens
- **Special Tokens**: `<BOS>`, `<EOS>`, `<PAD>`, `<UNK>`, etc.

### Dataset Splits
- **Pre-train**: Large corpus for initial training
- **Fine-tune**: Task-specific data
- **Eval/Test**: Held-out evaluation set
- **Leakage**: Ensure no overlap between splits

### RAG Data Prep
- **Chunking**: Split documents into manageable pieces (typically 256-512 tokens)
- **Overlap**: 10-20% overlap between chunks to preserve context
- **Metadata**: Store source, position, embeddings for retrieval

---

## 4. Model Optimization (17%)

### Quantization
| Type | Precision | Memory Reduction | Speed Gain | Quality Loss |
|------|-----------|------------------|------------|--------------|
| FP32 | 32-bit | Baseline | Baseline | None |
| FP16 | 16-bit | 2x | 1.5-2x | Minimal |
| INT8 | 8-bit | 4x | 2-3x | Small |
| INT4 | 4-bit | 8x | 3-4x | Moderate |

- **Weight Quantization**: Quantize model weights
- **Activation Quantization**: Quantize activations (more complex)
- **Post-Training Quantization (PTQ)**: Quantize after training
- **Quantization-Aware Training (QAT)**: Train with quantization simulation

### TensorRT-LLM Optimizations
- **Graph Fusion**: Combine operations for efficiency
- **Kernel Auto-Tuning**: Optimize kernels for specific hardware
- **Quantization Flows**: Automated quantization pipelines
- **Throughput Gains**: Typically 2-5x speedup vs baseline

### Pruning & Distillation
- **Pruning**: Remove less important weights/neurons
- **Distillation**: Train smaller model to mimic larger model
- **Benefits**: Smaller models, faster inference, lower memory

### Decoding Strategies
- **Beam Search**: Explores multiple paths, better quality, slower
- **Sampling**: Random selection, faster, more diverse
  - **Temperature**: Controls randomness (0.7-1.0 typical)
  - **Top-k**: Sample from k most likely tokens
  - **Top-p (Nucleus)**: Sample from tokens with cumulative probability p

### KV Cache Optimization
- **Batch Size**: Larger batches = better GPU utilization but more memory
- **Max Tokens**: Longer sequences = more memory
- **Streaming**: Generate tokens incrementally

---

## 5. Fine-Tuning (13%)

### Fine-Tuning Types
| Type | Parameters Updated | VRAM Required | Speed | Use Case |
|------|-------------------|---------------|-------|----------|
| Full Fine-Tune | All | Very High | Slow | Major domain shift |
| LoRA | Adapters (~1%) | Low | Fast | Task adaptation |
| QLoRA | Quantized LoRA | Very Low | Fast | Limited resources |

### LoRA (Low-Rank Adaptation)
- **Rank (r)**: Typically 4-16, controls adapter size
- **Alpha**: Scaling factor, typically 2× rank
- **Target Modules**: Usually attention layers (Q, K, V, O)

### QLoRA (Quantized LoRA)
- **4-bit Quantization**: Base model quantized to 4-bit
- **LoRA on Top**: Train LoRA adapters in FP16
- **Memory**: ~75% reduction vs full fine-tune

### Hyperparameters
- **Learning Rate**: 1e-5 to 5e-4 (lower for full FT, higher for LoRA)
- **Warmup**: Gradual LR increase (typically 3-10% of steps)
- **Batch Size**: 1-8 for LoRA, larger for full FT
- **Epochs**: 1-5 typically sufficient
- **Early Stopping**: Stop when validation loss plateaus

### Fine-Tuning Types by Goal
- **Instruction Tuning**: Teach model to follow instructions
- **Domain Adaptation**: Adapt to specific domain (medical, legal, etc.)
- **Safety Tuning**: Reduce harmful outputs, improve alignment

### Avoiding Catastrophic Forgetting
- **Data Mixing**: Mix new data with original training data
- **Lower Learning Rate**: Use smaller LR for fine-tuning
- **Regularization**: Add constraints to prevent large weight changes

---

## 6. GPU Acceleration & Distributed Training (14%)

### Single-GPU Optimization
- **Tensor Cores**: Specialized units for matrix operations (A100, H100)
- **Mixed Precision**: FP16 training with FP32 master weights
- **Batch Size vs VRAM**: Larger batches = better utilization but more memory
- **Gradient Accumulation**: Simulate larger batches with limited VRAM

### Parallelism Types
| Type | How It Works | Use Case | Communication |
|------|--------------|----------|---------------|
| Data Parallelism | Split data across GPUs | Large batch training | All-reduce gradients |
| Tensor Parallelism | Split model layers | Very large models | Within-layer communication |
| Pipeline Parallelism | Split model stages | Long models | Sequential communication |

### Memory Optimization
- **Gradient Checkpointing**: Trade compute for memory
- **Offloading**: Move optimizer states to CPU
- **ZeRO**: Zero Redundancy Optimizer, partition optimizer states

### Profiling
- **Bottlenecks**: Identify under-utilized GPUs, memory issues
- **Tools**: NVIDIA Nsight, PyTorch Profiler
- **Metrics**: GPU utilization, memory usage, communication overhead

### Multi-Node Training
- **NCCL**: NVIDIA Collective Communications Library
- **All-Reduce**: Synchronize gradients across nodes
- **Communication Overhead**: Can be bottleneck for small models
- **Scaling Efficiency**: Typically 70-90% efficiency at scale

---

## 7. Model Deployment (9%)

### Triton Inference Server
- **Model Repository**: Directory structure for models
- **Model Config**: `config.pbtxt` defines model settings
- **Dynamic Batching**: Batch requests automatically
  - `max_batch_size`: Maximum batch size
  - `preferred_batch_size`: Optimal batch size
  - `max_queue_delay`: Max wait time for batching
- **Concurrent Models**: Run multiple models simultaneously
- **Protocols**: HTTP REST, gRPC, C API

### NIM (NVIDIA Inference Microservices)
- **Packaging**: Containerized model services
- **Routing**: Load balancing, request routing
- **Scaling**: Auto-scaling based on load
- **Integration**: Easy integration with applications

### Containerization
- **Docker**: Standard container format
- **GPU Runtime**: `nvidia-docker` or `--gpus` flag
- **Base Images**: `nvcr.io/nvidia` official images

### CI/CD & Rollout
- **Blue-Green**: Switch between two identical environments
- **Canary**: Gradual rollout to subset of users
- **Shadow**: Test new version alongside production (no user impact)

### Performance Targets
- **Latency**: P50, P95, P99 percentiles
- **Throughput**: Requests/second, tokens/second
- **Autoscaling**: Scale based on queue depth, latency

---

## 8. Evaluation (7%)

### Intrinsic Metrics
- **Perplexity**: exp(cross-entropy loss), lower is better
  - Measures model's uncertainty
  - Typical values: 10-50 for good models
- **Log-Loss**: Negative log-likelihood, lower is better

### Task Metrics
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
  - ROUGE-L: Longest common subsequence
  - ROUGE-1: Unigram overlap
  - ROUGE-2: Bigram overlap
- **BLEU**: Bilingual Evaluation Understudy
  - Precision-based, n-gram overlap
  - Range: 0-1, higher is better
- **BERTScore**: Semantic similarity using BERT embeddings
  - More semantic than ROUGE/BLEU
- **Accuracy/F1**: For classification tasks

### Human Evaluation
- **Rubrics**: Structured evaluation criteria
- **Pairwise Comparison**: Compare two outputs
- **Likert Scales**: Rate outputs on scales

### Regression Testing
- **Test Harnesses**: Automated evaluation suites
- **Baseline Comparison**: Compare against previous versions
- **A/B Testing**: Compare two models in production

---

## 9. Production Monitoring & Reliability (7%)

### Key Metrics
- **Latency**: Response time (P50, P95, P99)
- **Throughput**: Tokens/second, requests/second
- **Error Rate**: Failed requests / total requests
- **Timeout Rate**: Requests exceeding timeout threshold
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
- **Alerting**: Set up alerts for SLO violations
- **Rollback**: Revert to previous model version
- **Capacity Planning**: Plan for traffic spikes

### Safe Rollout Strategies
- **Canary**: Gradual rollout (5% → 25% → 50% → 100%)
- **Blue-Green**: Instant switch between environments
- **Shadow**: Test without user impact

---

## 10. Safety, Ethics & Compliance (5%)

### Guardrails
- **Pre-Prompt Filters**: Check input before processing
- **Post-Generation Filters**: Check output before returning
- **Blocklists**: Block specific words/phrases
- **Allowlists**: Only allow specific content

### Content Categories
- **Violence**: Physical harm, threats
- **Hate Speech**: Discriminatory content
- **Sexual Content**: Explicit material
- **Self-Harm**: Suicide, self-injury content

### PII (Personally Identifiable Information)
- **Detection**: Identify PII in inputs/outputs
- **Redaction**: Remove or mask PII
- **Logging Policies**: Don't log PII, anonymize data

### Bias & Toxicity
- **Bias Detection**: Identify demographic biases
- **Toxicity Mitigation**: Reduce harmful outputs
- **Fairness**: Ensure equal treatment across groups

### Regulatory Compliance
- **GDPR**: EU data protection regulation
- **Auditability**: Log decisions, maintain records
- **Right to Explanation**: Explain model decisions
- **Data Retention**: Policies for data storage/deletion

---

## Common NVIDIA-Specific Terminology

- **DGX**: NVIDIA's AI supercomputer systems
- **CUDA**: Parallel computing platform
- **TensorRT**: Inference optimization SDK
- **TensorRT-LLM**: LLM-specific TensorRT optimizations
- **NeMo**: NVIDIA's framework for building conversational AI
- **Triton**: Inference server for production deployment
- **NIM**: NVIDIA Inference Microservices
- **NCCL**: NVIDIA Collective Communications Library
- **A100/H100**: Latest GPU architectures
- **NVLink**: High-speed GPU interconnect

---

## Quick Decision Trees

### When to Use Quantization?
- **INT8**: Good balance of speed/quality
- **INT4**: Maximum compression, acceptable quality loss
- **FP16**: Minimal quality loss, good speedup

### When to Use Fine-Tuning?
- **Full FT**: Major domain shift, sufficient resources
- **LoRA**: Task adaptation, limited resources
- **QLoRA**: Very limited resources, still need adaptation

### When to Use Parallelism?
- **Data Parallelism**: Model fits on single GPU, need larger batches
- **Tensor Parallelism**: Model too large for single GPU
- **Pipeline Parallelism**: Very long models, sequential processing

### When to Use Deployment Strategy?
- **Triton**: Need flexible batching, multiple models
- **NIM**: Need easy integration, microservices architecture
- **Direct**: Simple use case, single model

---

## Formula Reference

### Perplexity
```
Perplexity = exp(cross_entropy_loss)
```

### ROUGE-L
```
ROUGE-L = LCS(reference, candidate) / length(reference)
```

### BLEU
```
BLEU = BP × exp(Σ log(p_n))
where BP = brevity penalty, p_n = n-gram precision
```

### Memory Estimation (Approximate)
```
Model Memory ≈ (parameters × bytes_per_param) + activations + optimizer_states
```

---

*Last Updated: Day 1 of Study Plan*
*Review this guide daily during Week 2*
