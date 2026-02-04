# Flashcard Research Papers Summary

## Overview
Each enhanced flashcard now includes comprehensive research papers:
- **Foundational papers**: Original papers that introduced the concept
- **Influential papers**: Papers that have used/extended the method
- **Recent papers**: Latest developments and improvements

## Papers by Domain

### LLM Architecture

#### Multi-Head Attention
- **Foundational**: Attention Is All You Need (2017)
- **Applications**: BERT, GPT-3, GPT-4, LLaMA, PaLM
- **Total**: 7 papers + tutorials

#### Self-Attention (Q, K, V)
- **Foundational**: Attention Is All You Need
- **Improvements**: Efficient Attention, Layer Normalization
- **Total**: 5 papers + tutorials

#### KV Cache
- **Foundational**: FlashAttention (2022)
- **Improvements**: PagedAttention, vLLM
- **Total**: 4 papers + tutorials

#### RoPE (Rotary Positional Encoding)
- **Foundational**: RoFormer (2021)
- **Applications**: LLaMA, PaLM, GPT-NeoX
- **Total**: 5 papers + videos

#### Feed-Forward Blocks
- **Foundational**: Attention Is All You Need
- **Improvements**: GLU Variants, Swish Activation
- **Total**: 3 papers

#### Residual Connections
- **Foundational**: Deep Residual Learning (2015)
- **Applications**: Transformer architecture
- **Total**: 3 papers

#### Layer Normalization
- **Foundational**: Layer Normalization (2016)
- **Improvements**: Pre-norm vs Post-norm analysis
- **Total**: 3 papers

#### Decoder-Only Architecture
- **Foundational**: GPT-2, GPT-3
- **Applications**: LLaMA, PaLM, GPT-4
- **Total**: 5 papers

---

### Model Optimization

#### FP32/FP16
- **Foundational**: Mixed Precision Training (2017)
- **Improvements**: BFLOAT16, Training Deep Networks
- **Total**: 4 papers + tutorials

#### INT8 Quantization
- **Foundational**: Quantization and Training (2017)
- **Improvements**: Q8BERT, LLM.int8(), SmoothQuant, GPTQ, AWQ
- **Total**: 6 papers + tutorials

#### INT4 Quantization
- **Foundational**: QLoRA (2023)
- **Improvements**: GPTQ, AWQ, SmoothQuant
- **Total**: 4 papers

#### Temperature Sampling
- **Foundational**: Neural Text Degeneration (2019)
- **Applications**: GPT-2, GPT-3
- **Total**: 4 papers + tutorials

#### Top-k / Top-p Sampling
- **Foundational**: Neural Text Degeneration (2019)
- **Applications**: GPT-2, GPT-3
- **Total**: 2-3 papers each

#### Pruning
- **Foundational**: Learning Weights and Connections (2015)
- **Improvements**: Lottery Ticket Hypothesis, Movement Pruning, SparseGPT
- **Total**: 4 papers

#### Distillation
- **Foundational**: Distilling Knowledge (2015)
- **Improvements**: Patient KD for BERT, TinyBERT, MiniLM
- **Total**: 4 papers

---

### Fine-Tuning

#### LoRA
- **Foundational**: LoRA (2021)
- **Improvements**: QLoRA, AdaLoRA, LoRA+, DoRA, VeRA
- **Total**: 6 papers + tutorials

#### QLoRA
- **Foundational**: QLoRA (2023)
- **Related**: LoRA, LLM.int8(), GPTQ, BitsAndBytes
- **Total**: 5 papers + tutorials

#### Full Fine-Tune
- **Foundational**: BERT (2018)
- **Applications**: Instruction Tuning (FLAN)
- **Total**: 3 papers

---

### GPU Acceleration & Distributed Training

#### Data Parallelism
- **Foundational**: PyTorch DDP
- **Applications**: Megatron-LM, ZeRO
- **Total**: 6 papers + tutorials

#### Tensor Parallelism
- **Foundational**: Megatron-LM (2019)
- **Applications**: PaLM, GPT-3, Efficient Inference
- **Total**: 5 papers + videos

#### Pipeline Parallelism
- **Foundational**: GPipe (2019)
- **Improvements**: PipeDream, Megatron-LM
- **Total**: 4 papers

#### ZeRO
- **Foundational**: ZeRO (2019)
- **Improvements**: ZeRO-Offload, ZeRO-Infinity, DeepSpeed
- **Total**: 4 papers

---

### Prompt Engineering

#### Chain-of-Thought (CoT)
- **Foundational**: CoT Prompting (2022)
- **Improvements**: Self-Consistency, Zero-Shot Reasoners, Tree of Thoughts, ReAct, Auto-CoT
- **Total**: 6 papers + videos

#### Few-Shot Learning
- **Foundational**: GPT-3 (2020)
- **Improvements**: In-Context Learning, Induction Heads, What Makes ICL Work
- **Total**: 5 papers + tutorials

---

### Model Deployment

#### Dynamic Batching
- **Foundational**: Triton Inference Server
- **Improvements**: vLLM, Orca, FastServe
- **Total**: 5 papers + tutorials

---

### Data Preparation

#### BPE (Byte Pair Encoding)
- **Foundational**: Neural MT of Rare Words (2015)
- **Applications**: GPT-2, BERT (WordPiece), SentencePiece
- **Total**: 5 papers + videos

#### RAG Chunking
- **Foundational**: RAG (2020)
- **Improvements**: Dense Passage Retrieval, REALM, In-Context RAG
- **Total**: 4 papers

---

### Evaluation

#### Perplexity
- **Foundational**: Neural Machine Translation (2014)
- **Applications**: GPT-2, Scaling Laws, Chinchilla
- **Total**: 4 papers + tutorials

#### ROUGE-L
- **Foundational**: ROUGE (2004)
- **Related**: BLEU, BERTScore, METEOR, SummEval
- **Total**: 5 papers + videos

#### BLEU
- **Foundational**: BLEU (2002)
- **Related**: ROUGE, BERTScore
- **Total**: 3 papers

#### BERTScore
- **Foundational**: BERTScore (2019)
- **Related**: BLEU, ROUGE
- **Total**: 3 papers

---

### Production Monitoring & Reliability

#### SLO
- **Foundational**: Google SRE Book
- **Related**: Art of SLOs, Error Budgets
- **Total**: 3 papers/resources

---

### Safety, Ethics & Compliance

#### PII Detection
- **Foundational**: Presidio (2020)
- **Related**: Privacy-Preserving ML, GDPR Compliance
- **Total**: 3 papers

---

## Paper Categories

### Foundational Papers (Must Read)
- Attention Is All You Need (2017) - Transformers
- BERT (2018) - Bidirectional Transformers
- GPT-2/GPT-3 (2019/2020) - Decoder-only LLMs
- LoRA (2021) - Parameter-efficient fine-tuning
- QLoRA (2023) - Quantized LoRA
- ZeRO (2019) - Memory optimization
- Megatron-LM (2019) - Large-scale training

### Recent Important Papers (2022-2024)
- FlashAttention (2022) - Efficient attention
- Chain-of-Thought (2022) - Reasoning
- vLLM (2023) - Efficient serving
- QLoRA (2023) - Quantized fine-tuning
- Tree of Thoughts (2023) - Advanced reasoning

### NVIDIA-Specific Papers
- TensorRT-LLM optimizations
- Triton Inference Server
- CUDA and GPU acceleration papers
- DeepSpeed (ZeRO) papers

---

## How to Use Papers

1. **For Exam Prep**: Focus on foundational papers and NVIDIA-specific papers
2. **For Deep Understanding**: Read papers that use/extend the method
3. **For Latest Techniques**: Check recent papers (2022-2024)
4. **Quick Reference**: Use paper titles to search for summaries/explanations

---

## Accessing Papers

- **arXiv**: Most papers available at arxiv.org/abs/[paper-id]
- **ACL Anthology**: NLP papers at aclanthology.org
- **Google Scholar**: Search by paper title
- **Papers with Code**: Implementation code available

---

**Total Enhanced Flashcards**: 25+ with comprehensive papers
**Total Research Papers**: 100+ papers across all flashcards
