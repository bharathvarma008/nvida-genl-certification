import streamlit as st
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flashcard_content_enhanced import FLASHCARD_CONTENT_ENHANCED, get_flashcard_content

# Page configuration
st.set_page_config(
    page_title="NCP-GENL Study Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data file
DATA_FILE = "study_data.json"

# Domain information
DOMAINS = {
    "Model Optimization": {"weight": 17, "target": 75},
    "GPU Acceleration & Distributed Training": {"weight": 14, "target": 75},
    "Prompt Engineering": {"weight": 13, "target": 75},
    "Fine-Tuning": {"weight": 13, "target": 75},
    "Data Preparation": {"weight": 9, "target": 75},
    "Model Deployment": {"weight": 9, "target": 75},
    "Evaluation": {"weight": 7, "target": 75},
    "Production Monitoring & Reliability": {"weight": 7, "target": 75},
    "LLM Architecture": {"weight": 6, "target": 75},
    "Safety, Ethics & Compliance": {"weight": 5, "target": 75}
}

# Flashcard topics by domain
FLASHCARD_TOPICS = {
    "LLM Architecture": [
        "Multi-Head Attention", "Feed-Forward Blocks", "Residual Connections",
        "Layer Normalization", "Self-Attention (Q, K, V)", "KV Cache",
        "Absolute Positional Encoding", "Rotary Positional Encoding (RoPE)",
        "Decoder-Only Architecture", "Encoder-Decoder Architecture", "Scaling Laws"
    ],
    "Prompt Engineering": [
        "Zero-Shot", "One-Shot", "Few-Shot", "Chain-of-Thought (CoT)",
        "Self-Consistency", "System Message", "User Message", "Tool/Function Messages",
        "JSON-Only Output", "Delimiters", "Content Filters", "Temperature",
        "Domain Adaptation via Prompts", "RAG Prompting"
    ],
    "Data Preparation": [
        "Data Collection", "Data Cleaning", "De-duplication", "Filtering",
        "BPE (Byte Pair Encoding)", "WordPiece", "Vocab Size", "Special Tokens",
        "Pre-train Split", "Fine-tune Split", "Eval/Test Split", "Leakage",
        "RAG Chunking", "Overlap", "Metadata"
    ],
    "Model Optimization": [
        "FP32", "FP16", "INT8", "INT4", "Weight Quantization",
        "Activation Quantization", "PTQ", "QAT", "TensorRT-LLM Graph Fusion",
        "Kernel Auto-Tuning", "Pruning", "Distillation", "Beam Search",
        "Sampling", "Temperature", "Top-k Sampling", "Top-p Sampling", "KV Cache Optimization"
    ],
    "Fine-Tuning": [
        "Full Fine-Tune", "LoRA", "QLoRA", "Rank (r)", "Alpha",
        "Target Modules", "Learning Rate", "Warmup", "Batch Size",
        "Epochs", "Early Stopping", "Instruction Tuning", "Domain Adaptation",
        "Safety Tuning", "Catastrophic Forgetting", "Data Mixing"
    ],
    "GPU Acceleration & Distributed Training": [
        "Tensor Cores", "Mixed Precision", "Batch Size vs VRAM",
        "Gradient Accumulation", "Data Parallelism", "Tensor Parallelism",
        "Pipeline Parallelism", "Gradient Checkpointing", "Offloading",
        "ZeRO", "NCCL", "All-Reduce", "Communication Overhead", "Scaling Efficiency"
    ],
    "Model Deployment": [
        "Triton Model Repository", "Model Config (config.pbtxt)", "Dynamic Batching",
        "Concurrent Models", "HTTP REST/gRPC", "NIM Packaging", "NIM Routing",
        "NIM Scaling", "Docker", "GPU Runtime", "Blue-Green", "Canary", "Shadow"
    ],
    "Evaluation": [
        "Perplexity", "Log-Loss", "ROUGE-L", "ROUGE-1", "ROUGE-2",
        "BLEU", "BERTScore", "Accuracy/F1", "Human Evaluation Rubrics",
        "Pairwise Comparison", "Test Harnesses", "A/B Testing"
    ],
    "Production Monitoring & Reliability": [
        "Latency (P50, P95, P99)", "Throughput", "Error Rate", "Timeout Rate",
        "Cache Hit Rate", "SLO", "SLI", "Error Budget", "Data Drift",
        "Concept Drift", "Model Drift", "Alerting", "Rollback", "Capacity Planning"
    ],
    "Safety, Ethics & Compliance": [
        "Pre-Prompt Filters", "Post-Generation Filters", "Blocklists", "Allowlists",
        "Violence Content", "Hate Speech", "PII Detection", "PII Redaction",
        "Bias Detection", "Toxicity Mitigation", "GDPR", "Auditability"
    ]
}

# Flashcard content with detailed explanations, formulas, diagrams, code, and links
FLASHCARD_CONTENT = {
    "LLM Architecture": {
        "Multi-Head Attention": {
            "definition": "Parallel attention mechanisms that allow the model to focus on different aspects simultaneously. Each head learns different attention patterns, enabling richer representations.",
            "formula": "MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O\nwhere headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)\nand Attention(Q, K, V) = softmax(QK^T/√dₖ)V",
            "diagram": """graph LR
    A[Input] --> B[Q, K, V<br/>Projections]
    B --> C1[Head 1]
    B --> C2[Head 2]
    B --> C3[Head h]
    C1 --> D[Concat]
    C2 --> D
    C3 --> D
    D --> E[Output<br/>Projection]
    E --> F[Output]""",
            "code_example": None,
            "links": [
                {"title": "Attention Is All You Need Paper", "url": "https://arxiv.org/abs/1706.03762"},
                {"title": "Illustrated Transformer", "url": "https://jalammar.github.io/illustrated-transformer/"}
            ]
        },
        "Feed-Forward Blocks": "Two linear transformations with an activation function (ReLU/GELU) between them. Processes the attended information from the attention layer.",
        "Residual Connections": "Skip connections that add the input directly to the output. Prevents vanishing gradients and allows deeper networks to train effectively.",
        "Layer Normalization": "Normalizes inputs to each layer, stabilizing training. Pre-norm (before attention) vs post-norm (after attention) are common variants.",
        "Self-Attention (Q, K, V)": "Attention mechanism using Query (Q), Key (K), and Value (V) matrices. Formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V. Allows tokens to attend to each other.",
        "KV Cache": "Stores computed Key and Value matrices for previous tokens during generation. Avoids recomputation, significantly speeding up autoregressive generation.",
        "Absolute Positional Encoding": "Fixed sinusoidal patterns added to embeddings to encode token positions. Used in original Transformer architecture.",
        "Rotary Positional Encoding (RoPE)": "Relative positional encoding that rotates embeddings based on position. Enables longer context windows (e.g., 32K+ tokens) compared to absolute encoding.",
        "Decoder-Only Architecture": "GPT-style architecture that generates text autoregressively. Most modern LLMs use this. Simpler and better for text generation tasks.",
        "Encoder-Decoder Architecture": "BART/T5-style with both encoder and decoder. Good for tasks requiring bidirectional understanding but more complex than decoder-only.",
        "Scaling Laws": "Relationships between model size (depth/width), parameters, context length, and performance. Deeper vs wider networks, parameter count vs performance trade-offs."
    },
    "Prompt Engineering": {
        "Zero-Shot": "No examples provided, just direct instruction. Model relies on pre-training knowledge. Quick but may have limited accuracy.",
        "One-Shot": "Single example provided in the prompt. Helps model understand the task format. Better than zero-shot for structured tasks.",
        "Few-Shot": "Multiple examples (typically 2-5) provided in the prompt. Demonstrates the task pattern. Improves performance over zero/one-shot.",
        "Chain-of-Thought (CoT)": "Prompting technique that encourages step-by-step reasoning. Improves complex reasoning tasks by breaking problems into smaller steps.",
        "Self-Consistency": "Generate multiple CoT reasoning paths and take majority vote. Improves accuracy by aggregating multiple solutions.",
        "System Message": "Sets the model's behavior, role, and constraints. Persistent across the conversation. Used to control model personality and output style.",
        "User Message": "The actual query or task from the user. Contains the specific request or question to be answered.",
        "Tool/Function Messages": "Messages for function calling and tool use. Allows LLMs to interact with external tools and APIs.",
        "JSON-Only Output": "Forces structured output using JSON schema. Ensures parseable, consistent responses. Useful for API integrations.",
        "Delimiters": "Markers used to separate sections in prompts (e.g., ###, ---, ```). Helps model distinguish between different parts of the input.",
        "Content Filters": "Pre-generation (check input) or post-generation (check output) filters. Used to block harmful or inappropriate content.",
        "Temperature": "Sampling parameter controlling randomness. Lower (0-0.7) = more deterministic, Higher (0.7-1.5) = more creative. Typical range: 0.7-1.0.",
        "Domain Adaptation via Prompts": "Quick adaptation without training. Limited effectiveness but fast. Use when fine-tuning isn't feasible.",
        "RAG Prompting": "Retrieval-Augmented Generation. Uses retrieval prompts to find relevant context, includes in prompt. Enables access to external knowledge."
    },
    "Data Preparation": {
        "Data Collection": "Sourcing data from various sources (web, books, code, etc.). First step in data pipeline. Quality of collection affects final model.",
        "Data Cleaning": "Remove noise, fix formatting, normalize text. Critical for model quality. Includes fixing encoding, removing HTML, etc.",
        "De-duplication": "Remove duplicate content from dataset. Prevents model from overfitting to repeated examples. Important for fair evaluation.",
        "Filtering": "Remove offensive content, PII, low-quality data. Ensures safety and quality. Can use automated filters or human review.",
        "BPE (Byte Pair Encoding)": "Subword tokenization algorithm used by GPT models. Merges most frequent pairs iteratively. Handles out-of-vocabulary words well.",
        "WordPiece": "Similar to BPE, used by BERT. Slightly different merging strategy. Also handles subword tokenization effectively.",
        "Vocab Size": "Size of token vocabulary. Typically 30K-50K tokens for LLMs. Larger vocab = more tokens but better coverage. Trade-off with memory.",
        "Special Tokens": "Special purpose tokens: <BOS> (beginning), <EOS> (end), <PAD> (padding), <UNK> (unknown). Used for model control and formatting.",
        "Pre-train Split": "Large corpus for initial model training. Typically millions/billions of tokens. Foundation for all downstream tasks.",
        "Fine-tune Split": "Task-specific data for fine-tuning. Smaller than pre-train data. Used to adapt model to specific domain or task.",
        "Eval/Test Split": "Held-out evaluation set. Never seen during training. Used to measure true model performance. Critical for fair assessment.",
        "Leakage": "When test data appears in training data. Causes inflated performance metrics. Must ensure strict separation between splits.",
        "RAG Chunking": "Split documents into manageable pieces for retrieval. Typical sizes: 256-512 tokens. Balance between context and granularity.",
        "Overlap": "10-20% overlap between chunks. Preserves context across chunk boundaries. Important for maintaining semantic continuity.",
        "Metadata": "Store source, position, embeddings for each chunk. Enables retrieval and citation. Critical for RAG systems."
    },
    "Model Optimization": {
        "FP32": "32-bit floating point precision. Baseline format. Full precision, no quality loss. Highest memory usage.",
        "FP16": "16-bit floating point. 2x memory reduction, 1.5-2x speed gain. Minimal quality loss. Common for training with mixed precision.",
        "INT8": "8-bit integer quantization. 4x memory reduction, 2-3x speed gain. Small quality loss. Good balance for inference.",
        "INT4": "4-bit integer quantization. 8x memory reduction, 3-4x speed gain. Moderate quality loss. Maximum compression.",
        "Weight Quantization": "Quantize model weights only. Simpler than activation quantization. Common approach for inference optimization.",
        "Activation Quantization": "Quantize activations during inference. More complex, requires calibration. Better compression but harder to implement.",
        "PTQ": "Post-Training Quantization. Quantize model after training. No retraining needed. Faster but may have more quality loss.",
        "QAT": "Quantization-Aware Training. Train with quantization simulation. Better quality retention. Requires retraining but better results.",
        "TensorRT-LLM Graph Fusion": "Combine multiple operations into single kernel. Reduces overhead. Part of TensorRT-LLM optimization pipeline.",
        "Kernel Auto-Tuning": "Automatically optimize kernels for specific hardware. Maximizes GPU utilization. TensorRT-LLM feature.",
        "Pruning": "Remove less important weights or neurons. Reduces model size. Can be structured (channels) or unstructured (individual weights).",
        "Distillation": "Train smaller student model to mimic larger teacher model. Reduces size while preserving knowledge. Knowledge transfer technique.",
        "Beam Search": "Explores multiple generation paths, keeps top-k candidates. Better quality but slower. Used when quality is priority.",
        "Sampling": "Random selection from probability distribution. Faster than beam search, more diverse outputs. Used for creative tasks.",
        "Temperature": "Controls randomness in sampling. Lower = more deterministic, Higher = more creative. Typical: 0.7-1.0 for balanced output.",
        "Top-k Sampling": "Sample from k most likely tokens. Limits vocabulary to top candidates. Reduces low-probability tokens.",
        "Top-p Sampling": "Nucleus sampling. Sample from tokens with cumulative probability p. Dynamic vocabulary size. Often better than top-k.",
        "KV Cache Optimization": "Optimize Key-Value cache storage. Batch size, max tokens affect memory. Critical for generation speed."
    },
    "Fine-Tuning": {
        "Full Fine-Tune": "Update all model parameters. Very high VRAM, slow training. Use for major domain shifts. Best quality but expensive.",
        "LoRA": "Low-Rank Adaptation. Train small adapters (~1% parameters). Low VRAM, fast training. Use for task adaptation. Most popular PEFT method.",
        "QLoRA": "Quantized LoRA. 4-bit base model + FP16 LoRA adapters. Very low VRAM (~75% reduction). Use when resources are limited.",
        "Rank (r)": "LoRA rank parameter. Controls adapter size. Typically 4-16. Higher rank = more capacity but more parameters.",
        "Alpha": "LoRA scaling factor. Typically 2× rank. Controls adapter strength. Higher alpha = stronger adaptation.",
        "Target Modules": "Which layers to apply LoRA. Usually attention layers (Q, K, V, O). Can target specific modules for efficiency.",
        "Learning Rate": "Step size for weight updates. 1e-5 to 5e-4 typical. Lower for full FT, higher for LoRA. Critical hyperparameter.",
        "Warmup": "Gradual learning rate increase. Typically 3-10% of training steps. Stabilizes early training. Prevents divergence.",
        "Batch Size": "Number of examples per update. 1-8 for LoRA, larger for full FT. Affects memory and gradient quality.",
        "Epochs": "Number of full dataset passes. Typically 1-5 sufficient. More epochs risk overfitting. Monitor validation loss.",
        "Early Stopping": "Stop training when validation loss plateaus. Prevents overfitting. Saves compute. Common practice.",
        "Instruction Tuning": "Fine-tune to follow instructions. Teaches model to respond to prompts. Foundation for chat models.",
        "Domain Adaptation": "Adapt model to specific domain (medical, legal, etc.). Improves domain-specific performance. Requires domain data.",
        "Safety Tuning": "Fine-tune to reduce harmful outputs. Improves alignment. Important for production deployment. Uses safety datasets.",
        "Catastrophic Forgetting": "Model forgets original knowledge when fine-tuned. Problem when adapting to new domain. Mitigated by data mixing.",
        "Data Mixing": "Mix new fine-tuning data with original training data. Prevents catastrophic forgetting. Maintains general knowledge."
    },
    "GPU Acceleration & Distributed Training": {
        "Tensor Cores": "Specialized units for matrix operations in A100, H100 GPUs. Accelerate training significantly. Use mixed precision to leverage.",
        "Mixed Precision": "FP16 training with FP32 master weights. 2x speedup, lower memory. Standard practice. Requires careful implementation.",
        "Batch Size vs VRAM": "Larger batches = better GPU utilization but more memory. Balance based on available VRAM. Key optimization parameter.",
        "Gradient Accumulation": "Simulate larger batches by accumulating gradients. Use when VRAM limits batch size. Effective workaround.",
        "Data Parallelism": "Split data across GPUs, each GPU has full model. All-reduce gradients. Use when model fits on single GPU. Most common.",
        "Tensor Parallelism": "Split model layers across GPUs. Each GPU has part of model. Use for very large models. Requires within-layer communication.",
        "Pipeline Parallelism": "Split model into stages, each GPU handles one stage. Sequential processing. Use for very long models. Lower communication.",
        "Gradient Checkpointing": "Trade compute for memory. Recompute activations instead of storing. Reduces memory by ~50%. Slower but enables larger models.",
        "Offloading": "Move optimizer states to CPU. Reduces GPU memory. Slower but enables larger models. Part of ZeRO optimization.",
        "ZeRO": "Zero Redundancy Optimizer. Partitions optimizer states, gradients, parameters. Reduces memory footprint. Enables massive models.",
        "NCCL": "NVIDIA Collective Communications Library. Optimized communication for multi-GPU. Used for all-reduce, all-gather operations.",
        "All-Reduce": "Synchronize gradients across GPUs/nodes. Sums gradients from all devices. Critical for distributed training. Uses NCCL.",
        "Communication Overhead": "Time spent synchronizing gradients. Can be bottleneck for small models. Less important for large models. Consider when scaling.",
        "Scaling Efficiency": "How well performance scales with GPUs. Typically 70-90% at scale. Diminishing returns due to communication overhead."
    },
    "Model Deployment": {
        "Triton Model Repository": "Directory structure for Triton models. Organized by model name/version. Standard format for model serving.",
        "Model Config (config.pbtxt)": "Triton configuration file. Defines model settings, batching, inputs/outputs. Required for each model.",
        "Dynamic Batching": "Automatically batch requests. max_batch_size, preferred_batch_size, max_queue_delay. Improves throughput significantly.",
        "Concurrent Models": "Run multiple models simultaneously on same server. Efficient resource utilization. Triton feature.",
        "HTTP REST/gRPC": "Protocols for Triton. HTTP REST for simple use, gRPC for better performance. Choose based on requirements.",
        "NIM Packaging": "Containerized model services. Easy packaging and distribution. NVIDIA Inference Microservices format.",
        "NIM Routing": "Load balancing and request routing. Distributes load across instances. Part of NIM architecture.",
        "NIM Scaling": "Auto-scaling based on load. Handles traffic spikes. Cloud-native deployment. NIM feature.",
        "Docker": "Standard container format. Package models and dependencies. Ensures consistent environments. Industry standard.",
        "GPU Runtime": "nvidia-docker or --gpus flag. Enables GPU access in containers. Required for GPU-accelerated inference.",
        "Blue-Green": "Switch between two identical environments. Zero-downtime deployment. Instant rollback capability.",
        "Canary": "Gradual rollout (5% → 25% → 50% → 100%). Test with subset of users. Safe deployment strategy.",
        "Shadow": "Test new version alongside production. No user impact. Compare performance. Safe testing method."
    },
    "Evaluation": {
        "Perplexity": "exp(cross-entropy loss). Lower is better. Measures model uncertainty. Typical values: 10-50 for good models. Intrinsic metric.",
        "Log-Loss": "Negative log-likelihood. Lower is better. Measures prediction confidence. Related to perplexity. Intrinsic metric.",
        "ROUGE-L": "Longest Common Subsequence between reference and candidate. Recall-oriented. Good for summarization. Range: 0-1.",
        "ROUGE-1": "Unigram overlap between reference and candidate. Simple but effective. Common metric. Range: 0-1.",
        "ROUGE-2": "Bigram overlap between reference and candidate. More strict than ROUGE-1. Better for longer texts. Range: 0-1.",
        "BLEU": "Bilingual Evaluation Understudy. N-gram precision with brevity penalty. Range: 0-1, higher is better. Common for translation.",
        "BERTScore": "Semantic similarity using BERT embeddings. More semantic than ROUGE/BLEU. Better captures meaning. Range: 0-1.",
        "Accuracy/F1": "Classification metrics. Accuracy = correct/total. F1 = harmonic mean of precision/recall. Use for classification tasks.",
        "Human Evaluation Rubrics": "Structured criteria for human evaluation. More reliable than automatic metrics. Expensive but gold standard.",
        "Pairwise Comparison": "Compare two model outputs. Human judges which is better. Common evaluation method. Reduces bias.",
        "Test Harnesses": "Automated evaluation suites. Run regression tests. Ensure model quality. Part of CI/CD pipeline.",
        "A/B Testing": "Compare two models in production. Real user feedback. Most realistic evaluation. Requires infrastructure."
    },
    "Production Monitoring & Reliability": {
        "Latency (P50, P95, P99)": "Response time percentiles. P50 = median, P95 = 95th percentile, P99 = 99th percentile. Key performance metric.",
        "Throughput": "Requests/second or tokens/second. Measures system capacity. Higher is better. Critical for scaling decisions.",
        "Error Rate": "Failed requests / total requests. Should be <1%. Key reliability metric. Monitor closely.",
        "Timeout Rate": "Requests exceeding timeout threshold. Indicates performance issues. Should be minimal. Alert on spikes.",
        "Cache Hit Rate": "Percentage of cached responses. Higher = better performance. Reduces load. Monitor cache effectiveness.",
        "SLO": "Service Level Objective. Target for reliability (e.g., 99.9% uptime). Business requirement. Defines acceptable performance.",
        "SLI": "Service Level Indicator. Measured metric (e.g., actual uptime). Tracks against SLO. Used for monitoring.",
        "Error Budget": "Allowable failures before SLO violation. Guides deployment decisions. Depleted budget = no deployments.",
        "Data Drift": "Input distribution changes over time. Model may degrade. Detect with statistical tests. Requires monitoring.",
        "Concept Drift": "Relationship between input/output changes. Model predictions become less accurate. Harder to detect than data drift.",
        "Model Drift": "Model performance degrades over time. Requires retraining. Monitor with evaluation metrics. Part of ML lifecycle.",
        "Alerting": "Set up alerts for SLO violations. Critical for production. Use PagerDuty, etc. Fast response required.",
        "Rollback": "Revert to previous model version. Quick fix for issues. Requires versioning. Critical capability.",
        "Capacity Planning": "Plan for traffic spikes. Scale proactively. Avoid outages. Based on historical data and trends."
    },
    "Safety, Ethics & Compliance": {
        "Pre-Prompt Filters": "Check input before processing. Block harmful requests early. First line of defense. Fast and efficient.",
        "Post-Generation Filters": "Check output before returning to user. Catch harmful content. Second line of defense. More thorough.",
        "Blocklists": "Block specific words/phrases. Simple but effective. Can be bypassed. Use with other methods.",
        "Allowlists": "Only allow specific content. Very restrictive. Use for high-security scenarios. Limits functionality.",
        "Violence Content": "Physical harm, threats, graphic content. Category for filtering. Important safety concern. Must detect.",
        "Hate Speech": "Discriminatory content targeting groups. Serious safety issue. Must filter. Legal and ethical requirement.",
        "PII Detection": "Identify Personally Identifiable Information (names, SSN, etc.). Critical for privacy. Use NER models.",
        "PII Redaction": "Remove or mask PII from outputs. Protect user privacy. Required by regulations. Use before logging.",
        "Bias Detection": "Identify demographic biases in outputs. Fairness concern. Use evaluation metrics. Monitor regularly.",
        "Toxicity Mitigation": "Reduce harmful, toxic outputs. Improve model safety. Use safety tuning. Ongoing effort.",
        "GDPR": "EU General Data Protection Regulation. Right to explanation, data deletion. Legal requirement. Must comply.",
        "Auditability": "Log decisions, maintain records. Required for compliance. Enables debugging. Important for production."
    }
}

def load_data():
    """Load study data from JSON file"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            # Ensure flashcards are initialized for all topics
            if "flashcards" not in data:
                data["flashcards"] = {}
            
            # Initialize papers tracking if not exists
            if "papers" not in data:
                data["papers"] = {}
            
            # Add any missing flashcards
            for domain, topics in FLASHCARD_TOPICS.items():
                if domain not in data["flashcards"]:
                    data["flashcards"][domain] = {}
                for topic in topics:
                    if topic not in data["flashcards"][domain]:
                        data["flashcards"][domain][topic] = {
                            "status": "new",
                            "last_reviewed": None,
                            "review_count": 0,
                            "has_content": topic in FLASHCARD_CONTENT.get(domain, {})
                        }
                    # Ensure has_content flag is set (check both old and enhanced)
                    if "has_content" not in data["flashcards"][domain][topic]:
                        has_old = topic in FLASHCARD_CONTENT.get(domain, {})
                        has_enhanced = get_flashcard_content(domain, topic) is not None
                        data["flashcards"][domain][topic]["has_content"] = has_old or has_enhanced
                    
                    # Initialize papers for this flashcard
                    card_content = get_flashcard_content(domain, topic)
                    if card_content and card_content.get("links"):
                        for link in card_content["links"]:
                            if link.get("type") == "paper":
                                paper_id = f"{domain}_{topic}_{link['url']}"
                                if paper_id not in data["papers"]:
                                    data["papers"][paper_id] = {
                                        "title": link["title"],
                                        "url": link["url"],
                                        "domain": domain,
                                        "topic": topic,
                                        "read": False,
                                        "notes": "",
                                        "difficulty": None,
                                        "priority": "optional",
                                        "summary": "",
                                        "key_takeaways": [],
                                        "read_date": None,
                                        "tags": []
                                    }
            
            # Ensure all domains exist in domain_mastery
            for domain in DOMAINS:
                if domain not in data.get("domain_mastery", {}):
                    if "domain_mastery" not in data:
                        data["domain_mastery"] = {}
                    data["domain_mastery"][domain] = 0
            
            return data
    return initialize_data()

def initialize_data():
    """Initialize empty data structure"""
    # Initialize flashcards with content if available
    flashcards_init = {}
    for domain, topics in FLASHCARD_TOPICS.items():
        flashcards_init[domain] = {}
        for topic in topics:
            flashcards_init[domain][topic] = {
                "status": "new",
                "last_reviewed": None,
                "review_count": 0,
                "has_content": topic in FLASHCARD_CONTENT.get(domain, {})
            }
    
    return {
        "study_start_date": None,
        "target_exam_date": None,
        "study_hours": {
            "target": 50,
            "completed": 0,
            "daily_log": []
        },
        "practice_questions": {
            "target": 400,
            "total_attempted": 0,
            "total_correct": 0,
            "by_domain": {domain: {"attempted": 0, "correct": 0, "incorrect": 0} for domain in DOMAINS}
        },
        "domain_mastery": {domain: 0 for domain in DOMAINS},
        "mock_exams": {
            "mock_1": None,
            "mock_2": None
        },
        "flashcards": flashcards_init,
        "papers": {}
    }

def save_data(data):
    """Save study data to JSON file"""
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def get_status_emoji(accuracy):
    """Get status emoji based on accuracy"""
    if accuracy == 0:
        return "⬜"
    elif accuracy < 40:
        return "⬜"
    elif accuracy < 70:
        return "🟡"
    elif accuracy < 85:
        return "🟢"
    else:
        return "✅"

def main():
    st.title("🎓 NCP-GENL Study Dashboard")
    st.markdown("---")
    
    # Load data
    data = load_data()
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["📊 Progress Dashboard", "📝 Mock Exams", "🃏 Flashcards", "📄 Papers"]
    )
    
    # Initialize dates if not set
    if data["study_start_date"] is None:
        with st.sidebar:
            st.info("👋 Welcome! Please set your study dates.")
            study_start = st.date_input("Study Start Date", value=datetime.now().date())
            exam_date = st.date_input("Target Exam Date", value=datetime.now().date() + timedelta(days=14))
            if st.button("Save Dates"):
                data["study_start_date"] = str(study_start)
                data["target_exam_date"] = str(exam_date)
                save_data(data)
                st.success("Dates saved!")
                st.rerun()
    
    if page == "📊 Progress Dashboard":
        show_progress_dashboard(data)
    elif page == "📝 Mock Exams":
        show_mock_exams(data)
    elif page == "🃏 Flashcards":
        show_flashcards(data)
    elif page == "📄 Papers":
        show_papers(data)

def show_progress_dashboard(data):
    st.header("📊 Progress Dashboard")
    
    # Overall Progress
    col1, col2, col3 = st.columns(3)
    
    with col1:
        days_remaining = 0
        if data["study_start_date"] and data["target_exam_date"]:
            start = datetime.strptime(data["study_start_date"], "%Y-%m-%d").date()
            target = datetime.strptime(data["target_exam_date"], "%Y-%m-%d").date()
            days_remaining = (target - datetime.now().date()).days
        
        st.metric("Days Remaining", days_remaining)
        if data["study_start_date"]:
            st.caption(f"Started: {data['study_start_date']}")
        if data["target_exam_date"]:
            st.caption(f"Target: {data['target_exam_date']}")
    
    with col2:
        completed_hours = data["study_hours"]["completed"]
        target_hours = data["study_hours"]["target"]
        progress_hours = min(100, (completed_hours / target_hours * 100)) if target_hours > 0 else 0
        st.metric("Study Hours", f"{completed_hours}/{target_hours}", f"{progress_hours:.1f}%")
        st.progress(progress_hours / 100)
    
    with col3:
        total_q = data["practice_questions"]["total_attempted"]
        target_q = data["practice_questions"]["target"]
        progress_q = min(100, (total_q / target_q * 100)) if target_q > 0 else 0
        st.metric("Practice Questions", f"{total_q}/{target_q}", f"{progress_q:.1f}%")
        st.progress(progress_q / 100)
    
    st.markdown("---")
    
    # Domain Mastery
    st.subheader("Domain Mastery Levels")
    
    domain_data = []
    for domain, info in DOMAINS.items():
        accuracy = data["domain_mastery"][domain]
        domain_data.append({
            "Domain": domain,
            "Weight": f"{info['weight']}%",
            "Target": f">{info['target']}%",
            "Current": f"{accuracy}%",
            "Status": get_status_emoji(accuracy),
            "Accuracy": accuracy
        })
    
    df_domains = pd.DataFrame(domain_data)
    
    # Visual chart
    fig = px.bar(
        df_domains,
        x="Domain",
        y="Accuracy",
        color="Accuracy",
        color_continuous_scale=["red", "yellow", "green"],
        title="Domain Mastery Levels",
        labels={"Accuracy": "Accuracy (%)"}
    )
    fig.add_hline(y=75, line_dash="dash", line_color="green", annotation_text="Target: 75%")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Domain table
    st.dataframe(
        df_domains[["Domain", "Weight", "Target", "Current", "Status"]],
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Practice Question Statistics
    st.subheader("Practice Question Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    total_attempted = data["practice_questions"]["total_attempted"]
    total_correct = data["practice_questions"]["total_correct"]
    total_incorrect = total_attempted - total_correct
    overall_accuracy = (total_correct / total_attempted * 100) if total_attempted > 0 else 0
    
    with col1:
        st.metric("Total Attempted", total_attempted)
    with col2:
        st.metric("Total Correct", total_correct)
    with col3:
        st.metric("Total Incorrect", total_incorrect)
    with col4:
        st.metric("Overall Accuracy", f"{overall_accuracy:.1f}%")
    
    # Domain breakdown
    domain_stats = []
    for domain in DOMAINS:
        stats = data["practice_questions"]["by_domain"][domain]
        attempted = stats["attempted"]
        correct = stats["correct"]
        accuracy = (correct / attempted * 100) if attempted > 0 else 0
        domain_stats.append({
            "Domain": domain,
            "Attempted": attempted,
            "Correct": correct,
            "Incorrect": stats["incorrect"],
            "Accuracy": accuracy
        })
    
    df_stats = pd.DataFrame(domain_stats)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Daily Study Hours Log
    st.subheader("Daily Study Hours Log")
    
    with st.expander("Add Study Hours"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            study_date = st.date_input("Date", value=datetime.now().date())
        with col2:
            morning_hours = st.number_input("Morning Hours", min_value=0.0, max_value=10.0, step=0.5, value=0.0)
        with col3:
            afternoon_hours = st.number_input("Afternoon Hours", min_value=0.0, max_value=10.0, step=0.5, value=0.0)
        with col4:
            evening_hours = st.number_input("Evening Hours", min_value=0.0, max_value=10.0, step=0.5, value=0.0)
        
        notes = st.text_input("Notes (optional)")
        
        if st.button("Add Hours"):
            total_day_hours = morning_hours + afternoon_hours + evening_hours
            data["study_hours"]["daily_log"].append({
                "date": str(study_date),
                "total": total_day_hours,
                "morning": morning_hours,
                "afternoon": afternoon_hours,
                "evening": evening_hours,
                "notes": notes
            })
            data["study_hours"]["completed"] += total_day_hours
            save_data(data)
            st.success(f"Added {total_day_hours} hours for {study_date}")
            st.rerun()
    
    # Display study log
    if data["study_hours"]["daily_log"]:
        log_df = pd.DataFrame(data["study_hours"]["daily_log"])
        log_df = log_df.sort_values("date", ascending=False)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        
        # Chart
        fig_hours = px.line(
            log_df.sort_values("date"),
            x="date",
            y="total",
            title="Daily Study Hours Trend",
            labels={"date": "Date", "total": "Hours"}
        )
        st.plotly_chart(fig_hours, use_container_width=True)
    else:
        st.info("No study hours logged yet. Add your first study session above!")
    
    st.markdown("---")
    
    # Mock Exam Results Summary
    st.subheader("Mock Exam Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if data["mock_exams"]["mock_1"]:
            mock1 = data["mock_exams"]["mock_1"]
            st.metric("Mock Exam 1", f"{mock1.get('score', 0)}%", f"{mock1.get('correct', 0)}/65")
            st.caption(f"Date: {mock1.get('date', 'N/A')}")
        else:
            st.info("Mock Exam 1: Not completed")
    
    with col2:
        if data["mock_exams"]["mock_2"]:
            mock2 = data["mock_exams"]["mock_2"]
            improvement = mock2.get('score', 0) - (mock1.get('score', 0) if data["mock_exams"]["mock_1"] else 0)
            st.metric("Mock Exam 2", f"{mock2.get('score', 0)}%", f"{improvement:+.1f}%")
            st.caption(f"Date: {mock2.get('date', 'N/A')}")
        else:
            st.info("Mock Exam 2: Not completed")
    
    st.markdown("---")
    
    # Success Metrics
    st.subheader("Success Metrics")
    
    metrics_data = {
        "Metric": ["Practice Questions", "Domain Accuracy (All)", "Mock Exam 1 Score", "Mock Exam 2 Score", "Study Hours"],
        "Target": ["400+", ">75%", "Baseline", ">80%", "50+"],
        "Current": [
            f"{total_attempted}",
            f"{min([data['domain_mastery'][d] for d in DOMAINS]) if any(data['domain_mastery'].values()) else 0}%",
            f"{mock1.get('score', 0)}%" if data["mock_exams"]["mock_1"] else "N/A",
            f"{mock2.get('score', 0)}%" if data["mock_exams"]["mock_2"] else "N/A",
            f"{completed_hours}"
        ],
        "Status": [
            "✅" if total_attempted >= 400 else "⬜",
            "✅" if all(data['domain_mastery'][d] >= 75 for d in DOMAINS if data['domain_mastery'][d] > 0) else "⬜",
            "✅" if data["mock_exams"]["mock_1"] else "⬜",
            "✅" if (data["mock_exams"]["mock_2"] and mock2.get('score', 0) >= 80) else "⬜",
            "✅" if completed_hours >= 50 else "⬜"
        ]
    }
    
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

def show_mock_exams(data):
    st.header("📝 Mock Exam Management")
    
    exam_type = st.radio("Select Mock Exam", ["Mock Exam 1 (Day 7)", "Mock Exam 2 (Day 14)"], horizontal=True)
    exam_key = "mock_1" if exam_type == "Mock Exam 1 (Day 7)" else "mock_2"
    
    st.markdown("---")
    
    # Input form
    with st.form(f"mock_exam_form_{exam_key}"):
        st.subheader(f"Enter {exam_type} Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            exam_date = st.date_input("Exam Date", value=datetime.now().date())
        with col2:
            total_questions = st.number_input("Total Questions", min_value=1, max_value=100, value=65)
        with col3:
            correct_answers = st.number_input("Correct Answers", min_value=0, max_value=total_questions, value=0)
        
        score = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        st.metric("Score", f"{score:.1f}%")
        
        time_taken = st.number_input("Time Taken (minutes)", min_value=0, max_value=180, value=120)
        
        st.subheader("Domain Breakdown")
        domain_results = {}
        for domain in DOMAINS:
            col1, col2, col3 = st.columns(3)
            with col1:
                questions = st.number_input(f"{domain} - Questions", min_value=0, value=0, key=f"q_{exam_key}_{domain}")
            with col2:
                correct = st.number_input(f"{domain} - Correct", min_value=0, max_value=questions, value=0, key=f"c_{exam_key}_{domain}")
            with col3:
                accuracy = (correct / questions * 100) if questions > 0 else 0
                st.metric("Accuracy", f"{accuracy:.1f}%")
            
            domain_results[domain] = {
                "questions": questions,
                "correct": correct,
                "incorrect": questions - correct,
                "accuracy": accuracy
            }
        
        submitted = st.form_submit_button("Save Mock Exam Results")
        
        if submitted:
            data["mock_exams"][exam_key] = {
                "date": str(exam_date),
                "total_questions": total_questions,
                "correct": correct_answers,
                "score": score,
                "time_taken": time_taken,
                "domain_breakdown": domain_results
            }
            
            # Update domain mastery based on mock exam
            for domain, results in domain_results.items():
                if results["questions"] > 0:
                    # Update mastery if mock exam accuracy is higher or if no previous data
                    if data["domain_mastery"][domain] == 0 or results["accuracy"] > data["domain_mastery"][domain]:
                        data["domain_mastery"][domain] = results["accuracy"]
            
            save_data(data)
            st.success(f"{exam_type} results saved!")
            st.rerun()
    
    st.markdown("---")
    
    # Display existing results
    if data["mock_exams"][exam_key]:
        st.subheader(f"{exam_type} - Results")
        mock = data["mock_exams"][exam_key]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Score", f"{mock['score']:.1f}%")
        with col2:
            st.metric("Correct", f"{mock['correct']}/{mock['total_questions']}")
        with col3:
            st.metric("Time Taken", f"{mock['time_taken']} min")
        with col4:
            st.metric("Date", mock['date'])
        
        # Domain breakdown chart
        domain_breakdown = mock.get("domain_breakdown", {})
        if domain_breakdown:
            breakdown_data = []
            for domain, results in domain_breakdown.items():
                breakdown_data.append({
                    "Domain": domain,
                    "Accuracy": results["accuracy"],
                    "Weight": DOMAINS[domain]["weight"]
                })
            
            df_breakdown = pd.DataFrame(breakdown_data)
            fig = px.bar(
                df_breakdown,
                x="Domain",
                y="Accuracy",
                color="Accuracy",
                color_continuous_scale=["red", "yellow", "green"],
                title=f"{exam_type} - Domain Breakdown"
            )
            fig.add_hline(y=75, line_dash="dash", line_color="green", annotation_text="Target: 75%")
            st.plotly_chart(fig, use_container_width=True)
            
            # Weak areas
            weak_areas = [d for d, r in domain_breakdown.items() if r["accuracy"] < 70]
            if weak_areas:
                st.warning(f"⚠️ Weak Areas (<70%): {', '.join(weak_areas)}")
    
    # Comparison between Mock 1 and Mock 2
    if data["mock_exams"]["mock_1"] and data["mock_exams"]["mock_2"]:
        st.markdown("---")
        st.subheader("Mock Exam Comparison")
        
        mock1 = data["mock_exams"]["mock_1"]
        mock2 = data["mock_exams"]["mock_2"]
        
        comparison_data = []
        for domain in DOMAINS:
            acc1 = mock1.get("domain_breakdown", {}).get(domain, {}).get("accuracy", 0)
            acc2 = mock2.get("domain_breakdown", {}).get(domain, {}).get("accuracy", 0)
            improvement = acc2 - acc1
            comparison_data.append({
                "Domain": domain,
                "Mock 1": acc1,
                "Mock 2": acc2,
                "Improvement": improvement
            })
        
        df_comp = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Mock 1", x=df_comp["Domain"], y=df_comp["Mock 1"]))
        fig.add_trace(go.Bar(name="Mock 2", x=df_comp["Domain"], y=df_comp["Mock 2"]))
        fig.update_layout(title="Mock Exam Comparison by Domain", barmode="group", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

def show_flashcards(data):
    st.header("🃏 Flashcard Review System")
    
    # Domain selector
    selected_domain = st.selectbox("Select Domain", ["All Domains"] + list(DOMAINS.keys()))
    
    # Filter flashcards
    flashcards_to_show = {}
    if selected_domain == "All Domains":
        flashcards_to_show = data["flashcards"]
    else:
        flashcards_to_show = {selected_domain: data["flashcards"][selected_domain]}
    
    st.markdown("---")
    
    # Statistics
    total_cards = sum(len(topics) for topics in FLASHCARD_TOPICS.values())
    mastered = sum(1 for domain_flashcards in data["flashcards"].values() 
                   for card in domain_flashcards.values() if card["status"] == "mastered")
    need_review = sum(1 for domain_flashcards in data["flashcards"].values() 
                      for card in domain_flashcards.values() if card["status"] == "need_review")
    new_cards = sum(1 for domain_flashcards in data["flashcards"].values() 
                    for card in domain_flashcards.values() if card["status"] == "new")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cards", total_cards)
    with col2:
        st.metric("Mastered", mastered, f"{mastered/total_cards*100:.1f}%")
    with col3:
        st.metric("Need Review", need_review)
    with col4:
        st.metric("New Cards", new_cards)
    
    st.markdown("---")
    
    # Flashcard review interface
    st.subheader("Flashcard Review")
    
    # Get cards to review (prioritize need_review, then new)
    cards_to_review = []
    for domain, topics in flashcards_to_show.items():
        for topic, card_data in topics.items():
            cards_to_review.append({
                "domain": domain,
                "topic": topic,
                "status": card_data["status"],
                "review_count": card_data.get("review_count", 0)
            })
    
    # Sort by priority: need_review > new > mastered
    def sort_priority(card):
        status_order = {"need_review": 0, "new": 1, "mastered": 2}
        return (status_order.get(card["status"], 3), -card["review_count"])
    
    cards_to_review.sort(key=sort_priority)
    
    if cards_to_review:
        # Display current card
        if "current_card_index" not in st.session_state:
            st.session_state.current_card_index = 0
        if "show_answer" not in st.session_state:
            st.session_state.show_answer = {}
        
        current_index = st.session_state.current_card_index % len(cards_to_review)
        current_card = cards_to_review[current_index]
        card_key = f"{current_card['domain']}_{current_card['topic']}"
        
        # Card display
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"**Domain**: {current_card['domain']}")
            st.subheader(f"📌 {current_card['topic']}")
        
        with col2:
            status_emoji = {"new": "🆕", "need_review": "🔄", "mastered": "✅"}
            st.metric("Status", status_emoji.get(current_card['status'], '❓'))
            st.caption(f"Reviews: {current_card['review_count']}")
        
        st.markdown("---")
        
        # Get flashcard content (enhanced format)
        card_content = get_flashcard_content(current_card['domain'], current_card['topic'])
        
        # Fallback to old format if not in enhanced
        if card_content is None:
            card_content = FLASHCARD_CONTENT.get(current_card['domain'], {}).get(current_card['topic'])
            if isinstance(card_content, str):
                card_content = {"definition": card_content}
        
        # Show answer toggle
        show_answer = st.session_state.show_answer.get(card_key, False)
        
        if card_content:
            if show_answer:
                st.success("**📚 Detailed Explanation:**")
                
                # Definition
                if isinstance(card_content, dict) and "definition" in card_content:
                    st.markdown("### 📖 Definition")
                    st.markdown(card_content["definition"])
                
                # Formula
                if isinstance(card_content, dict) and card_content.get("formula"):
                    st.markdown("### 🔢 Formula / Math")
                    st.code(card_content["formula"], language="text")
                
                # Diagram (Mermaid)
                if isinstance(card_content, dict) and card_content.get("diagram"):
                    st.markdown("### 📊 Diagram")
                    st.code(card_content["diagram"], language="mermaid")
                    st.info("💡 **Tip**: Copy the diagram code above and paste it into [mermaid.live](https://mermaid.live) to visualize it!")
                
                # Code Example
                if isinstance(card_content, dict) and card_content.get("code_example"):
                    st.markdown("### 💻 Code Example")
                    st.code(card_content["code_example"], language="python")
                
                # Links - organized by type
                if isinstance(card_content, dict) and card_content.get("links"):
                    # Separate papers, tutorials, and videos
                    papers = [l for l in card_content["links"] if l.get("type") == "paper"]
                    tutorials = [l for l in card_content["links"] if l.get("type") == "tutorial"]
                    videos = [l for l in card_content["links"] if l.get("type") == "video"]
                    other_links = [l for l in card_content["links"] if l.get("type") not in ["paper", "tutorial", "video"]]
                    
                    if papers:
                        st.markdown("### 📄 Research Papers")
                        for paper in papers:
                            st.markdown(f"- [{paper['title']}]({paper['url']})")
                    
                    if tutorials:
                        st.markdown("### 📚 Tutorials & Guides")
                        for tutorial in tutorials:
                            st.markdown(f"- [{tutorial['title']}]({tutorial['url']})")
                    
                    if videos:
                        st.markdown("### 🎥 Videos")
                        for video in videos:
                            st.markdown(f"- [{video['title']}]({video['url']})")
                    
                    if other_links:
                        st.markdown("### 🔗 Additional Resources")
                        for link in other_links:
                            st.markdown(f"- [{link['title']}]({link['url']})")
                
                # Fallback for old string format
                elif isinstance(card_content, str):
                    st.markdown(f"💡 {card_content}")
                
                if st.button("🙈 Hide Answer", use_container_width=True):
                    st.session_state.show_answer[card_key] = False
                    st.rerun()
            else:
                st.info("💭 **Think about the answer, then click below to reveal detailed explanation.**")
                if isinstance(card_content, dict):
                    # Show preview
                    if "formula" in card_content:
                        st.caption("📌 Contains: Definition, Formula, Diagram, Code, Links")
                    else:
                        st.caption("📌 Contains: Definition")
                if st.button("👁️ Show Answer", use_container_width=True, type="primary"):
                    st.session_state.show_answer[card_key] = True
                    st.rerun()
        else:
            st.warning("⚠️ Content not available for this flashcard. Please refer to study materials.")
            st.markdown("**💡 Study Tip**: Review the concept in your study materials, then mark your status below.")
        
        st.markdown("---")
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Need Review", use_container_width=True):
                update_card_status(data, current_card['domain'], current_card['topic'], "need_review")
                st.session_state.show_answer[card_key] = False
                st.rerun()
        
        with col2:
            if st.button("✅ Mastered", use_container_width=True):
                update_card_status(data, current_card['domain'], current_card['topic'], "mastered")
                st.session_state.show_answer[card_key] = False
                st.rerun()
        
        with col3:
            if st.button("⏭️ Next Card", use_container_width=True):
                st.session_state.current_card_index += 1
                st.session_state.show_answer[card_key] = False
                st.rerun()
        
        with col4:
            if st.button("🔄 Reset to First", use_container_width=True):
                st.session_state.current_card_index = 0
                st.session_state.show_answer = {}
                st.rerun()
        
        # Progress
        progress = (current_index + 1) / len(cards_to_review)
        st.progress(progress)
        st.caption(f"Card {current_index + 1} of {len(cards_to_review)}")
        
        # Navigation shortcuts
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏮️ Previous Card", use_container_width=True):
                st.session_state.current_card_index = (st.session_state.current_card_index - 1) % len(cards_to_review)
                st.session_state.show_answer[card_key] = False
                st.rerun()
        with col2:
            if st.button("🔀 Random Card", use_container_width=True):
                import random
                st.session_state.current_card_index = random.randint(0, len(cards_to_review) - 1)
                st.session_state.show_answer[card_key] = False
                st.rerun()
    else:
        st.info("No flashcards available for the selected domain.")
    
    st.markdown("---")
    
    # Flashcard browser/viewer
    view_mode = st.radio("View Mode", ["Review Mode", "Browse All"], horizontal=True)
    
    if view_mode == "Browse All":
        st.subheader("Browse All Flashcards")
        
        # Search functionality
        search_term = st.text_input("🔍 Search flashcards", placeholder="Enter topic name...")
        
        for domain, topics in flashcards_to_show.items():
            # Filter by search term
            filtered_topics = {k: v for k, v in topics.items() 
                             if search_term.lower() in k.lower() or search_term == ""}
            
            if filtered_topics:
                with st.expander(f"{domain} ({len(filtered_topics)} cards)", expanded=False):
                    status_counts = {"new": 0, "need_review": 0, "mastered": 0}
                    for topic, card_data in filtered_topics.items():
                        status = card_data["status"]
                        status_counts[status] = status_counts.get(status, 0) + 1
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("New", status_counts["new"])
                    with col2:
                        st.metric("Need Review", status_counts["need_review"])
                    with col3:
                        st.metric("Mastered", status_counts["mastered"])
                    
                    st.markdown("---")
                    
                    # Display all cards with content
                    for topic, card_data in filtered_topics.items():
                        status_emoji = {"new": "🆕", "need_review": "🔄", "mastered": "✅"}
                        emoji = status_emoji.get(card_data["status"], "❓")
                        
                        st.markdown(f"### {emoji} {topic}")
                        st.caption(f"Status: {card_data['status']} | Reviews: {card_data.get('review_count', 0)}")
                        
                        # Show content if available (enhanced format)
                        card_content = get_flashcard_content(domain, topic)
                        if card_content is None:
                            card_content = FLASHCARD_CONTENT.get(domain, {}).get(topic)
                            if isinstance(card_content, str):
                                card_content = {"definition": card_content}
                        
                        if card_content:
                            with st.expander("View Detailed Explanation"):
                                if isinstance(card_content, dict):
                                    if "definition" in card_content:
                                        st.markdown("**Definition:**")
                                        st.markdown(card_content["definition"])
                                    
                                    if card_content.get("formula"):
                                        st.markdown("**Formula:**")
                                        st.code(card_content["formula"], language="text")
                                    
                                    if card_content.get("diagram"):
                                        st.markdown("**Diagram:**")
                                        st.code(card_content["diagram"], language="mermaid")
                                        st.caption("💡 Copy to mermaid.live to visualize")
                                    
                                    if card_content.get("code_example"):
                                        st.markdown("**Code Example:**")
                                        st.code(card_content["code_example"], language="python")
                                    
                                    if card_content.get("links"):
                                        # Organize links by type
                                        papers = [l for l in card_content["links"] if l.get("type") == "paper"]
                                        tutorials = [l for l in card_content["links"] if l.get("type") == "tutorial"]
                                        videos = [l for l in card_content["links"] if l.get("type") == "video"]
                                        other_links = [l for l in card_content["links"] if l.get("type") not in ["paper", "tutorial", "video"]]
                                        
                                        if papers:
                                            st.markdown("**📄 Research Papers:**")
                                            for paper in papers:
                                                st.markdown(f"- [{paper['title']}]({paper['url']})")
                                        
                                        if tutorials:
                                            st.markdown("**📚 Tutorials:**")
                                            for tutorial in tutorials:
                                                st.markdown(f"- [{tutorial['title']}]({tutorial['url']})")
                                        
                                        if videos:
                                            st.markdown("**🎥 Videos:**")
                                            for video in videos:
                                                st.markdown(f"- [{video['title']}]({video['url']})")
                                        
                                        if other_links:
                                            st.markdown("**🔗 Other Resources:**")
                                            for link in other_links:
                                                st.markdown(f"- [{link['title']}]({link['url']})")
                                else:
                                    st.markdown(card_content)
                        else:
                            st.info("Content not available - refer to study materials")
                        
                        # Quick status update buttons
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button(f"🔄 Need Review", key=f"br_{domain}_{topic}_review"):
                                update_card_status(data, domain, topic, "need_review")
                                st.rerun()
                        with col2:
                            if st.button(f"✅ Mastered", key=f"br_{domain}_{topic}_mastered"):
                                update_card_status(data, domain, topic, "mastered")
                                st.rerun()
                        with col3:
                            if st.button(f"🆕 Reset to New", key=f"br_{domain}_{topic}_new"):
                                update_card_status(data, domain, topic, "new")
                                st.rerun()
                        
                        st.markdown("---")
    else:
        # Flashcard list by domain (compact view)
        st.subheader("Flashcard Status by Domain")
        
        for domain, topics in flashcards_to_show.items():
            with st.expander(f"{domain} ({len(topics)} cards)"):
                status_counts = {"new": 0, "need_review": 0, "mastered": 0}
                for topic, card_data in topics.items():
                    status = card_data["status"]
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("New", status_counts["new"])
                with col2:
                    st.metric("Need Review", status_counts["need_review"])
                with col3:
                    st.metric("Mastered", status_counts["mastered"])
                
                # List all cards
                for topic, card_data in topics.items():
                    status_emoji = {"new": "🆕", "need_review": "🔄", "mastered": "✅"}
                    emoji = status_emoji.get(card_data["status"], "❓")
                    has_content = "📚" if topic in FLASHCARD_CONTENT.get(domain, {}) else ""
                    st.write(f"{emoji} {has_content} **{topic}** (Reviews: {card_data.get('review_count', 0)})")

def update_card_status(data, domain, topic, new_status):
    """Update flashcard status"""
    if domain in data["flashcards"] and topic in data["flashcards"][domain]:
        data["flashcards"][domain][topic]["status"] = new_status
        data["flashcards"][domain][topic]["last_reviewed"] = str(datetime.now())
        data["flashcards"][domain][topic]["review_count"] = data["flashcards"][domain][topic].get("review_count", 0) + 1
        save_data(data)

def show_papers(data):
    """Display paper reading tracker and management"""
    st.header("📄 Research Papers Tracker")
    
    # Initialize papers if needed
    if "papers" not in data:
        data["papers"] = {}
    
    papers = data["papers"]
    
    # Statistics
    total_papers = len(papers)
    read_papers = sum(1 for p in papers.values() if p.get("read", False))
    unread_papers = total_papers - read_papers
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Papers", total_papers)
    with col2:
        st.metric("Read", read_papers)
    with col3:
        st.metric("Unread", unread_papers)
    with col4:
        progress = (read_papers / total_papers * 100) if total_papers > 0 else 0
        st.metric("Progress", f"{progress:.1f}%")
    
    if total_papers > 0:
        st.progress(read_papers / total_papers)
    
    st.markdown("---")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filter_status = st.selectbox("Filter by Status", ["All", "Read", "Unread"])
    with col2:
        filter_priority = st.selectbox("Filter by Priority", ["All", "Must Read", "Optional", "Reference"])
    with col3:
        filter_difficulty = st.selectbox("Filter by Difficulty", ["All", "Easy", "Medium", "Hard"])
    with col4:
        filter_domain = st.selectbox("Filter by Domain", ["All"] + DOMAINS)
    
    # Apply filters
    filtered_papers = {}
    for paper_id, paper in papers.items():
        if filter_status != "All":
            if filter_status == "Read" and not paper.get("read", False):
                continue
            if filter_status == "Unread" and paper.get("read", False):
                continue
        
        if filter_priority != "All" and paper.get("priority") != filter_priority.lower():
            continue
        
        if filter_difficulty != "All" and paper.get("difficulty") != filter_difficulty.lower():
            continue
        
        if filter_domain != "All" and paper.get("domain") != filter_domain:
            continue
        
        filtered_papers[paper_id] = paper
    
    st.markdown(f"**Showing {len(filtered_papers)} of {total_papers} papers**")
    st.markdown("---")
    
    # Display papers
    if filtered_papers:
        # Sort by priority and read status
        sorted_papers = sorted(
            filtered_papers.items(),
            key=lambda x: (
                x[1].get("priority") == "must read",
                not x[1].get("read", False),
                x[1].get("domain", "")
            ),
            reverse=True
        )
        
        for paper_id, paper in sorted_papers:
            with st.expander(f"{'✅' if paper.get('read') else '📄'} {paper.get('title', 'Untitled')}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Domain**: {paper.get('domain', 'Unknown')}")
                    st.markdown(f"**Topic**: {paper.get('topic', 'Unknown')}")
                    st.markdown(f"**URL**: [{paper.get('url', '')}]({paper.get('url', '')})")
                    
                    # Paper metadata
                    if paper.get("summary"):
                        st.markdown("**Summary:**")
                        st.info(paper["summary"])
                    
                    if paper.get("key_takeaways"):
                        st.markdown("**Key Takeaways:**")
                        for takeaway in paper["key_takeaways"]:
                            st.markdown(f"- {takeaway}")
                    
                    notes_key = f"notes_{paper_id}"
                    notes = st.text_area("Your Notes", value=paper.get("notes", ""), key=notes_key)
                    if st.button("Save Notes", key=f"save_notes_{paper_id}"):
                        update_paper_notes(data, paper_id, notes)
                        st.success("Notes saved!")
                        st.rerun()
                
                with col2:
                    # Read status
                    read_key = f"read_{paper_id}"
                    read_status = st.checkbox(
                        "Read",
                        value=paper.get("read", False),
                        key=read_key
                    )
                    if read_status != paper.get("read", False):
                        toggle_paper_read(data, paper_id, read_status)
                        st.rerun()
                    
                    # Priority
                    priority_key = f"priority_{paper_id}"
                    priority_options = ["Optional", "Must Read", "Reference"]
                    current_priority = paper.get("priority", "optional").title()
                    if current_priority not in priority_options:
                        current_priority = "Optional"
                    priority_index = priority_options.index(current_priority)
                    priority = st.selectbox(
                        "Priority",
                        priority_options,
                        index=priority_index,
                        key=priority_key
                    )
                    if priority.lower() != paper.get("priority", "optional"):
                        update_paper_priority(data, paper_id, priority)
                        st.rerun()
                    
                    # Difficulty
                    difficulty_key = f"difficulty_{paper_id}"
                    difficulty_options = ["", "Easy", "Medium", "Hard"]
                    current_difficulty = paper.get("difficulty", "").title() if paper.get("difficulty") else ""
                    if current_difficulty not in difficulty_options:
                        current_difficulty = ""
                    difficulty_index = difficulty_options.index(current_difficulty)
                    difficulty = st.selectbox(
                        "Difficulty",
                        difficulty_options,
                        index=difficulty_index,
                        key=difficulty_key
                    )
                    if difficulty.lower() if difficulty else None != paper.get("difficulty"):
                        update_paper_difficulty(data, paper_id, difficulty)
                        st.rerun()
                    
                    if paper.get("read_date"):
                        st.caption(f"Read: {paper.get('read_date')}")
                
                # Summary and takeaways input
                with st.expander("Add Summary & Takeaways"):
                    summary_key = f"summary_{paper_id}"
                    summary = st.text_area("Summary (2-3 sentences)", value=paper.get("summary", ""), key=summary_key)
                    takeaways_key = f"takeaways_{paper_id}"
                    takeaways_input = st.text_area("Key Takeaways (one per line)", value="\n".join(paper.get("key_takeaways", [])), key=takeaways_key)
                    
                    if st.button("Save Summary", key=f"save_summary_{paper_id}"):
                        update_paper_summary(data, paper_id, summary, takeaways_input)
                        st.success("Summary saved!")
                        st.rerun()
                
                st.markdown("---")
    else:
        st.info("No papers found. Papers will be automatically added when you view flashcards with research papers.")
    
    # Paper reading statistics by domain
    if papers:
        st.markdown("### 📊 Statistics by Domain")
        domain_stats = {}
        for paper in papers.values():
            domain = paper.get("domain", "Unknown")
            if domain not in domain_stats:
                domain_stats[domain] = {"total": 0, "read": 0}
            domain_stats[domain]["total"] += 1
            if paper.get("read", False):
                domain_stats[domain]["read"] += 1
        
        for domain, stats in sorted(domain_stats.items()):
            progress = (stats["read"] / stats["total"] * 100) if stats["total"] > 0 else 0
            st.markdown(f"**{domain}**: {stats['read']}/{stats['total']} read ({progress:.1f}%)")
            st.progress(stats["read"] / stats["total"] if stats["total"] > 0 else 0)

def toggle_paper_read(data, paper_id, read_status):
    """Toggle paper read status"""
    if "papers" not in data:
        data["papers"] = {}
    if paper_id in data["papers"]:
        data["papers"][paper_id]["read"] = read_status
        if read_status:
            data["papers"][paper_id]["read_date"] = str(datetime.now().date())
        else:
            data["papers"][paper_id]["read_date"] = None
        save_data(data)

def update_paper_notes(data, paper_id, notes):
    """Update paper notes"""
    if "papers" not in data:
        data["papers"] = {}
    if paper_id in data["papers"]:
        data["papers"][paper_id]["notes"] = notes
        save_data(data)

def update_paper_priority(data, paper_id, priority):
    """Update paper priority"""
    if "papers" not in data:
        data["papers"] = {}
    if paper_id in data["papers"]:
        data["papers"][paper_id]["priority"] = priority.lower()
        save_data(data)

def update_paper_difficulty(data, paper_id, difficulty):
    """Update paper difficulty"""
    if "papers" not in data:
        data["papers"] = {}
    if paper_id in data["papers"]:
        data["papers"][paper_id]["difficulty"] = difficulty.lower() if difficulty else None
        save_data(data)

def update_paper_summary(data, paper_id, summary, takeaways_input):
    """Update paper summary and takeaways"""
    if "papers" not in data:
        data["papers"] = {}
    if paper_id in data["papers"]:
        data["papers"][paper_id]["summary"] = summary
        takeaways = [t.strip() for t in takeaways_input.split("\n") if t.strip()]
        data["papers"][paper_id]["key_takeaways"] = takeaways
        save_data(data)

if __name__ == "__main__":
    main()
