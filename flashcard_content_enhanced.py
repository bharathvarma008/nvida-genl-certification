"""
Enhanced flashcard content with definitions, formulas, diagrams, code examples, and links
"""

FLASHCARD_CONTENT_ENHANCED = {
    "LLM Architecture": {
        "Multi-Head Attention": {
            "definition": "Parallel attention mechanisms that allow the model to focus on different aspects simultaneously. Each head learns different attention patterns, enabling richer representations.",
            "formula": "MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O\nwhere headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)\nand Attention(Q, K, V) = softmax(QK^T/√dₖ)V",
            "diagram": """graph LR
    A[Input Embeddings] --> B[Q, K, V<br/>Projections]
    B --> C1[Head 1<br/>Attention]
    B --> C2[Head 2<br/>Attention]
    B --> C3[Head h<br/>Attention]
    C1 --> D[Concat]
    C2 --> D
    C3 --> D
    D --> E[Output<br/>Projection W^O]
    E --> F[Output]""",
            "code_example": """# PyTorch-like pseudocode
def multi_head_attention(Q, K, V, num_heads):
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    # Split into multiple heads
    Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    
    # Scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    
    # Concat and project
    output = output.transpose(1, 2).contiguous()
    output = output.view(batch_size, seq_len, d_model)
    return output""",
            "links": [
                {"title": "📄 Attention Is All You Need (Original Paper)", "url": "https://arxiv.org/abs/1706.03762", "type": "paper"},
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 GPT-3: Language Models are Few-Shot Learners", "url": "https://arxiv.org/abs/2005.14165", "type": "paper"},
                {"title": "📄 GPT-4 Technical Report", "url": "https://arxiv.org/abs/2303.08774", "type": "paper"},
                {"title": "📄 LLaMA: Open and Efficient Foundation Language Models", "url": "https://arxiv.org/abs/2302.13971", "type": "paper"},
                {"title": "📄 PaLM: Scaling Language Modeling with Pathways", "url": "https://arxiv.org/abs/2204.02311", "type": "paper"},
                {"title": "📚 Illustrated Transformer (Jay Alammar)", "url": "https://jalammar.github.io/illustrated-transformer/", "type": "tutorial"},
                {"title": "🎥 Transformer Architecture Explained", "url": "https://www.youtube.com/watch?v=4Bdc55j80l8", "type": "video"}
            ]
        },
        "Self-Attention (Q, K, V)": {
            "definition": "Attention mechanism using Query (Q), Key (K), and Value (V) matrices. Allows tokens to attend to each other by computing similarity between queries and keys, then using values.",
            "formula": "Attention(Q, K, V) = softmax(QK^T/√dₖ)V\n\nWhere:\n- Q (Query): What information am I looking for?\n- K (Key): What information do I have?\n- V (Value): The actual information content\n- √dₖ: Scaling factor to prevent softmax saturation",
            "diagram": """graph TD
    A[Input Tokens] --> B[Linear Projections]
    B --> C[Q Matrix]
    B --> D[K Matrix]
    B --> E[V Matrix]
    C --> F[Q × K^T]
    D --> F
    F --> G[Scale by √dₖ]
    G --> H[Softmax]
    H --> I[Attention Weights]
    I --> J[× V Matrix]
    E --> J
    J --> K[Output]""",
            "code_example": """def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply to values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights""",
            "links": [
                {"title": "📄 Attention Is All You Need (Original)", "url": "https://arxiv.org/abs/1706.03762", "type": "paper"},
                {"title": "📄 On Layer Normalization in the Transformer Architecture", "url": "https://arxiv.org/abs/2002.04745", "type": "paper"},
                {"title": "📄 The Annotated Transformer", "url": "http://nlp.seas.harvard.edu/annotated-transformer/", "type": "tutorial"},
                {"title": "📄 Efficient Attention: Attention with Linear Complexities", "url": "https://arxiv.org/abs/1812.01243", "type": "paper"},
                {"title": "📚 The Illustrated Transformer", "url": "https://jalammar.github.io/illustrated-transformer/", "type": "tutorial"},
                {"title": "🎥 Attention Mechanism Explained", "url": "https://www.youtube.com/watch?v=4Bdc55j80l8", "type": "video"}
            ]
        },
        "KV Cache": {
            "definition": "Stores computed Key and Value matrices for previous tokens during generation. Avoids recomputation, significantly speeding up autoregressive generation by reusing cached K,V values.",
            "formula": "For token at position t:\n- Without cache: Compute K₁...Kₜ, V₁...Vₜ (O(t²))\n- With cache: Reuse K₁...Kₜ₋₁, V₁...Vₜ₋₁, compute only Kₜ, Vₜ (O(t))",
            "diagram": """graph LR
    A[Token t-1] --> B[Compute Kₜ₋₁, Vₜ₋₁]
    B --> C[KV Cache]
    A --> D[Token t]
    D --> E[Compute Kₜ, Vₜ]
    C --> F[Concat with<br/>Kₜ, Vₜ]
    E --> F
    F --> G[Attention<br/>with all tokens]
    G --> H[Output]""",
            "code_example": """# KV Cache implementation
class KVCache:
    def __init__(self):
        self.cached_k = None
        self.cached_v = None
    
    def forward(self, new_k, new_v):
        if self.cached_k is None:
            # First token
            self.cached_k = new_k
            self.cached_v = new_v
        else:
            # Append new K, V to cache
            self.cached_k = torch.cat([self.cached_k, new_k], dim=-2)
            self.cached_v = torch.cat([self.cached_v, new_v], dim=-2)
        
        return self.cached_k, self.cached_v""",
            "links": [
                {"title": "📄 FlashAttention: Fast and Memory-Efficient Exact Attention", "url": "https://arxiv.org/abs/2205.14135", "type": "paper"},
                {"title": "📄 PagedAttention: From Interface to Implementation", "url": "https://arxiv.org/abs/2309.06180", "type": "paper"},
                {"title": "📄 vLLM: Easy, Fast, and Cheap LLM Serving", "url": "https://arxiv.org/abs/2309.06180", "type": "paper"},
                {"title": "📄 Efficient Memory Management for Large Language Models", "url": "https://arxiv.org/abs/2309.06180", "type": "paper"},
                {"title": "📚 KV Cache Optimization Blog", "url": "https://lilianweng.github.io/posts/2023-01-10-inference-optimization/", "type": "tutorial"},
                {"title": "🎥 Faster LLM Inference", "url": "https://www.youtube.com/watch?v=5iX8qQSM3b0", "type": "video"}
            ]
        },
        "Rotary Positional Encoding (RoPE)": {
            "definition": "Relative positional encoding that rotates embeddings based on position. Enables longer context windows (e.g., 32K+ tokens) compared to absolute encoding by encoding relative positions.",
            "formula": "For position m, apply rotation:\nRₘ = [cos(mθᵢ)  -sin(mθᵢ)]\n     [sin(mθᵢ)   cos(mθᵢ)]\n\nwhere θᵢ = 10000^(-2i/d) for dimension i",
            "diagram": """graph TD
    A[Token Embedding] --> B[Split into Pairs]
    B --> C[Apply Rotation<br/>Matrix Rₘ]
    C --> D[Rotated Embedding]
    E[Position m] --> C
    D --> F[Attention Computation]""",
            "code_example": """def apply_rotary_pos_emb(x, freqs):
    # x: [batch, seq_len, n_heads, head_dim]
    # Split into pairs
    x1, x2 = x.chunk(2, dim=-1)
    
    # Apply rotation
    cos, sin = freqs.cos(), freqs.sin()
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    
    return rotated""",
            "links": [
                {"title": "📄 RoFormer: Enhanced Transformer with Rotary Position Embedding", "url": "https://arxiv.org/abs/2104.09864", "type": "paper"},
                {"title": "📄 LLaMA: Open and Efficient Foundation Language Models (Uses RoPE)", "url": "https://arxiv.org/abs/2302.13971", "type": "paper"},
                {"title": "📄 PaLM: Scaling Language Modeling (Uses RoPE)", "url": "https://arxiv.org/abs/2204.02311", "type": "paper"},
                {"title": "📄 GPT-NeoX-20B: An Open-Source Autoregressive Language Model", "url": "https://arxiv.org/abs/2204.06745", "type": "paper"},
                {"title": "📄 Extending Context Window of Large Language Models", "url": "https://arxiv.org/abs/2308.03281", "type": "paper"},
                {"title": "🎥 RoPE Explained", "url": "https://www.youtube.com/watch?v=o29P0Kpobz0", "type": "video"}
            ]
        },
        "Feed-Forward Blocks": {
            "definition": "Two linear transformations with an activation function (ReLU/GELU) between them. Processes the attended information from the attention layer. Typically expands dimension 4x then contracts back.",
            "formula": "FFN(x) = max(0, xW₁ + b₁)W₂ + b₂\nor\nFFN(x) = GELU(xW₁ + b₁)W₂ + b₂\n\nTypically: d_model → 4×d_model → d_model",
            "diagram": """graph LR
    A[Attention Output] --> B[Linear 1<br/>d → 4d]
    B --> C[Activation<br/>ReLU/GELU]
    C --> D[Linear 2<br/>4d → d]
    D --> E[Output]""",
            "code_example": """class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))""",
            "links": [
                {"title": "📄 Attention Is All You Need (Original)", "url": "https://arxiv.org/abs/1706.03762", "type": "paper"},
                {"title": "📄 GLU Variants Improve Transformer", "url": "https://arxiv.org/abs/2002.05202", "type": "paper"},
                {"title": "📄 Swish: A Self-Gated Activation Function", "url": "https://arxiv.org/abs/1710.05941", "type": "paper"}
            ]
        },
        "Residual Connections": {
            "definition": "Skip connections that add the input directly to the output. Prevents vanishing gradients and allows deeper networks to train effectively. Enables gradient flow through many layers.",
            "formula": "output = x + F(x)\n\nwhere F(x) is the transformation (attention or FFN)\n\nGradient: ∂L/∂x = ∂L/∂output × (1 + ∂F/∂x)",
            "diagram": """graph TD
    A[Input x] --> B[Transformation F]
    A --> C[Skip Connection]
    B --> D[F x]
    C --> E[Add]
    D --> E
    E --> F[Output x + F x]""",
            "code_example": """# Residual connection
def transformer_block(x):
    # Self-attention with residual
    x = x + self.attention(x)
    x = self.layer_norm1(x)
    
    # FFN with residual
    x = x + self.feed_forward(x)
    x = self.layer_norm2(x)
    
    return x""",
            "links": [
                {"title": "📄 Deep Residual Learning for Image Recognition", "url": "https://arxiv.org/abs/1512.03385", "type": "paper"},
                {"title": "📄 Attention Is All You Need (Uses Residuals)", "url": "https://arxiv.org/abs/1706.03762", "type": "paper"},
                {"title": "📄 Identity Mappings in Deep Residual Networks", "url": "https://arxiv.org/abs/1603.05027", "type": "paper"}
            ]
        },
        "Layer Normalization": {
            "definition": "Normalizes inputs to each layer, stabilizing training. Pre-norm (before attention) vs post-norm (after attention) are common variants. Reduces internal covariate shift.",
            "formula": "LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β\n\nwhere:\nμ = mean(x)\nσ² = variance(x)\nγ, β = learnable parameters\nε = small constant",
            "diagram": """graph LR
    A[Input] --> B[Compute Mean μ]
    A --> C[Compute Variance σ²]
    B --> D[Normalize<br/>x - μ / √σ²]
    C --> D
    D --> E[Scale & Shift<br/>γ * norm + β]
    E --> F[Output]""",
            "code_example": """import torch.nn as nn

# Pre-norm (before attention)
class PreNormTransformerBlock(nn.Module):
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# Post-norm (after attention)
class PostNormTransformerBlock(nn.Module):
    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ffn(x))
        return x""",
            "links": [
                {"title": "📄 Layer Normalization (Original)", "url": "https://arxiv.org/abs/1607.06450", "type": "paper"},
                {"title": "📄 On Layer Normalization in Transformer", "url": "https://arxiv.org/abs/2002.04745", "type": "paper"},
                {"title": "📄 Pre-Layer Normalization Transformer", "url": "https://arxiv.org/abs/2002.04745", "type": "paper"}
            ]
        },
        "Decoder-Only Architecture": {
            "definition": "GPT-style architecture that generates text autoregressively. Most modern LLMs use this. Simpler and better for text generation tasks. Uses masked self-attention.",
            "formula": "For position i:\nAttention can only attend to positions ≤ i\n\nMask: M[i,j] = 0 if j ≤ i, else -∞",
            "diagram": """graph TD
    A[Token 1] --> B[Self-Attention<br/>Masked]
    C[Token 2] --> B
    D[Token 3] --> B
    B --> E[FFN]
    E --> F[Next Token<br/>Prediction]""",
            "code_example": """# Decoder-only with causal mask
def causal_attention_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# GPT-style generation
def generate(model, prompt, max_length):
    for _ in range(max_length):
        logits = model(prompt)
        next_token = sample(logits[:, -1, :])
        prompt = torch.cat([prompt, next_token], dim=1)
    return prompt""",
            "links": [
                {"title": "📄 Language Models are Unsupervised Multitask Learners (GPT-2)", "url": "https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", "type": "paper"},
                {"title": "📄 Language Models are Few-Shot Learners (GPT-3)", "url": "https://arxiv.org/abs/2005.14165", "type": "paper"},
                {"title": "📄 LLaMA: Open and Efficient Foundation Language Models", "url": "https://arxiv.org/abs/2302.13971", "type": "paper"},
                {"title": "📄 PaLM: Scaling Language Modeling", "url": "https://arxiv.org/abs/2204.02311", "type": "paper"},
                {"title": "📄 GPT-4 Technical Report", "url": "https://arxiv.org/abs/2303.08774", "type": "paper"}
            ]
        },
        "Absolute Positional Encoding": {
            "definition": "Fixed sinusoidal patterns added to token embeddings to encode position information. Used in original Transformer. Each position gets a unique encoding based on sine/cosine functions of different frequencies.",
            "formula": "PE(pos, 2i) = sin(pos / 10000^(2i/d_model))\nPE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))\n\nwhere:\npos = position in sequence\ni = dimension index\nd_model = embedding dimension",
            "diagram": """graph TD
    A[Token Embedding] --> C[Add Positional Encoding]
    B[Positional Encoding<br/>Sinusoidal] --> C
    C --> D[Position-Aware<br/>Embedding]""",
            "code_example": """import torch
import math

def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# Add to embeddings
embeddings = token_embeddings + positional_encoding(seq_len, d_model)""",
            "links": [
                {"title": "📄 Attention Is All You Need (Original)", "url": "https://arxiv.org/abs/1706.03762", "type": "paper"},
                {"title": "📄 BERT: Pre-training Uses Learned Positional Embeddings", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context", "url": "https://arxiv.org/abs/1901.02860", "type": "paper"}
            ]
        },
        "Encoder-Decoder Architecture": {
            "definition": "Transformer architecture with both encoder (bidirectional) and decoder (autoregressive). Encoder processes input, decoder generates output. Used in BART, T5. Good for tasks requiring bidirectional understanding.",
            "formula": "Encoder: Processes input bidirectionally\nDecoder: Generates output autoregressively\n\nCross-Attention: Decoder attends to encoder output\n\nOutput = Decoder(Encoder(input), target_sequence)",
            "diagram": """graph TD
    A[Input Tokens] --> B[Encoder<br/>Bidirectional]
    B --> C[Encoder Output]
    C --> D[Cross-Attention]
    E[Target Tokens] --> F[Decoder<br/>Autoregressive]
    F --> D
    D --> G[Output Tokens]""",
            "code_example": """# Encoder-Decoder Architecture
from transformers import EncoderDecoderModel

# BART-style encoder-decoder
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "bert-base-uncased",  # Encoder
    "gpt2"  # Decoder
)

# Forward pass
encoder_outputs = model.encoder(input_ids=input_ids)
decoder_outputs = model.decoder(
    input_ids=target_ids,
    encoder_hidden_states=encoder_outputs.last_hidden_state
)""",
            "links": [
                {"title": "📄 Attention Is All You Need (Original)", "url": "https://arxiv.org/abs/1706.03762", "type": "paper"},
                {"title": "📄 BART: Denoising Sequence-to-Sequence Pre-training", "url": "https://arxiv.org/abs/1910.13461", "type": "paper"},
                {"title": "📄 T5: Text-To-Text Transfer Transformer", "url": "https://arxiv.org/abs/1910.10683", "type": "paper"},
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"}
            ]
        },
        "Scaling Laws": {
            "definition": "Empirical relationships between model size (parameters, depth, width), compute, data, and performance. Guide decisions on model architecture and training. Key insight: performance scales predictably with size.",
            "formula": "Performance ∝ (Parameters)^α × (Compute)^β × (Data)^γ\n\nKey relationships:\n- Depth vs Width: Deeper often better than wider\n- Parameters: More parameters → better performance (up to point)\n- Context Length: Longer context → more memory, slower\n- Throughput: Batch size × tokens/second",
            "diagram": """graph LR
    A[Model Size] --> D[Performance]
    B[Compute] --> D
    C[Data] --> D
    E[Depth] --> F[Trade-off]
    G[Width] --> F
    F --> D""",
            "code_example": """# Scaling Laws Analysis
# From Kaplan et al. (2020) - Scaling Laws for Neural Language Models

# Performance scales as:
# L(N) ≈ (N_c / N)^α
# where N = parameters, N_c = critical parameter count

# Typical scaling exponents:
# α ≈ 0.076 for loss
# β ≈ 0.095 for compute
# γ ≈ 0.095 for data

# Example: Doubling parameters improves loss by ~5%
loss_reduction = 1 - (2 ** -0.076)  # ≈ 5%""",
            "links": [
                {"title": "📄 Scaling Laws for Neural Language Models (Original)", "url": "https://arxiv.org/abs/2001.08361", "type": "paper"},
                {"title": "📄 Training Compute-Optimal Large Language Models (Chinchilla)", "url": "https://arxiv.org/abs/2203.15556", "type": "paper"},
                {"title": "📄 GPT-3: Language Models are Few-Shot Learners", "url": "https://arxiv.org/abs/2005.14165", "type": "paper"},
                {"title": "📄 PaLM: Scaling Language Modeling with Pathways", "url": "https://arxiv.org/abs/2204.02311", "type": "paper"},
                {"title": "📄 LLaMA: Open and Efficient Foundation Language Models", "url": "https://arxiv.org/abs/2302.13971", "type": "paper"}
            ]
        }
    },
    "Model Optimization": {
        "FP32": {
            "definition": "32-bit floating point precision. Baseline format with full precision, no quality loss. Highest memory usage but most accurate.",
            "formula": "Memory per parameter: 4 bytes\nTotal memory ≈ 4 × num_parameters bytes",
            "diagram": None,
            "code_example": """# FP32 model
model = model.float()  # PyTorch
# or
model = model.to(torch.float32)""",
            "links": [
                {"title": "📄 Mixed Precision Training (Original)", "url": "https://arxiv.org/abs/1710.03740", "type": "paper"},
                {"title": "📄 Training Deep Neural Networks with 8-bit Floating Point Numbers", "url": "https://arxiv.org/abs/1812.08011", "type": "paper"},
                {"title": "📄 BFLOAT16: The Secret to High Performance ML", "url": "https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus", "type": "paper"},
                {"title": "📚 PyTorch Mixed Precision", "url": "https://pytorch.org/docs/stable/amp.html", "type": "tutorial"},
                {"title": "🎥 Mixed Precision Explained", "url": "https://www.youtube.com/watch?v=OqCrNkjN_PM", "type": "video"}
            ]
        },
        "FP16": {
            "definition": "16-bit floating point. 2x memory reduction, 1.5-2x speed gain. Minimal quality loss. Common for training with mixed precision.",
            "formula": "Memory per parameter: 2 bytes\nSpeedup: ~1.5-2x\nMemory reduction: 2x",
            "diagram": """graph LR
    A[FP32 Model<br/>4 bytes/param] --> B[Mixed Precision]
    B --> C[FP16 Forward<br/>2 bytes/param]
    B --> D[FP32 Gradients<br/>4 bytes/param]
    C --> E[2x Speedup]""",
            "code_example": """# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()""",
            "links": [
                {"title": "📄 Mixed Precision Training (Original)", "url": "https://arxiv.org/abs/1710.03740", "type": "paper"},
                {"title": "📄 FP16 Training: Faster Training with Minimal Accuracy Loss", "url": "https://arxiv.org/abs/1710.03740", "type": "paper"},
                {"title": "📄 BFLOAT16 Training", "url": "https://arxiv.org/abs/1905.12322", "type": "paper"},
                {"title": "📚 PyTorch Mixed Precision", "url": "https://pytorch.org/docs/stable/amp.html", "type": "tutorial"},
                {"title": "🎥 Mixed Precision Explained", "url": "https://www.youtube.com/watch?v=OqCrNkjN_PM", "type": "video"}
            ]
        },
        "INT8": {
            "definition": "8-bit integer quantization. 4x memory reduction, 2-3x speed gain. Small quality loss. Good balance for inference.",
            "formula": "Memory per parameter: 1 byte\nQuantization: x_int8 = round(x_fp32 / scale)\nDequantization: x_fp32 ≈ x_int8 × scale",
            "diagram": """graph LR
    A[FP32 Weights] --> B[Calibration<br/>Find Scale]
    B --> C[Quantize to INT8]
    C --> D[INT8 Model<br/>1 byte/param]
    D --> E[4x Memory<br/>Reduction]""",
            "code_example": """# INT8 Quantization (TensorRT-LLM)
import tensorrt_llm

# Build engine with INT8
builder = tensorrt_llm.Builder()
network = builder.create_network()
# ... configure network ...

# Enable INT8 quantization
builder_config = builder.create_builder_config()
builder_config.set_flag(tensorrt_llm.BuilderFlag.INT8)

engine = builder.build_engine(network, builder_config)""",
            "links": [
                {"title": "📄 Quantization and Training of Neural Networks", "url": "https://arxiv.org/abs/1712.05877", "type": "paper"},
                {"title": "📄 Q8BERT: Quantized 8Bit BERT", "url": "https://arxiv.org/abs/1910.06188", "type": "paper"},
                {"title": "📄 LLM.int8(): 8-bit Matrix Multiplication", "url": "https://arxiv.org/abs/2208.07339", "type": "paper"},
                {"title": "📄 SmoothQuant: Accurate and Efficient Post-Training Quantization", "url": "https://arxiv.org/abs/2211.10438", "type": "paper"},
                {"title": "📄 GPTQ: Accurate Post-Training Quantization", "url": "https://arxiv.org/abs/2210.17323", "type": "paper"},
                {"title": "📄 AWQ: Activation-aware Weight Quantization", "url": "https://arxiv.org/abs/2306.00978", "type": "paper"},
                {"title": "📚 TensorRT-LLM Quantization", "url": "https://nvidia.github.io/TensorRT-LLM/quantization.html", "type": "tutorial"},
                {"title": "🎥 Quantization Explained", "url": "https://www.youtube.com/watch?v=OhFdxC3ph0k", "type": "video"}
            ]
        },
        "Temperature": {
            "definition": "Sampling parameter controlling randomness. Lower (0-0.7) = more deterministic, Higher (0.7-1.5) = more creative. Typical range: 0.7-1.0.",
            "formula": "P'(token) = exp(logit / temperature) / Σ exp(logitᵢ / temperature)\n\nTemperature effects:\n- T → 0: Deterministic (always pick highest)\n- T = 1: Original distribution\n- T → ∞: Uniform distribution",
            "diagram": """graph LR
    A[Logits] --> B[Divide by T]
    B --> C[Softmax]
    C --> D[Sample]
    E[T=0.1<br/>Deterministic] --> B
    F[T=1.0<br/>Balanced] --> B
    G[T=2.0<br/>Creative] --> B""",
            "code_example": """def sample_with_temperature(logits, temperature=1.0):
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Apply softmax
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Sample
    token = torch.multinomial(probs, num_samples=1)
    return token

# Usage
logits = model(input_ids)
token = sample_with_temperature(logits, temperature=0.7)  # More deterministic
token = sample_with_temperature(logits, temperature=1.2)  # More creative""",
            "links": [
                {"title": "📄 The Curious Case of Neural Text Degeneration", "url": "https://arxiv.org/abs/1904.09751", "type": "paper"},
                {"title": "📄 Language Models are Unsupervised Multitask Learners (GPT-2)", "url": "https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", "type": "paper"},
                {"title": "📄 Language Models are Few-Shot Learners (GPT-3)", "url": "https://arxiv.org/abs/2005.14165", "type": "paper"},
                {"title": "📄 Nucleus Sampling", "url": "https://arxiv.org/abs/1904.09751", "type": "paper"},
                {"title": "📚 Sampling Strategies (HuggingFace)", "url": "https://huggingface.co/blog/how-to-generate", "type": "tutorial"},
                {"title": "🎥 Temperature Explained", "url": "https://www.youtube.com/watch?v=MPmr6uL5U2c", "type": "video"}
            ]
        }
    },
    "Fine-Tuning": {
        "LoRA": {
            "definition": "Low-Rank Adaptation. Train small adapters (~1% parameters) instead of full model. Low VRAM, fast training. Use for task adaptation. Most popular PEFT method.",
            "formula": "W' = W + ΔW\nwhere ΔW = BA\nB ∈ R^(d×r), A ∈ R^(r×k)\nr << min(d, k) (rank, typically 4-16)\n\nParameters: d×r + r×k << d×k",
            "diagram": """graph TD
    A[Frozen Base Model W] --> B[Input]
    B --> C[LoRA Adapter<br/>ΔW = BA]
    C --> D[Output = Wx + BAx]
    E[Trainable:<br/>B and A only] --> C
    F[Frozen:<br/>W] --> A""",
            "code_example": """# LoRA implementation (PEFT library)
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.1,
)

model = get_peft_model(base_model, config)

# Only B and A matrices are trainable
# W remains frozen""",
            "links": [
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models (Original)", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 QLoRA: Efficient Finetuning of Quantized LLMs", "url": "https://arxiv.org/abs/2305.14314", "type": "paper"},
                {"title": "📄 AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning", "url": "https://arxiv.org/abs/2303.10512", "type": "paper"},
                {"title": "📄 LoRA+: Efficient Low Rank Adaptation", "url": "https://arxiv.org/abs/2402.12354", "type": "paper"},
                {"title": "📄 DoRA: Weight-Decomposed Low-Rank Adaptation", "url": "https://arxiv.org/abs/2402.09353", "type": "paper"},
                {"title": "📄 VeRA: Vector-based Random Matrix Adaptation", "url": "https://arxiv.org/abs/2310.11454", "type": "paper"},
                {"title": "📚 PEFT Library (HuggingFace)", "url": "https://github.com/huggingface/peft", "type": "tutorial"},
                {"title": "🎥 LoRA Explained", "url": "https://www.youtube.com/watch?v=YV7_5q3l9V0", "type": "video"}
            ]
        },
        "QLoRA": {
            "definition": "Quantized LoRA. 4-bit base model + FP16 LoRA adapters. Very low VRAM (~75% reduction). Use when resources are limited.",
            "formula": "Memory ≈ 4-bit weights + FP16 LoRA\n≈ 0.5 bytes/param (base) + 2 bytes/param (LoRA)\nTotal: ~75% reduction vs FP32 full fine-tune",
            "diagram": """graph TD
    A[Base Model] --> B[4-bit Quantization]
    B --> C[Frozen 4-bit Model]
    C --> D[LoRA Adapters<br/>FP16]
    D --> E[Trainable<br/>B and A]
    F[Memory: ~75%<br/>reduction] --> E""",
            "code_example": """# QLoRA with bitsandbytes
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)

# Add LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, ...)
model = get_peft_model(model, lora_config)""",
            "links": [
                {"title": "📄 QLoRA: Efficient Finetuning of Quantized LLMs (Original)", "url": "https://arxiv.org/abs/2305.14314", "type": "paper"},
                {"title": "📄 LoRA: Low-Rank Adaptation (Foundation)", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 LLM.int8(): 8-bit Matrix Multiplication", "url": "https://arxiv.org/abs/2208.07339", "type": "paper"},
                {"title": "📄 GPTQ: Accurate Post-Training Quantization", "url": "https://arxiv.org/abs/2210.17323", "type": "paper"},
                {"title": "📄 4-bit Quantization via BitsAndBytes", "url": "https://arxiv.org/abs/2305.14314", "type": "paper"},
                {"title": "📚 QLoRA Tutorial", "url": "https://www.youtube.com/watch?v=J_3hDqSvpmg", "type": "video"},
                {"title": "📚 BitsAndBytes Library", "url": "https://github.com/TimDettmers/bitsandbytes", "type": "tutorial"}
            ]
        },
        "Full Fine-Tune": {
            "definition": "Update all model parameters. Very high VRAM, slow training. Use for major domain shifts. Best quality but expensive. Requires full model copy in memory.",
            "formula": "Memory ≈ 4 × num_parameters bytes (FP32)\n≈ 2 × num_parameters bytes (FP16)\n\nTraining time: O(num_parameters × dataset_size)",
            "diagram": """graph TD
    A[Pre-trained Model] --> B[All Parameters<br/>Trainable]
    B --> C[Forward Pass]
    C --> D[Backward Pass]
    D --> E[Update All Weights]
    E --> F[Fine-tuned Model]""",
            "code_example": """# Full fine-tuning
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("gpt2")

# All parameters are trainable
for param in model.parameters():
    param.requires_grad = True

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    # Requires large VRAM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()""",
            "links": [
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 Fine-tuning Language Models", "url": "https://arxiv.org/abs/2001.08361", "type": "paper"},
                {"title": "📄 Instruction Tuning with FLAN", "url": "https://arxiv.org/abs/2109.01652", "type": "paper"}
            ]
        },
        "Rank (r)": {
            "definition": "Rank parameter in LoRA. Controls adapter size. Lower rank = fewer parameters but less capacity. Higher rank = more parameters but more capacity. Typical values: 4-16. Most common: 8.",
            "formula": "LoRA rank:\n- Rank r: Dimension of low-rank matrices\n- Parameters: 2 × r × d_model per layer\n- Typical r: 4-16\n- Most common: r = 8\n\nLower r = fewer params, less capacity\nHigher r = more params, more capacity",
            "diagram": """graph LR
    A[Weight Matrix<br/>d × d] --> B[LoRA Decomposition]
    B --> C[Low-Rank Matrices<br/>d × r and r × d]
    C --> D[Rank r Controls Size]""",
            "code_example": """# LoRA Rank (r)
from peft import LoraConfig, get_peft_model

# Rank r controls adapter size
# r=4: Small adapter, fewer parameters
# r=8: Medium adapter (most common)
# r=16: Large adapter, more parameters

lora_config = LoraConfig(
    r=8,  # Rank - controls adapter size
    lora_alpha=16,  # Typically 2× rank
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

model = get_peft_model(base_model, lora_config)

# Parameters added: 2 × r × d_model per target module
# r=8, d_model=768: ~12K params per module""",
            "links": [
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models (Original)", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 QLoRA: Efficient Finetuning of Quantized LLMs", "url": "https://arxiv.org/abs/2305.14314", "type": "paper"},
                {"title": "📚 PEFT Library Documentation", "url": "https://huggingface.co/docs/peft", "type": "tutorial"}
            ]
        },
        "Alpha": {
            "definition": "Scaling factor in LoRA. Controls adapter contribution. Typically set to 2× rank. Higher alpha = stronger adapter influence. Lower alpha = weaker adapter influence. Formula: output = base + (alpha/r) × adapter.",
            "formula": "LoRA Alpha:\n- Scaling factor for adapter\n- Typically: alpha = 2 × r\n- Formula: output = base + (alpha/r) × adapter_output\n\nHigher alpha = stronger adapter\nLower alpha = weaker adapter",
            "diagram": """graph LR
    A[Base Model Output] --> C[Final Output]
    B[Adapter Output] --> D[Scale by alpha/r]
    D --> C""",
            "code_example": """# LoRA Alpha (scaling factor)
from peft import LoraConfig

# Alpha controls adapter strength
# Typical: alpha = 2 × rank
# r=8 → alpha=16

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,  # Scaling factor (typically 2× rank)
    target_modules=["q_proj", "v_proj"]
)

# LoRA formula:
# W' = W + (alpha/r) × ΔW
# where ΔW = B × A (low-rank decomposition)

# Higher alpha: Stronger adapter influence
# Lower alpha: Weaker adapter influence
# alpha/r ratio determines scaling""",
            "links": [
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models (Original)", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📚 PEFT Library Documentation", "url": "https://huggingface.co/docs/peft", "type": "tutorial"}
            ]
        },
        "Target Modules": {
            "definition": "Which model modules to apply LoRA adapters to. Typically attention layers: Q, K, V, O projections. Can also target FFN layers. More modules = more parameters but potentially better performance.",
            "formula": "Target Modules:\n- Attention: q_proj, k_proj, v_proj, o_proj\n- FFN: gate_proj, up_proj, down_proj\n- All: All linear layers\n\nMore modules = more parameters\nFewer modules = fewer parameters",
            "diagram": """graph TD
    A[Model Layers] --> B{Target Modules?}
    B -->|Attention Only| C[Q, K, V, O]
    B -->|Attention + FFN| D[Q, K, V, O, Gate, Up, Down]
    B -->|All| E[All Linear Layers]
    C --> F[LoRA Adapters]
    D --> F
    E --> F""",
            "code_example": """# LoRA Target Modules
from peft import LoraConfig

# Option 1: Attention layers only (most common)
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention only
)

# Option 2: Attention + FFN
lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"  # FFN
    ]
)

# Option 3: All linear layers
lora_config = LoraConfig(
    r=8,
    target_modules="all-linear"  # All linear layers
)

# More modules = more parameters but potentially better performance""",
            "links": [
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models (Original)", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 QLoRA: Efficient Finetuning of Quantized LLMs", "url": "https://arxiv.org/abs/2305.14314", "type": "paper"},
                {"title": "📚 PEFT Library Documentation", "url": "https://huggingface.co/docs/peft", "type": "tutorial"}
            ]
        },
        "Learning Rate": {
            "definition": "Step size for parameter updates during fine-tuning. Too high = unstable training. Too low = slow convergence. Typical ranges: 1e-5 to 5e-4. LoRA typically uses higher LR than full fine-tuning.",
            "formula": "Learning Rate:\n- Typical: 1e-5 to 5e-4\n- LoRA: 1e-4 to 5e-4 (higher)\n- Full FT: 1e-5 to 1e-4 (lower)\n\nUpdate: θ = θ - lr × ∇θ L(θ)",
            "diagram": """graph LR
    A[High LR] --> B[Fast but Unstable]
    C[Optimal LR] --> D[Good Convergence]
    E[Low LR] --> F[Stable but Slow]""",
            "code_example": """# Learning Rate for Fine-Tuning
from transformers import TrainingArguments

# LoRA: Higher learning rate (1e-4 to 5e-4)
training_args = TrainingArguments(
    learning_rate=2e-4,  # Typical for LoRA
    per_device_train_batch_size=4,
    num_train_epochs=3
)

# Full Fine-Tuning: Lower learning rate (1e-5 to 1e-4)
training_args = TrainingArguments(
    learning_rate=5e-5,  # Typical for full FT
    per_device_train_batch_size=2,
    num_train_epochs=3
)

# Learning rate schedule
training_args = TrainingArguments(
    learning_rate=2e-4,
    lr_scheduler_type="cosine",  # Cosine decay
    warmup_steps=100,  # Warmup phase
    num_train_epochs=3
)""",
            "links": [
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📚 HuggingFace Training Guide", "url": "https://huggingface.co/docs/transformers/training", "type": "tutorial"}
            ]
        },
        "Warmup": {
            "definition": "Gradually increase learning rate at start of training. Prevents early instability. Typical: 3-10% of total steps. Helps model adapt gradually. Common schedules: linear, cosine.",
            "formula": "Warmup:\n- Steps: 3-10% of total training steps\n- Schedule: Linear or cosine\n- LR increases from 0 to target LR\n\nExample: 1000 steps, 10% warmup = 100 warmup steps",
            "diagram": """graph LR
    A[LR = 0] --> B[Warmup Phase<br/>Gradually Increase]
    B --> C[LR = Target]
    C --> D[Training Phase<br/>Constant/Decay]""",
            "code_example": """# Learning Rate Warmup
from transformers import TrainingArguments

# Warmup: 10% of total steps
training_args = TrainingArguments(
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_steps=100,  # 10% of 1000 steps
    lr_scheduler_type="linear"  # Linear warmup
)

# Warmup schedule:
# Step 0-100: LR increases linearly from 0 to 2e-4
# Step 100+: LR = 2e-4 (or decays)

# Cosine warmup
training_args = TrainingArguments(
    learning_rate=2e-4,
    warmup_steps=100,
    lr_scheduler_type="cosine"  # Cosine warmup + decay
)""",
            "links": [
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762", "type": "paper"},
                {"title": "📚 HuggingFace Training Guide", "url": "https://huggingface.co/docs/transformers/training", "type": "tutorial"}
            ]
        },
        "Batch Size": {
            "definition": "Number of examples processed per training step. Larger batch = more stable gradients but more memory. Smaller batch = less memory but noisier gradients. Typical for LoRA: 1-8. Can use gradient accumulation.",
            "formula": "Batch Size:\n- LoRA: 1-8 (limited by adapter size)\n- Full FT: 1-4 (limited by model size)\n- Effective batch = batch_size × gradient_accumulation\n\nMemory: O(batch_size × seq_len × hidden_size)",
            "diagram": """graph LR
    A[Small Batch<br/>1-2] --> B[Less Memory<br/>Noisier Gradients]
    C[Medium Batch<br/>4-8] --> D[Balanced]
    E[Large Batch<br/>16+] --> F[More Memory<br/>Stable Gradients]""",
            "code_example": """# Batch Size for Fine-Tuning
from transformers import TrainingArguments

# LoRA: Can use larger batch (1-8)
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # Batch size per GPU
    gradient_accumulation_steps=2,  # Effective batch = 4 × 2 = 8
    num_train_epochs=3
)

# Full Fine-Tuning: Smaller batch (1-4)
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Smaller due to memory
    gradient_accumulation_steps=4,  # Effective batch = 2 × 4 = 8
    num_train_epochs=3
)

# Gradient Accumulation simulates larger batch:
# Accumulate gradients over N steps before updating
# Effective batch = batch_size × gradient_accumulation_steps""",
            "links": [
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 QLoRA: Efficient Finetuning of Quantized LLMs", "url": "https://arxiv.org/abs/2305.14314", "type": "paper"},
                {"title": "📚 HuggingFace Training Guide", "url": "https://huggingface.co/docs/transformers/training", "type": "tutorial"}
            ]
        },
        "Epochs": {
            "definition": "Number of complete passes through training dataset. Typically 1-5 epochs sufficient for fine-tuning. More epochs = better fit but risk overfitting. Early stopping can prevent overfitting.",
            "formula": "Epochs:\n- Fine-tuning: 1-5 epochs typically sufficient\n- Pre-training: Hundreds of epochs\n- Early stopping: Stop when validation loss stops improving\n\nMore epochs = more training but risk overfitting",
            "diagram": """graph LR
    A[Epoch 1] --> B[Epoch 2]
    B --> C[Epoch 3]
    C --> D{Converged?}
    D -->|Yes| E[Stop]
    D -->|No| F[Continue]""",
            "code_example": """# Training Epochs
from transformers import TrainingArguments, EarlyStoppingCallback

# Typical: 1-5 epochs for fine-tuning
training_args = TrainingArguments(
    num_train_epochs=3,  # 3 epochs typically sufficient
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",  # Evaluate each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,  # Load best checkpoint
    metric_for_best_model="eval_loss"
)

# Early stopping prevents overfitting
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stop if no improvement for 2 epochs
)

trainer.train()""",
            "links": [
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📚 HuggingFace Training Guide", "url": "https://huggingface.co/docs/transformers/training", "type": "tutorial"}
            ]
        },
        "Early Stopping": {
            "definition": "Stop training when validation performance stops improving. Prevents overfitting. Monitors validation loss/metric. Stops after N epochs without improvement. Saves best model checkpoint.",
            "formula": "Early Stopping:\n- Monitor: validation loss/metric\n- Patience: N epochs without improvement\n- Stop: When no improvement for patience epochs\n- Save: Best model checkpoint\n\nPrevents overfitting",
            "diagram": """graph TD
    A[Training] --> B[Evaluate Validation]
    B --> C{Improved?}
    C -->|Yes| D[Save Best Model]
    C -->|No| E[Increment Counter]
    D --> F[Continue Training]
    E --> G{Counter > Patience?}
    G -->|Yes| H[Stop Training]
    G -->|No| F""",
            "code_example": """# Early Stopping
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

training_args = TrainingArguments(
    num_train_epochs=10,  # Max epochs
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3  # Stop after 3 epochs without improvement
        )
    ]
)

trainer.train()
# Training stops early if validation loss doesn't improve
# Best model checkpoint is loaded automatically""",
            "links": [
                {"title": "📄 Early Stopping - But When?", "url": "https://link.springer.com/article/10.1023/A:1022487305947", "type": "paper"},
                {"title": "📚 HuggingFace Training Guide", "url": "https://huggingface.co/docs/transformers/training", "type": "tutorial"}
            ]
        },
        "Instruction Tuning": {
            "definition": "Fine-tune model to follow instructions. Train on instruction-response pairs. Enables zero-shot task performance. Makes model more helpful and aligned. Examples: FLAN, Alpaca, InstructGPT.",
            "formula": "Instruction Tuning:\n- Format: Instruction + Response pairs\n- Dataset: Human-written or synthetic\n- Goal: Follow instructions, be helpful\n- Enables: Zero-shot task performance\n\nExamples: \"Translate to French: Hello\" → \"Bonjour\"",
            "diagram": """graph LR
    A[Base Model] --> B[Instruction Dataset]
    B --> C[Fine-Tune]
    C --> D[Instruction-Following<br/>Model]""",
            "code_example": """# Instruction Tuning
from transformers import Trainer, TrainingArguments

# Instruction dataset format
instruction_dataset = [
    {
        "instruction": "Translate to French",
        "input": "Hello",
        "output": "Bonjour"
    },
    {
        "instruction": "Summarize",
        "input": "Long article...",
        "output": "Summary..."
    }
]

# Fine-tune on instructions
training_args = TrainingArguments(
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=instruction_dataset
)

trainer.train()
# Model learns to follow instructions
# Enables zero-shot task performance""",
            "links": [
                {"title": "📄 FLAN: Scaling Instruction-Finetuned Language Models", "url": "https://arxiv.org/abs/2109.01652", "type": "paper"},
                {"title": "📄 InstructGPT: Training Language Models to Follow Instructions", "url": "https://arxiv.org/abs/2203.02155", "type": "paper"},
                {"title": "📄 Alpaca: A Strong Open-Source Instruction-Following Model", "url": "https://crfm.stanford.edu/2023/03/13/alpaca.html", "type": "paper"}
            ]
        },
        "Domain Adaptation": {
            "definition": "Adapt model to specific domain (medical, legal, code, etc.). Fine-tune on domain-specific data. Improves performance in target domain. Can use LoRA for efficient adaptation. Balances domain knowledge with general knowledge.",
            "formula": "Domain Adaptation:\n- Target: Specific domain (medical, legal, etc.)\n- Method: Fine-tune on domain data\n- Balance: Domain knowledge vs general knowledge\n- Efficiency: Use LoRA for parameter-efficient adaptation",
            "diagram": """graph LR
    A[General Model] --> B[Domain Data]
    B --> C[Fine-Tune]
    C --> D[Domain-Adapted<br/>Model]""",
            "code_example": """# Domain Adaptation
from peft import LoraConfig, get_peft_model
from transformers import Trainer

# Load domain-specific dataset
medical_dataset = load_medical_corpus()

# Use LoRA for efficient adaptation
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(base_model, lora_config)

# Fine-tune on domain data
trainer = Trainer(
    model=model,
    train_dataset=medical_dataset,
    args=training_args
)

trainer.train()
# Model adapted to medical domain
# Still retains general knowledge""",
            "links": [
                {"title": "📄 Domain-Specific Language Model Pretraining", "url": "https://arxiv.org/abs/2002.05645", "type": "paper"},
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 BioBERT: Pre-trained Biomedical Language Model", "url": "https://arxiv.org/abs/1901.08746", "type": "paper"}
            ]
        },
        "Catastrophic Forgetting": {
            "definition": "Model forgets previously learned knowledge when fine-tuned on new data. Problem in continual learning. Mitigated by: data mixing, lower learning rate, regularization. Important to preserve general knowledge.",
            "formula": "Catastrophic Forgetting:\n- Problem: Model forgets old knowledge\n- Cause: Fine-tuning on new data\n- Mitigation:\n  - Data mixing: Mix old + new data\n  - Lower LR: Preserve existing weights\n  - Regularization: Constrain weight changes",
            "diagram": """graph TD
    A[Pre-trained Knowledge] --> B[Fine-Tune on New Data]
    B --> C{Forgetting?}
    C -->|Yes| D[Lost Old Knowledge]
    C -->|No| E[Retained Knowledge]""",
            "code_example": """# Preventing Catastrophic Forgetting

# Method 1: Data Mixing
# Mix pre-training data with fine-tuning data
mixed_dataset = concatenate_datasets([
    pretrain_data.sample(frac=0.1),  # 10% of pre-training data
    finetune_data  # 100% of fine-tuning data
])

# Method 2: Lower Learning Rate
training_args = TrainingArguments(
    learning_rate=1e-5,  # Lower LR preserves existing weights
    num_train_epochs=3
)

# Method 3: Regularization (Elastic Weight Consolidation)
# Constrain weight changes to important weights
# Preserves important connections

# Method 4: LoRA (Parameter-Efficient)
# Only updates small adapter, preserves base model
lora_config = LoraConfig(r=8)
model = get_peft_model(base_model, lora_config)""",
            "links": [
                {"title": "📄 Catastrophic Forgetting in Neural Networks", "url": "https://arxiv.org/abs/1312.6211", "type": "paper"},
                {"title": "📄 Elastic Weight Consolidation", "url": "https://arxiv.org/abs/1612.00796", "type": "paper"},
                {"title": "📄 LoRA: Low-Rank Adaptation (Prevents Forgetting)", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"}
            ]
        },
        "Data Mixing": {
            "definition": "Mix pre-training data with fine-tuning data during training. Prevents catastrophic forgetting. Typical ratio: 10-20% pre-training data, 80-90% fine-tuning data. Helps preserve general knowledge.",
            "formula": "Data Mixing:\n- Pre-train data: 10-20%\n- Fine-tune data: 80-90%\n- Mix during training\n- Prevents forgetting general knowledge\n\nRatio depends on task and domain shift",
            "diagram": """graph LR
    A[Pre-train Data<br/>10-20%] --> C[Mixed Dataset]
    B[Fine-tune Data<br/>80-90%] --> C
    C --> D[Training]""",
            "code_example": """# Data Mixing
from datasets import concatenate_datasets

# Load datasets
pretrain_data = load_pretrain_corpus()
finetune_data = load_finetune_corpus()

# Mix: 10% pre-train + 100% fine-tune
mixed_pretrain = pretrain_data.shuffle().select(range(len(pretrain_data) // 10))
mixed_dataset = concatenate_datasets([mixed_pretrain, finetune_data])

# Shuffle mixed dataset
mixed_dataset = mixed_dataset.shuffle(seed=42)

# Train on mixed dataset
trainer = Trainer(
    model=model,
    train_dataset=mixed_dataset,
    args=training_args
)

# Model learns new task while preserving general knowledge""",
            "links": [
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 LoRA: Low-Rank Adaptation", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📚 HuggingFace Datasets Guide", "url": "https://huggingface.co/docs/datasets", "type": "tutorial"}
            ]
        }
    },
    "Evaluation": {
        "Perplexity": {
            "definition": "exp(cross-entropy loss). Lower is better. Measures model uncertainty. Typical values: 10-50 for good models. Intrinsic metric.",
            "formula": "Perplexity = exp(H)\nwhere H = -1/N Σ log P(wᵢ|w₁...wᵢ₋₁)\n\nInterpretation:\n- Perplexity = k means model is as confused as if it had to choose uniformly from k options",
            "diagram": """graph LR
    A[Model Predictions] --> B[Cross-Entropy Loss]
    B --> C[Average Loss H]
    C --> D[exp H]
    D --> E[Perplexity]
    F[Lower = Better] --> E""",
            "code_example": """import torch.nn.functional as F

def compute_perplexity(logits, targets):
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                           targets.view(-1), 
                           reduction='mean')
    
    # Perplexity = exp(loss)
    perplexity = torch.exp(loss)
    return perplexity.item()

# Example: perplexity of 20 means model is as uncertain
# as choosing uniformly from 20 words""",
            "links": [
                {"title": "📄 Neural Machine Translation by Jointly Learning", "url": "https://arxiv.org/abs/1409.3215", "type": "paper"},
                {"title": "📄 Language Models are Unsupervised Multitask Learners", "url": "https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", "type": "paper"},
                {"title": "📄 Scaling Laws for Neural Language Models", "url": "https://arxiv.org/abs/2001.08361", "type": "paper"},
                {"title": "📄 Training Compute-Optimal Large Language Models", "url": "https://arxiv.org/abs/2203.15556", "type": "paper"},
                {"title": "📚 Perplexity Explained", "url": "https://towardsdatascience.com/perplexity-in-language-models-87a196019a94", "type": "tutorial"},
                {"title": "🎥 Evaluation Metrics", "url": "https://www.youtube.com/watch?v=O9insN7qX0k", "type": "video"}
            ]
        },
        "ROUGE-L": {
            "definition": "Longest Common Subsequence between reference and candidate. Recall-oriented. Good for summarization. Range: 0-1.",
            "formula": "ROUGE-L = LCS(reference, candidate) / length(reference)\n\nwhere LCS = Longest Common Subsequence\n\nExample:\nReference: 'the cat sat on the mat'\nCandidate: 'the cat on mat'\nLCS = 'the cat on mat' (length 5)\nROUGE-L = 5/6 = 0.833",
            "diagram": """graph LR
    A[Reference Text] --> C[Find LCS]
    B[Candidate Text] --> C
    C --> D[LCS Length]
    D --> E[Divide by<br/>Reference Length]
    E --> F[ROUGE-L Score]""",
            "code_example": """from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

reference = "the cat sat on the mat"
candidate = "the cat on mat"

scores = scorer.score(reference, candidate)
rouge_l = scores['rougeL'].fmeasure
print(f"ROUGE-L: {rouge_l:.3f}")  # ~0.833""",
            "links": [
                {"title": "📄 ROUGE: A Package for Automatic Evaluation of Summaries (Original)", "url": "https://aclanthology.org/W04-1013/", "type": "paper"},
                {"title": "📄 BLEU: A Method for Automatic Evaluation", "url": "https://aclanthology.org/P02-1040/", "type": "paper"},
                {"title": "📄 BERTScore: Evaluating Text Generation with BERT", "url": "https://arxiv.org/abs/1904.09675", "type": "paper"},
                {"title": "📄 METEOR: An Automatic Metric for MT Evaluation", "url": "https://aclanthology.org/W05-0909/", "type": "paper"},
                {"title": "📄 SummEval: Re-evaluating Summarization Evaluation", "url": "https://arxiv.org/abs/2007.12626", "type": "paper"},
                {"title": "🎥 ROUGE Explained", "url": "https://www.youtube.com/watch?v=TE6ity2u3mo", "type": "video"}
            ]
        }
    },
    "GPU Acceleration & Distributed Training": {
        "Data Parallelism": {
            "definition": "Split data across GPUs, each GPU has full model. All-reduce gradients. Use when model fits on single GPU. Most common parallelism strategy.",
            "formula": "Gradient synchronization:\ng_avg = (g₁ + g₂ + ... + gₙ) / n\n\nCommunication: O(parameters) per step",
            "diagram": """graph TD
    A[Batch] --> B[Split into N parts]
    B --> C1[GPU 1<br/>Model Copy]
    B --> C2[GPU 2<br/>Model Copy]
    B --> C3[GPU N<br/>Model Copy]
    C1 --> D1[Gradients 1]
    C2 --> D2[Gradients 2]
    C3 --> D3[Gradients N]
    D1 --> E[All-Reduce<br/>Average]
    D2 --> E
    D3 --> E
    E --> F[Update All Models]""",
            "code_example": """# PyTorch DataParallel
import torch.nn as nn

model = nn.DataParallel(model)  # Simple but limited

# DistributedDataParallel (better)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

model = DistributedDataParallel(model)

# Each process handles different data shard
# Gradients are synchronized via all-reduce""",
            "links": [
                {"title": "📄 PyTorch DistributedDataParallel", "url": "https://pytorch.org/tutorials/intermediate/ddp_tutorial.html", "type": "paper"},
                {"title": "📄 Megatron-LM: Training Multi-Billion Parameter Models", "url": "https://arxiv.org/abs/1909.08053", "type": "paper"},
                {"title": "📄 ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", "url": "https://arxiv.org/abs/1910.02054", "type": "paper"},
                {"title": "📄 ZeRO-Offload: Democratizing Billion-Scale Model Training", "url": "https://arxiv.org/abs/2101.06840", "type": "paper"},
                {"title": "📄 ZeRO-Infinity: Breaking GPU Memory Wall", "url": "https://arxiv.org/abs/2104.07857", "type": "paper"},
                {"title": "📄 Efficient Large-Scale Language Model Training on GPU Clusters", "url": "https://arxiv.org/abs/2104.04473", "type": "paper"},
                {"title": "📚 PyTorch DDP Tutorial", "url": "https://pytorch.org/tutorials/intermediate/ddp_tutorial.html", "type": "tutorial"},
                {"title": "🎥 Distributed Training", "url": "https://www.youtube.com/watch?v=5q2E2O7a0hY", "type": "video"}
            ]
        },
        "Tensor Parallelism": {
            "definition": "Split model layers across GPUs. Each GPU has part of model. Use for very large models. Requires within-layer communication.",
            "formula": "For layer with weights W:\nW = [W₁, W₂, ..., Wₙ]\nEach GPU i holds Wᵢ\n\nOutput: Concat([W₁x, W₂x, ..., Wₙx])",
            "diagram": """graph TD
    A[Input] --> B[Split across GPUs]
    B --> C1[GPU 1<br/>W₁]
    B --> C2[GPU 2<br/>W₂]
    B --> C3[GPU N<br/>Wₙ]
    C1 --> D1[Partial Output 1]
    C2 --> D2[Partial Output 2]
    C3 --> D3[Partial Output N]
    D1 --> E[All-Gather]
    D2 --> E
    D3 --> E
    E --> F[Concat Output]""",
            "code_example": """# Tensor Parallelism (simplified)
# Each GPU processes part of the layer
class TensorParallelLinear:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
    
    def forward(self, x):
        # Each GPU computes its part
        local_output = F.linear(x, self.local_weight)
        
        # Gather all parts
        outputs = [torch.zeros_like(local_output) 
                   for _ in range(self.world_size)]
        dist.all_gather(outputs, local_output)
        
        # Concatenate
        return torch.cat(outputs, dim=-1)""",
            "links": [
                {"title": "📄 Megatron-LM: Training Multi-Billion Parameter Models (Original)", "url": "https://arxiv.org/abs/1909.08053", "type": "paper"},
                {"title": "📄 Efficient Large-Scale Language Model Training", "url": "https://arxiv.org/abs/2104.04473", "type": "paper"},
                {"title": "📄 PaLM: Scaling Language Modeling (Uses Tensor Parallelism)", "url": "https://arxiv.org/abs/2204.02311", "type": "paper"},
                {"title": "📄 GPT-3: Language Models (Uses Tensor Parallelism)", "url": "https://arxiv.org/abs/2005.14165", "type": "paper"},
                {"title": "📄 Efficiently Scaling Transformer Inference", "url": "https://arxiv.org/abs/2211.05102", "type": "paper"},
                {"title": "🎥 Tensor Parallelism Explained", "url": "https://www.youtube.com/watch?v=MPmr6uL5U2c", "type": "video"}
            ]
        },
        "Tensor Cores": {
            "definition": "Specialized hardware units in NVIDIA GPUs (V100, A100, H100) for matrix operations. Optimized for mixed-precision (FP16, INT8). Provide massive speedup for deep learning. 10-100x faster than regular CUDA cores for matrix ops.",
            "formula": "Tensor Cores:\n- Available: V100, A100, H100, RTX series\n- Operations: Matrix multiply-accumulate\n- Precision: FP16, INT8, BF16\n- Speedup: 10-100x for matrix ops\n\nEnable: Mixed precision training, fast inference",
            "diagram": """graph LR
    A[Matrix A<br/>FP16] --> C[Tensor Core]
    B[Matrix B<br/>FP16] --> C
    C --> D[Result<br/>FP32 Accumulate]
    D --> E[10-100x Faster]""",
            "code_example": """# Tensor Cores Usage (Automatic with FP16)
import torch

# Enable Tensor Cores with mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
model = model.cuda()

# FP16 operations use Tensor Cores automatically
with autocast():
    output = model(input)
    loss = criterion(output, target)

# Tensor Cores provide massive speedup
# V100: 640 Tensor Cores
# A100: 432 Tensor Cores
# H100: 528 Tensor Cores""",
            "links": [
                {"title": "📄 Mixed Precision Training (Uses Tensor Cores)", "url": "https://arxiv.org/abs/1710.03740", "type": "paper"},
                {"title": "📄 NVIDIA Tensor Core Architecture", "url": "https://www.nvidia.com/en-us/data-center/tensor-cores/", "type": "paper"},
                {"title": "📚 CUDA Tensor Cores Guide", "url": "https://developer.nvidia.com/tensor-cores", "type": "tutorial"}
            ]
        },
        "Gradient Accumulation": {
            "definition": "Accumulate gradients over multiple batches before updating weights. Simulates larger batch size. Useful when GPU memory limits batch size. Effective batch = batch_size × accumulation_steps.",
            "formula": "Gradient Accumulation:\n- Accumulate gradients over N steps\n- Effective batch = batch_size × N\n- Update weights every N steps\n\nExample: batch_size=2, N=4 → effective batch=8",
            "diagram": """graph LR
    A[Batch 1] --> E[Accumulate Gradients]
    B[Batch 2] --> E
    C[Batch 3] --> E
    D[Batch 4] --> E
    E --> F[Update Weights]""",
            "code_example": """# Gradient Accumulation
from transformers import TrainingArguments

# Simulate larger batch size
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Small batch (limited by memory)
    gradient_accumulation_steps=4,  # Accumulate over 4 steps
    # Effective batch size = 2 × 4 = 8
    num_train_epochs=3
)

# Manual implementation
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()""",
            "links": [
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📚 PyTorch Gradient Accumulation", "url": "https://pytorch.org/docs/stable/notes/amp_examples.html", "type": "tutorial"}
            ]
        },
        "Gradient Checkpointing": {
            "definition": "Trade compute for memory. Recompute activations during backward pass instead of storing them. Reduces memory usage significantly. Increases compute time. Useful for large models.",
            "formula": "Gradient Checkpointing:\n- Memory: O(√n) instead of O(n)\n- Compute: 2x forward passes\n- Trade-off: Memory vs Compute\n\nSaves activations only at checkpoints",
            "diagram": """graph TD
    A[Forward Pass] --> B{Checkpoint?}
    B -->|Yes| C[Save Activation]
    B -->|No| D[Don't Save]
    E[Backward Pass] --> F{Need Activation?}
    F -->|Saved| G[Use Saved]
    F -->|Not Saved| H[Recompute]""",
            "code_example": """# Gradient Checkpointing
import torch
from torch.utils.checkpoint import checkpoint

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Or manually
class CheckpointedModel(nn.Module):
    def forward(self, x):
        # Checkpoint every N layers
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = checkpoint(self.layer3, x)
        return x

# Memory reduced significantly
# But requires recomputation during backward
# Trade-off: Memory vs Compute""",
            "links": [
                {"title": "📄 Training Deep Nets with Sublinear Memory Cost", "url": "https://arxiv.org/abs/1604.06174", "type": "paper"},
                {"title": "📄 Checkpointing for Large-Scale Training", "url": "https://arxiv.org/abs/1604.06174", "type": "paper"},
                {"title": "📚 PyTorch Checkpointing", "url": "https://pytorch.org/docs/stable/checkpoint.html", "type": "tutorial"}
            ]
        },
        "Offloading": {
            "definition": "Move model parameters/optimizer states to CPU memory. Reduces GPU memory usage. Slower but enables larger models. Used with ZeRO-Offload. CPU memory is larger but slower.",
            "formula": "Offloading:\n- Move: Parameters, optimizer states, gradients\n- To: CPU memory (larger, slower)\n- Benefit: Reduced GPU memory\n- Cost: Slower training (CPU-GPU transfer)\n\nZeRO-Offload: Automates offloading",
            "diagram": """graph TD
    A[GPU Memory<br/>Limited] --> B[Offload to CPU]
    B --> C[CPU Memory<br/>Larger]
    D[Need Parameter] --> E{In GPU?}
    E -->|Yes| F[Use GPU]
    E -->|No| G[Transfer from CPU]""",
            "code_example": """# Parameter Offloading
from deepspeed import ZeROConfig

# ZeRO-Offload: Automatically offloads to CPU
ds_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",  # Offload optimizer states
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",  # Offload parameters
            "pin_memory": True
        }
    }
}

# Reduces GPU memory significantly
# Enables training larger models
# Slower due to CPU-GPU transfers""",
            "links": [
                {"title": "📄 ZeRO-Offload: Democratizing Billion-Scale Model Training", "url": "https://arxiv.org/abs/2101.06840", "type": "paper"},
                {"title": "📄 ZeRO-Infinity: Breaking GPU Memory Wall", "url": "https://arxiv.org/abs/2104.07857", "type": "paper"},
                {"title": "📚 DeepSpeed ZeRO Guide", "url": "https://www.deepspeed.ai/tutorials/zero/", "type": "tutorial"}
            ]
        },
        "NCCL": {
            "definition": "NVIDIA Collective Communications Library. Optimized communication primitives for multi-GPU. Implements all-reduce, all-gather, broadcast. Critical for distributed training. Highly optimized for NVIDIA GPUs.",
            "formula": "NCCL Primitives:\n- All-Reduce: Sum gradients across GPUs\n- All-Gather: Gather data from all GPUs\n- Broadcast: Send data to all GPUs\n- Reduce-Scatter: Scatter and reduce\n\nOptimized for NVLink, InfiniBand",
            "diagram": """graph TD
    A[GPU 1] --> E[NCCL]
    B[GPU 2] --> E
    C[GPU 3] --> E
    D[GPU N] --> E
    E --> F[All-Reduce<br/>All-Gather<br/>Broadcast]""",
            "code_example": """# NCCL Usage (via PyTorch)
import torch.distributed as dist

# Initialize NCCL backend
dist.init_process_group(
    backend='nccl',  # NVIDIA Collective Communications Library
    init_method='env://'
)

# All-Reduce: Sum gradients
tensor = torch.ones(10).cuda()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# All GPUs have sum of all tensors

# All-Gather: Gather from all GPUs
output_list = [torch.zeros(10).cuda() for _ in range(world_size)]
dist.all_gather(output_list, tensor)
# output_list contains tensors from all GPUs

# Broadcast: Send to all GPUs
dist.broadcast(tensor, src=0)
# All GPUs have same tensor""",
            "links": [
                {"title": "📄 NCCL: Optimized Primitives for Multi-GPU Communication", "url": "https://developer.nvidia.com/nccl", "type": "paper"},
                {"title": "📄 Efficient Large-Scale Language Model Training", "url": "https://arxiv.org/abs/2104.04473", "type": "paper"},
                {"title": "📚 PyTorch Distributed Guide", "url": "https://pytorch.org/tutorials/intermediate/ddp_tutorial.html", "type": "tutorial"}
            ]
        },
        "All-Reduce": {
            "definition": "Collective operation that sums (or other ops) values across all GPUs and distributes result to all. Critical for gradient synchronization in data parallelism. NCCL optimizes all-reduce.",
            "formula": "All-Reduce:\n- Input: Local value on each GPU\n- Operation: Sum (or max, min, etc.)\n- Output: Sum distributed to all GPUs\n\nRing All-Reduce: O(n) complexity\nTree All-Reduce: O(log n) complexity",
            "diagram": """graph LR
    A[GPU 1: g₁] --> E[All-Reduce]
    B[GPU 2: g₂] --> E
    C[GPU 3: g₃] --> E
    D[GPU N: gₙ] --> E
    E --> F[All GPUs: Σgᵢ]""",
            "code_example": """# All-Reduce for Gradient Synchronization
import torch.distributed as dist

# Each GPU computes local gradient
local_grad = compute_gradient()

# All-Reduce: Sum gradients across all GPUs
dist.all_reduce(local_grad, op=dist.ReduceOp.SUM)

# Now all GPUs have sum of all gradients
# local_grad = g₁ + g₂ + ... + gₙ

# Average (divide by world size)
local_grad /= world_size

# All GPUs have same averaged gradient
# Ready for weight update

# NCCL optimizes all-reduce using:
# - Ring algorithm: O(n) steps
# - Tree algorithm: O(log n) steps
# - NVLink/InfiniBand for fast communication""",
            "links": [
                {"title": "📄 NCCL: Optimized Primitives for Multi-GPU Communication", "url": "https://developer.nvidia.com/nccl", "type": "paper"},
                {"title": "📄 Ring All-Reduce", "url": "https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/", "type": "paper"},
                {"title": "📚 PyTorch Distributed Guide", "url": "https://pytorch.org/tutorials/intermediate/ddp_tutorial.html", "type": "tutorial"}
            ]
        }
    },
    "Prompt Engineering": {
        "Chain-of-Thought (CoT)": {
            "definition": "Prompting technique that encourages step-by-step reasoning. Improves complex reasoning tasks by breaking problems into smaller steps.",
            "formula": "Standard: Q → A\nCoT: Q → Reasoning Steps → A\n\nExample:\nQ: If 3x + 5 = 20, what is x?\nCoT: First, subtract 5 from both sides: 3x = 15. Then divide by 3: x = 5.",
            "diagram": """graph LR
    A[Question] --> B[Step 1<br/>Reasoning]
    B --> C[Step 2<br/>Reasoning]
    C --> D[Step N<br/>Reasoning]
    D --> E[Final Answer]""",
            "code_example": """# Chain-of-Thought Prompting
prompt = \"\"\"
Q: A store has 15 apples. They sell 6. How many are left?

A: Let's think step by step:
1. Start with 15 apples
2. Sell 6 apples
3. Remaining = 15 - 6 = 9 apples

So the answer is 9.
\"\"\"

response = model.generate(prompt)""",
            "links": [
                {"title": "📄 Chain-of-Thought Prompting Elicits Reasoning (Original)", "url": "https://arxiv.org/abs/2201.11903", "type": "paper"},
                {"title": "📄 Self-Consistency Improves Chain of Thought Reasoning", "url": "https://arxiv.org/abs/2203.11171", "type": "paper"},
                {"title": "📄 Large Language Models are Zero-Shot Reasoners", "url": "https://arxiv.org/abs/2205.11916", "type": "paper"},
                {"title": "📄 Tree of Thoughts: Deliberate Problem Solving", "url": "https://arxiv.org/abs/2305.10601", "type": "paper"},
                {"title": "📄 ReAct: Synergizing Reasoning and Acting", "url": "https://arxiv.org/abs/2210.03629", "type": "paper"},
                {"title": "📄 Auto-CoT: Automatic Chain of Thought Prompting", "url": "https://arxiv.org/abs/2210.03493", "type": "paper"},
                {"title": "🎥 CoT Prompting Explained", "url": "https://www.youtube.com/watch?v=5u9XkKJXJ4E", "type": "video"}
            ]
        },
        "Few-Shot": {
            "definition": "Multiple examples (typically 2-5) provided in the prompt. Demonstrates the task pattern. Improves performance over zero/one-shot.",
            "formula": "Prompt = Example₁ + Example₂ + ... + Exampleₙ + Query\n\nTypical n = 2-5 examples",
            "diagram": """graph LR
    A[Example 1<br/>Input → Output] --> D[Prompt]
    B[Example 2<br/>Input → Output] --> D
    C[Example N<br/>Input → Output] --> D
    D --> E[Query<br/>Input]
    E --> F[Model<br/>Generates Output]""",
            "code_example": """# Few-Shot Prompting
few_shot_prompt = \"\"\"
Translate English to French:

Example 1:
English: Hello
French: Bonjour

Example 2:
English: Good morning
French: Bonjour

Example 3:
English: Thank you
French: Merci

Now translate:
English: Goodbye
French:\"\"\"

response = model.generate(few_shot_prompt)""",
            "links": [
                {"title": "📄 Language Models are Few-Shot Learners (GPT-3)", "url": "https://arxiv.org/abs/2005.14165", "type": "paper"},
                {"title": "📄 In-Context Learning and Induction Heads", "url": "https://arxiv.org/abs/2209.11895", "type": "paper"},
                {"title": "📄 What Makes In-Context Learning Work?", "url": "https://arxiv.org/abs/2202.12837", "type": "paper"},
                {"title": "📄 Rethinking the Role of Demonstrations", "url": "https://arxiv.org/abs/2202.12837", "type": "paper"},
                {"title": "📄 An Explanation of In-Context Learning", "url": "https://arxiv.org/abs/2212.10559", "type": "paper"},
                {"title": "📚 Prompt Engineering Guide", "url": "https://www.promptingguide.ai/techniques/fewshot", "type": "tutorial"}
            ]
        },
        "Zero-Shot": {
            "definition": "No examples provided, just direct instruction. Model relies entirely on pre-training knowledge. Quick and simple but may have limited accuracy for complex tasks.",
            "formula": "Prompt = Instruction + Query\n\nNo examples provided\nModel uses pre-trained knowledge only",
            "diagram": """graph LR
    A[Instruction] --> C[Prompt]
    B[Query] --> C
    C --> D[Model]
    D --> E[Output<br/>Based on Pre-training]""",
            "code_example": """# Zero-Shot Prompting
zero_shot_prompt = \"\"\"
Translate the following English text to French:

English: Hello, how are you?
French:\"\"\"

response = model.generate(zero_shot_prompt)
# Model relies on pre-training knowledge only""",
            "links": [
                {"title": "📄 Language Models are Few-Shot Learners (GPT-3)", "url": "https://arxiv.org/abs/2005.14165", "type": "paper"},
                {"title": "📄 Language Models are Unsupervised Multitask Learners (GPT-2)", "url": "https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", "type": "paper"},
                {"title": "📄 What Makes In-Context Learning Work?", "url": "https://arxiv.org/abs/2202.12837", "type": "paper"}
            ]
        },
        "One-Shot": {
            "definition": "Single example provided in the prompt. Demonstrates the task pattern once. Better than zero-shot but may not capture task complexity fully.",
            "formula": "Prompt = Example₁ + Query\n\nSingle example demonstrates task pattern",
            "diagram": """graph LR
    A[Example 1<br/>Input → Output] --> C[Prompt]
    B[Query<br/>Input] --> C
    C --> D[Model]
    D --> E[Output]""",
            "code_example": """# One-Shot Prompting
one_shot_prompt = \"\"\"
Translate English to French:

Example:
English: Hello
French: Bonjour

Now translate:
English: Good morning
French:\"\"\"

response = model.generate(one_shot_prompt)""",
            "links": [
                {"title": "📄 Language Models are Few-Shot Learners (GPT-3)", "url": "https://arxiv.org/abs/2005.14165", "type": "paper"},
                {"title": "📄 Rethinking the Role of Demonstrations", "url": "https://arxiv.org/abs/2202.12837", "type": "paper"},
                {"title": "📚 Prompt Engineering Guide", "url": "https://www.promptingguide.ai/techniques/oneshot", "type": "tutorial"}
            ]
        },
        "System Message": {
            "definition": "Persistent message that sets model behavior, role, and constraints. Used in chat-based APIs. Defines the assistant's personality and guidelines. Not part of conversation history.",
            "formula": "Messages = [\n  {\"role\": \"system\", \"content\": \"You are a helpful assistant...\"},\n  {\"role\": \"user\", \"content\": \"...\"}\n]\n\nSystem message persists across conversation",
            "diagram": """graph TD
    A[System Message<br/>Sets Behavior] --> B[Conversation Context]
    C[User Message 1] --> B
    D[User Message 2] --> B
    B --> E[Model Response]""",
            "code_example": """# System Message Usage
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant specialized in machine learning. Always provide accurate, well-sourced information."
    },
    {
        "role": "user",
        "content": "What is attention?"
    }
]

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages
)""",
            "links": [
                {"title": "📄 GPT-4 Technical Report", "url": "https://arxiv.org/abs/2303.08774", "type": "paper"},
                {"title": "📄 ChatGPT: Optimizing Language Models for Dialogue", "url": "https://openai.com/research/chatgpt", "type": "paper"},
                {"title": "📚 OpenAI API Documentation", "url": "https://platform.openai.com/docs/guides/chat", "type": "tutorial"}
            ]
        },
        "User Message": {
            "definition": "The actual query or task from the user. Part of conversation history. Can be a question, instruction, or request. Model responds to user messages.",
            "formula": "User Message = Query/Task/Instruction\n\nPart of conversation flow\nModel generates response based on user message",
            "diagram": """graph LR
    A[User] --> B[User Message]
    B --> C[Model]
    C --> D[Assistant Response]
    D --> E[User]""",
            "code_example": """# User Message in Chat API
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain transformer architecture"},
    {"role": "assistant", "content": "Transformers use attention..."},
    {"role": "user", "content": "What about multi-head attention?"}  # User follow-up
]""",
            "links": [
                {"title": "📄 GPT-4 Technical Report", "url": "https://arxiv.org/abs/2303.08774", "type": "paper"},
                {"title": "📚 OpenAI Chat API Guide", "url": "https://platform.openai.com/docs/guides/chat", "type": "tutorial"}
            ]
        },
        "Tool/Function Messages": {
            "definition": "Messages used for function calling/tool use. Allow models to call external functions/APIs. Tool messages contain function results. Enables models to interact with external systems.",
            "formula": "Messages = [\n  {\"role\": \"user\", \"content\": \"What's the weather?\"},\n  {\"role\": \"assistant\", \"content\": null, \"function_call\": {...}},\n  {\"role\": \"function\", \"name\": \"get_weather\", \"content\": \"72°F\"}\n]",
            "diagram": """graph TD
    A[User: Query] --> B[Model: Function Call]
    B --> C[Execute Function]
    C --> D[Tool Message: Result]
    D --> E[Model: Final Response]""",
            "code_example": """# Function Calling Example
messages = [
    {"role": "user", "content": "What's the weather in San Francisco?"}
]

# Model decides to call function
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    functions=[{
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }]
)

# Add function result
messages.append({
    "role": "function",
    "name": "get_weather",
    "content": "72°F, sunny"
})

# Model generates final response
final_response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages
)""",
            "links": [
                {"title": "📄 GPT-4 Technical Report (Function Calling)", "url": "https://arxiv.org/abs/2303.08774", "type": "paper"},
                {"title": "📄 ReAct: Synergizing Reasoning and Acting", "url": "https://arxiv.org/abs/2210.03629", "type": "paper"},
                {"title": "📚 OpenAI Function Calling Guide", "url": "https://platform.openai.com/docs/guides/function-calling", "type": "tutorial"}
            ]
        },
        "JSON-Only Output": {
            "definition": "Force model to output only valid JSON. Use structured output for APIs, data extraction. Prevents malformed JSON. Use response_format parameter or prompt instructions.",
            "formula": "Prompt = Instruction + \"Output JSON only\"\n\nOr use: response_format={\"type\": \"json_object\"}",
            "diagram": """graph LR
    A[User Query] --> B[Model with<br/>JSON Constraint]
    B --> C{Valid JSON?}
    C -->|Yes| D[Return JSON]
    C -->|No| E[Retry/Error]""",
            "code_example": """# JSON-Only Output
prompt = \"\"\"
Extract information from the text and return as JSON:
Text: John Doe is 30 years old and works as a software engineer.

Return JSON with keys: name, age, occupation
\"\"\"

# Method 1: Prompt instruction
response = model.generate(prompt + "\\nOutput JSON only:")

# Method 2: Response format (OpenAI)
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"}
)""",
            "links": [
                {"title": "📄 GPT-4 Technical Report (Structured Outputs)", "url": "https://arxiv.org/abs/2303.08774", "type": "paper"},
                {"title": "📚 OpenAI JSON Mode Guide", "url": "https://platform.openai.com/docs/guides/text-generation/json-mode", "type": "tutorial"}
            ]
        },
        "Delimiters": {
            "definition": "Special markers (triple quotes, XML tags, etc.) used to separate different parts of prompt. Helps model distinguish instructions from data. Prevents prompt injection.",
            "formula": "Prompt = Instruction + Delimiter + Data + Delimiter + Task\n\nCommon delimiters: ```, <tag></tag>, ###, ---",
            "diagram": """graph LR
    A[Instruction] --> C[Delimiter]
    B[Data] --> C
    C --> D[Task]
    D --> E[Model<br/>Processes Separately]""",
            "code_example": """# Using Delimiters
prompt = \"\"\"
Summarize the following text:

```
The transformer architecture was introduced in 2017...
```

Provide a 2-sentence summary.
\"\"\"

# XML-style delimiters
prompt = \"\"\"
<text>
The transformer architecture...
</text>

<task>Summarize this text</task>
\"\"\"

# Prevents prompt injection
user_input = "Ignore previous instructions..."
safe_prompt = f\"\"\"
Summarize: <user_input>{user_input}</user_input>
\"\"\"
""",
            "links": [
                {"title": "📄 Prompt Injection Attacks", "url": "https://arxiv.org/abs/2302.12173", "type": "paper"},
                {"title": "📚 Prompt Engineering Best Practices", "url": "https://platform.openai.com/docs/guides/prompt-engineering", "type": "tutorial"}
            ]
        },
        "Content Filters": {
            "definition": "Filters to detect and block harmful content. Pre-generation filters check input. Post-generation filters check output. Used for safety, compliance, moderation.",
            "formula": "Pre-filter: Check input before generation\nPost-filter: Check output after generation\n\nFilter Categories: Violence, Hate Speech, Sexual Content, Self-Harm",
            "diagram": """graph TD
    A[User Input] --> B{Pre-Filter}
    B -->|Safe| C[Generate]
    B -->|Unsafe| D[Block/Modify]
    C --> E[Output]
    E --> F{Post-Filter}
    F -->|Safe| G[Return]
    F -->|Unsafe| H[Filter/Block]""",
            "code_example": """# Content Filtering
def pre_filter(text):
    # Check for harmful content
    harmful_keywords = ["violence", "hate", ...]
    if any(keyword in text.lower() for keyword in harmful_keywords):
        return False, "Content violates policy"
    return True, None

def post_filter(output):
    # Check generated content
    if is_harmful(output):
        return None, "Generated content filtered"
    return output, None

# Usage
is_safe, error = pre_filter(user_input)
if not is_safe:
    return error

output = model.generate(user_input)
filtered_output, error = post_filter(output)""",
            "links": [
                {"title": "📄 GPT-4 System Card (Safety)", "url": "https://cdn.openai.com/papers/gpt-4-system-card.pdf", "type": "paper"},
                {"title": "📄 RealToxicityPrompts: Evaluating Neural Toxic Degeneration", "url": "https://arxiv.org/abs/2009.11462", "type": "paper"},
                {"title": "📄 Constitutional AI: Harmlessness from AI Feedback", "url": "https://arxiv.org/abs/2212.08073", "type": "paper"}
            ]
        },
        "Domain Adaptation via Prompts": {
            "definition": "Adapt model to specific domain using prompts only. No fine-tuning required. Quick and flexible but limited by model's pre-training. Use domain-specific examples and terminology.",
            "formula": "Prompt = Domain Context + Examples + Query\n\nPros: Fast, flexible, no training\nCons: Limited by pre-training, context window limits",
            "diagram": """graph LR
    A[Domain Context] --> C[Prompt]
    B[Domain Examples] --> C
    C --> D[Model]
    D --> E[Domain-Adapted<br/>Output]""",
            "code_example": """# Domain Adaptation via Prompts
medical_prompt = \"\"\"
You are a medical AI assistant. Use medical terminology accurately.

Example:
Patient: 45-year-old male with chest pain
Assessment: Possible myocardial infarction, recommend ECG

Now assess:
Patient: 30-year-old female with headache
Assessment:\"\"\"

response = model.generate(medical_prompt)""",
            "links": [
                {"title": "📄 Language Models are Few-Shot Learners", "url": "https://arxiv.org/abs/2005.14165", "type": "paper"},
                {"title": "📄 Domain-Specific Language Model Pretraining", "url": "https://arxiv.org/abs/2002.05645", "type": "paper"},
                {"title": "📚 Prompt Engineering for Domain Adaptation", "url": "https://www.promptingguide.ai/techniques/domain_adaptation", "type": "tutorial"}
            ]
        },
        "RAG Prompting": {
            "definition": "Retrieval-Augmented Generation. Include retrieved context in prompt. Model uses external knowledge. Better than relying on pre-training alone. Enables up-to-date information.",
            "formula": "Prompt = Retrieved Context + Query\n\nRAG = Retrieve + Augment + Generate\n\nContext from vector store, documents, knowledge base",
            "diagram": """graph TD
    A[Query] --> B[Retrieve<br/>Relevant Docs]
    B --> C[Context]
    C --> D[Prompt:<br/>Context + Query]
    D --> E[Model]
    E --> F[Generated<br/>Response]""",
            "code_example": """# RAG Prompting
def rag_prompt(query, retrieved_docs):
    context = "\\n\\n".join([doc["text"] for doc in retrieved_docs])
    
    prompt = f\"\"\"
Use the following context to answer the question:

Context:
{context}

Question: {query}

Answer based on the context provided.
\"\"\"
    
    return model.generate(prompt)

# Usage
docs = retrieve(query, vector_store, top_k=3)
response = rag_prompt(query, docs)""",
            "links": [
                {"title": "📄 Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG Original)", "url": "https://arxiv.org/abs/2005.11401", "type": "paper"},
                {"title": "📄 Dense Passage Retrieval for Open-Domain Question Answering", "url": "https://arxiv.org/abs/2004.04906", "type": "paper"},
                {"title": "📄 REALM: Retrieval-Augmented Language Model Pre-Training", "url": "https://arxiv.org/abs/2002.08909", "type": "paper"}
            ]
        }
    },
    "Model Deployment": {
        "Dynamic Batching": {
            "definition": "Automatically batch requests. max_batch_size, preferred_batch_size, max_queue_delay. Improves throughput significantly by grouping requests.",
            "formula": "Throughput improvement:\nWithout batching: N requests × latency₁\nWith batching: (N/batch_size) × latency_batch\n\nlatency_batch << batch_size × latency₁",
            "diagram": """graph TD
    A[Request 1] --> E[Queue]
    B[Request 2] --> E
    C[Request 3] --> E
    D[Request 4] --> E
    E --> F{Wait for<br/>max_queue_delay<br/>or batch_size?}
    F -->|Yes| G[Batch Requests]
    G --> H[Process Batch]
    H --> I[Return Results]""",
            "code_example": """# Triton Dynamic Batching Config
# config.pbtxt
dynamic_batching {
    preferred_batch_size: [4, 8]
    max_queue_delay_microseconds: 100
    max_batch_size: 16
}

# Requests are automatically batched
# when queue reaches preferred_batch_size
# or max_queue_delay is exceeded""",
            "links": [
                {"title": "📄 TensorRT Inference Server (Triton)", "url": "https://developer.nvidia.com/blog/triton-inference-server/", "type": "paper"},
                {"title": "📄 vLLM: Easy, Fast, and Cheap LLM Serving", "url": "https://arxiv.org/abs/2309.06180", "type": "paper"},
                {"title": "📄 Efficient Memory Management for Large Language Models", "url": "https://arxiv.org/abs/2309.06180", "type": "paper"},
                {"title": "📄 Orca: A Distributed Serving System", "url": "https://www.usenix.org/conference/osdi22/presentation-yu", "type": "paper"},
                {"title": "📄 FastServe: Efficient LLM Serving", "url": "https://arxiv.org/abs/2305.05920", "type": "paper"},
                {"title": "📚 Triton Dynamic Batching Docs", "url": "https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher", "type": "tutorial"},
                {"title": "🎥 Batching Explained", "url": "https://www.youtube.com/watch?v=5iX8qQSM3b0", "type": "video"}
            ]
        },
        "Triton Model Repository": {
            "definition": "Directory structure for Triton Inference Server models. Contains model files and config.pbtxt. Supports versioning. Multiple models can be served simultaneously. Standard format for production deployment.",
            "formula": "Model Repository Structure:\nmodel_repository/\n  model_name/\n    config.pbtxt\n    1/\n      model.plan\n    2/\n      model.plan\n\nVersions: 1, 2, 3... (numeric)",
            "diagram": """graph TD
    A[Model Repository] --> B[Model 1/<br/>config.pbtxt]
    A --> C[Model 2/<br/>config.pbtxt]
    B --> D[Version 1/<br/>model.plan]
    B --> E[Version 2/<br/>model.plan]""",
            "code_example": """# Triton Model Repository Structure
model_repository/
├── llama-7b/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.plan
│   └── 2/
│       └── model.plan
└── gpt-3.5/
    ├── config.pbtxt
    └── 1/
        └── model.plan

# config.pbtxt example
name: "llama-7b"
platform: "tensorrt_llm"
max_batch_size: 8
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]
output [
  {
    name: "output_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]""",
            "links": [
                {"title": "📄 TensorRT Inference Server (Triton)", "url": "https://developer.nvidia.com/blog/triton-inference-server/", "type": "paper"},
                {"title": "📚 Triton Model Repository Guide", "url": "https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md", "type": "tutorial"},
                {"title": "📚 Triton Documentation", "url": "https://docs.nvidia.com/deeplearning/triton-inference-server/", "type": "tutorial"}
            ]
        },
        "Model Config (config.pbtxt)": {
            "definition": "Configuration file for Triton models. Defines model name, platform, inputs/outputs, batching, optimization settings. Required for each model. Protobuf text format.",
            "formula": "config.pbtxt contains:\n- name: Model identifier\n- platform: tensorrt_llm, onnx, pytorch\n- max_batch_size: Maximum batch size\n- input/output: Tensor specifications\n- dynamic_batching: Batching configuration\n- instance_group: GPU assignment",
            "diagram": """graph LR
    A[config.pbtxt] --> B[Model Name]
    A --> C[Platform]
    A --> D[Inputs/Outputs]
    A --> E[Batching Config]
    A --> F[Optimization]""",
            "code_example": """# config.pbtxt example
name: "llama-7b"
platform: "tensorrt_llm"
max_batch_size: 8

# Input specification
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]  # Variable length
  }
]

# Output specification
output [
  {
    name: "output_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]

# Dynamic batching
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100
  max_batch_size: 16
}

# Instance group (GPU assignment)
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]""",
            "links": [
                {"title": "📄 TensorRT Inference Server (Triton)", "url": "https://developer.nvidia.com/blog/triton-inference-server/", "type": "paper"},
                {"title": "📚 Triton Model Configuration Guide", "url": "https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md", "type": "tutorial"}
            ]
        },
        "Concurrent Models": {
            "definition": "Run multiple models simultaneously on Triton server. Each model can use different GPUs. Enables multi-model serving. Efficient resource utilization. Models isolated from each other.",
            "formula": "Concurrent Models:\n- Multiple models in repository\n- Each model can use different GPUs\n- Isolated execution\n- Shared GPU resources\n\nExample: Model A on GPU 0, Model B on GPU 1",
            "diagram": """graph TD
    A[Triton Server] --> B[Model A<br/>GPU 0]
    A --> C[Model B<br/>GPU 1]
    A --> D[Model C<br/>GPU 0,1]
    B --> E[Concurrent<br/>Serving]""",
            "code_example": """# Concurrent Models in Triton
# Model repository with multiple models
model_repository/
├── llama-7b/
│   ├── config.pbtxt
│   └── 1/model.plan
├── gpt-3.5/
│   ├── config.pbtxt
│   └── 1/model.plan
└── bert-base/
    ├── config.pbtxt
    └── 1/model.plan

# Each model config specifies GPU assignment
# llama-7b/config.pbtxt
instance_group [
  { count: 1, kind: KIND_GPU, gpus: [ 0 ] }
]

# gpt-3.5/config.pbtxt
instance_group [
  { count: 1, kind: KIND_GPU, gpus: [ 1 ] }
]

# Models run concurrently
# Each handles requests independently""",
            "links": [
                {"title": "📄 TensorRT Inference Server (Triton)", "url": "https://developer.nvidia.com/blog/triton-inference-server/", "type": "paper"},
                {"title": "📚 Triton Multi-Model Serving", "url": "https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups", "type": "tutorial"}
            ]
        },
        "HTTP REST/gRPC": {
            "definition": "Communication protocols for Triton. HTTP REST: Simple, JSON-based. gRPC: Binary, more efficient. C API: Lowest latency. Choose based on use case. REST for simplicity, gRPC for performance.",
            "formula": "Protocols:\n- HTTP REST: JSON, simple, widely supported\n- gRPC: Binary, efficient, streaming\n- C API: Lowest latency, direct\n\nChoose: REST for simplicity, gRPC for performance",
            "diagram": """graph LR
    A[Client] --> B{Protocol?}
    B -->|REST| C[HTTP/JSON]
    B -->|gRPC| D[gRPC/Binary]
    B -->|C API| E[C API/Direct]
    C --> F[Triton Server]
    D --> F
    E --> F""",
            "code_example": """# HTTP REST API
import requests

url = "http://localhost:8000/v2/models/llama-7b/infer"
payload = {
    "inputs": [
        {
            "name": "input_ids",
            "shape": [1, 10],
            "datatype": "INT64",
            "data": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        }
    ]
}

response = requests.post(url, json=payload)
result = response.json()

# gRPC API
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="localhost:8001")
inputs = [grpcclient.InferInput("input_ids", [1, 10], "INT64")]
inputs[0].set_data_from_numpy(input_array)

outputs = [grpcclient.InferRequestedOutput("output_ids")]
result = client.infer("llama-7b", inputs, outputs=outputs)""",
            "links": [
                {"title": "📄 TensorRT Inference Server (Triton)", "url": "https://developer.nvidia.com/blog/triton-inference-server/", "type": "paper"},
                {"title": "📚 Triton Client Libraries", "url": "https://github.com/triton-inference-server/client", "type": "tutorial"}
            ]
        },
        "NIM Packaging": {
            "definition": "NVIDIA Inference Microservices packaging format. Containerized models with runtime. Easy deployment. Pre-configured optimizations. One-command deployment. Simplifies production deployment.",
            "formula": "NIM Packaging:\n- Containerized model + runtime\n- Pre-configured optimizations\n- Standardized format\n- Easy deployment\n\nDeploy: docker run nvcr.io/nim/llama-2-7b",
            "diagram": """graph LR
    A[Model] --> B[NIM Package]
    B --> C[Container]
    C --> D[One-Command<br/>Deployment]""",
            "code_example": """# NIM Packaging and Deployment
# NIM packages are pre-built containers
# Available from NVIDIA Container Registry

# Pull NIM package
docker pull nvcr.io/nim/llama-2-7b:latest

# Run NIM
docker run --gpus all -p 8000:8000 \\
  nvcr.io/nim/llama-2-7b:latest

# NIM includes:
# - Optimized model
# - Runtime environment
# - API server
# - Pre-configured settings

# Query NIM
curl http://localhost:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "llama-2-7b",
    "prompt": "Hello",
    "max_tokens": 100
  }'""",
            "links": [
                {"title": "📄 NVIDIA NIM Documentation", "url": "https://build.nvidia.com/nim", "type": "paper"},
                {"title": "📚 NIM Getting Started", "url": "https://build.nvidia.com/nim/docs/getting-started", "type": "tutorial"}
            ]
        },
        "NIM Routing": {
            "definition": "Load balancing and routing for NIM services. Distributes requests across multiple instances. Handles failover. Enables scaling. Can use Kubernetes ingress or load balancer.",
            "formula": "NIM Routing:\n- Load balancer distributes requests\n- Multiple NIM instances\n- Health checks for failover\n- Auto-scaling based on load\n\nRoutes: /v1/completions, /v1/embeddings, etc.",
            "diagram": """graph TD
    A[Client Requests] --> B[Load Balancer]
    B --> C[NIM Instance 1]
    B --> D[NIM Instance 2]
    B --> E[NIM Instance N]
    C --> F[Response]
    D --> F
    E --> F""",
            "code_example": """# NIM Routing with Kubernetes
apiVersion: v1
kind: Service
metadata:
  name: nim-llama-service
spec:
  selector:
    app: nim-llama
  ports:
    - port: 8000
      targetPort: 8000
  type: LoadBalancer

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-llama
spec:
  replicas: 3  # Multiple instances
  selector:
    matchLabels:
      app: nim-llama
  template:
    metadata:
      labels:
        app: nim-llama
    spec:
      containers:
      - name: nim
        image: nvcr.io/nim/llama-2-7b:latest
        resources:
          limits:
            nvidia.com/gpu: 1

# Load balancer routes to healthy instances""",
            "links": [
                {"title": "📄 NVIDIA NIM Documentation", "url": "https://build.nvidia.com/nim", "type": "paper"},
                {"title": "📚 Kubernetes Ingress Guide", "url": "https://kubernetes.io/docs/concepts/services-networking/ingress/", "type": "tutorial"}
            ]
        },
        "NIM Scaling": {
            "definition": "Auto-scaling NIM instances based on load. Horizontal scaling: add/remove instances. Vertical scaling: increase resources. Kubernetes HPA for auto-scaling. Scales based on metrics (CPU, GPU, queue depth).",
            "formula": "NIM Scaling:\n- Horizontal: Add/remove instances\n- Vertical: Increase resources\n- Auto-scaling: Based on metrics\n- Metrics: CPU, GPU, queue depth, latency\n\nScale: min replicas to max replicas",
            "diagram": """graph TD
    A[Load Metrics] --> B{Threshold?}
    B -->|High| C[Scale Up]
    B -->|Low| D[Scale Down]
    C --> E[Add Instances]
    D --> F[Remove Instances]""",
            "code_example": """# NIM Auto-Scaling with Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nim-llama-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nim-llama
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: queue_depth
      target:
        type: AverageValue
        averageValue: "10"

# Auto-scales based on GPU utilization and queue depth
# Scales up when utilization > 70% or queue > 10
# Scales down when utilization < 50%""",
            "links": [
                {"title": "📄 NVIDIA NIM Documentation", "url": "https://build.nvidia.com/nim", "type": "paper"},
                {"title": "📚 Kubernetes HPA Guide", "url": "https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/", "type": "tutorial"}
            ]
        },
        "Docker": {
            "definition": "Containerization platform for deploying models. Standard format for packaging. Includes model, runtime, dependencies. Portable across environments. Used by Triton, NIM, and custom deployments.",
            "formula": "Docker:\n- Container: Isolated environment\n- Image: Snapshot of container\n- Dockerfile: Build instructions\n- Registry: Store images (Docker Hub, NGC)\n\nBenefits: Portability, consistency, isolation",
            "diagram": """graph LR
    A[Dockerfile] --> B[Build Image]
    B --> C[Docker Image]
    C --> D[Run Container]
    D --> E[Deployed Model]""",
            "code_example": """# Dockerfile for Model Deployment
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# Copy model files
COPY model.pt /app/model.pt
COPY requirements.txt /app/

# Install dependencies
RUN pip install -r requirements.txt

# Expose API port
EXPOSE 8000

# Run inference server
CMD ["python", "inference_server.py"]

# Build image
# docker build -t my-model:latest .

# Run container
# docker run --gpus all -p 8000:8000 my-model:latest

# NVIDIA GPU Runtime
# docker run --gpus all nvidia/cuda:11.8.0-base""",
            "links": [
                {"title": "📄 Docker Documentation", "url": "https://docs.docker.com/", "type": "paper"},
                {"title": "📚 NVIDIA Container Toolkit", "url": "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/", "type": "tutorial"}
            ]
        },
        "GPU Runtime": {
            "definition": "Docker runtime for GPU access. Required to use GPUs in containers. Options: nvidia-docker (legacy) or --gpus flag (modern). Enables GPU access from containers. Required for model inference.",
            "formula": "GPU Runtime:\n- nvidia-docker: Legacy method\n- --gpus flag: Modern method\n- nvidia-container-toolkit: Required\n\nUsage: docker run --gpus all image:tag",
            "diagram": """graph LR
    A[Docker Container] --> B[GPU Runtime]
    B --> C[NVIDIA GPU]
    C --> D[GPU Access]""",
            "code_example": """# GPU Runtime Setup
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \\
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Run container with GPU access
docker run --gpus all nvcr.io/nvidia/pytorch:23.10-py3

# Specify specific GPUs
docker run --gpus '"device=0,1"' nvcr.io/nvidia/pytorch:23.10-py3

# Legacy method (deprecated)
nvidia-docker run nvcr.io/nvidia/pytorch:23.10-py3""",
            "links": [
                {"title": "📄 NVIDIA Container Toolkit", "url": "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/", "type": "paper"},
                {"title": "📚 Docker GPU Support", "url": "https://docs.docker.com/config/containers/resource_constraints/#gpu", "type": "tutorial"}
            ]
        },
        "Blue-Green": {
            "definition": "Deployment strategy with two identical environments. Blue: current production. Green: new version. Switch traffic instantly. Zero downtime. Easy rollback. Requires duplicate infrastructure.",
            "formula": "Blue-Green Deployment:\n- Blue: Current version (production)\n- Green: New version (staging)\n- Deploy: New version to green\n- Switch: Route traffic to green\n- Rollback: Switch back to blue\n\nZero downtime, instant switch",
            "diagram": """graph TD
    A[Traffic] --> B{Version?}
    B -->|Blue| C[Current Version<br/>Production]
    B -->|Green| D[New Version<br/>Staging]
    E[Deploy New] --> D
    F[Switch Traffic] --> B""",
            "code_example": """# Blue-Green Deployment
# Step 1: Deploy new version to green environment
kubectl apply -f deployment-green.yaml

# Step 2: Test green environment
curl https://green.example.com/health

# Step 3: Switch traffic (load balancer config)
# Update load balancer to route to green

# Step 4: Monitor green
# If issues: Switch back to blue
# If OK: Keep green, decommission blue

# Kubernetes example
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  selector:
    version: green  # Switch from blue to green
  ports:
    - port: 80

# Rollback: Change selector back to blue""",
            "links": [
                {"title": "📄 Blue-Green Deployment", "url": "https://martinfowler.com/bliki/BlueGreenDeployment.html", "type": "paper"},
                {"title": "📚 Kubernetes Deployment Strategies", "url": "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/", "type": "tutorial"}
            ]
        },
        "Canary": {
            "definition": "Gradual rollout deployment. Start with small percentage (5-10%). Gradually increase (25%, 50%, 100%). Monitor metrics. Rollback if issues. Lower risk than blue-green. Requires traffic splitting.",
            "formula": "Canary Deployment:\n- Start: 5-10% traffic to new version\n- Gradual: Increase to 25%, 50%, 100%\n- Monitor: Metrics, errors, latency\n- Rollback: If issues detected\n- Lower risk than blue-green",
            "diagram": """graph LR
    A[100% Traffic] --> B[95% Old<br/>5% New]
    B --> C[75% Old<br/>25% New]
    C --> D[50% Old<br/>50% New]
    D --> E[0% Old<br/>100% New]""",
            "code_example": """# Canary Deployment
# Step 1: Deploy canary (5% traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-canary
spec:
  replicas: 1  # Small percentage
  template:
    metadata:
      labels:
        version: canary
    spec:
      containers:
      - name: app
        image: app:v2

# Step 2: Route 5% traffic to canary
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  selector:
    app: app
  # Use Istio/Linkerd for traffic splitting
  # 95% to stable, 5% to canary

# Step 3: Monitor metrics
# If OK: Increase to 25%, then 50%, then 100%
# If issues: Rollback (remove canary)

# Istio VirtualService for traffic splitting
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: app
spec:
  hosts:
  - app.example.com
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: app
        subset: canary
      weight: 5  # 5% traffic
  - route:
    - destination:
        host: app
        subset: stable
      weight: 95  # 95% traffic""",
            "links": [
                {"title": "📄 Canary Releases", "url": "https://martinfowler.com/bliki/CanaryRelease.html", "type": "paper"},
                {"title": "📚 Istio Traffic Management", "url": "https://istio.io/latest/docs/tasks/traffic-management/", "type": "tutorial"}
            ]
        },
        "Shadow": {
            "definition": "Deploy new version alongside production. Process requests but don't return responses to users. Compare outputs. Test without user impact. No risk to production. Requires duplicate processing.",
            "formula": "Shadow Deployment:\n- Deploy: New version alongside production\n- Process: All requests in parallel\n- Compare: Outputs between versions\n- Monitor: Performance, correctness\n- No user impact\n\nTest without risk",
            "diagram": """graph TD
    A[User Request] --> B[Production<br/>Returns Response]
    A --> C[Shadow<br/>Processes Only]
    B --> D[User]
    C --> E[Comparison<br/>Logging]""",
            "code_example": """# Shadow Deployment
# Production service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-production
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: app
        version: production
    spec:
      containers:
      - name: app
        image: app:v1

# Shadow service (processes but doesn't respond)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-shadow
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: app
        version: shadow
    spec:
      containers:
      - name: app
        image: app:v2
        env:
        - name: SHADOW_MODE
          value: "true"  # Don't return responses

# Mirror traffic to shadow
# Compare outputs
# Monitor performance
# No user impact
# Safe testing""",
            "links": [
                {"title": "📄 Shadow Testing", "url": "https://martinfowler.com/articles/blue-green-deployment.html", "type": "paper"},
                {"title": "📚 Istio Mirroring", "url": "https://istio.io/latest/docs/tasks/traffic-management/mirroring/", "type": "tutorial"}
            ]
        }
    },
    "Data Preparation": {
        "BPE (Byte Pair Encoding)": {
            "definition": "Subword tokenization algorithm used by GPT models. Merges most frequent pairs iteratively. Handles out-of-vocabulary words well.",
            "formula": "Algorithm:\n1. Start with characters\n2. Find most frequent pair (A, B)\n3. Merge into single token AB\n4. Repeat until vocab size reached\n\nExample: 'low' + 'est' → 'lowest'",
            "diagram": """graph LR
    A[Text: 'lowest'] --> B[Split: l, o, w, e, s, t]
    B --> C[Count Pairs]
    C --> D[Most Frequent:<br/>'lo', 'we', 'st']
    D --> E[Merge Pairs]
    E --> F[Tokens: 'low', 'est']
    F --> G[Merge Again]
    G --> H[Final: 'lowest']""",
            "code_example": """# BPE Algorithm (simplified)
from collections import Counter

def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    bigram = ' '.join(pair)
    new_vocab = {}
    for word in vocab:
        new_word = word.replace(bigram, ''.join(pair))
        new_vocab[new_word] = vocab[word]
    return new_vocab

# Iteratively merge most frequent pairs""",
            "links": [
                {"title": "📄 Neural Machine Translation of Rare Words (BPE Original)", "url": "https://arxiv.org/abs/1508.07909", "type": "paper"},
                {"title": "📄 BERT: Pre-training Uses WordPiece", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 SentencePiece: A Language-Independent Subword Tokenizer", "url": "https://arxiv.org/abs/1808.06226", "type": "paper"},
                {"title": "📄 Unigram Language Model for Subword Tokenization", "url": "https://arxiv.org/abs/1804.10959", "type": "paper"},
                {"title": "📄 GPT-2 Uses BPE", "url": "https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", "type": "paper"},
                {"title": "🎥 BPE Explained", "url": "https://www.youtube.com/watch?v=HEikzVL-lZU", "type": "video"}
            ]
        },
        "RAG Chunking": {
            "definition": "Split documents into manageable pieces for retrieval. Typical sizes: 256-512 tokens. Balance between context and granularity. Critical for RAG systems.",
            "formula": "Chunk size: 256-512 tokens\nOverlap: 10-20% of chunk size\n\nNumber of chunks ≈ document_length / (chunk_size × (1 - overlap))",
            "diagram": """graph LR
    A[Document] --> B[Split into<br/>Chunks]
    B --> C[Chunk 1<br/>256 tokens]
    B --> D[Chunk 2<br/>256 tokens<br/>20% overlap]
    B --> E[Chunk N<br/>256 tokens]
    C --> F[Embeddings]
    D --> F
    E --> F
    F --> G[Vector Store]""",
            "code_example": """# RAG Chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,  # ~10% overlap
    length_function=len,
)

chunks = text_splitter.split_text(document)

# Create embeddings
embeddings = embed_model.encode(chunks)
vector_store.add(embeddings, chunks)""",
            "links": [
                {"title": "📄 Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG Original)", "url": "https://arxiv.org/abs/2005.11401", "type": "paper"},
                {"title": "📄 Dense Passage Retrieval for Open-Domain Question Answering", "url": "https://arxiv.org/abs/2004.04906", "type": "paper"},
                {"title": "📄 REALM: Retrieval-Augmented Language Model Pre-Training", "url": "https://arxiv.org/abs/2002.08909", "type": "paper"},
                {"title": "📄 In-Context Retrieval-Augmented Language Models", "url": "https://arxiv.org/abs/2302.00083", "type": "paper"}
            ]
        },
        "INT4": {
            "definition": "4-bit integer quantization. 8x memory reduction, 3-4x speed gain. Moderate quality loss. Maximum compression. Often uses group-wise quantization.",
            "formula": "Memory per parameter: 0.5 bytes\nQuantization: x_int4 = round(x_fp32 / scale)\n\nGroup-wise: Different scale per group of weights",
            "diagram": """graph LR
    A[FP32 Weights] --> B[Group into Blocks]
    B --> C[Find Scale per Block]
    C --> D[Quantize to 4-bit]
    D --> E[INT4 Model<br/>0.5 bytes/param]""",
            "code_example": """# INT4 Quantization (GPTQ-style)
import bitsandbytes as bnb

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)""",
            "links": [
                {"title": "📄 QLoRA: Efficient Finetuning of Quantized LLMs", "url": "https://arxiv.org/abs/2305.14314", "type": "paper"},
                {"title": "📄 GPTQ: Accurate Post-Training Quantization", "url": "https://arxiv.org/abs/2210.17323", "type": "paper"},
                {"title": "📄 AWQ: Activation-aware Weight Quantization", "url": "https://arxiv.org/abs/2306.00978", "type": "paper"},
                {"title": "📄 SmoothQuant: Accurate Post-Training Quantization", "url": "https://arxiv.org/abs/2211.10438", "type": "paper"}
            ]
        },
        "Top-k Sampling": {
            "definition": "Sample from k most likely tokens. Limits vocabulary to top candidates. Reduces low-probability tokens. Common decoding strategy.",
            "formula": "1. Sort tokens by probability\n2. Select top k tokens\n3. Renormalize: P'(i) = P(i) / Σ P(top_k)\n4. Sample from top k",
            "diagram": """graph LR
    A[All Tokens<br/>Probabilities] --> B[Sort by Prob]
    B --> C[Select Top k]
    C --> D[Renormalize]
    D --> E[Sample from<br/>Top k]""",
            "code_example": """def top_k_sampling(logits, k=50):
    # Get top k values and indices
    top_k_values, top_k_indices = torch.topk(logits, k)
    
    # Renormalize probabilities
    probs = F.softmax(top_k_values, dim=-1)
    
    # Sample from top k
    sampled_idx = torch.multinomial(probs, 1)
    return top_k_indices[sampled_idx]""",
            "links": [
                {"title": "📄 The Curious Case of Neural Text Degeneration", "url": "https://arxiv.org/abs/1904.09751", "type": "paper"},
                {"title": "📄 Language Models are Unsupervised Multitask Learners", "url": "https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", "type": "paper"}
            ]
        },
        "Top-p Sampling": {
            "definition": "Nucleus sampling. Sample from tokens with cumulative probability p. Dynamic vocabulary size. Often better than top-k. More adaptive.",
            "formula": "1. Sort tokens by probability (descending)\n2. Find smallest set S where Σ P(i) ≥ p\n3. Sample from set S\n\np typically 0.9-0.95",
            "diagram": """graph LR
    A[Sorted Probabilities] --> B[Calculate Cumulative]
    B --> C[Find Set where<br/>Cumulative ≥ p]
    C --> D[Sample from<br/>Nucleus Set]""",
            "code_example": """def top_p_sampling(logits, p=0.9):
    # Sort probabilities
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    # Cumulative probabilities
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find nucleus
    nucleus = cum_probs <= p
    nucleus[0] = True  # Always include top token
    
    # Sample from nucleus
    nucleus_probs = sorted_probs[nucleus]
    nucleus_indices = sorted_indices[nucleus]
    
    sampled_idx = torch.multinomial(nucleus_probs, 1)
    return nucleus_indices[sampled_idx]""",
            "links": [
                {"title": "📄 The Curious Case of Neural Text Degeneration (Original)", "url": "https://arxiv.org/abs/1904.09751", "type": "paper"},
                {"title": "📄 Language Models are Few-Shot Learners", "url": "https://arxiv.org/abs/2005.14165", "type": "paper"}
            ]
        },
        "Full Fine-Tune": {
            "definition": "Update all model parameters. Very high VRAM, slow training. Use for major domain shifts. Best quality but expensive. Requires full model copy in memory.",
            "formula": "Memory ≈ 4 × num_parameters bytes (FP32)\n≈ 2 × num_parameters bytes (FP16)\n\nTraining time: O(num_parameters × dataset_size)",
            "diagram": """graph TD
    A[Pre-trained Model] --> B[All Parameters<br/>Trainable]
    B --> C[Forward Pass]
    C --> D[Backward Pass]
    D --> E[Update All Weights]
    E --> F[Fine-tuned Model]""",
            "code_example": """# Full fine-tuning
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("gpt2")

# All parameters are trainable
for param in model.parameters():
    param.requires_grad = True

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    # Requires large VRAM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()""",
            "links": [
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 Fine-tuning Language Models", "url": "https://arxiv.org/abs/2001.08361", "type": "paper"},
                {"title": "📄 Instruction Tuning with FLAN", "url": "https://arxiv.org/abs/2109.01652", "type": "paper"}
            ]
        },
        "Pipeline Parallelism": {
            "definition": "Split model into stages, each GPU handles one stage. Sequential processing. Use for very long models. Lower communication overhead than tensor parallelism.",
            "formula": "Model split: [Stage₁, Stage₂, ..., Stageₙ]\nEach GPU i processes Stageᵢ\n\nPipeline efficiency: 1 / (1 + (p-1)/m)\nwhere p = pipeline stages, m = micro-batches",
            "diagram": """graph TD
    A[Input] --> B[GPU 1<br/>Stage 1]
    B --> C[GPU 2<br/>Stage 2]
    C --> D[GPU 3<br/>Stage 3]
    D --> E[GPU N<br/>Stage N]
    E --> F[Output]
    G[Micro-batch 1] --> B
    H[Micro-batch 2] --> B
    I[Micro-batch 3] --> B""",
            "code_example": """# Pipeline Parallelism (simplified)
from torch.distributed.pipeline.sync import Pipe

# Split model into stages
stage1 = model.layers[0:4].to('cuda:0')
stage2 = model.layers[4:8].to('cuda:1')
stage3 = model.layers[8:12].to('cuda:2')

# Create pipeline
model = Pipe(stage1, stage2, stage3, chunks=4)

# Forward pass automatically pipelines
output = model(input)""",
            "links": [
                {"title": "📄 GPipe: Efficient Training of Giant Neural Networks", "url": "https://arxiv.org/abs/1811.06965", "type": "paper"},
                {"title": "📄 PipeDream: Fast and Efficient Pipeline Parallelism", "url": "https://arxiv.org/abs/1806.03377", "type": "paper"},
                {"title": "📄 Megatron-LM: Training Multi-Billion Parameter Models", "url": "https://arxiv.org/abs/1909.08053", "type": "paper"},
                {"title": "📄 Efficient Large-Scale Language Model Training", "url": "https://arxiv.org/abs/2104.04473", "type": "paper"}
            ]
        },
        "ZeRO": {
            "definition": "Zero Redundancy Optimizer. Partitions optimizer states, gradients, parameters across GPUs. Reduces memory footprint. Enables massive models. Part of DeepSpeed.",
            "formula": "ZeRO-1: Partition optimizer states\nZeRO-2: + Partition gradients\nZeRO-3: + Partition parameters\n\nMemory reduction: 1/N per GPU\nwhere N = number of GPUs",
            "diagram": """graph TD
    A[Model Parameters] --> B[ZeRO-1:<br/>Partition Optimizer States]
    B --> C[ZeRO-2:<br/>+ Partition Gradients]
    C --> D[ZeRO-3:<br/>+ Partition Parameters]
    D --> E[Memory: 1/N per GPU]""",
            "code_example": """# ZeRO with DeepSpeed
from transformers import Trainer
from transformers.integrations import DeepSpeedConfig

# DeepSpeed config
ds_config = {
    "zero_optimization": {
        "stage": 3,  # ZeRO-3
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    deepspeed=ds_config
)""",
            "links": [
                {"title": "📄 ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Original)", "url": "https://arxiv.org/abs/1910.02054", "type": "paper"},
                {"title": "📄 ZeRO-Offload: Democratizing Billion-Scale Model Training", "url": "https://arxiv.org/abs/2101.06840", "type": "paper"},
                {"title": "📄 ZeRO-Infinity: Breaking GPU Memory Wall", "url": "https://arxiv.org/abs/2104.07857", "type": "paper"},
                {"title": "📄 DeepSpeed: Extreme-Scale Model Training", "url": "https://arxiv.org/abs/1910.02054", "type": "paper"}
            ]
        },
        "BLEU": {
            "definition": "Bilingual Evaluation Understudy. N-gram precision with brevity penalty. Range: 0-1, higher is better. Common for translation. Measures n-gram overlap.",
            "formula": "BLEU = BP × exp(Σ log(pₙ))\n\nwhere:\npₙ = n-gram precision\nBP = brevity penalty = min(1, exp(1 - r/c))\nr = reference length, c = candidate length",
            "diagram": """graph LR
    A[Reference] --> C[Compute N-gram<br/>Precision]
    B[Candidate] --> C
    C --> D[Apply Brevity<br/>Penalty]
    D --> E[BLEU Score]""",
            "code_example": """from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = [['the', 'cat', 'is', 'on', 'the', 'mat']]
candidate = ['the', 'cat', 'sat', 'on', 'the', 'mat']

# BLEU score
smoothing = SmoothingFunction().method1
score = sentence_bleu(reference, candidate, smoothing_function=smoothing)
print(f"BLEU: {score:.4f}")""",
            "links": [
                {"title": "📄 BLEU: A Method for Automatic Evaluation (Original)", "url": "https://aclanthology.org/P02-1040/", "type": "paper"},
                {"title": "📄 ROUGE: A Package for Automatic Evaluation", "url": "https://aclanthology.org/W04-1013/", "type": "paper"},
                {"title": "📄 BERTScore: Evaluating Text Generation with BERT", "url": "https://arxiv.org/abs/1904.09675", "type": "paper"}
            ]
        },
        "BERTScore": {
            "definition": "Semantic similarity using BERT embeddings. More semantic than ROUGE/BLEU. Better captures meaning. Range: 0-1. Uses contextual embeddings.",
            "formula": "BERTScore = 1/|candidate| Σ max(sim(cᵢ, rⱼ))\n\nwhere:\nsim = cosine similarity of BERT embeddings\ncᵢ = candidate token i\nrⱼ = reference token j",
            "diagram": """graph LR
    A[Reference] --> C[BERT<br/>Embeddings]
    B[Candidate] --> C
    C --> D[Cosine<br/>Similarity]
    D --> E[Match Tokens]
    E --> F[BERTScore]""",
            "code_example": """from bert_score import score

candidates = ["the cat sat on the mat"]
references = [["the cat is on the mat"]]

P, R, F1 = score(candidates, references, lang='en', verbose=True)
# P = Precision, R = Recall, F1 = F1 score
print(f"BERTScore F1: {F1.item():.4f}")""",
            "links": [
                {"title": "📄 BERTScore: Evaluating Text Generation with BERT (Original)", "url": "https://arxiv.org/abs/1904.09675", "type": "paper"},
                {"title": "📄 BLEU: A Method for Automatic Evaluation", "url": "https://aclanthology.org/P02-1040/", "type": "paper"},
                {"title": "📄 ROUGE: A Package for Automatic Evaluation", "url": "https://aclanthology.org/W04-1013/", "type": "paper"}
            ]
        },
        "Log-Loss": {
            "definition": "Logarithmic loss (cross-entropy loss). Measures prediction quality. Lower is better. Range: 0 to ∞. Used for classification tasks. Penalizes confident wrong predictions more.",
            "formula": "Log-Loss = -1/N Σ [y_i × log(p_i) + (1-y_i) × log(1-p_i)]\n\nwhere:\ny_i = true label (0 or 1)\np_i = predicted probability\nN = number of samples\n\nLower is better",
            "diagram": """graph LR
    A[True Label] --> C[Log-Loss]
    B[Predicted Prob] --> C
    C --> D[Penalty for<br/>Wrong Predictions]""",
            "code_example": """# Log-Loss (Cross-Entropy)
import torch.nn.functional as F

# Binary classification
true_labels = torch.tensor([1, 0, 1, 1])
pred_probs = torch.tensor([0.9, 0.2, 0.8, 0.7])

# Log-loss
log_loss = F.binary_cross_entropy(pred_probs, true_labels.float())
# Lower is better

# Multi-class
true_labels = torch.tensor([2, 0, 1])
pred_logits = torch.tensor([
    [0.1, 0.2, 0.7],
    [0.8, 0.1, 0.1],
    [0.2, 0.6, 0.2]
])

log_loss = F.cross_entropy(pred_logits, true_labels)""",
            "links": [
                {"title": "📄 Cross-Entropy Loss", "url": "https://en.wikipedia.org/wiki/Cross_entropy", "type": "paper"},
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"}
            ]
        },
        "ROUGE-1": {
            "definition": "ROUGE metric measuring unigram (word) overlap between reference and candidate. Recall-oriented. Measures word-level similarity. Good for content coverage assessment.",
            "formula": "ROUGE-1 = Count of overlapping unigrams / Count of unigrams in reference\n\nROUGE-1 Recall = |unigrams(candidate) ∩ unigrams(reference)| / |unigrams(reference)|\nROUGE-1 Precision = |unigrams(candidate) ∩ unigrams(reference)| / |unigrams(candidate)|",
            "diagram": """graph LR
    A[Reference<br/>Unigrams] --> C[Overlap]
    B[Candidate<br/>Unigrams] --> C
    C --> D[ROUGE-1 Score]""",
            "code_example": """# ROUGE-1 Calculation
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

reference = "the cat sat on the mat"
candidate = "the cat is on the mat"

scores = scorer.score(reference, candidate)
rouge1 = scores['rouge1']

# ROUGE-1 measures word overlap
# Unigrams: 'the', 'cat', 'sat', 'on', 'the', 'mat'
# Overlap: 'the', 'cat', 'on', 'the', 'mat'
# ROUGE-1 Recall = 5/6 = 0.83""",
            "links": [
                {"title": "📄 ROUGE: A Package for Automatic Evaluation of Summaries (Original)", "url": "https://aclanthology.org/W04-1013/", "type": "paper"},
                {"title": "📄 BLEU: A Method for Automatic Evaluation", "url": "https://aclanthology.org/P02-1040/", "type": "paper"}
            ]
        },
        "ROUGE-2": {
            "definition": "ROUGE metric measuring bigram (2-word pairs) overlap. More strict than ROUGE-1. Measures phrase-level similarity. Better captures word order and fluency.",
            "formula": "ROUGE-2 = Count of overlapping bigrams / Count of bigrams in reference\n\nROUGE-2 Recall = |bigrams(candidate) ∩ bigrams(reference)| / |bigrams(reference)|\n\nBigrams: consecutive word pairs",
            "diagram": """graph LR
    A[Reference<br/>Bigrams] --> C[Overlap]
    B[Candidate<br/>Bigrams] --> C
    C --> D[ROUGE-2 Score]""",
            "code_example": """# ROUGE-2 Calculation
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)

reference = "the cat sat on the mat"
candidate = "the cat is on the mat"

scores = scorer.score(reference, candidate)
rouge2 = scores['rouge2']

# Bigrams in reference:
# 'the cat', 'cat sat', 'sat on', 'on the', 'the mat'
# Bigrams in candidate:
# 'the cat', 'cat is', 'is on', 'on the', 'the mat'
# Overlap: 'the cat', 'on the', 'the mat'
# ROUGE-2 Recall = 3/5 = 0.6""",
            "links": [
                {"title": "📄 ROUGE: A Package for Automatic Evaluation of Summaries (Original)", "url": "https://aclanthology.org/W04-1013/", "type": "paper"},
                {"title": "📄 BLEU: A Method for Automatic Evaluation", "url": "https://aclanthology.org/P02-1040/", "type": "paper"}
            ]
        },
        "Pruning": {
            "definition": "Remove less important weights or neurons. Reduces model size. Can be structured (channels) or unstructured (individual weights). Enables faster inference.",
            "formula": "Magnitude-based pruning:\nRemove weights where |w| < threshold\n\nSparsity: S = (removed_weights) / (total_weights)",
            "diagram": """graph LR
    A[Dense Model] --> B[Identify<br/>Low Magnitude Weights]
    B --> C[Remove Weights]
    C --> D[Sparse Model]
    D --> E[Fine-tune]""",
            "code_example": """# Magnitude-based pruning
import torch.nn.utils.prune as prune

# Prune 20% of weights
prune.l1_unstructured(module, name='weight', amount=0.2)

# Remove pruned weights permanently
prune.remove(module, 'weight')""",
            "links": [
                {"title": "📄 The Lottery Ticket Hypothesis", "url": "https://arxiv.org/abs/1803.03635", "type": "paper"},
                {"title": "📄 Learning both Weights and Connections", "url": "https://arxiv.org/abs/1506.02626", "type": "paper"},
                {"title": "📄 Movement Pruning: Adaptive Sparsity", "url": "https://arxiv.org/abs/2005.07683", "type": "paper"},
                {"title": "📄 SparseGPT: Massive Language Models", "url": "https://arxiv.org/abs/2301.00774", "type": "paper"}
            ]
        },
        "Distillation": {
            "definition": "Train smaller student model to mimic larger teacher model. Reduces size while preserving knowledge. Knowledge transfer technique. Enables efficient deployment.",
            "formula": "Loss = α × Hard_Loss + (1-α) × Soft_Loss\n\nHard_Loss = CrossEntropy(student, labels)\nSoft_Loss = KL(student_logits/T, teacher_logits/T)\nT = temperature",
            "diagram": """graph TD
    A[Teacher Model<br/>Large] --> B[Soft Labels]
    C[Student Model<br/>Small] --> D[Predictions]
    B --> E[Knowledge Distillation<br/>Loss]
    D --> E
    E --> F[Train Student]""",
            "code_example": """# Knowledge Distillation
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    # Hard loss
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Soft loss
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)
    
    # Combined
    return alpha * hard_loss + (1 - alpha) * soft_loss""",
            "links": [
                {"title": "📄 Distilling the Knowledge in a Neural Network (Original)", "url": "https://arxiv.org/abs/1503.02531", "type": "paper"},
                {"title": "📄 Patient Knowledge Distillation for BERT", "url": "https://arxiv.org/abs/1908.09355", "type": "paper"},
                {"title": "📄 TinyBERT: Distilling BERT", "url": "https://arxiv.org/abs/1909.10351", "type": "paper"},
                {"title": "📄 MiniLM: Deep Self-Attention Distillation", "url": "https://arxiv.org/abs/2002.10957", "type": "paper"}
            ]
        }
    },
    "Data Preparation": {
        "RAG Chunking": {
            "definition": "Split documents into manageable pieces for retrieval. Typical sizes: 256-512 tokens. Balance between context and granularity. Critical for RAG systems.",
            "formula": "Chunk size: 256-512 tokens\nOverlap: 10-20% of chunk size\n\nNumber of chunks ≈ document_length / (chunk_size × (1 - overlap))",
            "diagram": """graph LR
    A[Document] --> B[Split into<br/>Chunks]
    B --> C[Chunk 1<br/>256 tokens]
    B --> D[Chunk 2<br/>256 tokens<br/>20% overlap]
    B --> E[Chunk N<br/>256 tokens]
    C --> F[Embeddings]
    D --> F
    E --> F
    F --> G[Vector Store]""",
            "code_example": """# RAG Chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,  # ~10% overlap
    length_function=len,
)

chunks = text_splitter.split_text(document)

# Create embeddings
embeddings = embed_model.encode(chunks)
vector_store.add(embeddings, chunks)""",
            "links": [
                {"title": "📄 Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG Original)", "url": "https://arxiv.org/abs/2005.11401", "type": "paper"},
                {"title": "📄 Dense Passage Retrieval for Open-Domain Question Answering", "url": "https://arxiv.org/abs/2004.04906", "type": "paper"},
                {"title": "📄 REALM: Retrieval-Augmented Language Model Pre-Training", "url": "https://arxiv.org/abs/2002.08909", "type": "paper"},
                {"title": "📄 In-Context Retrieval-Augmented Language Models", "url": "https://arxiv.org/abs/2302.00083", "type": "paper"}
            ]
        },
        "WordPiece": {
            "definition": "Subword tokenization algorithm used by BERT. Similar to BPE but uses word-level statistics. Splits words into subword units. Handles out-of-vocabulary words.",
            "formula": "Algorithm:\n1. Initialize vocab with characters\n2. For each word, find best subword segmentation\n3. Merge most frequent subword pairs\n4. Repeat until vocab size reached\n\nDiffers from BPE: Uses word-level likelihood",
            "diagram": """graph LR
    A[Word: 'playing'] --> B[Split: 'play' + '##ing']
    B --> C[Subword Units]
    C --> D[Token IDs]""",
            "code_example": """# WordPiece Tokenization (BERT-style)
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# WordPiece tokenization
text = "playing"
tokens = tokenizer.tokenize(text)
# Output: ['play', '##ing']

# Full tokenization
encoded = tokenizer.encode(text)
# Output: [101, 2345, 2346, 102]  # [CLS] play ##ing [SEP]""",
            "links": [
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers (Uses WordPiece)", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 Neural Machine Translation of Rare Words (BPE Original)", "url": "https://arxiv.org/abs/1508.07909", "type": "paper"},
                {"title": "📄 SentencePiece: A Language-Independent Subword Tokenizer", "url": "https://arxiv.org/abs/1808.06226", "type": "paper"}
            ]
        },
        "Vocab Size": {
            "definition": "Size of tokenizer vocabulary. Typical ranges: 30K-50K tokens. Larger vocab = fewer subwords but more memory. Smaller vocab = more subwords but less memory. Balance between coverage and efficiency.",
            "formula": "Typical vocab sizes:\n- GPT-2: 50,257 tokens\n- BERT: 30,522 tokens (WordPiece)\n- T5: 32,128 tokens\n\nTrade-off: Coverage vs Memory",
            "diagram": """graph LR
    A[Small Vocab<br/>~10K] --> B[More Subwords<br/>More Memory]
    C[Medium Vocab<br/>~30-50K] --> D[Balanced]
    E[Large Vocab<br/>~100K+] --> F[Fewer Subwords<br/>Less Memory]""",
            "code_example": """# Vocabulary size impact
from transformers import GPT2Tokenizer, BertTokenizer

# GPT-2: 50,257 tokens
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print(f"GPT-2 vocab size: {len(gpt2_tokenizer)}")

# BERT: 30,522 tokens
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(f"BERT vocab size: {len(bert_tokenizer)}")

# Larger vocab = fewer subword splits
text = "unhappiness"
gpt2_tokens = gpt2_tokenizer.tokenize(text)  # ['un', 'happiness']
bert_tokens = bert_tokenizer.tokenize(text)  # ['un', '##happy', '##ness']""",
            "links": [
                {"title": "📄 Language Models are Unsupervised Multitask Learners (GPT-2)", "url": "https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", "type": "paper"},
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 SentencePiece: A Language-Independent Subword Tokenizer", "url": "https://arxiv.org/abs/1808.06226", "type": "paper"}
            ]
        },
        "Special Tokens": {
            "definition": "Reserved tokens with special meaning. Common: <BOS> (beginning of sequence), <EOS> (end of sequence), <PAD> (padding), <UNK> (unknown), <SEP> (separator), <CLS> (classification). Used for model control and structure.",
            "formula": "Common special tokens:\n- <BOS>/<SOS>: Start of sequence\n- <EOS>: End of sequence\n- <PAD>: Padding token (ID: 0)\n- <UNK>: Unknown/out-of-vocab\n- <SEP>: Separator (BERT)\n- <CLS>: Classification token (BERT)\n- <MASK>: Masking token (BERT)",
            "diagram": """graph LR
    A[Input Text] --> B[Add Special Tokens]
    B --> C[<BOS> text <EOS>]
    C --> D[Tokenize]
    D --> E[Token IDs]""",
            "code_example": """# Special tokens usage
from transformers import GPT2Tokenizer, BertTokenizer

# GPT-2 special tokens
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# <|endoftext|> = end token
text = "Hello world"
tokens = gpt2_tokenizer.encode(text)
# [15496, 995, 50256]  # 50256 = <|endoftext|>

# BERT special tokens
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# [CLS] = 101, [SEP] = 102, [PAD] = 0, [UNK] = 100
text = "Hello world"
tokens = bert_tokenizer.encode(text)
# [101, 7592, 2088, 102]  # [CLS] hello world [SEP]""",
            "links": [
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 Language Models are Unsupervised Multitask Learners (GPT-2)", "url": "https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", "type": "paper"}
            ]
        },
        "Pre-train Split": {
            "definition": "Dataset split used for pre-training. Typically 80-90% of data. Large, diverse corpus. Used to learn general language patterns. Foundation for all downstream tasks.",
            "formula": "Typical splits:\n- Pre-train: 80-90%\n- Fine-tune: 5-10%\n- Eval/Test: 5-10%\n\nPre-train size: Millions to billions of tokens",
            "diagram": """graph TD
    A[Total Dataset] --> B[Pre-train Split<br/>80-90%]
    A --> C[Fine-tune Split<br/>5-10%]
    A --> D[Eval/Test Split<br/>5-10%]
    B --> E[General Language<br/>Learning]""",
            "code_example": """# Dataset splitting
from datasets import load_dataset

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2")

# Split for pre-training
train_test = dataset['train'].train_test_split(test_size=0.1)
pretrain_data = train_test['train']  # 90% for pre-training
eval_data = train_test['test']  # 10% for evaluation

# Pre-train split: Large, diverse corpus
# Used to learn general language patterns
# Examples: Common Crawl, Wikipedia, Books""",
            "links": [
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 Language Models are Few-Shot Learners (GPT-3)", "url": "https://arxiv.org/abs/2005.14165", "type": "paper"},
                {"title": "📄 Training Compute-Optimal Large Language Models (Chinchilla)", "url": "https://arxiv.org/abs/2203.15556", "type": "paper"}
            ]
        },
        "Fine-tune Split": {
            "definition": "Dataset split used for fine-tuning. Typically 5-10% of data. Task-specific or domain-specific. Smaller than pre-train but focused. Used to adapt model to specific tasks.",
            "formula": "Fine-tune split:\n- Size: 5-10% of total data\n- Task-specific examples\n- Domain-specific data\n- Quality > Quantity",
            "diagram": """graph TD
    A[Total Dataset] --> B[Pre-train Split<br/>80-90%]
    A --> C[Fine-tune Split<br/>5-10%]
    C --> D[Task-Specific<br/>Learning]""",
            "code_example": """# Fine-tuning split
from datasets import load_dataset

# Load task-specific dataset
dataset = load_dataset("glue", "sst2")  # Sentiment classification

# Split for fine-tuning
train_test = dataset['train'].train_test_split(test_size=0.1)
finetune_data = train_test['train']  # Fine-tuning split
eval_data = train_test['test']  # Evaluation

# Fine-tune split: Task-specific examples
# Smaller but focused on target task
# Examples: GLUE, SuperGLUE, domain-specific datasets""",
            "links": [
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 GLUE: A Multi-Task Benchmark", "url": "https://arxiv.org/abs/1804.07461", "type": "paper"},
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"}
            ]
        },
        "Eval/Test Split": {
            "definition": "Dataset split used for evaluation/testing. Typically 5-10% of data. Held-out, never used for training. Measures model performance. Critical for generalization assessment.",
            "formula": "Eval/Test split:\n- Size: 5-10% of total data\n- Held-out (never seen during training)\n- Used only for evaluation\n- Measures generalization",
            "diagram": """graph TD
    A[Total Dataset] --> B[Train Split<br/>80-90%]
    A --> C[Eval/Test Split<br/>5-10%]
    C --> D[Held-Out<br/>Never Used for Training]
    D --> E[Performance<br/>Evaluation]""",
            "code_example": """# Evaluation split
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load dataset
dataset = load_dataset("glue", "sst2")

# Split: Train, Validation, Test
train_val = dataset['train'].train_test_split(test_size=0.1)
train_data = train_val['train']  # Training
val_data = train_val['test']  # Validation (for hyperparameter tuning)
test_data = dataset['validation']  # Test (held-out, final evaluation)

# Test split: Never used for training
# Used only for final performance evaluation
# Critical for measuring generalization""",
            "links": [
                {"title": "📄 GLUE: A Multi-Task Benchmark", "url": "https://arxiv.org/abs/1804.07461", "type": "paper"},
                {"title": "📄 SuperGLUE: A Stickier Benchmark", "url": "https://arxiv.org/abs/1905.00537", "type": "paper"},
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"}
            ]
        },
        "Leakage": {
            "definition": "Data leakage occurs when test data information leaks into training data. Causes overfitting and unrealistic performance. Must prevent by strict train/test separation. Common in time-series or similar data.",
            "formula": "Leakage types:\n- Target leakage: Future info in training\n- Train-test leakage: Test data in training\n- Temporal leakage: Future data in past\n\nPrevention: Strict splits, no overlap",
            "diagram": """graph TD
    A[Dataset] --> B{Proper Split?}
    B -->|Yes| C[Train/Test Separation]
    B -->|No| D[Data Leakage]
    C --> E[Accurate Evaluation]
    D --> F[Overfitting<br/>Unrealistic Performance]""",
            "code_example": """# Preventing data leakage
from sklearn.model_selection import train_test_split
import pandas as pd

# WRONG: Shuffle before splitting (can cause leakage)
# df = df.sample(frac=1).reset_index(drop=True)  # Don't do this!

# CORRECT: Time-based split (for time-series)
df = pd.read_csv('data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Split by time
split_date = '2023-01-01'
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]

# Check for leakage
train_ids = set(train['id'])
test_ids = set(test['id'])
assert len(train_ids & test_ids) == 0, "Leakage detected!"

# CORRECT: Random split (for i.i.d. data)
train, test = train_test_split(df, test_size=0.2, random_state=42)""",
            "links": [
                {"title": "📄 Preventing Data Leakage in Machine Learning", "url": "https://arxiv.org/abs/1807.04153", "type": "paper"},
                {"title": "📄 Time Series Cross-Validation", "url": "https://robjhyndman.com/hyndsight/tscv/", "type": "paper"}
            ]
        },
        "Weight Quantization": {
            "definition": "Quantize model weights (parameters) to lower precision. Reduces model size and memory. Weights stored in INT8/INT4 instead of FP32. Inference uses quantized weights. Training typically uses FP32.",
            "formula": "Weight quantization:\n- FP32 → INT8: 4x size reduction\n- FP32 → INT4: 8x size reduction\n\nQuantization: w_int = round(w_fp32 / scale)\nDequantization: w_fp32 ≈ w_int × scale",
            "diagram": """graph LR
    A[FP32 Weights] --> B[Find Scale]
    B --> C[Quantize to INT8/INT4]
    C --> D[Quantized Model<br/>Smaller Size]""",
            "code_example": """# Weight Quantization
import torch
import torch.quantization as quantization

# Model with FP32 weights
model = torch.nn.Linear(100, 50)

# Post-training quantization
model.qconfig = quantization.get_default_qconfig('fbgemm')
quantized_model = quantization.prepare(model, inplace=False)
quantized_model = quantization.convert(quantized_model)

# Quantized weights are INT8
# Model size reduced by 4x
# Inference uses quantized weights""",
            "links": [
                {"title": "📄 Quantization and Training of Neural Networks", "url": "https://arxiv.org/abs/1712.05877", "type": "paper"},
                {"title": "📄 Q8BERT: Quantized 8Bit BERT", "url": "https://arxiv.org/abs/1910.06188", "type": "paper"},
                {"title": "📄 GPTQ: Accurate Post-Training Quantization", "url": "https://arxiv.org/abs/2210.17323", "type": "paper"}
            ]
        },
        "Activation Quantization": {
            "definition": "Quantize activations (intermediate outputs) during inference. More complex than weight quantization. Requires calibration data. Can cause accuracy loss. Used for maximum speed/memory reduction.",
            "formula": "Activation quantization:\n- Quantize activations during forward pass\n- Requires calibration to find scales\n- More complex than weight quantization\n\nCalibration: Use sample data to find activation ranges",
            "diagram": """graph TD
    A[Input] --> B[FP32 Activation 1]
    B --> C[Quantize to INT8]
    C --> D[INT8 Activation 2]
    D --> E[Quantize to INT8]
    E --> F[Output]""",
            "code_example": """# Activation Quantization
import torch
import torch.quantization as quantization

model = torch.nn.Sequential(
    torch.nn.Linear(100, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 10)
)

# Quantize both weights and activations
model.qconfig = quantization.get_default_qconfig('fbgemm')

# Calibration: Use sample data to find activation scales
calibration_data = [torch.randn(1, 100) for _ in range(100)]
quantized_model = quantization.prepare(model)
for data in calibration_data:
    quantized_model(data)
quantized_model = quantization.convert(quantized_model)

# Both weights and activations are INT8
# Maximum speed/memory reduction""",
            "links": [
                {"title": "📄 Quantization and Training of Neural Networks", "url": "https://arxiv.org/abs/1712.05877", "type": "paper"},
                {"title": "📄 SmoothQuant: Accurate Post-Training Quantization", "url": "https://arxiv.org/abs/2211.10438", "type": "paper"},
                {"title": "📄 LLM.int8(): 8-bit Matrix Multiplication", "url": "https://arxiv.org/abs/2208.07339", "type": "paper"}
            ]
        },
        "PTQ": {
            "definition": "Post-Training Quantization. Quantize model after training is complete. No retraining needed. Fast but may have accuracy loss. Uses calibration data to find quantization scales.",
            "formula": "PTQ Process:\n1. Train model in FP32\n2. Collect calibration data\n3. Find quantization scales\n4. Quantize weights/activations\n5. No retraining needed",
            "diagram": """graph LR
    A[FP32 Trained Model] --> B[Calibration Data]
    B --> C[Find Scales]
    C --> D[Quantize]
    D --> E[Quantized Model]""",
            "code_example": """# Post-Training Quantization (PTQ)
import torch
import torch.quantization as quantization

# Step 1: Train model in FP32 (already done)
model = load_pretrained_model()

# Step 2: Prepare for quantization
model.qconfig = quantization.get_default_qconfig('fbgemm')
quantized_model = quantization.prepare(model)

# Step 3: Calibration (find scales)
calibration_data = load_calibration_dataset()
for data in calibration_data:
    quantized_model(data)

# Step 4: Convert to quantized model
quantized_model = quantization.convert(quantized_model)

# Model is now quantized, ready for inference
# No retraining needed""",
            "links": [
                {"title": "📄 Quantization and Training of Neural Networks", "url": "https://arxiv.org/abs/1712.05877", "type": "paper"},
                {"title": "📄 GPTQ: Accurate Post-Training Quantization", "url": "https://arxiv.org/abs/2210.17323", "type": "paper"},
                {"title": "📄 SmoothQuant: Accurate Post-Training Quantization", "url": "https://arxiv.org/abs/2211.10438", "type": "paper"}
            ]
        },
        "QAT": {
            "definition": "Quantization-Aware Training. Train model with quantization simulation. Better accuracy than PTQ. Model learns to work with quantized weights. More time-consuming but better results.",
            "formula": "QAT Process:\n1. Start with FP32 model\n2. Simulate quantization during training\n3. Train with quantized simulation\n4. Model adapts to quantization\n5. Convert to quantized model",
            "diagram": """graph LR
    A[FP32 Model] --> B[Add Quantization<br/>Simulation]
    B --> C[Train with<br/>Simulated Quantization]
    C --> D[Model Adapts]
    D --> E[Quantized Model]""",
            "code_example": """# Quantization-Aware Training (QAT)
import torch
import torch.quantization as quantization

# Step 1: Start with FP32 model
model = create_model()

# Step 2: Add quantization simulation
model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
qat_model = quantization.prepare_qat(model)

# Step 3: Train with quantization simulation
optimizer = torch.optim.Adam(qat_model.parameters())
for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = qat_model(data)  # Quantization simulated
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Step 4: Convert to quantized model
quantized_model = quantization.convert(qat_model)

# Better accuracy than PTQ, but requires retraining""",
            "links": [
                {"title": "📄 Quantization and Training of Neural Networks", "url": "https://arxiv.org/abs/1712.05877", "type": "paper"},
                {"title": "📄 PACT: Parameterized Clipping Activation", "url": "https://arxiv.org/abs/1805.06085", "type": "paper"},
                {"title": "📄 LSQ: Learned Step Size Quantization", "url": "https://arxiv.org/abs/1902.08153", "type": "paper"}
            ]
        },
        "Beam Search": {
            "definition": "Decoding strategy that explores multiple paths. Keeps top-k candidates at each step. Better quality than greedy but slower. Used for tasks requiring high quality output.",
            "formula": "Beam Search:\n- Beam width = k (typically 4-10)\n- Keep top-k sequences at each step\n- Score = log probability sum\n- Select best sequence at end\n\nTime complexity: O(k × vocab_size × length)",
            "diagram": """graph TD
    A[Start] --> B[Generate k candidates]
    B --> C[Expand each candidate]
    C --> D[Keep top-k]
    D --> E{End?}
    E -->|No| C
    E -->|Yes| F[Select Best]""",
            "code_example": """# Beam Search Decoding
import torch
import torch.nn.functional as F

def beam_search(model, start_token, beam_width=5, max_length=50):
    # Initialize beams: (sequence, score)
    beams = [([start_token], 0.0)]
    
    for step in range(max_length):
        new_beams = []
        
        for sequence, score in beams:
            # Get next token probabilities
            logits = model(torch.tensor([sequence]))
            probs = F.log_softmax(logits[0, -1], dim=0)
            
            # Get top-k tokens
            top_k_probs, top_k_indices = torch.topk(probs, beam_width)
            
            # Expand each beam
            for prob, idx in zip(top_k_probs, top_k_indices):
                new_sequence = sequence + [idx.item()]
                new_score = score + prob.item()
                new_beams.append((new_sequence, new_score))
        
        # Keep top-k beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Check for end token
        if beams[0][0][-1] == end_token_id:
            break
    
    return beams[0][0]  # Best sequence""",
            "links": [
                {"title": "📄 Neural Machine Translation by Jointly Learning", "url": "https://arxiv.org/abs/1409.3215", "type": "paper"},
                {"title": "📄 The Curious Case of Neural Text Degeneration", "url": "https://arxiv.org/abs/1904.09751", "type": "paper"},
                {"title": "📄 Language Models are Unsupervised Multitask Learners", "url": "https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", "type": "paper"}
            ]
        },
        "Sampling": {
            "definition": "Decoding strategy that randomly samples from probability distribution. More diverse outputs than greedy. Faster than beam search. Uses temperature to control randomness.",
            "formula": "Sampling:\n- Sample from P(token | context)\n- Temperature controls randomness:\n  - Low T (0.1-0.5): More deterministic\n  - High T (1.0-2.0): More random\n- Can use top-k or top-p filtering",
            "diagram": """graph LR
    A[Probability Distribution] --> B[Apply Temperature]
    B --> C[Sample Token]
    C --> D[Add to Sequence]
    D --> E{End?}
    E -->|No| A
    E -->|Yes| F[Output]""",
            "code_example": """# Sampling Decoding
import torch
import torch.nn.functional as F

def sample_decode(model, start_token, temperature=1.0, max_length=50):
    sequence = [start_token]
    
    for step in range(max_length):
        # Get logits
        logits = model(torch.tensor([sequence]))
        logits = logits[0, -1] / temperature  # Apply temperature
        
        # Sample from distribution
        probs = F.softmax(logits, dim=0)
        next_token = torch.multinomial(probs, 1).item()
        
        sequence.append(next_token)
        
        if next_token == end_token_id:
            break
    
    return sequence

# Temperature effect:
# Low (0.1): More deterministic, less diverse
# High (2.0): More random, more diverse""",
            "links": [
                {"title": "📄 The Curious Case of Neural Text Degeneration", "url": "https://arxiv.org/abs/1904.09751", "type": "paper"},
                {"title": "📄 Language Models are Unsupervised Multitask Learners", "url": "https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", "type": "paper"}
            ]
        },
        "TensorRT-LLM Graph Fusion": {
            "definition": "TensorRT-LLM optimization that fuses multiple operations into single kernels. Reduces kernel launch overhead. Improves GPU utilization. Significant speedup (2-5x). Combines operations like Add+LayerNorm, Attention operations.",
            "formula": "Graph Fusion Benefits:\n- Reduces kernel launches\n- Better GPU utilization\n- Lower memory bandwidth\n- Typical speedup: 2-5x\n\nFused operations: Add+Norm, Attention patterns, FFN patterns",
            "diagram": """graph LR
    A[Operation 1] --> C[Fused Kernel]
    B[Operation 2] --> C
    C --> D[Single GPU Kernel]
    D --> E[Faster Execution]""",
            "code_example": """# TensorRT-LLM Graph Fusion (conceptual)
# Before fusion:
# 1. Add operation
# 2. LayerNorm operation
# 3. Two kernel launches

# After fusion:
# 1. Fused Add+LayerNorm kernel
# 2. Single kernel launch
# 3. Better performance

# TensorRT-LLM automatically fuses:
# - Add + LayerNorm
# - Attention operations
# - FFN operations
# - Activation functions

# Usage:
from tensorrt_llm import Builder

builder = Builder()
# TensorRT-LLM automatically applies graph fusion
optimized_model = builder.build(model)""",
            "links": [
                {"title": "📄 TensorRT-LLM Documentation", "url": "https://nvidia.github.io/TensorRT-LLM/", "type": "paper"},
                {"title": "📄 TensorRT: High-Performance Deep Learning Inference", "url": "https://developer.nvidia.com/tensorrt", "type": "paper"},
                {"title": "📚 TensorRT-LLM Optimization Guide", "url": "https://nvidia.github.io/TensorRT-LLM/optimization.html", "type": "tutorial"}
            ]
        },
        "Kernel Auto-Tuning": {
            "definition": "Automatically tune GPU kernels for specific hardware. Tests different kernel configurations. Selects best performing kernel. Optimizes for specific GPU architecture. Part of TensorRT-LLM optimization.",
            "formula": "Auto-Tuning Process:\n1. Generate kernel variants\n2. Benchmark each variant\n3. Select best performing kernel\n4. Cache optimal configuration\n\nOptimizes: Block size, thread count, memory access patterns",
            "diagram": """graph TD
    A[Kernel Variants] --> B[Benchmark Each]
    B --> C[Measure Performance]
    C --> D[Select Best]
    D --> E[Cache Configuration]
    E --> F[Use Optimal Kernel]""",
            "code_example": """# Kernel Auto-Tuning (TensorRT-LLM)
# TensorRT-LLM automatically tunes kernels for:
# - Specific GPU architecture (A100, H100, etc.)
# - Input shapes
# - Batch sizes
# - Precision (FP16, INT8)

from tensorrt_llm import Builder

builder = Builder()
# Auto-tuning happens during build
# Tests different kernel configurations
# Selects optimal for your hardware
optimized_model = builder.build(
    model,
    auto_tune=True  # Enable auto-tuning
)

# Tuned kernels cached for reuse
# Significant performance improvement""",
            "links": [
                {"title": "📄 TensorRT-LLM Documentation", "url": "https://nvidia.github.io/TensorRT-LLM/", "type": "paper"},
                {"title": "📄 CUDA Kernel Optimization", "url": "https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/", "type": "paper"},
                {"title": "📚 TensorRT Optimization Guide", "url": "https://docs.nvidia.com/deeplearning/tensorrt/", "type": "tutorial"}
            ]
        },
        "KV Cache Optimization": {
            "definition": "Optimize KV cache memory usage and access patterns. Critical for efficient generation. Techniques: PagedAttention, FlashAttention, compression. Reduces memory footprint and improves throughput.",
            "formula": "KV Cache Memory:\nMemory = batch_size × seq_len × num_layers × hidden_size × 2 (K+V) × bytes_per_param\n\nOptimization techniques:\n- PagedAttention: Reduce fragmentation\n- Compression: Reduce precision\n- Eviction: Remove old cache entries",
            "diagram": """graph TD
    A[KV Cache] --> B{Optimization?}
    B -->|PagedAttention| C[Reduce Fragmentation]
    B -->|Compression| D[Reduce Size]
    B -->|Eviction| E[Remove Old]
    C --> F[Better Performance]
    D --> F
    E --> F""",
            "code_example": """# KV Cache Optimization
# Standard KV cache: O(batch × seq_len × hidden_size)

# Optimization 1: PagedAttention (vLLM)
from vllm import LLM

llm = LLM(
    model="gpt-3.5-turbo",
    enable_prefix_caching=True,  # Reuse KV cache for prefixes
    max_model_len=4096
)

# Optimization 2: FlashAttention (memory efficient)
from flash_attn import flash_attn_func

# Reduces memory from O(n²) to O(n)
output = flash_attn_func(q, k, v)

# Optimization 3: Compression
# Use INT8 for KV cache instead of FP16
kv_cache_int8 = quantize_kv_cache(kv_cache_fp16)""",
            "links": [
                {"title": "📄 FlashAttention: Fast and Memory-Efficient Exact Attention", "url": "https://arxiv.org/abs/2205.14135", "type": "paper"},
                {"title": "📄 PagedAttention: From Interface to Implementation", "url": "https://arxiv.org/abs/2309.06180", "type": "paper"},
                {"title": "📄 vLLM: Easy, Fast, and Cheap LLM Serving", "url": "https://arxiv.org/abs/2309.06180", "type": "paper"}
            ]
        },
        "Rank (r)": {
            "definition": "Rank parameter in LoRA. Controls adapter size. Lower rank = fewer parameters but less capacity. Higher rank = more parameters but more capacity. Typical values: 4-16. Most common: 8.",
            "formula": "LoRA rank:\n- Rank r: Dimension of low-rank matrices\n- Parameters: 2 × r × d_model per layer\n- Typical r: 4-16\n- Most common: r = 8\n\nLower r = fewer params, less capacity\nHigher r = more params, more capacity",
            "diagram": """graph LR
    A[Weight Matrix<br/>d × d] --> B[LoRA Decomposition]
    B --> C[Low-Rank Matrices<br/>d × r and r × d]
    C --> D[Rank r Controls Size]""",
            "code_example": """# LoRA Rank (r)
from peft import LoraConfig, get_peft_model

# Rank r controls adapter size
# r=4: Small adapter, fewer parameters
# r=8: Medium adapter (most common)
# r=16: Large adapter, more parameters

lora_config = LoraConfig(
    r=8,  # Rank - controls adapter size
    lora_alpha=16,  # Typically 2× rank
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

model = get_peft_model(base_model, lora_config)

# Parameters added: 2 × r × d_model per target module
# r=8, d_model=768: ~12K params per module""",
            "links": [
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models (Original)", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 QLoRA: Efficient Finetuning of Quantized LLMs", "url": "https://arxiv.org/abs/2305.14314", "type": "paper"},
                {"title": "📚 PEFT Library Documentation", "url": "https://huggingface.co/docs/peft", "type": "tutorial"}
            ]
        },
        "Alpha": {
            "definition": "Scaling factor in LoRA. Controls adapter contribution. Typically set to 2× rank. Higher alpha = stronger adapter influence. Lower alpha = weaker adapter influence. Formula: output = base + (alpha/r) × adapter.",
            "formula": "LoRA Alpha:\n- Scaling factor for adapter\n- Typically: alpha = 2 × r\n- Formula: output = base + (alpha/r) × adapter_output\n\nHigher alpha = stronger adapter\nLower alpha = weaker adapter",
            "diagram": """graph LR
    A[Base Model Output] --> C[Final Output]
    B[Adapter Output] --> D[Scale by alpha/r]
    D --> C""",
            "code_example": """# LoRA Alpha (scaling factor)
from peft import LoraConfig

# Alpha controls adapter strength
# Typical: alpha = 2 × rank
# r=8 → alpha=16

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,  # Scaling factor (typically 2× rank)
    target_modules=["q_proj", "v_proj"]
)

# LoRA formula:
# W' = W + (alpha/r) × ΔW
# where ΔW = B × A (low-rank decomposition)

# Higher alpha: Stronger adapter influence
# Lower alpha: Weaker adapter influence
# alpha/r ratio determines scaling""",
            "links": [
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models (Original)", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📚 PEFT Library Documentation", "url": "https://huggingface.co/docs/peft", "type": "tutorial"}
            ]
        },
        "Target Modules": {
            "definition": "Which model modules to apply LoRA adapters to. Typically attention layers: Q, K, V, O projections. Can also target FFN layers. More modules = more parameters but potentially better performance.",
            "formula": "Target Modules:\n- Attention: q_proj, k_proj, v_proj, o_proj\n- FFN: gate_proj, up_proj, down_proj\n- All: All linear layers\n\nMore modules = more parameters\nFewer modules = fewer parameters",
            "diagram": """graph TD
    A[Model Layers] --> B{Target Modules?}
    B -->|Attention Only| C[Q, K, V, O]
    B -->|Attention + FFN| D[Q, K, V, O, Gate, Up, Down]
    B -->|All| E[All Linear Layers]
    C --> F[LoRA Adapters]
    D --> F
    E --> F""",
            "code_example": """# LoRA Target Modules
from peft import LoraConfig

# Option 1: Attention layers only (most common)
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention only
)

# Option 2: Attention + FFN
lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"  # FFN
    ]
)

# Option 3: All linear layers
lora_config = LoraConfig(
    r=8,
    target_modules="all-linear"  # All linear layers
)

# More modules = more parameters but potentially better performance""",
            "links": [
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models (Original)", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 QLoRA: Efficient Finetuning of Quantized LLMs", "url": "https://arxiv.org/abs/2305.14314", "type": "paper"},
                {"title": "📚 PEFT Library Documentation", "url": "https://huggingface.co/docs/peft", "type": "tutorial"}
            ]
        },
        "Learning Rate": {
            "definition": "Step size for parameter updates during fine-tuning. Too high = unstable training. Too low = slow convergence. Typical ranges: 1e-5 to 5e-4. LoRA typically uses higher LR than full fine-tuning.",
            "formula": "Learning Rate:\n- Typical: 1e-5 to 5e-4\n- LoRA: 1e-4 to 5e-4 (higher)\n- Full FT: 1e-5 to 1e-4 (lower)\n\nUpdate: θ = θ - lr × ∇θ L(θ)",
            "diagram": """graph LR
    A[High LR] --> B[Fast but Unstable]
    C[Optimal LR] --> D[Good Convergence]
    E[Low LR] --> F[Stable but Slow]""",
            "code_example": """# Learning Rate for Fine-Tuning
from transformers import TrainingArguments

# LoRA: Higher learning rate (1e-4 to 5e-4)
training_args = TrainingArguments(
    learning_rate=2e-4,  # Typical for LoRA
    per_device_train_batch_size=4,
    num_train_epochs=3
)

# Full Fine-Tuning: Lower learning rate (1e-5 to 1e-4)
training_args = TrainingArguments(
    learning_rate=5e-5,  # Typical for full FT
    per_device_train_batch_size=2,
    num_train_epochs=3
)

# Learning rate schedule
training_args = TrainingArguments(
    learning_rate=2e-4,
    lr_scheduler_type="cosine",  # Cosine decay
    warmup_steps=100,  # Warmup phase
    num_train_epochs=3
)""",
            "links": [
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📚 HuggingFace Training Guide", "url": "https://huggingface.co/docs/transformers/training", "type": "tutorial"}
            ]
        },
        "Warmup": {
            "definition": "Gradually increase learning rate at start of training. Prevents early instability. Typical: 3-10% of total steps. Helps model adapt gradually. Common schedules: linear, cosine.",
            "formula": "Warmup:\n- Steps: 3-10% of total training steps\n- Schedule: Linear or cosine\n- LR increases from 0 to target LR\n\nExample: 1000 steps, 10% warmup = 100 warmup steps",
            "diagram": """graph LR
    A[LR = 0] --> B[Warmup Phase<br/>Gradually Increase]
    B --> C[LR = Target]
    C --> D[Training Phase<br/>Constant/Decay]""",
            "code_example": """# Learning Rate Warmup
from transformers import TrainingArguments

# Warmup: 10% of total steps
training_args = TrainingArguments(
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_steps=100,  # 10% of 1000 steps
    lr_scheduler_type="linear"  # Linear warmup
)

# Warmup schedule:
# Step 0-100: LR increases linearly from 0 to 2e-4
# Step 100+: LR = 2e-4 (or decays)

# Cosine warmup
training_args = TrainingArguments(
    learning_rate=2e-4,
    warmup_steps=100,
    lr_scheduler_type="cosine"  # Cosine warmup + decay
)""",
            "links": [
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762", "type": "paper"},
                {"title": "📚 HuggingFace Training Guide", "url": "https://huggingface.co/docs/transformers/training", "type": "tutorial"}
            ]
        },
        "Batch Size": {
            "definition": "Number of examples processed per training step. Larger batch = more stable gradients but more memory. Smaller batch = less memory but noisier gradients. Typical for LoRA: 1-8. Can use gradient accumulation.",
            "formula": "Batch Size:\n- LoRA: 1-8 (limited by adapter size)\n- Full FT: 1-4 (limited by model size)\n- Effective batch = batch_size × gradient_accumulation\n\nMemory: O(batch_size × seq_len × hidden_size)",
            "diagram": """graph LR
    A[Small Batch<br/>1-2] --> B[Less Memory<br/>Noisier Gradients]
    C[Medium Batch<br/>4-8] --> D[Balanced]
    E[Large Batch<br/>16+] --> F[More Memory<br/>Stable Gradients]""",
            "code_example": """# Batch Size for Fine-Tuning
from transformers import TrainingArguments

# LoRA: Can use larger batch (1-8)
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # Batch size per GPU
    gradient_accumulation_steps=2,  # Effective batch = 4 × 2 = 8
    num_train_epochs=3
)

# Full Fine-Tuning: Smaller batch (1-4)
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Smaller due to memory
    gradient_accumulation_steps=4,  # Effective batch = 2 × 4 = 8
    num_train_epochs=3
)

# Gradient Accumulation simulates larger batch:
# Accumulate gradients over N steps before updating
# Effective batch = batch_size × gradient_accumulation_steps""",
            "links": [
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 QLoRA: Efficient Finetuning of Quantized LLMs", "url": "https://arxiv.org/abs/2305.14314", "type": "paper"},
                {"title": "📚 HuggingFace Training Guide", "url": "https://huggingface.co/docs/transformers/training", "type": "tutorial"}
            ]
        },
        "Epochs": {
            "definition": "Number of complete passes through training dataset. Typically 1-5 epochs sufficient for fine-tuning. More epochs = better fit but risk overfitting. Early stopping can prevent overfitting.",
            "formula": "Epochs:\n- Fine-tuning: 1-5 epochs typically sufficient\n- Pre-training: Hundreds of epochs\n- Early stopping: Stop when validation loss stops improving\n\nMore epochs = more training but risk overfitting",
            "diagram": """graph LR
    A[Epoch 1] --> B[Epoch 2]
    B --> C[Epoch 3]
    C --> D{Converged?}
    D -->|Yes| E[Stop]
    D -->|No| F[Continue]""",
            "code_example": """# Training Epochs
from transformers import TrainingArguments, EarlyStoppingCallback

# Typical: 1-5 epochs for fine-tuning
training_args = TrainingArguments(
    num_train_epochs=3,  # 3 epochs typically sufficient
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",  # Evaluate each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,  # Load best checkpoint
    metric_for_best_model="eval_loss"
)

# Early stopping prevents overfitting
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stop if no improvement for 2 epochs
)

trainer.train()""",
            "links": [
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📚 HuggingFace Training Guide", "url": "https://huggingface.co/docs/transformers/training", "type": "tutorial"}
            ]
        },
        "Early Stopping": {
            "definition": "Stop training when validation performance stops improving. Prevents overfitting. Monitors validation loss/metric. Stops after N epochs without improvement. Saves best model checkpoint.",
            "formula": "Early Stopping:\n- Monitor: validation loss/metric\n- Patience: N epochs without improvement\n- Stop: When no improvement for patience epochs\n- Save: Best model checkpoint\n\nPrevents overfitting",
            "diagram": """graph TD
    A[Training] --> B[Evaluate Validation]
    B --> C{Improved?}
    C -->|Yes| D[Save Best Model]
    C -->|No| E[Increment Counter]
    D --> F[Continue Training]
    E --> G{Counter > Patience?}
    G -->|Yes| H[Stop Training]
    G -->|No| F""",
            "code_example": """# Early Stopping
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

training_args = TrainingArguments(
    num_train_epochs=10,  # Max epochs
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3  # Stop after 3 epochs without improvement
        )
    ]
)

trainer.train()
# Training stops early if validation loss doesn't improve
# Best model checkpoint is loaded automatically""",
            "links": [
                {"title": "📄 Early Stopping - But When?", "url": "https://link.springer.com/article/10.1023/A:1022487305947", "type": "paper"},
                {"title": "📚 HuggingFace Training Guide", "url": "https://huggingface.co/docs/transformers/training", "type": "tutorial"}
            ]
        },
        "Instruction Tuning": {
            "definition": "Fine-tune model to follow instructions. Train on instruction-response pairs. Enables zero-shot task performance. Makes model more helpful and aligned. Examples: FLAN, Alpaca, InstructGPT.",
            "formula": "Instruction Tuning:\n- Format: Instruction + Response pairs\n- Dataset: Human-written or synthetic\n- Goal: Follow instructions, be helpful\n- Enables: Zero-shot task performance\n\nExamples: \"Translate to French: Hello\" → \"Bonjour\"",
            "diagram": """graph LR
    A[Base Model] --> B[Instruction Dataset]
    B --> C[Fine-Tune]
    C --> D[Instruction-Following<br/>Model]""",
            "code_example": """# Instruction Tuning
from transformers import Trainer, TrainingArguments

# Instruction dataset format
instruction_dataset = [
    {
        "instruction": "Translate to French",
        "input": "Hello",
        "output": "Bonjour"
    },
    {
        "instruction": "Summarize",
        "input": "Long article...",
        "output": "Summary..."
    }
]

# Fine-tune on instructions
training_args = TrainingArguments(
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=instruction_dataset
)

trainer.train()
# Model learns to follow instructions
# Enables zero-shot task performance""",
            "links": [
                {"title": "📄 FLAN: Scaling Instruction-Finetuned Language Models", "url": "https://arxiv.org/abs/2109.01652", "type": "paper"},
                {"title": "📄 InstructGPT: Training Language Models to Follow Instructions", "url": "https://arxiv.org/abs/2203.02155", "type": "paper"},
                {"title": "📄 Alpaca: A Strong Open-Source Instruction-Following Model", "url": "https://crfm.stanford.edu/2023/03/13/alpaca.html", "type": "paper"}
            ]
        },
        "Domain Adaptation": {
            "definition": "Adapt model to specific domain (medical, legal, code, etc.). Fine-tune on domain-specific data. Improves performance in target domain. Can use LoRA for efficient adaptation. Balances domain knowledge with general knowledge.",
            "formula": "Domain Adaptation:\n- Target: Specific domain (medical, legal, etc.)\n- Method: Fine-tune on domain data\n- Balance: Domain knowledge vs general knowledge\n- Efficiency: Use LoRA for parameter-efficient adaptation",
            "diagram": """graph LR
    A[General Model] --> B[Domain Data]
    B --> C[Fine-Tune]
    C --> D[Domain-Adapted<br/>Model]""",
            "code_example": """# Domain Adaptation
from peft import LoraConfig, get_peft_model
from transformers import Trainer

# Load domain-specific dataset
medical_dataset = load_medical_corpus()

# Use LoRA for efficient adaptation
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(base_model, lora_config)

# Fine-tune on domain data
trainer = Trainer(
    model=model,
    train_dataset=medical_dataset,
    args=training_args
)

trainer.train()
# Model adapted to medical domain
# Still retains general knowledge""",
            "links": [
                {"title": "📄 Domain-Specific Language Model Pretraining", "url": "https://arxiv.org/abs/2002.05645", "type": "paper"},
                {"title": "📄 LoRA: Low-Rank Adaptation of Large Language Models", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📄 BioBERT: Pre-trained Biomedical Language Model", "url": "https://arxiv.org/abs/1901.08746", "type": "paper"}
            ]
        },
        "Catastrophic Forgetting": {
            "definition": "Model forgets previously learned knowledge when fine-tuned on new data. Problem in continual learning. Mitigated by: data mixing, lower learning rate, regularization. Important to preserve general knowledge.",
            "formula": "Catastrophic Forgetting:\n- Problem: Model forgets old knowledge\n- Cause: Fine-tuning on new data\n- Mitigation:\n  - Data mixing: Mix old + new data\n  - Lower LR: Preserve existing weights\n  - Regularization: Constrain weight changes",
            "diagram": """graph TD
    A[Pre-trained Knowledge] --> B[Fine-Tune on New Data]
    B --> C{Forgetting?}
    C -->|Yes| D[Lost Old Knowledge]
    C -->|No| E[Retained Knowledge]""",
            "code_example": """# Preventing Catastrophic Forgetting

# Method 1: Data Mixing
# Mix pre-training data with fine-tuning data
mixed_dataset = concatenate_datasets([
    pretrain_data.sample(frac=0.1),  # 10% of pre-training data
    finetune_data  # 100% of fine-tuning data
])

# Method 2: Lower Learning Rate
training_args = TrainingArguments(
    learning_rate=1e-5,  # Lower LR preserves existing weights
    num_train_epochs=3
)

# Method 3: Regularization (Elastic Weight Consolidation)
# Constrain weight changes to important weights
# Preserves important connections

# Method 4: LoRA (Parameter-Efficient)
# Only updates small adapter, preserves base model
lora_config = LoraConfig(r=8)
model = get_peft_model(base_model, lora_config)""",
            "links": [
                {"title": "📄 Catastrophic Forgetting in Neural Networks", "url": "https://arxiv.org/abs/1312.6211", "type": "paper"},
                {"title": "📄 Elastic Weight Consolidation", "url": "https://arxiv.org/abs/1612.00796", "type": "paper"},
                {"title": "📄 LoRA: Low-Rank Adaptation (Prevents Forgetting)", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"}
            ]
        },
        "Data Mixing": {
            "definition": "Mix pre-training data with fine-tuning data during training. Prevents catastrophic forgetting. Typical ratio: 10-20% pre-training data, 80-90% fine-tuning data. Helps preserve general knowledge.",
            "formula": "Data Mixing:\n- Pre-train data: 10-20%\n- Fine-tune data: 80-90%\n- Mix during training\n- Prevents forgetting general knowledge\n\nRatio depends on task and domain shift",
            "diagram": """graph LR
    A[Pre-train Data<br/>10-20%] --> C[Mixed Dataset]
    B[Fine-tune Data<br/>80-90%] --> C
    C --> D[Training]""",
            "code_example": """# Data Mixing
from datasets import concatenate_datasets

# Load datasets
pretrain_data = load_pretrain_corpus()
finetune_data = load_finetune_corpus()

# Mix: 10% pre-train + 100% fine-tune
mixed_pretrain = pretrain_data.shuffle().select(range(len(pretrain_data) // 10))
mixed_dataset = concatenate_datasets([mixed_pretrain, finetune_data])

# Shuffle mixed dataset
mixed_dataset = mixed_dataset.shuffle(seed=42)

# Train on mixed dataset
trainer = Trainer(
    model=model,
    train_dataset=mixed_dataset,
    args=training_args
)

# Model learns new task while preserving general knowledge""",
            "links": [
                {"title": "📄 BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/abs/1810.04805", "type": "paper"},
                {"title": "📄 LoRA: Low-Rank Adaptation", "url": "https://arxiv.org/abs/2106.09685", "type": "paper"},
                {"title": "📚 HuggingFace Datasets Guide", "url": "https://huggingface.co/docs/datasets", "type": "tutorial"}
            ]
        }
    },
    "Production Monitoring & Reliability": {
        "SLO": {
            "definition": "Service Level Objective. Target for reliability (e.g., 99.9% uptime). Business requirement. Defines acceptable performance. Used with SLI to measure.",
            "formula": "SLO Examples:\n- Availability: 99.9% uptime = 8.76 hours downtime/year\n- Latency: P95 < 200ms\n- Error Rate: < 0.1%\n\nError Budget = 100% - SLO",
            "diagram": """graph LR
    A[Business Requirements] --> B[SLO Definition]
    B --> C[SLI Measurement]
    C --> D{SLI meets SLO?}
    D -->|Yes| E[Within Budget]
    D -->|No| F[Alert<br/>Action Required]""",
            "code_example": """# SLO Monitoring
slo = {
    "availability": 0.999,  # 99.9%
    "latency_p95": 200,  # ms
    "error_rate": 0.001  # 0.1%
}

# Monitor SLI
sli = {
    "availability": actual_uptime,
    "latency_p95": measure_p95_latency(),
    "error_rate": failed_requests / total_requests
}

# Check if SLO met
if sli["availability"] < slo["availability"]:
    alert("SLO violation: availability")""",
            "links": [
                {"title": "📄 Site Reliability Engineering (Google SRE Book)", "url": "https://sre.google/books/", "type": "paper"},
                {"title": "📄 The Art of SLOs", "url": "https://sre.google/workbook/slo-document/", "type": "paper"},
                {"title": "📄 Error Budgets in Practice", "url": "https://sre.google/workbook/error-budget-policy/", "type": "paper"}
            ]
        },
        "Latency (P50, P95, P99)": {
            "definition": "Response time percentiles. P50 (median): 50% of requests faster. P95: 95% of requests faster. P99: 99% of requests faster. P99 captures tail latency. Critical for user experience.",
            "formula": "Latency Percentiles:\n- P50: Median latency (50th percentile)\n- P95: 95% of requests faster\n- P99: 99% of requests faster\n- P99.9: 99.9% of requests faster\n\nP99 captures worst-case tail latency",
            "diagram": """graph LR
    A[Request Latencies] --> B[Sort]
    B --> C[P50: Median]
    B --> D[P95: 95th Percentile]
    B --> E[P99: 99th Percentile]
    C --> F[User Experience]""",
            "code_example": """# Latency Percentiles
import numpy as np

# Latency measurements (ms)
latencies = [50, 55, 52, 48, 60, 45, 58, 53, 49, 120, 150, 200]

# Calculate percentiles
p50 = np.percentile(latencies, 50)  # Median
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)
p999 = np.percentile(latencies, 99.9)

print(f"P50: {p50:.1f}ms")   # 50% of requests faster
print(f"P95: {p95:.1f}ms")   # 95% of requests faster
print(f"P99: {p99:.1f}ms")   # 99% of requests faster
print(f"P99.9: {p999:.1f}ms")  # 99.9% of requests faster

# P99 captures tail latency (worst cases)
# Important for user experience
# SLO example: P99 < 200ms""",
            "links": [
                {"title": "📄 Site Reliability Engineering (Google SRE Book)", "url": "https://sre.google/books/", "type": "paper"},
                {"title": "📄 The Tail at Scale", "url": "https://cacm.acm.org/magazines/2013/2/160173-the-tail-at-scale/abstract", "type": "paper"}
            ]
        },
        "Throughput": {
            "definition": "Requests processed per unit time. Tokens/second or requests/second. Measures system capacity. Higher is better. Affected by batch size, model size, hardware.",
            "formula": "Throughput:\n- Requests/second: Total requests / time\n- Tokens/second: Total tokens / time\n- Higher is better\n- Affected by: Batch size, model size, hardware\n\nThroughput = Batch size × Tokens per request / Latency",
            "diagram": """graph LR
    A[Requests] --> B[Process]
    B --> C[Throughput<br/>req/s or tokens/s]
    D[Batch Size] --> B
    E[Hardware] --> B""",
            "code_example": """# Throughput Measurement
import time

# Measure tokens per second
start_time = time.time()
tokens_generated = 0
num_requests = 100

for _ in range(num_requests):
    tokens = generate_tokens(model, prompt)
    tokens_generated += len(tokens)

elapsed_time = time.time() - start_time

throughput_tokens_per_sec = tokens_generated / elapsed_time
throughput_requests_per_sec = num_requests / elapsed_time

print(f"Throughput: {throughput_tokens_per_sec:.1f} tokens/sec")
print(f"Throughput: {throughput_requests_per_sec:.1f} requests/sec")

# Factors affecting throughput:
# - Batch size: Larger batches = higher throughput
# - Model size: Smaller models = higher throughput
# - Hardware: Better GPUs = higher throughput
# - Optimization: TensorRT, quantization = higher throughput""",
            "links": [
                {"title": "📄 vLLM: Easy, Fast, and Cheap LLM Serving", "url": "https://arxiv.org/abs/2309.06180", "type": "paper"},
                {"title": "📄 Efficient Memory Management for Large Language Models", "url": "https://arxiv.org/abs/2309.06180", "type": "paper"}
            ]
        },
        "Error Rate": {
            "definition": "Percentage of failed requests. Failed requests / Total requests. Lower is better. Target: < 1%. Critical metric for reliability. Part of SLO.",
            "formula": "Error Rate = Failed Requests / Total Requests × 100%\n\nTarget: < 1%\nSLO example: Error rate < 0.1%\n\nErrors: 500s, timeouts, invalid responses",
            "diagram": """graph LR
    A[Total Requests] --> B{Success?}
    B -->|Yes| C[Success]
    B -->|No| D[Error]
    C --> E[Error Rate]
    D --> E""",
            "code_example": """# Error Rate Monitoring
total_requests = 10000
failed_requests = 25

error_rate = (failed_requests / total_requests) * 100
print(f"Error Rate: {error_rate:.2f}%")

# Error types
error_types = {
    "500": 10,  # Server errors
    "timeout": 8,  # Timeouts
    "invalid": 5,  # Invalid responses
    "other": 2
}

# Monitor error rate over time
# Alert if error rate > threshold (e.g., 1%)
# Part of SLO monitoring

# SLO example: Error rate < 0.1%
if error_rate > 0.1:
    alert("SLO violation: Error rate too high")""",
            "links": [
                {"title": "📄 Site Reliability Engineering (Google SRE Book)", "url": "https://sre.google/books/", "type": "paper"},
                {"title": "📄 The Art of SLOs", "url": "https://sre.google/workbook/slo-document/", "type": "paper"}
            ]
        },
        "Data Drift": {
            "definition": "Input data distribution changes over time. Model trained on old distribution. Performance degrades. Detect by comparing input distributions. Retrain or adapt model.",
            "formula": "Data Drift:\n- Input distribution P(X) changes\n- Model trained on P_old(X)\n- Production sees P_new(X)\n- Performance degrades\n\nDetection: Compare distributions (KS test, MMD)",
            "diagram": """graph TD
    A[Training Data<br/>P_old X] --> B[Model]
    C[Production Data<br/>P_new X] --> D{Drift?}
    D -->|Yes| E[Performance<br/>Degrades]
    D -->|No| F[Stable]""",
            "code_example": """# Data Drift Detection
from scipy import stats
import numpy as np

# Training data distribution
train_data = np.random.normal(0, 1, 1000)

# Production data (may have drifted)
prod_data = np.random.normal(0.5, 1.2, 1000)  # Shifted distribution

# Kolmogorov-Smirnov test
ks_stat, p_value = stats.ks_2samp(train_data, prod_data)
print(f"KS statistic: {ks_stat:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < 0.05:
    print("Data drift detected!")
    # Actions:
    # 1. Retrain model on new data
    # 2. Adapt model (fine-tuning)
    # 3. Monitor performance closely

# Monitor continuously
# Alert when drift detected
# Retrain periodically""",
            "links": [
                {"title": "📄 Detecting Dataset Shift", "url": "https://arxiv.org/abs/1807.04153", "type": "paper"},
                {"title": "📄 Monitoring ML Models in Production", "url": "https://arxiv.org/abs/2007.06299", "type": "paper"}
            ]
        },
        "Concept Drift": {
            "definition": "Relationship between input and output changes. P(Y|X) changes over time. Model predictions become less accurate. More subtle than data drift. Requires monitoring model performance.",
            "formula": "Concept Drift:\n- Relationship P(Y|X) changes\n- Input distribution may be same\n- Model performance degrades\n- Detection: Monitor accuracy/metrics\n\nMore subtle than data drift",
            "diagram": """graph TD
    A[Old Relationship<br/>P_old Y|X] --> B[Model]
    C[New Relationship<br/>P_new Y|X] --> D{Drift?}
    D -->|Yes| E[Accuracy<br/>Degrades]
    D -->|No| F[Stable]""",
            "code_example": """# Concept Drift Detection
# Monitor model performance over time

# Baseline performance (training)
baseline_accuracy = 0.90

# Production performance over time
production_accuracy = [
    0.89, 0.88, 0.87, 0.85, 0.83, 0.81  # Declining
]

# Detect concept drift
current_accuracy = production_accuracy[-1]
accuracy_drop = baseline_accuracy - current_accuracy

if accuracy_drop > 0.05:  # 5% drop
    print("Concept drift detected!")
    print(f"Accuracy dropped by {accuracy_drop:.2%}")
    
    # Actions:
    # 1. Investigate cause
    # 2. Collect new labeled data
    # 3. Retrain model
    # 4. Fine-tune on recent data

# Continuous monitoring
# Alert on performance degradation
# Periodic retraining""",
            "links": [
                {"title": "📄 Learning under Concept Drift", "url": "https://arxiv.org/abs/2007.05341", "type": "paper"},
                {"title": "📄 Monitoring ML Models in Production", "url": "https://arxiv.org/abs/2007.06299", "type": "paper"}
            ]
        },
        "Model Drift": {
            "definition": "Model performance degrades over time. Accuracy/metrics decrease. Can be due to data drift or concept drift. Requires monitoring and retraining. Part of ML operations.",
            "formula": "Model Drift:\n- Performance degrades over time\n- Accuracy/metrics decrease\n- Causes: Data drift, concept drift\n- Detection: Monitor metrics\n- Solution: Retrain/adapt model",
            "diagram": """graph LR
    A[Time] --> B[Performance]
    B --> C[Baseline: 90%]
    B --> D[Current: 85%]
    D --> E{Drift?}
    E -->|Yes| F[Retrain]""",
            "code_example": """# Model Drift Monitoring
import pandas as pd

# Performance over time
performance_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=30, freq='D'),
    'accuracy': [0.90] * 10 + [0.89] * 5 + [0.87] * 5 + [0.85] * 10
})

baseline_accuracy = 0.90
current_accuracy = performance_data['accuracy'].iloc[-1]

# Detect drift
if current_accuracy < baseline_accuracy - 0.03:  # 3% threshold
    print("Model drift detected!")
    print(f"Accuracy dropped from {baseline_accuracy:.2%} to {current_accuracy:.2%}")
    
    # Alert and retrain
    alert("Model drift: Retrain required")
    retrain_model()

# Continuous monitoring
# Set up alerts
# Schedule periodic retraining
# Track performance trends""",
            "links": [
                {"title": "📄 Monitoring ML Models in Production", "url": "https://arxiv.org/abs/2007.06299", "type": "paper"},
                {"title": "📄 ML Operations (MLOps)", "url": "https://arxiv.org/abs/2005.13225", "type": "paper"}
            ]
        }
    },
    "Safety, Ethics & Compliance": {
        "PII Detection": {
            "definition": "Identify Personally Identifiable Information (names, SSN, email, phone, etc.). Critical for privacy. Use NER models or rule-based systems. Required for compliance.",
            "formula": "PII Types:\n- Direct: SSN, credit card, email\n- Indirect: Name + ZIP code\n- Quasi: IP address, device ID\n\nDetection: NER models or regex patterns",
            "diagram": """graph LR
    A[Input Text] --> B[PII Detection]
    B --> C[NER Model]
    B --> D[Regex Patterns]
    C --> E[PII Entities]
    D --> E
    E --> F[Redaction or<br/>Masking]""",
            "code_example": """# PII Detection
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Detect PII
text = "Contact John Doe at john@example.com or 555-1234"
results = analyzer.analyze(text=text, language='en')

# Anonymize
anonymized = anonymizer.anonymize(
    text=text,
    analyzer_results=results
)
# Output: "Contact <PERSON> at <EMAIL> or <PHONE_NUMBER>"
""",
            "links": [
                {"title": "📄 Presidio: Context-Aware PII Detection", "url": "https://arxiv.org/abs/2003.07911", "type": "paper"},
                {"title": "📄 Privacy-Preserving Machine Learning", "url": "https://arxiv.org/abs/1811.04017", "type": "paper"},
                {"title": "📄 GDPR Compliance in ML Systems", "url": "https://arxiv.org/abs/1805.08210", "type": "paper"}
            ]
        },
        "Pre-Prompt Filters": {
            "definition": "Check input before generation. Detect harmful content in user prompts. Block or modify before processing. Prevents generation of harmful content. First line of defense.",
            "formula": "Pre-Prompt Filters:\n- Check input before generation\n- Detect: Violence, hate speech, PII, etc.\n- Action: Block, modify, or flag\n- Prevents harmful generation\n\nFirst line of defense",
            "diagram": """graph TD
    A[User Input] --> B{Pre-Prompt<br/>Filter}
    B -->|Safe| C[Generate]
    B -->|Unsafe| D[Block/Modify]
    C --> E[Model Generation]""",
            "code_example": """# Pre-Prompt Filtering
from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()

def pre_prompt_filter(user_input):
    # Check for PII
    pii_results = analyzer.analyze(text=user_input, language='en')
    if pii_results:
        return False, "Input contains PII"
    
    # Check for harmful content
    harmful_keywords = ["violence", "hate", "illegal"]
    if any(keyword in user_input.lower() for keyword in harmful_keywords):
        return False, "Input violates content policy"
    
    # Check for prompt injection
    if "ignore previous instructions" in user_input.lower():
        return False, "Potential prompt injection detected"
    
    return True, None

# Usage
is_safe, error = pre_prompt_filter(user_input)
if not is_safe:
    return {"error": error}

# Safe to generate
response = model.generate(user_input)""",
            "links": [
                {"title": "📄 GPT-4 System Card (Safety)", "url": "https://cdn.openai.com/papers/gpt-4-system-card.pdf", "type": "paper"},
                {"title": "📄 RealToxicityPrompts: Evaluating Neural Toxic Degeneration", "url": "https://arxiv.org/abs/2009.11462", "type": "paper"},
                {"title": "📄 Prompt Injection Attacks", "url": "https://arxiv.org/abs/2302.12173", "type": "paper"}
            ]
        },
        "Post-Generation Filters": {
            "definition": "Check output after generation. Detect harmful content in model responses. Filter or block unsafe outputs. Second line of defense. Catches issues pre-prompt filters miss.",
            "formula": "Post-Generation Filters:\n- Check output after generation\n- Detect: Harmful content, PII leakage, etc.\n- Action: Filter, block, or regenerate\n- Second line of defense\n\nCatches issues pre-filters miss",
            "diagram": """graph TD
    A[Model Output] --> B{Post-Generation<br/>Filter}
    B -->|Safe| C[Return to User]
    B -->|Unsafe| D[Filter/Block/Regenerate]
    D --> E[Safe Output]""",
            "code_example": """# Post-Generation Filtering
from presidio_analyzer import AnalyzerEngine
from transformers import pipeline

classifier = pipeline("text-classification", model="unitary/toxic-bert")
analyzer = AnalyzerEngine()

def post_generation_filter(output):
    # Check for toxicity
    toxicity_result = classifier(output)[0]
    if toxicity_result['label'] == 'toxic' and toxicity_result['score'] > 0.7:
        return None, "Generated content is toxic"
    
    # Check for PII leakage
    pii_results = analyzer.analyze(text=output, language='en')
    if pii_results:
        # Redact PII
        from presidio_anonymizer import AnonymizerEngine
        anonymizer = AnonymizerEngine()
        anonymized = anonymizer.anonymize(text=output, analyzer_results=pii_results)
        return anonymized.text, None
    
    # Check for harmful content
    harmful_patterns = ["violence", "illegal", "harmful"]
    if any(pattern in output.lower() for pattern in harmful_patterns):
        return None, "Generated content violates policy"
    
    return output, None

# Usage
output = model.generate(user_input)
filtered_output, error = post_generation_filter(output)
if error:
    return {"error": error}
return {"output": filtered_output}""",
            "links": [
                {"title": "📄 GPT-4 System Card (Safety)", "url": "https://cdn.openai.com/papers/gpt-4-system-card.pdf", "type": "paper"},
                {"title": "📄 RealToxicityPrompts: Evaluating Neural Toxic Degeneration", "url": "https://arxiv.org/abs/2009.11462", "type": "paper"},
                {"title": "📄 Constitutional AI: Harmlessness from AI Feedback", "url": "https://arxiv.org/abs/2212.08073", "type": "paper"}
            ]
        },
        "Blocklists": {
            "definition": "List of prohibited words/phrases. Block content containing these terms. Simple but effective. Can be bypassed with variations. Used with other methods. Part of content filtering.",
            "formula": "Blocklists:\n- List of prohibited terms\n- Check: Input/output against list\n- Action: Block if match found\n- Limitations: Can be bypassed\n- Use with: Other filtering methods",
            "diagram": """graph LR
    A[Content] --> B{In Blocklist?}
    B -->|Yes| C[Block]
    B -->|No| D[Allow]""",
            "code_example": """# Blocklists
blocklist = [
    "illegal",
    "harmful",
    "violence",
    # ... more terms
]

def check_blocklist(text):
    text_lower = text.lower()
    for term in blocklist:
        if term in text_lower:
            return False, f"Content blocked: contains '{term}'"
    return True, None

# Usage
is_allowed, error = check_blocklist(user_input)
if not is_allowed:
    return {"error": error}

# Limitations:
# - Can be bypassed with variations ("illegal" → "il1egal")
# - May block legitimate content
# - Use with ML-based filters for better results""",
            "links": [
                {"title": "📄 Content Moderation in AI Systems", "url": "https://arxiv.org/abs/2009.11462", "type": "paper"},
                {"title": "📄 GPT-4 System Card (Safety)", "url": "https://cdn.openai.com/papers/gpt-4-system-card.pdf", "type": "paper"}
            ]
        },
        "Allowlists": {
            "definition": "List of allowed words/phrases. Only allow content matching these terms. Very restrictive. Use for specific domains. Prevents most harmful content. May be too restrictive.",
            "formula": "Allowlists:\n- List of allowed terms\n- Only allow: Content matching list\n- Very restrictive\n- Use for: Specific domains\n- Prevents: Most harmful content",
            "diagram": """graph LR
    A[Content] --> B{In Allowlist?}
    B -->|Yes| C[Allow]
    B -->|No| D[Block]""",
            "code_example": """# Allowlists
allowlist = [
    "hello",
    "help",
    "information",
    "question",
    # ... domain-specific terms
]

def check_allowlist(text):
    words = text.lower().split()
    allowed_words = [w for w in words if w in allowlist]
    
    # Require majority of words to be in allowlist
    if len(allowed_words) / len(words) < 0.7:
        return False, "Content not in allowlist"
    
    return True, None

# Usage
is_allowed, error = check_allowlist(user_input)
if not is_allowed:
    return {"error": "Content not allowed"}

# Very restrictive
# Use for specific domains (e.g., customer service)
# Prevents most harmful content
# May block legitimate requests""",
            "links": [
                {"title": "📄 Content Moderation in AI Systems", "url": "https://arxiv.org/abs/2009.11462", "type": "paper"},
                {"title": "📄 GPT-4 System Card (Safety)", "url": "https://cdn.openai.com/papers/gpt-4-system-card.pdf", "type": "paper"}
            ]
        },
        "Bias Detection": {
            "definition": "Identify unfair treatment across groups. Measure performance differences. Detect demographic bias. Use fairness metrics. Important for ethical AI. Ongoing monitoring required.",
            "formula": "Bias Detection:\n- Measure: Performance across groups\n- Metrics: Demographic parity, equalized odds\n- Groups: Gender, race, age, etc.\n- Goal: Equal performance\n\nFairness: Equal treatment across groups",
            "diagram": """graph TD
    A[Model Predictions] --> B[Group by<br/>Demographics]
    B --> C[Measure Performance]
    C --> D{Equal?}
    D -->|No| E[Bias Detected]
    D -->|Yes| F[Fair]""",
            "code_example": """# Bias Detection
import pandas as pd
from sklearn.metrics import accuracy_score

# Predictions with demographics
results = pd.DataFrame({
    'prediction': [1, 0, 1, 0, 1, 0],
    'true_label': [1, 0, 1, 1, 1, 0],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F']
})

# Measure performance by group
male_results = results[results['gender'] == 'M']
female_results = results[results['gender'] == 'F']

male_accuracy = accuracy_score(male_results['true_label'], male_results['prediction'])
female_accuracy = accuracy_score(female_results['true_label'], female_results['prediction'])

print(f"Male accuracy: {male_accuracy:.3f}")
print(f"Female accuracy: {female_accuracy:.3f}")

# Detect bias
if abs(male_accuracy - female_accuracy) > 0.05:  # 5% threshold
    print("Bias detected: Performance differs significantly")

# Fairness metrics:
# - Demographic parity: Equal positive rates
# - Equalized odds: Equal TPR and FPR
# - Equal opportunity: Equal TPR""",
            "links": [
                {"title": "📄 Fairness in Machine Learning", "url": "https://arxiv.org/abs/1908.09635", "type": "paper"},
                {"title": "📄 Gender Bias in Word Embeddings", "url": "https://arxiv.org/abs/1607.06520", "type": "paper"},
                {"title": "📄 Measuring and Mitigating Bias", "url": "https://arxiv.org/abs/1908.09635", "type": "paper"}
            ]
        },
        "Toxicity Mitigation": {
            "definition": "Reduce harmful, toxic outputs. Improve model safety. Use safety tuning, filtering, RLHF. Ongoing effort. Important for production. Multiple approaches needed.",
            "formula": "Toxicity Mitigation:\n- Safety tuning: Fine-tune for safety\n- Filtering: Pre/post-generation filters\n- RLHF: Reinforcement learning from human feedback\n- Monitoring: Track toxicity metrics\n\nGoal: Reduce harmful outputs",
            "diagram": """graph TD
    A[Model] --> B{Toxic Output?}
    B -->|Yes| C[Mitigation]
    C --> D[Safety Tuning]
    C --> E[Filtering]
    C --> F[RLHF]
    B -->|No| G[Safe Output]""",
            "code_example": """# Toxicity Mitigation
from transformers import pipeline

# Toxicity classifier
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")

def mitigate_toxicity(output):
    # Check toxicity
    toxicity_score = toxicity_classifier(output)[0]['score']
    
    if toxicity_score > 0.7:
        # High toxicity - filter or regenerate
        return None, "Output filtered due to toxicity"
    
    # Medium toxicity - warn
    if toxicity_score > 0.5:
        return output, "Warning: Content may be sensitive"
    
    return output, None

# Safety tuning: Fine-tune on safe examples
# RLHF: Train with human feedback on safety
# Filtering: Pre/post-generation filters
# Monitoring: Track toxicity over time

# Multiple approaches needed
# Ongoing effort""",
            "links": [
                {"title": "📄 RealToxicityPrompts: Evaluating Neural Toxic Degeneration", "url": "https://arxiv.org/abs/2009.11462", "type": "paper"},
                {"title": "📄 Constitutional AI: Harmlessness from AI Feedback", "url": "https://arxiv.org/abs/2212.08073", "type": "paper"},
                {"title": "📄 InstructGPT: Training Language Models to Follow Instructions", "url": "https://arxiv.org/abs/2203.02155", "type": "paper"}
            ]
        },
        "GDPR": {
            "definition": "EU General Data Protection Regulation. Right to explanation, data deletion, data portability. Legal requirement for EU users. Must comply. Affects model deployment and data handling.",
            "formula": "GDPR Requirements:\n- Right to explanation: Explain decisions\n- Right to deletion: Delete user data\n- Data portability: Export user data\n- Consent: Explicit consent for data use\n- Privacy by design: Built-in privacy\n\nLegal requirement for EU",
            "diagram": """graph TD
    A[User Data] --> B{GDPR<br/>Compliant?}
    B -->|Yes| C[Process]
    B -->|No| D[Reject/Modify]
    C --> E[Right to Explanation]
    C --> F[Right to Deletion]
    C --> G[Data Portability]""",
            "code_example": """# GDPR Compliance
# Right to explanation
def explain_prediction(user_input, prediction):
    # Provide explanation for model decision
    explanation = {
        "prediction": prediction,
        "reasoning": "Based on input features...",
        "confidence": 0.85,
        "key_factors": ["feature1", "feature2"]
    }
    return explanation

# Right to deletion
def delete_user_data(user_id):
    # Delete all user data
    delete_from_database(user_id)
    delete_from_logs(user_id)
    delete_from_cache(user_id)
    return "User data deleted"

# Data portability
def export_user_data(user_id):
    # Export all user data
    user_data = {
        "inputs": get_user_inputs(user_id),
        "outputs": get_user_outputs(user_id),
        "metadata": get_user_metadata(user_id)
    }
    return json.dumps(user_data)

# Consent management
def check_consent(user_id, purpose):
    consent = get_user_consent(user_id, purpose)
    if not consent:
        return False, "Consent required"
    return True, None

# Privacy by design
# - Minimize data collection
# - Encrypt data
# - Anonymize when possible
# - Regular audits""",
            "links": [
                {"title": "📄 GDPR Compliance in ML Systems", "url": "https://arxiv.org/abs/1805.08210", "type": "paper"},
                {"title": "📄 Explainable AI for GDPR Compliance", "url": "https://arxiv.org/abs/1806.08939", "type": "paper"},
                {"title": "📚 GDPR Official Documentation", "url": "https://gdpr.eu/", "type": "tutorial"}
            ]
        },
        "Auditability": {
            "definition": "Log decisions, maintain records. Required for compliance. Enables debugging. Important for production. Track: inputs, outputs, decisions, metadata. Support investigations.",
            "formula": "Auditability:\n- Log: All decisions and inputs/outputs\n- Store: Metadata, timestamps, user IDs\n- Retain: For compliance period\n- Access: For investigations\n- Privacy: Anonymize sensitive data\n\nRequired for compliance",
            "diagram": """graph TD
    A[Model Decision] --> B[Log Entry]
    B --> C[Input]
    B --> D[Output]
    B --> E[Metadata]
    B --> F[Timestamp]
    C --> G[Audit Trail]
    D --> G
    E --> G
    F --> G""",
            "code_example": """# Auditability
import logging
from datetime import datetime

# Audit logger
audit_logger = logging.getLogger('audit')
audit_logger.setLevel(logging.INFO)

def log_decision(user_id, input_text, output_text, metadata):
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": hash_user_id(user_id),  # Anonymize
        "input_hash": hash(input_text),  # Don't log PII
        "output_hash": hash(output_text),
        "metadata": metadata,
        "model_version": "llama-7b-v1",
        "decision": "approved"  # or "rejected"
    }
    
    audit_logger.info(json.dumps(audit_entry))
    
    # Store in audit database
    store_audit_entry(audit_entry)

# Usage
log_decision(
    user_id="user123",
    input_text=user_input,
    output_text=model_output,
    metadata={"confidence": 0.85, "latency_ms": 120}
)

# Audit trail enables:
# - Compliance verification
# - Debugging issues
# - Performance analysis
# - Security investigations

# Retention: Keep for compliance period (e.g., 7 years)
# Privacy: Anonymize/hash sensitive data
# Access: Controlled access for investigations""",
            "links": [
                {"title": "📄 Auditability in ML Systems", "url": "https://arxiv.org/abs/2007.06299", "type": "paper"},
                {"title": "📄 GDPR Compliance in ML Systems", "url": "https://arxiv.org/abs/1805.08210", "type": "paper"}
            ]
        }
    }
}

# Helper function to get content (supports both old string format and new dict format)
def get_flashcard_content(domain, topic):
    """Get flashcard content, handling both old and new formats"""
    content = FLASHCARD_CONTENT_ENHANCED.get(domain, {}).get(topic)
    if content is None:
        return None
    if isinstance(content, str):
        # Old format - convert to new format
        return {"definition": content}
    return content
