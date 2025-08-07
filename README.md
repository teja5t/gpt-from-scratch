# GPT from Scratch

A PyTorch implementation of GPT (Generative Pre-trained Transformer) built from scratch, following Andrej Karpathy's educational series. This repository contains both educational and production-ready implementations of the transformer architecture.

## Overview

This project implements the GPT architecture, demonstrating how modern language models work under the hood. The implementation includes:

- **Educational implementation** (`gpt_from_scratch.py`) - Step-by-step transformer with detailed comments
- **Production implementation** (`transformer.py`) - Optimized GPT model compatible with GPT-2 weights
- **Training infrastructure** - Complete training pipeline with distributed training support
- **Multiple experiments** - Jupyter notebooks exploring different aspects of the model

## How Transformers Work

### The Big Picture

A transformer is a neural network architecture that processes sequences of tokens (like words or characters) and learns to predict the next token in a sequence. The key innovation is the **attention mechanism**, which allows the model to focus on relevant parts of the input when making predictions.

### Key Components

#### 1. **Token and Positional Embeddings**
```python
# Convert tokens to dense vectors and add position information
tok_emb = self.transformer.wte(idx)  # Token embeddings
pos_emb = self.transformer.wpe(pos)  # Positional embeddings  
x = tok_emb + pos_emb  # Combine both
```

The model first converts discrete tokens into continuous vector representations and adds positional information since transformers don't inherently understand sequence order.

#### 2. **Self-Attention Mechanism**
The core of the transformer is self-attention, which computes how much each token should "attend to" every other token:

```python
# Simplified attention computation
q, k, v = qkv.split(self.n_embd, dim=2)  # Query, Key, Value
att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))  # Attention scores
att = att.masked_fill(mask == 0, float('-inf'))  # Causal masking
att = F.softmax(att, dim=-1)  # Normalize to probabilities
y = att @ v  # Apply attention to values
```

**Multi-Head Attention** runs multiple attention operations in parallel, allowing the model to focus on different types of relationships simultaneously.

#### 3. **Feed-Forward Network (MLP)**
After attention, each token's representation is processed through a simple neural network:

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)  # Expand
        self.GELU = nn.GELU(approximate='tanh')  # Non-linearity
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)  # Contract
```

#### 4. **Residual Connections and Layer Normalization**
Each component uses residual connections (skip connections) and layer normalization for stable training:

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))  # Attention with residual connection
    x = x + self.mlp(self.ln_2(x))   # MLP with residual connection
    return x
```

#### 5. **Causal Masking**
For language modeling, we mask future tokens so the model can only attend to previous tokens:

```python
# Ensure model can't "cheat" by looking at future tokens
self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)))
```

## Implementation Details

### Architecture Overview

The repository contains two main implementations:

#### Educational Implementation (`gpt_from_scratch.py`)
- **MultiHeadAttention**: Explicit implementation showing individual attention heads
- **Clear structure**: Easy to understand step-by-step attention computation
- **Training loop**: Complete training procedure with loss tracking

#### Production Implementation (`transformer.py`) 
- **CausalSelfAttention**: Optimized attention using PyTorch's fused implementation
- **GPT-2 compatibility**: Can load pre-trained GPT-2 weights
- **Distributed training**: Support for multi-GPU training

### Key Classes

#### `CausalSelfAttention`
Implements multi-head self-attention with causal masking:
```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)  # Q, K, V in one
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)     # Output projection
        self.n_head = config.n_head
        self.n_embd = config.n_embd
```

#### `Block` 
A transformer block combining attention and MLP:
```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)      # Pre-attention norm
        self.attn = CausalSelfAttention(config)       # Self-attention
        self.ln_2 = nn.LayerNorm(config.n_embd)      # Pre-MLP norm  
        self.mlp = MLP(config)                        # Feed-forward
```

#### `GPT`
The main model class:
```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),    # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),    # Position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer blocks
            ln_f = nn.LayerNorm(config.n_embd)                       # Final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Output projection
```

### Configuration

The model configuration is defined in `config.py`:
```python
@dataclass
class GPTConfig:
    block_size: int = 1024    # Maximum sequence length
    vocab_size: int = 50257   # Size of vocabulary  
    n_layer: int = 12         # Number of transformer blocks
    n_head: int = 12          # Number of attention heads
    n_embd: int = 768         # Embedding dimension
```

## Usage

### Basic Training
```python
from transformer import GPT
from config import GPTConfig

# Create model
config = GPTConfig()
model = GPT(config)

# Training loop (simplified)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for batch in dataloader:
    logits, loss = model(batch['input'], batch['target'])
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()
```

### Text Generation
```python
# Load trained model
model = GPT.from_pretrained('gpt2')
model.eval()

# Generate text
context = torch.tensor([[1, 2, 3]])  # Starting tokens
generated = model.generate(context, max_new_tokens=100)
```

### Training with the Educational Implementation
```python
# Run the educational version
python gpt_from_scratch.py
```

### Training the Production Model
```python
# Single GPU training
python train_gpt2.py

# Multi-GPU distributed training  
torchrun --nproc_per_node=4 train_gpt2.py
```

## Files Description

- **`transformer.py`** - Main GPT implementation with modern optimizations
- **`gpt_from_scratch.py`** - Educational step-by-step implementation
- **`train_gpt2.py`** - Training script with distributed training support
- **`config.py`** - Model configuration dataclass
- **`hellaswag.py`** - Evaluation on HellaSwag benchmark
- **`fineweb.py`** - Data processing utilities
- **Jupyter notebooks** - Various experiments and explorations

## Key Features

✅ **From-scratch implementation** - No high-level libraries, built with PyTorch primitives  
✅ **Educational focus** - Clear, commented code showing how transformers work  
✅ **Production ready** - Optimized implementation with GPT-2 compatibility  
✅ **Distributed training** - Multi-GPU training support  
✅ **Evaluation tools** - HellaSwag evaluation for model quality assessment  
✅ **Multiple examples** - Various notebooks showing different use cases  

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=2.0.0`
- `tiktoken>=0.5.0` 
- `transformers>=4.21.0`
- `numpy>=1.21.0`
- `datasets>=2.0.0`
- `tqdm>=4.62.0`

## Learning Path

1. **Start with theory** - Read this README to understand transformer concepts
2. **Educational implementation** - Study `gpt_from_scratch.py` for step-by-step learning
3. **Jupyter notebooks** - Explore `gpt_from_scratch.ipynb` for interactive learning
4. **Production code** - Examine `transformer.py` for optimized implementation
5. **Training** - Run `train_gpt2.py` to train your own model
6. **Evaluation** - Use `hellaswag.py` to evaluate model performance

## Acknowledgments

This implementation follows Andrej Karpathy's excellent educational content:
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Great visual explanation