# PM-MiniFinLLM

> A 110M-parameter GPT-2-style language model trained end-to-end on financial text — covering the complete pipeline from pre-training through SFT and DPO alignment.
---

## Overview

PM-MiniFinLLM is a from-scratch implementation of the full LLM training pipeline applied to the finance domain. The goal is not to build a production-competitive model — at 110M parameters, that would be unrealistic — but to deeply understand every stage of LLM development by implementing it.

A detailed walkthrough of every stage — architecture, pre-training, SFT, and DPO — is available in the [blog post on Medium](https://medium.com/@pankajmaulekhi32/introduction-a4415e165692).

This project covers:

- Custom transformer architecture with RoPE positional embeddings
- Pre-training on ~2.36B tokens from SEC EDGAR filings
- Supervised Fine-Tuning (SFT) on financial QA datasets
- Alignment via Direct Preference Optimization (DPO)
- HuggingFace Hub integration with `AutoModelForCausalLM` support


---

## Key Features

| Feature | Detail |
|---|---|
| Architecture | GPT-2 style transformer with RoPE (replaces learned positional embeddings) |
| Parameters | ~110M |
| Pre-training data | SEC EDGAR corpus — 2.36B tokens across 13 years (2008–2020) |
| Training hardware | 2× NVIDIA T4 (16GB VRAM each) on Kaggle free tier |
| Distributed training | PyTorch DDP (DistributedDataParallel) across 2 GPUs |
| Mixed precision | FP16 with `GradScaler` (T4 does not support BF16) |
| Alignment algorithm | DPO (Direct Preference Optimization) — avoids 4-model PPO overhead |
| HuggingFace support | `AutoModelForCausalLM`, custom config, tokenizer with chat template |

---

## Model Architecture

The model follows a standard decoder-only transformer design with one key modification

- **RoPE (Rotary Positional Embeddings)** instead of GPT-2's learned absolute positional embeddings — enabling better length generalization and relative position awareness

Weights for the embedding layer and output linear layer are tied to minimize total parameter count.

| Hyperparameter | Value |
| :--- | :--- |
| **Parameters** | ~110 Million |
| **Layers** | 12 |
| **Attention Heads** | 12 |
| **Embedding Dimension** | 768 |
| **Vocabulary Size** | 32,000 (Custom BPE) |
| **Positional Embedding** | RoPE |
| **Max Sequence Length** | 1,024 |



---

## Training Pipeline

### 1. Pre-training

**Dataset:** SEC EDGAR annual reports (10-K filings), sections 1, 1A, 3, 7, 7A — filtered for financial relevance.

- ~1.9B words, ~2.36B tokens after BPE tokenization
- Tokenizer trained from scratch using HuggingFace `tokenizers` (BPE, vocab size 32K)
- Dataset stored as a binary `.bin` file (uint16) and loaded via `np.memmap` — no full-RAM load required

**Training setup:**

- Batch size: 12 per GPU × 2 GPUs × 4 gradient accumulation steps = effective batch 96
- Sequence length: 1024 tokens
- Optimizer: AdamW with weight decay separation (decay on weight matrices only)
- LR schedule: cosine decay with linear warmup (1% of total steps)
- Training: 2 epochs (~40 tokens/parameter, above Chinchilla-optimal for this scale)
- Checkpointing: saves model, optimizer, scaler, scheduler, and Random state every 100 steps

**Memory budget:**

Static memory (model states under FP16 + optimizer) estimated at ~2GB, leaving ~14GB for activations. Theoretical max batch size at seq_len=1024 was ~13 — used 12 with a safety buffer.

### 2. Supervised Fine-Tuning (SFT)

Fine-tuned on three public financial QA datasets (~28K samples total):

- `virattt/financial-qa-10K` — context-grounded QA from 10-K filings
- `sweatSmile/FinanceQA` — context-grounded financial QA
- `LLukas22/fiqa` — open financial QA (no context)

Used HuggingFace `SFTTrainer` (trl) with `assistant_only_loss=True` — loss is masked on user prompt tokens using a custom `chat_template` with `{% generation %}` tags.

### 3. Alignment (DPO)

Implemented DPO from scratch (without trl's `DPOTrainer`) using:

- Policy model and frozen reference model (deepcopy of SFT checkpoint)
- Per-sequence log-probability computation with assistant-token masking
- DPO loss: `-log_sigmoid(β * (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))`
- Dataset: `argilla/ultrafeedback-binarized-preferences-cleaned` (1000 samples)
- β = 0.1, LR = 1e-6, gradient accumulation = 4 steps

---


## HuggingFace Integration

The model is registered with HuggingFace's AutoClass system and can be loaded with a single line:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Pankaj121212/PM_MiniFinLLM", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Pankaj121212/PM_MiniFinLLM", trust_remote_code=True)

messages = [{"role": "user", "content": "What are the main risk factors for a bank?"}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

output = model.generate(input_ids, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Custom components:**

- `PM_MiniFinLLM_config` — inherits `PretrainedConfig`, stores all hyperparameters
- `PM_MiniFinLLM_Model` — inherits `PreTrainedModel` + `GenerationMixin`, supports `.generate()` out of the box
- Weight tying handled explicitly via `tie_weights()` override

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch 2.x, `torch.cuda.amp`, `torch.nn.parallel.DistributedDataParallel` |
| Training | AdamW, cosine LR schedule, gradient clipping, FP16 mixed precision |
| Tokenization | HuggingFace `tokenizers` (BPE), `PreTrainedTokenizerFast` |
| Fine-tuning | HuggingFace `trl` (SFTTrainer), custom DPO loop |
| Model serving | HuggingFace Hub, `transformers` AutoClass, `GenerationMixin` |
| Data | EDGAR corpus (Zenodo), NumPy memmap, HuggingFace `datasets` |
| Hardware | 2× NVIDIA T4 (Kaggle free tier), 16GB VRAM each |

---

## Acknowledgements

- [EDGAR corpus on Zenodo](https://zenodo.org/records/5528490) — pre-training data
- Chinchilla scaling laws (Hoffmann et al., 2022) — for training compute estimates
- NVIDIA activation memory formula — for VRAM budgeting
- HuggingFace `trl` library — SFTTrainer implementation
- PyTorch DDP tutorial series — distributed training reference