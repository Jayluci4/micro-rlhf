# micro-rlhf

> **RLHF in ~150 lines of code**
> Understand how ChatGPT is aligned by building RLHF from scratch.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Part of Series](https://img.shields.io/badge/Part%20of-Build%20Modern%20AI-blue)](https://github.com/Jayluci4)

RLHF (Reinforcement Learning from Human Feedback) is how ChatGPT learned to be helpful, harmless, and honest. This repo shows you exactly how alignment works.

**No frameworks. No complexity. Just the algorithm.**

---

## Important Note

**This focuses on the ALIGNMENT phase** (what makes ChatGPT different from base GPT).

**For training foundations** (backpropagation, gradient descent), see:
- [**Karpathy's micrograd**](https://github.com/karpathy/micrograd) - Backpropagation from scratch
- [**Karpathy's makemore**](https://github.com/karpathy/makemore) - Training language models
- [**Karpathy's nanoGPT**](https://github.com/karpathy/nanoGPT) - GPT from scratch

**nanochat uses RLHF but doesn't teach it** - that's what this repo is for!

---

## What is RLHF?

RLHF is the secret sauce that transforms base GPT into ChatGPT.

**Without RLHF:**
- Model completes text (autocomplete)
- May be unhelpful or harmful
- Optimized for predicting next token

**With RLHF:**
- Model follows instructions
- Helpful, harmless, honest
- Optimized for human preferences

**This is how ChatGPT, Claude, and all assistants are trained.**

---

## The 3 Steps of RLHF

```
1. SFT (Supervised Fine-Tuning)
   → Train on (prompt, good_response) pairs
   → Model learns to follow instructions
   → See nanochat for implementation

2. Reward Model Training (This repo - Part 1)
   → Train on human preferences
   → "Response A > Response B"
   → Model learns what humans prefer

3. RL Fine-tuning with PPO (This repo - Part 2)
   → Use reward model to guide training
   → Policy learns to maximize reward
   → Model becomes aligned
```

---

## Quickstart

```python
from microrlhf import MicroRLHF
import numpy as np

# Create RLHF system
rlhf = MicroRLHF(embedding_dim=128)

# Step 2: Train reward model on preferences
preferences = [
    (helpful_response_embedding, unhelpful_response_embedding),
    (safe_response_embedding, unsafe_response_embedding),
    # ... more preference pairs
]

rlhf.train_reward_model(preferences)

# Step 3: Align policy using RL
prompts = [prompt_embedding_1, prompt_embedding_2, ...]
rewards = rlhf.align_policy(prompts, n_iterations=20)

# Generate aligned response
response, reward = rlhf.generate_aligned_response(new_prompt)
print(f"Reward: {reward:.3f}")  # Higher = better aligned
```

---

## What You'll Learn

### 1. Reward Modeling
```python
# Train model to predict: Is this response good?
reward = reward_model(response)  # Scalar value

# Trained on comparisons:
# "Response A > Response B" (human judgment)
```

### 2. Preference Learning
```python
# Loss: Chosen should have higher reward than rejected
loss = -log(sigmoid(reward_chosen - reward_rejected))
```

### 3. PPO (Proximal Policy Optimization)
```python
# Update policy to maximize reward
advantage = reward - baseline
policy_update ∝ advantage
```

### 4. The Alignment Loop
```
for iteration in range(N):
    1. Sample responses from policy
    2. Get rewards from reward model
    3. Update policy toward higher rewards
```

---

## Core Implementation

### Reward Model

```python
class RewardModel:
    def predict_reward(self, response):
        # Scalar output: How good is this response?
        return weights @ response + bias

    def train_on_preferences(self, pairs):
        # Learn from: chosen > rejected
        for chosen, rejected in pairs:
            reward_chosen = self.predict_reward(chosen)
            reward_rejected = self.predict_reward(rejected)

            # Loss: chosen should have higher reward
            loss = -log(sigmoid(reward_chosen - reward_rejected))

            # Update weights (gradient descent)
```

### PPO Alignment

```python
class SimplifiedPPO:
    def update_policy(self, prompt, reward):
        # Move policy toward higher rewards
        advantage = reward - baseline
        policy_weights += lr * advantage * gradient
```

**That's RLHF!** Everything else is optimization and engineering.

---

## Examples

### Example 1: Train Reward Model

```python
import numpy as np
from microrlhf import RewardModel

# Create reward model
rm = RewardModel(input_dim=128)

# Preference data: (chosen, rejected) pairs
preferences = [
    (helpful_embedding, unhelpful_embedding),
    (safe_embedding, unsafe_embedding),
]

# Train
rm.train_on_preferences(preferences, epochs=10)

# Predict
new_response = np.random.randn(128)
reward = rm.predict_reward(new_response)
print(f"Predicted reward: {reward:.3f}")
```

### Example 2: Align Policy

```python
from microrlhf import MicroRLHF

rlhf = MicroRLHF(embedding_dim=128)

# Train reward model first
rlhf.train_reward_model(preferences)

# Align policy
prompts = [np.random.randn(128) for _ in range(10)]
rewards_history = rlhf.align_policy(prompts, n_iterations=20)

# See improvement
print(f"Initial avg reward: {rewards_history[0]:.3f}")
print(f"Final avg reward: {rewards_history[-1]:.3f}")
```

### Example 3: Compare Before/After

```python
# Generate with unaligned policy
response_before, reward_before = rlhf.generate_aligned_response(prompt)

# Align the policy
rlhf.align_policy([prompt], n_iterations=50)

# Generate again
response_after, reward_after = rlhf.generate_aligned_response(prompt)

print(f"Reward before: {reward_before:.3f}")
print(f"Reward after: {reward_after:.3f}")  # Should be higher!
```

---

## Visualizations

### RLHF Pipeline

```python
from visualize import plot_rlhf_pipeline

plot_rlhf_pipeline()
```

Shows the 3-step process visually.

### Reward Training Progress

```python
from visualize import plot_reward_training

# After training reward model
plot_reward_training(loss_history)
```

Shows how reward model learns preferences.

### Alignment Progress

```python
from visualize import plot_alignment_progress

# After RL alignment
plot_alignment_progress(rewards_history)
```

Shows policy improving over iterations.

---

## Key Insights

### 1. Alignment ≠ Training

```
Training (micrograd/nanoGPT):
→ Learn language patterns
→ Predict next token
→ Result: Base model

Alignment (this repo):
→ Learn human preferences
→ Optimize for helpfulness
→ Result: Assistant model (ChatGPT)
```

**Different goals, different techniques.**

### 2. Reward Model is the Key

```
Without reward model:
→ No way to measure "goodness"

With reward model:
→ Can optimize policy
→ Can measure improvement
→ Can align to human values
```

### 3. RLHF is Expensive

```
Why ChatGPT costs more to train:
1. Human labeling (expensive)
2. Reward model training
3. RL fine-tuning (compute intensive)
4. Multiple iterations

But result: Much better model!
```

### 4. DPO is Simpler (Modern Alternative)

```
RLHF: Train reward model → Use PPO (2 steps)
DPO: Direct preference optimization (1 step)

Both achieve alignment. DPO is newer and simpler.
```

---

## Production vs Educational

**This Implementation:**
- Simplified PPO
- Small reward model
- Educational focus
- ~150 lines

**Production (ChatGPT):**
- Full PPO with clipping
- Large reward model (GPT-size)
- Distributed training
- Thousands of lines

**But the concept is identical!**

---

## Why This Matters

### ChatGPT's Secret:

```
Base GPT-3: Smart but unhelpful
    ↓ (RLHF)
ChatGPT: Smart AND helpful
```

**RLHF is the difference.**

### Understanding nanochat:

nanochat has RL alignment phase → Now you understand what it does!

---

## Learn More

### Papers
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT, 2022)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO, 2023)

### Prerequisites
- [micrograd](https://github.com/karpathy/micrograd) - Backpropagation
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Base model training
- [nanochat](https://github.com/karpathy/nanochat) - Complete pipeline

---

## Part of "Build Modern AI From Scratch" Series

**Foundations (see Karpathy):**
- Backpropagation → micrograd
- Training → makemore
- GPT → nanoGPT

**Modern AI (this series):**
0. [micro-tokenizer](https://github.com/Jayluci4/micro-tokenizer) - BPE tokenization
1. [micro-attention](https://github.com/Jayluci4/micro-attention) - Attention mechanism
2. [micro-transformer](https://github.com/Jayluci4/micro-transformer) - Complete transformers
3. **micro-rlhf** (you are here) - Alignment (~150 lines)
4. [micro-rag](https://github.com/Jayluci4/micro-rag) - RAG
5. [micro-agent](https://github.com/Jayluci4/micro-agent) - Agents
6. [micro-kg](https://github.com/Jayluci4/micro-kg) - Knowledge graphs

**Coming:**
- micro-diffusion - Image generation
- micro-lora - Efficient fine-tuning

---

## License

MIT License - see [LICENSE](LICENSE)

---

**Built with ❤️ for learning**

*"Now you know how ChatGPT becomes helpful."*
