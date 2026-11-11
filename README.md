# micro-rlhf

> **RLHF in ~250 lines of NumPy**
> Understand how ChatGPT is aligned by building RLHF from scratch.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Part of Series](https://img.shields.io/badge/Part%20of-Build%20Modern%20AI-blue)](https://github.com/Jayluci4)

RLHF (Reinforcement Learning from Human Feedback) is how ChatGPT learned to be helpful, harmless, and honest. This repo shows you exactly how the alignment loop works.

**No frameworks. No black boxes. Just the algorithm.**

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

**This is the classic RLHF recipe behind ChatGPT, Claude, and many assistants.**

---

## The 3 Steps of RLHF

```
1. SFT (Supervised Fine-Tuning)
   -> Train on (prompt, good_response) pairs
   -> Model learns to follow instructions
   -> See nanochat for implementation

2. Reward Model Training (This repo - Part 1)
   -> Train on human preferences
   -> "Response A > Response B"
   -> Model learns what humans prefer

3. RL Fine-tuning with PPO (This repo - Part 2)
   -> Use reward model to guide training
   -> Policy learns to maximize reward
   -> Model becomes aligned
```

---

## Quickstart

```python
import numpy as np
from microrlhf import MicroRLHF

rng = np.random.default_rng(0)
dim = 64

# Create the RLHF system
rlhf = MicroRLHF(embedding_dim=dim, seed=0)

# Step 2: Train the reward model on (chosen, rejected) pairs
preferences = [
    (rng.standard_normal(dim) + 1.0,   # chosen: helpful direction
     rng.standard_normal(dim) - 0.5)   # rejected: unhelpful direction
    for _ in range(20)
]
losses = rlhf.train_reward_model(preferences)

# Step 3: Align the policy
train_prompts = rng.standard_normal((8, dim))
eval_prompts = rng.standard_normal((64, dim))

before = rlhf.evaluate(eval_prompts)
history = rlhf.align_policy(
    train_prompts,
    n_iterations=20,
    eval_prompts=eval_prompts,
)
after = rlhf.evaluate(eval_prompts)

print(f"reward: {before:+.3f} -> {after:+.3f}")
print(f"Exact KL from reference: {rlhf.exact_kl_to_reference():.3f} nats")

# Generate with the aligned policy
response, reward = rlhf.generate_aligned_response(
    rng.standard_normal(dim), deterministic=True)
```

Run the full example:

```bash
python examples/simple.py
```

Typical output:

```text
Preference accuracy: 100%

[Step 3] Aligning Policy with 20 iterations...
  Iteration   5/20, Eval Reward: 0.5432, Exact KL: 0.2916
  Iteration  10/20, Eval Reward: 0.8591, Exact KL: 0.7876
  Iteration  20/20, Eval Reward: 0.8614, Exact KL: 0.7909

Eval reward before: +0.0655
Eval reward after:  +0.8614
Exact KL from ref: 0.7909 nats
```

The policy moves toward the reward model, stops near the KL-regularized
optimum, and the deterministic eval reward rises instead of merely the noisy
training reward.

---

## What You'll Learn

### 1. Reward Modeling

```python
# Train a model to predict: Is this response good?
reward = reward_model.predict_reward(response)  # scalar

# Trained on comparisons:
# "Response A > Response B" (human judgment)
```

### 2. Preference Learning

```python
# Bradley-Terry loss: chosen should have higher reward than rejected
diff = reward_chosen - reward_rejected
loss = -log(sigmoid(diff))      # == softplus(-diff), computed stably
```

### 3. Advantage and the Value Baseline

The advantage is **not** the raw reward. It is the shaped reward minus what the
value function expected:

```python
shaped_reward = reward_model(s, a) - beta * sample_kl
advantage = shaped_reward - value(prompt)
```

`sample_kl` is the per-sample KL estimate defined in Section 6. For multi-step
episodes, the same idea is computed recursively with GAE (`r_t` is the shaped
reward):

```python
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
```

This repo is a one-step (bandit) setting, so GAE reduces to
`A = shaped_reward - V(s)`. The general implementation is still in
`compute_gae` so the formula is not hidden.

### 4. The Score-Function Policy Gradient

The policy is a Gaussian stand-in for an LLM:

```python
pi(a | s) = Normal(mean=s + theta, std=sigma)
grad_theta log pi(a | s) = (a - s - theta) / sigma**2
```

The policy update must use that **score function**, not the prompt embedding.
The prompt is the state; it is not the gradient. This distinction is the
difference between a correct policy gradient and an arbitrary update rule.

### 5. PPO: the Old-Policy Ratio and Clipping

The gradient `A * grad log pi` is REINFORCE with a baseline. PPO uses that
same score-function gradient inside the old-policy probability ratio:

```python
rho = exp(log pi_new - log pi_old)
objective = min(rho * advantage,
                clip(rho, 1 - eps, 1 + eps) * advantage)
```

The ratio is 1 at the start of the batch. After a few gradient steps it moves;
the clip stops one batch from changing the policy too much. This is what makes
the update PPO rather than plain REINFORCE.

### 6. The KL Anchor: Sample vs Exact

The objective includes `KL(pi_theta || pi_ref)`. The code uses the per-sample
estimator in the reward:

```python
sample_kl = log pi(a|s) - log pi_ref(a|s)
shaped_reward = reward_model_reward - beta * sample_kl
```

With `a ~ pi`, `E[sample_kl] = KL`. It is noisy and can be negative.

For the Gaussian policy the exact KL is closed-form:

```python
exact_kl = ||theta - theta_ref||**2 / (2 * std**2)
```

The exact KL is for diagnostics and the analytic optimum, **not** for the
per-sample reward. At a fixed `theta` it is constant across actions, so if you
put it in the reward the score-function estimator sees no KL term:
`E[KL_exact * grad log pi] = KL_exact * E[grad log pi] = 0`. The sample KL is
action-dependent, and
`E[(log pi - log pi_ref) grad log pi] = grad KL`.

---

## The Math (and Three Traps to Avoid)

### Trap 1: "Advantage" is just the reward

The original teaching version returned `reward - 0.0`. That is not an
advantage; it is the raw reward. Every positive reward then pushes the policy,
even when the reward is exactly what the value function already predicted.
Here the reward in the advantage is the shaped reward (RM reward minus the
sample-KL penalty).

A baseline is safe to subtract because, for a fixed state,

```
E_{a ~ pi(.|s)}[ grad log pi(a|s) ] = 0
```

so `E[(r_shaped - V(s)) grad log pi] = E[r_shaped grad log pi]`. The expected
gradient is unchanged; the variance is lower. The code learns `V(s) = v . s + b`
by
regression on the returns.

### Trap 2: The prompt is not `grad log pi`

The policy gradient is

```
grad_theta J = E[ A(s,a) * grad_theta log pi(a|s) ]
```

and for a Gaussian policy the last factor is `(a - s - theta) / sigma^2`.
Using the prompt `s` instead is a different estimator: it ignores the sampled
action, and it is biased whenever the prompt distribution is not centered.

A clean counterexample: let the prompt direction be orthogonal to the reward
direction. The old `A * prompt` update moves the policy along the prompt, which
has zero reward projection, so it makes no progress. The score-function update
moves along `(a - mean)`, which contains the reward direction, and learns.

A second, subtler issue masked the bug: the reward model and policy both called
`np.random.seed(42)`, so their initial weight vectors were identical. The
policy started pointing along the reward model's initial weights, which made
the demo look like it was learning. This implementation uses independent
`np.random.default_rng` streams.

### REINFORCE vs PPO

The update `A * grad log pi` is REINFORCE with a baseline. PPO uses that same
score-function gradient inside an importance-weighted ratio:

```
rho = pi_theta(a|s) / pi_old(a|s)
L_clip = min(rho * A, clip(rho, 1-eps, 1+eps) * A)
```

The ratio lets the same rollout be reused for several gradient steps; the clip
keeps those steps close to the sampling policy. Without the ratio and clipping,
the update is REINFORCE, and calling it PPO would be wrong. The implementation
stores `old_log_probs` in `collect()` and uses them in
`clipped_surrogate_gradient()` — that is the PPO part.

### Trap 3: Maximizing a reward model without a leash

The reward model is a finite, imperfect approximation of human preferences. If
you optimize it without a constraint, the policy will find the model's blind
spots. That is reward hacking.

The KL-regularized objective is

```
max_theta  E_{a ~ pi_theta}[ r_RM(a) ] - beta * KL(pi_theta || pi_ref)
```

For an LLM the exact KL is intractable, so the implementation follows
InstructGPT and penalizes the per-sample estimate

```
k_hat = log pi_theta(a|s) - log pi_ref(a|s)
```

in the reward. For this Gaussian toy we can also compute the exact KL; it is
the expectation of `k_hat` and is used for diagnostics and for the closed-form
optimum, not as the per-sample reward (see Section 6 for why the difference
matters to the gradient).

For a linear reward model `r(a) = w . a + b` and a Gaussian policy with fixed
`sigma`, the expected reward is `w . (s + theta) + b`, and

```
KL(pi_theta || pi_ref) = ||theta - theta_ref||^2 / (2 * sigma^2)
```

Setting the derivative to zero gives the closed-form optimum

```
theta* = theta_ref + sigma^2 * w / beta
```

That is a useful sanity check: the policy should move in the reward direction,
but only as far as the KL budget allows. `tests/test_microrlhf.py` checks that
the trained policy lands near this optimum.

---

## Core Implementation

### Reward Model

```python
class RewardModel:
    def predict_reward(self, response):
        return self.weights @ response + self.bias

    def loss_and_grad(self, chosen, rejected):
        diff = (chosen - rejected) @ self.weights
        # stable softplus: logaddexp(0, -diff)
        loss = np.mean(np.logaddexp(0.0, -diff)) + 0.5 * wd * ||weights||^2
        # d/d(diff) logaddexp(0, -diff) = sigmoid(diff) - 1
        dloss = np.exp(-np.logaddexp(0.0, -diff)) - 1.0
        grad = (chosen - rejected).T @ dloss / n + wd * weights
        return loss, grad
```

Do **not** use `-np.log1p(np.exp(-diff))`: `exp(-diff)` overflows for large
negative `diff`. `np.logaddexp(0, -diff)` is the stable form. A shared bias
cancels in `r_chosen - r_rejected`, so it is not identifiable from pairwise
data and is left at zero.

### Gaussian Policy

```python
class GaussianPolicy:
    # pi(a|s) = Normal(s + theta, std^2 I)
    def mean(self, prompt):
        return prompt + self.theta

    def log_prob(self, prompt, response):
        z = (response - self.mean(prompt)) / self.std
        return -0.5 * z @ z - d * log(self.std) - 0.5 * d * log(2 * pi)

    def exact_kl(self, other):
        # exact KL for two diagonal Gaussians with the same std
        delta = self.theta - other.theta
        return 0.5 * delta @ delta / self.std**2
```

`exact_kl` is the closed-form expectation. The code's reward uses the
per-sample estimate `log pi - log pi_ref` instead; see below.

### Value Baseline and GAE

```python
class ValueFunction:
    # V(s) = weights . s + bias
    def predict_batch(self, prompts):
        return prompts @ self.weights + self.bias

    def train_step(self, prompts, returns):
        error = self.predict_batch(prompts) - returns
        self.weights -= lr * (prompts.T @ error / n + wd * self.weights)
        self.bias -= lr * mean(error)
```

`compute_gae` implements the general recursion; for this one-step setting it
reduces to `A = shaped_reward - V(prompt)`.

### PPO Alignment

```python
class SimplifiedPPO:
    def collect(self, prompts, reward_model, rng):
        responses = policy_mean + rng.standard_normal(...) * std
        old_logps = policy.log_prob_batch(prompts, responses)
        ref_logps = reference.log_prob_batch(prompts, responses)
        raw_reward = reward_model.predict_reward_batch(responses)

        # sample KL, not exact KL: this is the action-dependent penalty
        shaped_reward = raw_reward - beta * (old_logps - ref_logps)
        advantages, returns = compute_gae(shaped_reward, values, ...)
        return Rollout(...)

    def clipped_surrogate_gradient(self, prompts, responses, old_logps, adv):
        ratio = exp(log_prob(prompts, responses) - old_logps)
        unclipped = ratio * adv
        clipped = clip(ratio, 1 - eps, 1 + eps) * adv
        use_unclipped = unclipped <= clipped
        score = (responses - mean(prompts)) / std**2
        return mean((use_unclipped * adv * ratio)[:, None] * score, axis=0)
```

The score line is the REINFORCE gradient. `old_logps` and `ratio` are the PPO
part; `update()` runs several epochs on the same rollout and clips each step.

---

## Design Notes: What This Version Fixes

| Issue | Old behavior | Correct behavior |
|---|---|---|
| Value function | none | linear `V(s)`, trained on returns with MSE |
| Advantage | `reward - 0.0` | `shaped_reward - V(prompt)` via GAE |
| Policy gradient | `advantage * prompt` | `advantage * (a - mean) / std^2` |
| PPO | no ratio, no clipping | old-policy ratio + clipped surrogate, several epochs per batch |
| Reference policy | none | frozen snapshot of the initial policy |
| KL | none | sample KL in the reward; exact KL for diagnostics/optimum |
| Evaluation | noisy sampled training rewards | deterministic mean on eval prompts |
| Seeding | two models share `np.random.seed(42)` | independent RNG streams |
| Reward loss | unstable `log1p(exp(-diff))` | stable `np.logaddexp(0, -diff)` |

The reward-model gradient itself was already correct; the main errors were on
the policy/advantage side.

---

## Examples

### Example 1: Train the Reward Model

```python
import numpy as np
from microrlhf import RewardModel

rng = np.random.default_rng(0)
rm = RewardModel(input_dim=64, seed=0)

preferences = [
    (rng.standard_normal(64) + 1.0,
     rng.standard_normal(64) - 0.5)
    for _ in range(20)
]

losses = rm.train_on_preferences(preferences, epochs=50)

new_response = rng.standard_normal(64)
print(f"Predicted reward: {rm.predict_reward(new_response):+.3f}")
```

### Example 2: Align the Policy

```python
from microrlhf import MicroRLHF

rlhf = MicroRLHF(embedding_dim=64, seed=0)
rlhf.train_reward_model(preferences)

prompts = np.random.default_rng(1).standard_normal((8, 64))
history = rlhf.align_policy(prompts, n_iterations=20)

print(f"First eval reward: {history[0]:+.3f}")
print(f"Final eval reward: {history[-1]:+.3f}")
```

### Example 3: Compare Before and After

```python
eval_prompts = np.random.default_rng(2).standard_normal((64, 64))

before = rlhf.evaluate(eval_prompts)
rlhf.align_policy(prompts, n_iterations=20, eval_prompts=eval_prompts)
after = rlhf.evaluate(eval_prompts)

print(f"Before: {before:+.3f}")
print(f"After:  {after:+.3f}")
print(f"Exact KL: {rlhf.exact_kl_to_reference():.3f}")
```

---

## Tests

The math is checked against independent oracles, not just "does it run":

```bash
python -m unittest discover -s tests -v
```

The suite covers:

- Bradley-Terry gradient vs. finite differences
- Stable Bradley-Terry loss at extreme margins
- GAE vs. a brute-force recursive return calculation
- `log pi` and its gradient vs. finite differences
- Sample-KL estimator vs. the exact Gaussian KL (Monte Carlo)
- PPO clipped-surrogate gradient vs. finite differences
- End-to-end improvement and closeness to the closed-form KL optimum
- Seed reproducibility

---

## Visualizations

```python
from visualize import (plot_rlhf_pipeline, plot_reward_training,
                       plot_alignment_progress)

plot_rlhf_pipeline()

losses = rlhf.train_reward_model(preferences)
plot_reward_training(losses)

history = rlhf.align_policy(prompts, n_iterations=20)
plot_alignment_progress(history)
```

`plot_alignment_progress` now plots the deterministic eval reward, so the
curve is not dominated by sampling noise.

---

## Key Insights

### 1. Alignment != Training

```
Training (micrograd/nanoGPT):
-> Learn language patterns
-> Predict next token
-> Result: Base model

Alignment (this repo):
-> Learn human preferences
-> Optimize for helpfulness
-> Result: Assistant model (ChatGPT)
```

**Different goals, different techniques.**

### 2. The Reward Model is the Key

```
Without reward model:
-> No way to measure "goodness"

With reward model:
-> Can optimize policy
-> Can measure improvement
-> Can align to human values
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
RLHF: Train reward model -> Use PPO (2 steps)
DPO: Direct preference optimization (1 step)

Both achieve alignment. DPO is newer and simpler.
```

### 5. Three Details That Matter

```
1. Advantage = return - value baseline (not raw reward)
2. Policy gradient uses grad log pi (not the prompt)
3. KL penalty keeps the policy near the SFT reference
```

---

## Production vs Educational

**This Implementation:**
- Linear reward model
- Gaussian policy over response embeddings
- One-step episodes (one response = one episode)
- Fixed policy standard deviation
- Full-batch reward-model training
- ~250 lines of core NumPy, no deep-learning framework

**Production (ChatGPT):**
- Neural reward model at LLM scale
- Token-level policy with billions of parameters
- GAE over token sequences, reward normalization
- Distributed PPO with clipping and KL controllers
- Thousands of lines of infrastructure

**The objective is the same.** The scale is different.

### Limitations

This is a toy. It does **not** generate language, tokenize text, or use real
human data. The "responses" are embeddings and the "policy" is one learned
vector. The point is to isolate the alignment math: preference loss, advantage,
score-function gradient, PPO clipping, and the KL leash.

---

## Why This Matters

### ChatGPT's Secret:

```
Base GPT-3: Smart but unhelpful
    | (RLHF)
ChatGPT: Smart AND helpful
```

**RLHF is the difference.**

### Understanding nanochat:

nanochat has an RL alignment phase -> Now you understand what it does!

---

## Learn More

### Papers
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT, 2022)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (PPO, 2017)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO, 2023)

### Prerequisites
- [micrograd](https://github.com/karpathy/micrograd) - Backpropagation
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Base model training
- [nanochat](https://github.com/karpathy/nanochat) - Complete pipeline

---

## Part of "Build Modern AI From Scratch" Series

**Foundations (see Karpathy):**
- Backpropagation -> micrograd
- Training -> makemore
- GPT -> nanoGPT

**Modern AI (this series):**
0. [micro-tokenizer](https://github.com/Jayluci4/micro-tokenizer) - BPE tokenization
1. [micro-attention](https://github.com/Jayluci4/micro-attention) - Attention mechanism
2. [micro-transformer](https://github.com/Jayluci4/micro-transformer) - Complete transformers
3. **micro-rlhf** (you are here) - Alignment (~250 lines)
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

**Built with love for learning**

*"Now you know how ChatGPT becomes helpful."*
