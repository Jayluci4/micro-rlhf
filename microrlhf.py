"""
micro-rlhf: RLHF in ~250 lines

Understand how ChatGPT is aligned by building RLHF from scratch.

RLHF = Reinforcement Learning from Human Feedback
How models learn to be helpful, harmless, and honest.

NOTE: For training foundations (backprop, gradient descent), see:
- Karpathy's micrograd: https://github.com/karpathy/micrograd
- Karpathy's makemore: https://github.com/karpathy/makemore

This focuses on the ALIGNMENT phase - what makes ChatGPT different from base GPT.

The objective
-------------
    J(theta) = E_s E_{a ~ pi_theta(.|s)}[ r_RM(s, a) ]
               - beta * KL(pi_theta(.|s) || pi_ref(.|s))

Reward model (Bradley-Terry):
    L = -log sigmoid(r_chosen - r_rejected) = softplus(-diff)

Policy (a Gaussian stand-in for an LLM):
    pi(a|s) = N(s + theta, std^2 I)
    grad_theta log pi(a|s) = (a - s - theta) / std^2

Advantage (GAE with a learned value baseline):
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    For one-step episodes this is just A = r - V(s).

Policy gradient (REINFORCE with a baseline):
    g = A(s, a) * grad_theta log pi(a|s)

PPO (the old-policy ratio and clipping are what make it PPO):
    rho = pi_theta(a|s) / pi_old(a|s)
    L   = min(rho * A, clip(rho, 1 - eps, 1 + eps) * A)

KL: two different objects
    sample KL used in the reward:
        k_hat = log pi_theta(a|s) - log pi_ref(a|s)
        E_{a ~ pi_theta}[k_hat] = KL(pi_theta || pi_ref)
    exact KL for the Gaussian, used for diagnostics:
        KL = ||theta - theta_ref||^2 / (2 * std^2)

Inspired by Andrej Karpathy's educational approach.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ============================================================================
# PART 1: REWARD MODEL - Learning Human Preferences
# ============================================================================

class RewardModel:
    """
    Reward Model: Predicts how good a response is.

    Trained on human preferences:
    "Response A is better than Response B"

    This is step 2 of RLHF (step 1 is SFT, which is not implemented here).
    """

    def __init__(self, input_dim: int = 128, seed: int = 0,
                 weight_decay: float = 1e-3):
        """
        Args:
            input_dim: Dimension of response representation
            seed: RNG seed (no global np.random.seed!)
            weight_decay: L2 penalty, keeps the reward scale from exploding
        """
        # In production: This would be a neural network.
        # For education: a linear reward model is enough to see the mechanics.
        rng = np.random.default_rng(seed)
        self.weights = rng.standard_normal(input_dim) * 0.01
        self.bias = 0.0
        self.weight_decay = weight_decay

    def predict_reward(self, response_embedding: np.ndarray) -> float:
        """
        Predict reward for a response.

        Args:
            response_embedding: Vector representation of response

        Returns:
            reward: Scalar reward value (higher = better)
        """
        return float(self.weights @ response_embedding + self.bias)

    def predict_reward_batch(self, responses: np.ndarray) -> np.ndarray:
        """Predict rewards for a batch of response embeddings."""
        return responses @ self.weights + self.bias

    def loss_and_grad(self, chosen: np.ndarray, rejected: np.ndarray):
        """
        Bradley-Terry loss and its gradient with respect to the weights.

        L = mean(logaddexp(0, -(r_chosen - r_rejected))) + 0.5 * wd * ||w||^2

        logaddexp(0, -diff) is the stable form of -log sigmoid(diff). Do not
        use -log1p(exp(-diff)): exp(-diff) overflows for large negative diff.

        Note: a shared bias cancels in r_chosen - r_rejected, so it is not
        identifiable from pairwise data (and it cancels in advantages too).
        We therefore leave self.bias at zero.
        """
        chosen = np.asarray(chosen, dtype=float)
        rejected = np.asarray(rejected, dtype=float)

        diff = (chosen - rejected) @ self.weights
        loss = float(np.mean(np.logaddexp(0.0, -diff))
                     + 0.5 * self.weight_decay * (self.weights @ self.weights))

        # d/d(diff) logaddexp(0, -diff) = -sigmoid(-diff) = sigmoid(diff) - 1
        dloss_ddiff = np.exp(-np.logaddexp(0.0, -diff)) - 1.0
        grad_w = ((chosen - rejected).T @ dloss_ddiff / len(chosen)
                  + self.weight_decay * self.weights)
        return loss, grad_w

    def train_on_preferences(self, preferences: List[Tuple[np.ndarray, np.ndarray]],
                             learning_rate: float = 0.1, epochs: int = 50,
                             verbose: bool = True) -> List[float]:
        """
        Train reward model from preferences with full-batch gradient descent.

        Args:
            preferences: List of (chosen_response, rejected_response) pairs
            learning_rate: Learning rate
            epochs: Training epochs
            verbose: Print progress

        Returns:
            losses: Loss at the start of each epoch (for plotting)
        """
        if not preferences:
            raise ValueError("preferences must not be empty")

        chosen = np.stack([pair[0] for pair in preferences])
        rejected = np.stack([pair[1] for pair in preferences])

        if verbose:
            print(f"[OK] Training reward model on {len(preferences)} preferences...")

        losses = []
        for epoch in range(epochs):
            loss, grad = self.loss_and_grad(chosen, rejected)
            self.weights -= learning_rate * grad
            losses.append(loss)

            if verbose and ((epoch + 1) % 10 == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch + 1:3d}/{epochs}, Loss: {loss:.4f}")

        if verbose:
            print("[OK] Reward model trained!")
        return losses


# ============================================================================
# PART 2: PPO - Reinforcement Learning Algorithm
# ============================================================================

class GaussianPolicy:
    """
    A Gaussian policy over response embeddings:

        pi(a|s) = N(s + theta, std^2 * I)

    theta is the only learned parameter. It is a stand-in for all the weights
    of an LLM: the prompt s shifts the mean, and theta learns the response
    direction that the reward model likes.

    Keeping std fixed makes the policy-gradient and KL math exact and easy to
    read. Making std learnable is left as an exercise.
    """

    def __init__(self, dim: int, std: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.theta = rng.standard_normal(dim) * 0.01
        self.std = float(std)
        if self.std <= 0:
            raise ValueError("std must be positive")

    def snapshot(self) -> "GaussianPolicy":
        """A frozen copy, used as the reference (SFT) policy."""
        clone = GaussianPolicy(len(self.theta), std=self.std, seed=0)
        clone.theta = self.theta.copy()
        return clone

    def mean(self, prompt_embedding: np.ndarray) -> np.ndarray:
        """Deterministic action: the mean of the Gaussian."""
        prompt_embedding = np.asarray(prompt_embedding, dtype=float)
        if prompt_embedding.shape != self.theta.shape:
            raise ValueError("prompt dimension must match the policy dimension")
        return prompt_embedding + self.theta

    def sample(self, prompt_embedding: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample a response from the policy."""
        mu = self.mean(prompt_embedding)
        return mu + rng.standard_normal(mu.shape) * self.std

    def log_prob(self, prompt_embedding: np.ndarray,
                 response: np.ndarray) -> float:
        """log pi(response | prompt) for one response."""
        z = (np.asarray(response, dtype=float) - self.mean(prompt_embedding)) / self.std
        d = z.size
        return float(-0.5 * (z @ z) - d * np.log(self.std)
                     - 0.5 * d * np.log(2 * np.pi))

    def log_prob_batch(self, prompts: np.ndarray,
                       responses: np.ndarray) -> np.ndarray:
        """log pi(response | prompt) for a batch."""
        means = prompts + self.theta[None, :]
        z = (responses - means) / self.std
        d = prompts.shape[1]
        return (-0.5 * np.sum(z * z, axis=1) - d * np.log(self.std)
                - 0.5 * d * np.log(2 * np.pi))

    def exact_kl(self, other: "GaussianPolicy") -> float:
        """
        Exact KL(pi || other), for two Gaussians with the same std.

            KL = ||theta - theta_other||^2 / (2 * std^2)

        This is the closed-form expectation. It is NOT the per-sample estimate
        k_hat = log pi(a|s) - log pi_ref(a|s) used in the shaped reward; that
        estimate has mean equal to this value but is noisy and can be negative.
        """
        if self.theta.shape != other.theta.shape:
            raise ValueError("policies must have the same dimension")
        if not np.isclose(self.std, other.std):
            raise ValueError("exact_kl() assumes both policies have the same std")
        delta = self.theta - other.theta
        return float(0.5 * (delta @ delta) / (self.std ** 2))


class ValueFunction:
    """
    V(s) = weights . s + bias

    A learned baseline for the advantage. It does not change the expected
    policy gradient, but it reduces variance: a sample that is merely "good
    for this prompt" should not push the policy as hard as a genuine surprise.
    """

    def __init__(self, dim: int, learning_rate: float = 0.05,
                 weight_decay: float = 1e-3):
        self.weights = np.zeros(dim)
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def predict_batch(self, prompts: np.ndarray) -> np.ndarray:
        return prompts @ self.weights + self.bias

    def train_step(self, prompts: np.ndarray, returns: np.ndarray) -> None:
        """One gradient step on mean squared error."""
        n = len(prompts)
        error = self.predict_batch(prompts) - returns
        self.weights -= self.learning_rate * (
            prompts.T @ error / n + self.weight_decay * self.weights)
        self.bias -= self.learning_rate * np.mean(error)


def compute_gae(rewards: np.ndarray, values: np.ndarray,
                next_values: np.ndarray, dones: np.ndarray,
                gamma: float = 1.0, lam: float = 0.95):
    """
    Generalized Advantage Estimation (GAE).

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        returns = A + V

    For a one-step episode (done = 1), this reduces to A = r - V(s).
    That is exactly the bandit case used by this repo: one response = one
    episode. The general form is here so the formula is not hidden.

    Args:
        rewards:     Rewards, shape (T,) or (N,)
        values:      V(s_t), same shape
        next_values: V(s_{t+1}), same shape (zeros for terminal states)
        dones:       1.0 if the episode ended at t, else 0.0
        gamma:       Discount factor
        lam:         GAE smoothing parameter

    Returns:
        advantages, returns
    """
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    next_values = np.asarray(next_values, dtype=float)
    dones = np.asarray(dones, dtype=float)

    if not (len(rewards) == len(values) == len(next_values) == len(dones)):
        raise ValueError("rewards, values, next_values and dones must have equal length")

    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae

    return advantages, advantages + values


@dataclass
class Rollout:
    """One batch of on-policy experience, plus the values computed at sampling time."""
    prompts: np.ndarray
    responses: np.ndarray
    old_log_probs: np.ndarray
    ref_log_probs: np.ndarray
    raw_rewards: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray

    @property
    def sample_kl(self) -> np.ndarray:
        """
        Per-sample KL estimate: log pi(a|s) - log pi_ref(a|s).

        E_{a ~ pi}[sample_kl] = exact KL(pi || pi_ref). This is the quantity
        used in the shaped reward; the exact KL is for diagnostics.
        """
        return self.old_log_probs - self.ref_log_probs


class SimplifiedPPO:
    """
    Simplified PPO (Proximal Policy Optimization).

    The RL algorithm that aligns a policy to a reward model.
    Used in ChatGPT training.

    The underlying gradient is REINFORCE with a baseline. PPO uses that score
    function inside the old-policy importance ratio, then clips it, so one
    rollout can be reused for several gradient steps. Without the ratio and
    clipping, this would be REINFORCE, not PPO.

    This version keeps the important math:

    1. A learned value baseline and GAE for the advantage.
    2. The score-function gradient grad log pi, NOT the prompt.
    3. The old-policy ratio and PPO's clipped surrogate objective.
    4. A frozen reference policy and a sample-KL reward penalty.
    5. The exact Gaussian KL for diagnostics and the closed-form optimum.

    It is still a toy: the "LLM" is a single vector theta, and one response is
    one one-step episode.
    """

    def __init__(self, policy_dim: int, std: float = 1.0, beta: float = 0.5,
                 clip: float = 0.2, policy_lr: float = 0.02,
                 value_lr: float = 0.05, value_epochs: int = 2, seed: int = 0):
        """
        Args:
            policy_dim: Dimension of the response embedding
            std: Exploration scale of the Gaussian policy
            beta: KL penalty coefficient (0 = no anchor, larger = stay closer)
            clip: PPO clipping range epsilon
            policy_lr: Learning rate for theta
            value_lr: Learning rate for the value function
            value_epochs: Value regression steps per PPO minibatch
            seed: RNG seed for the initial policy
        """
        self.actor = GaussianPolicy(policy_dim, std=std, seed=seed)
        self.reference = self.actor.snapshot()
        self.critic = ValueFunction(policy_dim, learning_rate=value_lr)
        self.beta = beta
        self.clip = clip
        self.policy_lr = policy_lr
        self.value_epochs = value_epochs

    def collect(self, prompts: np.ndarray, reward_model: RewardModel,
                rng: np.random.Generator, samples_per_prompt: int = 16) -> Rollout:
        """
        Sample responses, score them, and compute advantages.

        Each (prompt, response) is a one-step episode, so:
            shaped_reward = r_RM - beta * (log pi - log pi_ref)
            advantage     = shaped_reward - V(prompt)
        """
        prompts = np.asarray(prompts, dtype=float)
        if prompts.ndim != 2:
            raise ValueError("prompts must be a 2D array of shape (n_prompts, dim)")
        if prompts.shape[1] != len(self.actor.theta):
            raise ValueError("prompt dimension must match the policy dimension")
        if samples_per_prompt < 1:
            raise ValueError("samples_per_prompt must be at least 1")

        # Repeat each prompt so every prompt contributes several samples.
        batch_prompts = np.repeat(prompts, samples_per_prompt, axis=0)
        means = batch_prompts + self.actor.theta[None, :]
        responses = means + rng.standard_normal(means.shape) * self.actor.std

        # PPO needs log pi from the policy that generated the batch:
        # rho = exp(log pi_new - log pi_old) in update().
        old_log_probs = self.actor.log_prob_batch(batch_prompts, responses)
        ref_log_probs = self.reference.log_prob_batch(batch_prompts, responses)
        raw_rewards = reward_model.predict_reward_batch(responses)

        # Use the *sample* KL estimate, not the exact Gaussian KL. The exact KL
        # is constant with respect to the sampled action, so it would vanish
        # from the score-function gradient; the sample KL is action-dependent,
        # and E[(log pi - log pi_ref) grad log pi] = grad KL.
        sample_kl = old_log_probs - ref_log_probs
        shaped_rewards = raw_rewards - self.beta * sample_kl

        values = self.critic.predict_batch(batch_prompts)
        advantages, returns = compute_gae(
            shaped_rewards,
            values,
            next_values=np.zeros_like(values),
            dones=np.ones_like(values),  # each response is a terminal one-step episode
        )

        return Rollout(batch_prompts, responses, old_log_probs, ref_log_probs,
                       raw_rewards, values, advantages, returns)

    def clipped_surrogate_gradient(self, prompts: np.ndarray, responses: np.ndarray,
                                   old_log_probs: np.ndarray,
                                   advantages: np.ndarray) -> np.ndarray:
        """
        Gradient of the PPO clipped surrogate with respect to theta.

        The underlying policy gradient is REINFORCE with a baseline. The ratio
        and clipping below are the PPO part: they let us reuse one batch for
        several gradient steps without moving the policy too far.

        For each sample:
            rho = exp(log pi_new - log pi_old)
            objective = min(rho * A, clip(rho, 1-eps, 1+eps) * A)

        The clipped branch is constant, so its gradient is zero. On the
        unclipped branch:
            d(rho * A)/dtheta = rho * A * (a - s - theta) / std^2

        The last factor is the score function. Using the prompt instead is a
        different (biased) estimator; see the README for a counterexample.
        """
        new_log_probs = self.actor.log_prob_batch(prompts, responses)
        # Guard the exponent against overflow; the PPO trust-region clip is below.
        ratio = np.exp(np.clip(new_log_probs - old_log_probs, -20.0, 20.0))

        unclipped = ratio * advantages
        clipped = np.clip(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages
        use_unclipped = unclipped <= clipped  # min() picks the unclipped branch

        score = (responses - (prompts + self.actor.theta[None, :])) / self.actor.std ** 2
        return np.mean((use_unclipped * advantages * ratio)[:, None] * score, axis=0)

    def update(self, rollout: Rollout, rng: np.random.Generator,
               epochs: int = 4, minibatch_size: int = 32) -> None:
        """
        Run several PPO epochs on one rollout.

        The old log-probs in the rollout are fixed, so the ratio moves as theta
        changes; the clip bounds each minibatch update.
        """
        advantages = rollout.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(advantages)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, minibatch_size):
                idx = order[start:start + minibatch_size]

                grad = self.clipped_surrogate_gradient(
                    rollout.prompts[idx],
                    rollout.responses[idx],
                    rollout.old_log_probs[idx],
                    advantages[idx],
                )
                self.actor.theta += self.policy_lr * grad

                for _ in range(self.value_epochs):
                    self.critic.train_step(rollout.prompts[idx], rollout.returns[idx])

    def evaluate(self, prompts: np.ndarray, reward_model: RewardModel) -> float:
        """
        Deterministic evaluation: use the policy mean, not a noisy sample.
        This makes before/after comparisons much less noisy.
        """
        prompts = np.asarray(prompts, dtype=float)
        if prompts.ndim != 2 or prompts.shape[1] != len(self.actor.theta):
            raise ValueError("prompts must have shape (n_prompts, policy_dim)")
        responses = prompts + self.actor.theta[None, :]
        return float(np.mean(reward_model.predict_reward_batch(responses)))

    def exact_kl_to_reference(self) -> float:
        """Exact KL(pi || pi_ref), in nats. Diagnostic, not the reward term."""
        return self.actor.exact_kl(self.reference)


# ============================================================================
# PART 3: RLHF PIPELINE - Putting It Together
# ============================================================================

class MicroRLHF:
    """
    Complete RLHF Pipeline (Simplified)

    The 3 steps of RLHF:
    1. Supervised Fine-Tuning (SFT) - not implemented (see nanochat)
    2. Reward Model Training - implemented
    3. RL Fine-tuning with PPO - implemented

    NOTE: This assumes you have a pretrained model (from nanochat/nanoGPT).
    This teaches the ALIGNMENT phase only.
    """

    def __init__(self, embedding_dim: int = 128, seed: int = 0,
                 std: float = 1.0, beta: float = 0.5):
        self.embedding_dim = embedding_dim
        self.rng = np.random.default_rng(seed)
        self.reward_model = RewardModel(embedding_dim, seed=seed)
        self.policy = SimplifiedPPO(embedding_dim, std=std, beta=beta,
                                    seed=seed + 1)

    def train_reward_model(self, preferences: List[Tuple[np.ndarray, np.ndarray]],
                           learning_rate: float = 0.1, epochs: int = 50,
                           verbose: bool = True) -> List[float]:
        """
        Step 2 of RLHF: Train reward model from human preferences.

        Args:
            preferences: List of (chosen_response, rejected_response) pairs
            learning_rate: Learning rate
            epochs: Training epochs
            verbose: Print progress

        Returns:
            losses: Loss at the start of each epoch
        """
        if verbose:
            print("\n[Step 2] Training Reward Model...")
        return self.reward_model.train_on_preferences(
            preferences, learning_rate=learning_rate, epochs=epochs,
            verbose=verbose)

    def align_policy(self, prompts: np.ndarray, n_iterations: int = 20,
                     eval_prompts: np.ndarray = None,
                     samples_per_prompt: int = 16,
                     verbose: bool = True) -> List[float]:
        """
        Step 3 of RLHF: Align policy using RL.

        Args:
            prompts: Training prompt embeddings, shape (n_prompts, dim)
            n_iterations: Number of RL training iterations
            eval_prompts: Held-out prompts for measuring progress. If None,
                the training prompts are used (deterministic policy mean).
            samples_per_prompt: Responses sampled per prompt per iteration
            verbose: Print progress

        Returns:
            rewards_history: Deterministic mean reward on eval_prompts after
                each iteration. This is much less noisy than the raw training
                rewards, so the learning curve actually means something.
        """
        prompts = np.asarray(prompts, dtype=float)
        if eval_prompts is None:
            eval_prompts = prompts
        else:
            eval_prompts = np.asarray(eval_prompts, dtype=float)

        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive")

        if verbose:
            print(f"\n[Step 3] Aligning Policy with {n_iterations} iterations...")

        rewards_history = []
        for iteration in range(n_iterations):
            rollout = self.policy.collect(
                prompts, self.reward_model, self.rng,
                samples_per_prompt=samples_per_prompt)
            self.policy.update(rollout, self.rng)

            avg_reward = self.policy.evaluate(eval_prompts, self.reward_model)
            rewards_history.append(avg_reward)

            if verbose and (iteration + 1) % 5 == 0:
                exact_kl = self.policy.exact_kl_to_reference()
                print(f"  Iteration {iteration + 1:3d}/{n_iterations}, "
                      f"Eval Reward: {avg_reward:.4f}, Exact KL: {exact_kl:.4f}")

        if verbose:
            print(f"[OK] Policy aligned! Final eval reward: {rewards_history[-1]:.4f}")

        return rewards_history

    def evaluate(self, prompts: np.ndarray) -> float:
        """Deterministic mean reward of the current policy on prompts."""
        return self.policy.evaluate(np.asarray(prompts, dtype=float),
                                    self.reward_model)

    def exact_kl_to_reference(self) -> float:
        """Exact KL(current policy || reference policy), in nats."""
        return self.policy.exact_kl_to_reference()

    def generate_aligned_response(self, prompt_embedding: np.ndarray,
                                  deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Generate a response with the aligned policy.

        Args:
            prompt_embedding: Prompt representation
            deterministic: If True, use the policy mean (no sampling noise)

        Returns:
            (response, reward): Generated response and its predicted reward
        """
        prompt_embedding = np.asarray(prompt_embedding, dtype=float)
        if deterministic:
            response = self.policy.actor.mean(prompt_embedding)
        else:
            response = self.policy.actor.sample(prompt_embedding, self.rng)

        reward = self.reward_model.predict_reward(response)
        return response, reward


# ============================================================================
# SUMMARY: The RLHF Process
# ============================================================================

"""
RLHF in 3 Steps:

1. SFT (Supervised Fine-Tuning)
   -> Train base model to follow instructions
   -> Uses examples: (prompt, good_response)
   -> See nanochat/nanoGPT for implementation

2. Reward Model (This repo - Part 1)
   -> Train model to predict human preferences
   -> Uses comparisons: response_A > response_B
   -> Implemented above

3. RL Fine-tuning (This repo - Part 2)
   -> Use PPO to maximize reward
   -> Policy learns to generate high-reward responses
   -> Implemented above

The three RL details that matter most:
  * Advantage = return - value baseline (not raw reward)
  * Policy gradient uses grad log pi = (a - mean) / std^2 (not the prompt)
  * KL penalty keeps the policy near the SFT reference (not reward hacking)

Result: Model that's helpful, harmless, honest (like ChatGPT)
"""

# That's RLHF. The alignment magic that makes ChatGPT behave well.
