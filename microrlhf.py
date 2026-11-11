"""
micro-rlhf: RLHF in ~150 lines

Understand how ChatGPT is aligned by building RLHF from scratch.

RLHF = Reinforcement Learning from Human Feedback
How models learn to be helpful, harmless, and honest.

NOTE: For training foundations (backprop, gradient descent), see:
- Karpathy's micrograd: https://github.com/karpathy/micrograd
- Karpathy's makemore: https://github.com/karpathy/makemore

This focuses on the ALIGNMENT phase - what makes ChatGPT different from base GPT.

Inspired by Andrej Karpathy's educational approach
"""

import numpy as np
from typing import List, Tuple

# ============================================================================
# PART 1: REWARD MODEL - Learning Human Preferences
# ============================================================================

class RewardModel:
    """
    Reward Model: Predicts how good a response is

    Trained on human preferences:
    "Response A is better than Response B"

    This is step 1 of RLHF.
    """

    def __init__(self, input_dim: int = 128, seed: int = 0):
        """
        Args:
            input_dim: Dimension of response representation
            seed: RNG seed (no global np.random.seed)
        """
        # In production: This would be a neural network
        # For education: Simple linear model
        rng = np.random.default_rng(seed)
        self.weights = rng.standard_normal(input_dim) * 0.01
        # A shared bias cancels in r_better - r_worse, so it is not learned.
        self.bias = 0.0

    def predict_reward(self, response_embedding: np.ndarray) -> float:
        """
        Predict reward for a response

        Args:
            response_embedding: Vector representation of response

        Returns:
            reward: Scalar reward value (higher = better)
        """
        reward = np.dot(self.weights, response_embedding) + self.bias
        return reward

    def train_on_preferences(self, preferences: List[Tuple[np.ndarray, np.ndarray]],
                            learning_rate: float = 0.01, epochs: int = 10):
        """
        Train reward model from preferences

        Args:
            preferences: List of (better_response, worse_response) pairs
            learning_rate: Learning rate
            epochs: Training epochs
        """
        print(f"[OK] Training reward model on {len(preferences)} preferences...")

        for epoch in range(epochs):
            total_loss = 0

            for better, worse in preferences:
                # Predict rewards
                reward_better = self.predict_reward(better)
                reward_worse = self.predict_reward(worse)

                # Loss: Reward(better) should be > Reward(worse)
                # Using logistic loss: -log(sigmoid(r_better - r_worse))
                diff = reward_better - reward_worse
                loss = np.logaddexp(0.0, -diff)  # softplus(-diff), stable

                # Gradient (simplified)
                # In production: Use automatic differentiation
                sigmoid_diff = np.exp(-np.logaddexp(0.0, -diff))
                gradient = (sigmoid_diff - 1) * (better - worse)

                # Update weights
                self.weights -= learning_rate * gradient

                total_loss += loss

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

        print(f"[OK] Reward model trained!")

# ============================================================================
# PART 2: PPO - Reinforcement Learning Algorithm
# ============================================================================

class SimplifiedPPO:
    """
    Simplified PPO (Proximal Policy Optimization)

    The RL algorithm that aligns models to reward model.
    Used in ChatGPT training.

    This is a simplified educational version. It implements the policy-gradient
    core (REINFORCE with a baseline). PPO's ratio clipping matters when a batch
    is reused for several gradient steps; it is a production detail.
    """

    def __init__(self, policy_dim: int = 128, std: float = 1.0, seed: int = 1):
        """
        Args:
            policy_dim: Dimension of policy representation
            std: Exploration scale of the Gaussian policy
            seed: RNG seed (no global np.random.seed)
        """
        # Policy parameters (in production: the LLM itself)
        self.rng = np.random.default_rng(seed)
        self.policy_weights = self.rng.standard_normal(policy_dim) * 0.01
        self.std = std
        if self.std <= 0:
            raise ValueError("std must be positive")

    def sample_response(self, prompt_embedding: np.ndarray) -> np.ndarray:
        """
        Sample response from current policy

        Args:
            prompt_embedding: Vector representation of prompt

        Returns:
            response_embedding: Sampled response
        """
        # In production: LLM generates tokens
        # For education: Add noise to prompt
        noise = self.rng.standard_normal(len(prompt_embedding)) * self.std
        response = prompt_embedding + noise + self.policy_weights[:len(prompt_embedding)]

        return response

    def compute_advantage(self, reward: float, baseline: float) -> float:
        """
        Compute advantage: How much better than expected?

        Args:
            reward: Reward from reward model
            baseline: Expected reward (e.g. the batch mean)

        Returns:
            advantage: reward - baseline
        """
        return reward - baseline

    def update_policy(self, prompt_embedding: np.ndarray, response: np.ndarray,
                     advantage: float, learning_rate: float = 0.01):
        """
        Update policy to maximize reward (simplified)

        Args:
            prompt_embedding: Prompt embedding
            response: Sampled response
            advantage: reward - baseline
            learning_rate: Learning rate
        """
        # Policy gradient for pi(a|s) = N(s + theta, std^2 I):
        # grad log pi(a|s) = (a - mean) / std^2
        # The prompt is the state, not the gradient.
        mean = prompt_embedding + self.policy_weights[:len(prompt_embedding)]
        score = (response - mean) / self.std ** 2
        self.policy_weights += learning_rate * advantage * score

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

    def __init__(self, embedding_dim: int = 128, seed: int = 0, std: float = 1.0):
        # Independent seeds: policy and reward model must not start identical.
        self.reward_model = RewardModel(embedding_dim, seed=seed)
        self.policy = SimplifiedPPO(embedding_dim, std=std, seed=seed + 1)
        self.embedding_dim = embedding_dim

    def train_reward_model(self, preferences: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Step 2 of RLHF: Train reward model from human preferences

        Args:
            preferences: List of (chosen_response, rejected_response) pairs
        """
        print("\n[Step 2] Training Reward Model...")
        self.reward_model.train_on_preferences(preferences)

    def align_policy(self, prompts: List[np.ndarray], n_iterations: int = 20):
        """
        Step 3 of RLHF: Align policy using RL

        Args:
            prompts: List of prompt embeddings
            n_iterations: Number of RL training iterations
        """
        print(f"\n[Step 3] Aligning Policy with {n_iterations} iterations...")

        rewards_history = []

        for iteration in range(n_iterations):
            for prompt in prompts:
                # Baseline: what the reward model expects from the policy mean.
                # Exact here because the reward model is linear. Production
                # RLHF learns a value function V(s) instead.
                mean_response = prompt + self.policy.policy_weights[:len(prompt)]
                baseline = self.reward_model.predict_reward(mean_response)

                # Sample response from current policy
                response = self.policy.sample_response(prompt)

                # Get reward from reward model
                reward = self.reward_model.predict_reward(response)

                # Advantage = how much better than expected
                advantage = self.policy.compute_advantage(reward, baseline)

                # Update policy with the score-function gradient
                self.policy.update_policy(
                    prompt, response, advantage, learning_rate=0.01)

            avg_reward = self.evaluate(prompts)
            rewards_history.append(avg_reward)

            if (iteration + 1) % 5 == 0:
                print(f"  Iteration {iteration + 1}/{n_iterations}, Mean Reward: {avg_reward:.4f}")

        print(f"[OK] Policy aligned! Final mean reward: {rewards_history[-1]:.4f}")

        return rewards_history

    def evaluate(self, prompts: List[np.ndarray]) -> float:
        """
        Deterministic mean reward for a set of prompts.

        Uses the policy mean (no sampling noise), so before/after comparisons
        and the learning curve are much easier to read.
        """
        rewards = []
        for prompt in prompts:
            response = prompt + self.policy.policy_weights[:len(prompt)]
            rewards.append(self.reward_model.predict_reward(response))
        return float(np.mean(rewards))

    def generate_aligned_response(self, prompt_embedding: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Generate response with aligned policy

        Args:
            prompt_embedding: Prompt representation

        Returns:
            (response, reward): Generated response and its reward
        """
        response = self.policy.sample_response(prompt_embedding)
        reward = self.reward_model.predict_reward(response)

        return response, reward

# ============================================================================
# SUMMARY: The RLHF Process
# ============================================================================

"""
RLHF in 3 Steps:

1. SFT (Supervised Fine-Tuning)
   → Train base model to follow instructions
   → Uses examples: (prompt, good_response)
   → See nanochat/nanoGPT for implementation

2. Reward Model (This repo - Part 1)
   → Train model to predict human preferences
   → Uses comparisons: response_A > response_B
   → Implemented above

3. RL Fine-tuning (This repo - Part 2)
   → Use PPO to maximize reward
   → Policy learns to generate high-reward responses
   → Implemented above

Result: Model that's helpful, harmless, honest (like ChatGPT)
"""

# That's RLHF. The alignment magic that makes ChatGPT behave well.
