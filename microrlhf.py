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

    def __init__(self, input_dim: int = 128):
        """
        Args:
            input_dim: Dimension of response representation
        """
        # In production: This would be a neural network
        # For education: Simple linear model
        np.random.seed(42)
        self.weights = np.random.randn(input_dim) * 0.01
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
                loss = -np.log(1 / (1 + np.exp(-diff)))  # Negative log likelihood

                # Gradient (simplified)
                # In production: Use automatic differentiation
                sigmoid_diff = 1 / (1 + np.exp(-diff))
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

    This is a simplified educational version.
    """

    def __init__(self, policy_dim: int = 128):
        """
        Args:
            policy_dim: Dimension of policy representation
        """
        # Policy parameters (in production: the LLM itself)
        np.random.seed(42)
        self.policy_weights = np.random.randn(policy_dim) * 0.01

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
        noise = np.random.randn(len(prompt_embedding)) * 0.1
        response = prompt_embedding + noise + self.policy_weights[:len(prompt_embedding)]

        return response

    def compute_advantage(self, reward: float, baseline: float = 0.0) -> float:
        """
        Compute advantage: How much better than expected?

        Args:
            reward: Reward from reward model
            baseline: Expected reward

        Returns:
            advantage: reward - baseline
        """
        return reward - baseline

    def update_policy(self, prompt_embedding: np.ndarray, reward: float,
                     learning_rate: float = 0.01):
        """
        Update policy to maximize reward (simplified)

        Args:
            prompt_embedding: Prompt embedding
            reward: Reward from reward model
            learning_rate: Learning rate
        """
        # Simplified gradient ascent (maximize reward)
        # In production: PPO clip, value function, etc.

        advantage = self.compute_advantage(reward)

        # Update policy in direction of higher reward
        gradient = advantage * prompt_embedding[:len(self.policy_weights)]
        self.policy_weights += learning_rate * gradient

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

    def __init__(self, embedding_dim: int = 128):
        self.reward_model = RewardModel(embedding_dim)
        self.policy = SimplifiedPPO(embedding_dim)
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
            iteration_rewards = []

            for prompt in prompts:
                # Sample response from current policy
                response = self.policy.sample_response(prompt)

                # Get reward from reward model
                reward = self.reward_model.predict_reward(response)

                # Update policy to maximize reward
                self.policy.update_policy(prompt, reward, learning_rate=0.01)

                iteration_rewards.append(reward)

            avg_reward = np.mean(iteration_rewards)
            rewards_history.append(avg_reward)

            if (iteration + 1) % 5 == 0:
                print(f"  Iteration {iteration + 1}/{n_iterations}, Avg Reward: {avg_reward:.4f}")

        print(f"[OK] Policy aligned! Final avg reward: {rewards_history[-1]:.4f}")

        return rewards_history

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
