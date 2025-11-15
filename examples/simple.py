"""
Simple RLHF example
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from microrlhf import MicroRLHF

def main():
    print("micro-rlhf: Simple Example")
    print("=" * 60)

    # Create RLHF system
    rlhf = MicroRLHF(embedding_dim=64)

    print("\n[Step 1] Supervised Fine-Tuning")
    print("(Not implemented - see nanochat/nanoGPT)")
    print("This trains base model to follow instructions")

    # Step 2: Train Reward Model
    print("\n[Step 2] Training Reward Model on Human Preferences...")
    print("-" * 60)

    # Simulated preference data
    # In production: These come from human labelers
    np.random.seed(42)
    preferences = []

    for i in range(20):
        # Simulate: helpful response vs unhelpful response
        helpful = np.random.randn(64) + np.array([1.0] * 64)  # Shifted positive
        unhelpful = np.random.randn(64) - np.array([0.5] * 64)  # Shifted negative

        preferences.append((helpful, unhelpful))

    rlhf.train_reward_model(preferences)

    # Step 3: Align Policy
    print("\n[Step 3] Aligning Policy with RL...")
    print("-" * 60)

    # Simulated prompts
    prompts = [np.random.randn(64) for _ in range(5)]

    rewards_history = rlhf.align_policy(prompts, n_iterations=20)

    # Show results
    print("\n[Results] Policy Alignment Progress:")
    print(f"  Initial avg reward: {rewards_history[0]:.4f}")
    print(f"  Final avg reward: {rewards_history[-1]:.4f}")
    print(f"  Improvement: {rewards_history[-1] - rewards_history[0]:+.4f}")

    # Generate with aligned policy
    print("\n[Step 4] Generating with Aligned Policy...")
    print("-" * 60)

    test_prompt = np.random.randn(64)
    response, reward = rlhf.generate_aligned_response(test_prompt)

    print(f"Generated response reward: {reward:.4f}")
    print("[OK] Higher reward = more aligned to human preferences")

    print("\n" + "=" * 60)
    print("[DONE] RLHF demo complete!")
    print("\nKey insight: RLHF optimizes for human preferences,")
    print("not just next-token prediction. This makes ChatGPT helpful!")

if __name__ == "__main__":
    main()
