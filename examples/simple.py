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

    rng = np.random.default_rng(42)
    embedding_dim = 64

    # Create RLHF system. Each component gets its own RNG stream.
    rlhf = MicroRLHF(embedding_dim=embedding_dim, seed=0)

    print("\n[Step 1] Supervised Fine-Tuning")
    print("(Not implemented - see nanochat/nanoGPT)")
    print("This trains a base model to follow instructions")

    # Step 2: Train Reward Model
    print("\n[Step 2] Training Reward Model on Human Preferences...")
    print("-" * 60)

    # Simulated preference data.
    # In production: these come from human labelers.
    preferences = []
    for _ in range(20):
        # Simulate: helpful response vs unhelpful response
        helpful = rng.standard_normal(embedding_dim) + 1.0
        unhelpful = rng.standard_normal(embedding_dim) - 0.5
        preferences.append((helpful, unhelpful))

    losses = rlhf.train_reward_model(preferences)

    accuracy = np.mean([
        rlhf.reward_model.predict_reward(chosen) >
        rlhf.reward_model.predict_reward(rejected)
        for chosen, rejected in preferences
    ])
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  Preference accuracy: {accuracy:.0%}")

    # Step 3: Align Policy
    print("\n[Step 3] Aligning Policy with RL...")
    print("-" * 60)

    train_prompts = rng.standard_normal((8, embedding_dim))
    eval_prompts = rng.standard_normal((64, embedding_dim))

    before = rlhf.evaluate(eval_prompts)
    rewards_history = rlhf.align_policy(
        train_prompts, n_iterations=20, eval_prompts=eval_prompts)
    after = rlhf.evaluate(eval_prompts)

    # Show results
    print("\n[Results] Policy Alignment Progress:")
    print(f"  Eval reward before: {before:+.4f}")
    print(f"  Eval reward after:  {after:+.4f}")
    print(f"  Improvement:        {after - before:+.4f}")
    print(f"  Exact KL from ref: {rlhf.exact_kl_to_reference():.4f} nats")

    # Generate with aligned policy
    print("\n[Step 4] Generating with Aligned Policy...")
    print("-" * 60)

    test_prompt = rng.standard_normal(embedding_dim)
    _, sampled_reward = rlhf.generate_aligned_response(test_prompt)
    print(f"Sampled response reward: {sampled_reward:+.4f}")
    print("(Single-prompt reward varies; the 64-prompt average above is the")
    print(" stable before/after measure.)")
    print("[OK] Higher reward = more aligned to human preferences")

    print("\n" + "=" * 60)
    print("[DONE] RLHF demo complete!")
    print("\nKey insight: RLHF optimizes for human preferences,")
    print("not just next-token prediction. This makes ChatGPT helpful!")


if __name__ == "__main__":
    main()
