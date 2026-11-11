"""
Visualization tools for micro-rlhf

Make alignment and RLHF intuitive and beautiful.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def plot_rlhf_pipeline(figsize=(14, 8)):
    """
    Visualize the 3-step RLHF process
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Step boxes
    steps = [
        ("Step 1:\nSupervised\nFine-Tuning (SFT)", 0.15, "See nanochat/nanoGPT", '#3498db'),
        ("Step 2:\nReward Model\nTraining", 0.45, "Implemented here", '#e74c3c'),
        ("Step 3:\nRL Fine-tuning\n(PPO)", 0.75, "Implemented here", '#2ecc71')
    ]

    for i, (title, x, note, color) in enumerate(steps):
        # Draw box
        box = plt.Rectangle((x-0.12, 0.4), 0.24, 0.2,
                           facecolor=color, edgecolor='black',
                           linewidth=2, alpha=0.7)
        ax.add_patch(box)

        # Title
        ax.text(x, 0.5, title, ha='center', va='center',
               fontsize=11, weight='bold', color='white')

        # Note
        ax.text(x, 0.25, note, ha='center', va='top',
               fontsize=9, style='italic')

        # Arrow to next step
        if i < len(steps) - 1:
            ax.arrow(x + 0.12, 0.5, 0.09, 0,
                    head_width=0.03, head_length=0.02,
                    fc='gray', ec='gray', linewidth=2)

    # Labels
    ax.text(0.15, 0.7, "Base Model\n→\nInstruction-Following", ha='center', fontsize=9)
    ax.text(0.45, 0.7, "Learn Human\nPreferences", ha='center', fontsize=9)
    ax.text(0.75, 0.7, "Optimize for\nReward", ha='center', fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)
    ax.axis('off')
    ax.set_title('RLHF: The 3-Step Process', fontsize=16, weight='bold', pad=20)

    plt.tight_layout()
    return fig

def plot_reward_training(losses: List[float], figsize=(10, 6)):
    """
    Visualize reward model training progress

    Args:
        losses: Training loss at the start of each epoch
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, linewidth=2, color='#e74c3c', marker='o')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Reward Model Training: Learning Human Preferences', fontsize=14, weight='bold')
    ax.grid(alpha=0.3)

    # Annotate start and end
    ax.annotate(f'Start: {losses[0]:.3f}',
               xy=(1, losses[0]), xytext=(10, 20),
               textcoords='offset points',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', lw=1.5))

    ax.annotate(f'End: {losses[-1]:.3f}',
               xy=(len(losses), losses[-1]), xytext=(10, -20),
               textcoords='offset points',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
               arrowprops=dict(arrowstyle='->', lw=1.5))

    plt.tight_layout()
    return fig

def plot_alignment_progress(rewards: List[float], figsize=(12, 6)):
    """
    Visualize RL alignment progress

    Args:
        rewards: Deterministic eval rewards over iterations
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    iterations = range(1, len(rewards) + 1)
    ax.plot(iterations, rewards, linewidth=2, color='#2ecc71', marker='o', markersize=6)

    ax.fill_between(iterations, rewards, alpha=0.3, color='#2ecc71')

    ax.set_xlabel('RL Iteration', fontsize=12)
    ax.set_ylabel('Deterministic Eval Reward', fontsize=12)
    ax.set_title('Policy Alignment: Increasing Reward Over Time', fontsize=14, weight='bold')
    ax.grid(alpha=0.3)

    # Show improvement (guard against a zero starting reward)
    improvement = rewards[-1] - rewards[0]
    start = abs(rewards[0]) if abs(rewards[0]) > 1e-9 else 1.0
    ax.text(len(rewards) * 0.7, max(rewards) * 0.9,
           f'Improvement: {improvement:+.3f}\n({improvement/start*100:.1f}%)',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
           fontsize=11, weight='bold')

    plt.tight_layout()
    return fig

def plot_preference_comparison(chosen: np.ndarray, rejected: np.ndarray,
                              chosen_reward: float, rejected_reward: float,
                              figsize=(12, 5)):
    """
    Visualize preference pair and their rewards

    Args:
        chosen: Chosen response embedding
        rejected: Rejected response embedding
        chosen_reward: Reward for chosen
        rejected_reward: Reward for rejected
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Show embeddings
    dims = range(min(20, len(chosen)))  # Show first 20 dims

    ax1.plot(dims, chosen[:20], label='Chosen Response', linewidth=2,
            color='#2ecc71', marker='o', markersize=4)
    ax1.plot(dims, rejected[:20], label='Rejected Response', linewidth=2,
            color='#e74c3c', marker='x', markersize=4)

    ax1.set_xlabel('Embedding Dimension', fontsize=11)
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('Response Embeddings', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Show rewards
    responses = ['Chosen', 'Rejected']
    rewards = [chosen_reward, rejected_reward]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax2.bar(responses, rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.3f}',
                ha='center', va='bottom' if reward > 0 else 'top',
                fontsize=12, weight='bold')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Reward', fontsize=11)
    ax2.set_title('Predicted Rewards', fontsize=12, weight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add annotation
    if chosen_reward > rejected_reward:
        result = "[OK] Reward model prefers chosen!"
        color = 'green'
    else:
        result = "[ERROR] Reward model prefers rejected!"
        color = 'red'

    ax2.text(0.5, max(rewards) * 0.8, result,
            ha='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    plt.suptitle('Preference Learning: Chosen vs Rejected', fontsize=14, weight='bold')
    plt.tight_layout()
    return fig

def plot_before_after_alignment(rewards_before: List[float],
                                rewards_after: List[float],
                                figsize=(12, 6)):
    """
    Compare model before and after RLHF alignment

    Args:
        rewards_before: Rewards before alignment
        rewards_after: Rewards after alignment
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Distribution before
    ax1.hist(rewards_before, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(rewards_before), color='darkred', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(rewards_before):.3f}')
    ax1.set_xlabel('Reward', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Before RLHF\n(Base Model)', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Distribution after
    ax2.hist(rewards_after, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(rewards_after), color='darkgreen', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(rewards_after):.3f}')
    ax2.set_xlabel('Reward', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('After RLHF\n(Aligned Model)', fontsize=12, weight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Overall improvement
    improvement = np.mean(rewards_after) - np.mean(rewards_before)
    plt.suptitle(f'RLHF Impact: Reward Improvement = {improvement:+.3f}',
                fontsize=14, weight='bold', color='green' if improvement > 0 else 'red')

    plt.tight_layout()
    return fig
