"""
Math tests for micro-rlhf.

Run from the repo root:
    python -m unittest discover -s tests -v

These tests check the implementation against independent oracles:
finite differences, brute-force GAE, and Monte Carlo KL estimates.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microrlhf import (GaussianPolicy, MicroRLHF, RewardModel,  # noqa: E402
                       SimplifiedPPO, compute_gae)


def make_preferences(rng, n=20, dim=16):
    """Linearly separable synthetic preferences: chosen ~ +1, rejected ~ -0.5."""
    return [(rng.standard_normal(dim) + 1.0,
             rng.standard_normal(dim) - 0.5) for _ in range(n)]


class TestRewardModel(unittest.TestCase):

    def test_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(0)
        model = RewardModel(5, seed=1, weight_decay=1e-3)
        base = rng.standard_normal(5) * 0.2
        model.weights = base.copy()

        chosen = rng.standard_normal((4, 5))
        rejected = rng.standard_normal((4, 5))
        _, grad = model.loss_and_grad(chosen, rejected)

        eps = 1e-6
        for i in range(5):
            plus = base.copy()
            plus[i] += eps
            model.weights = plus
            loss_plus, _ = model.loss_and_grad(chosen, rejected)

            minus = base.copy()
            minus[i] -= eps
            model.weights = minus
            loss_minus, _ = model.loss_and_grad(chosen, rejected)

            numeric = (loss_plus - loss_minus) / (2 * eps)
            self.assertAlmostEqual(grad[i], numeric, places=5)

        model.weights = base

    def test_loss_is_stable_at_extreme_margins(self):
        """logaddexp(0, -diff) must stay finite where log1p(exp(-diff)) overflows."""
        model = RewardModel(2, seed=0)
        model.weights = np.array([1.0, 0.0])
        chosen = np.array([[1000.0, 0.0]])
        rejected = np.array([[-1000.0, 0.0]])

        loss, grad = model.loss_and_grad(chosen, rejected)

        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_learns_preferences(self):
        rng = np.random.default_rng(1)
        preferences = make_preferences(rng, n=20, dim=16)
        model = RewardModel(16, seed=0)

        losses = model.train_on_preferences(
            preferences, learning_rate=0.1, epochs=50, verbose=False)

        self.assertLess(losses[-1], losses[0])
        accuracy = np.mean([
            model.predict_reward(chosen) > model.predict_reward(rejected)
            for chosen, rejected in preferences
        ])
        self.assertGreaterEqual(accuracy, 0.95)


class TestGAE(unittest.TestCase):

    def test_one_step_equals_reward_minus_value(self):
        rewards = np.array([1.0, -2.0, 3.0])
        values = np.array([0.5, 0.0, 1.0])

        advantages, returns = compute_gae(
            rewards, values,
            next_values=np.zeros_like(values),
            dones=np.ones_like(values),
        )

        np.testing.assert_allclose(advantages, rewards - values)
        np.testing.assert_allclose(returns, rewards)

    def test_matches_brute_force_multistep(self):
        rng = np.random.default_rng(2)
        T = 6
        rewards = rng.standard_normal(T)
        values = rng.standard_normal(T)
        next_values = rng.standard_normal(T)
        dones = np.array([0, 0, 0, 1, 0, 1], dtype=float)
        gamma, lam = 0.9, 0.8

        advantages, returns = compute_gae(
            rewards, values, next_values, dones, gamma=gamma, lam=lam)

        expected = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
            running = delta + gamma * lam * nonterminal * running
            expected[t] = running

        np.testing.assert_allclose(advantages, expected)
        np.testing.assert_allclose(returns, expected + values)


class TestPolicyMath(unittest.TestCase):

    def test_log_prob_matches_manual_gaussian(self):
        rng = np.random.default_rng(3)
        policy = GaussianPolicy(4, std=0.7, seed=0)
        policy.theta = rng.standard_normal(4) * 0.1
        prompt = rng.standard_normal(4)
        response = rng.standard_normal(4)

        z = (response - (prompt + policy.theta)) / policy.std
        manual = (-0.5 * (z @ z) - 4 * np.log(policy.std)
                  - 0.5 * 4 * np.log(2 * np.pi))

        self.assertAlmostEqual(policy.log_prob(prompt, response), manual, places=12)

        batch = policy.log_prob_batch(
            np.stack([prompt, prompt]), np.stack([response, response]))
        np.testing.assert_allclose(batch, [manual, manual])

    def test_score_function_matches_finite_difference(self):
        """
        grad_theta log pi(a|s) must be (a - s - theta)/std^2.

        This is the term the original implementation replaced with the prompt.
        """
        rng = np.random.default_rng(4)
        policy = GaussianPolicy(4, std=0.7, seed=0)
        base = rng.standard_normal(4) * 0.2
        policy.theta = base.copy()
        prompt = rng.standard_normal(4)
        response = rng.standard_normal(4)

        analytic = (response - policy.mean(prompt)) / policy.std ** 2

        eps = 1e-6
        numeric = np.zeros(4)
        for i in range(4):
            plus = base.copy()
            plus[i] += eps
            policy.theta = plus
            logp_plus = policy.log_prob(prompt, response)

            minus = base.copy()
            minus[i] -= eps
            policy.theta = minus
            logp_minus = policy.log_prob(prompt, response)

            numeric[i] = (logp_plus - logp_minus) / (2 * eps)

        policy.theta = base
        np.testing.assert_allclose(analytic, numeric, atol=1e-6)

    def test_sample_kl_estimates_exact_kl(self):
        """E[log pi - log pi_ref] equals the exact Gaussian KL."""
        rng = np.random.default_rng(5)
        policy = GaussianPolicy(4, std=0.8, seed=0)
        policy.theta = rng.standard_normal(4) * 0.3
        reference = policy.snapshot()
        reference.theta = np.zeros(4)

        n = 50_000
        prompts = np.zeros((n, 4))
        responses = (prompts + policy.theta
                     + rng.standard_normal((n, 4)) * policy.std)
        sample_kl_mean = np.mean(policy.log_prob_batch(prompts, responses)
                                 - reference.log_prob_batch(prompts, responses))

        self.assertAlmostEqual(sample_kl_mean, policy.exact_kl(reference),
                               delta=0.02)


class TestPPO(unittest.TestCase):

    def test_clipped_surrogate_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(6)
        ppo = SimplifiedPPO(4, std=1.0, clip=0.2, seed=0)
        base = rng.standard_normal(4) * 0.1
        ppo.actor.theta = base.copy()

        prompts = rng.standard_normal((6, 4))
        responses = (prompts + ppo.actor.theta
                     + rng.standard_normal((6, 4)) * ppo.actor.std)
        old_log_probs = ppo.actor.log_prob_batch(prompts, responses)
        advantages = rng.standard_normal(6)

        grad = ppo.clipped_surrogate_gradient(
            prompts, responses, old_log_probs, advantages)

        def objective(theta):
            ppo.actor.theta = theta
            new_log_probs = ppo.actor.log_prob_batch(prompts, responses)
            ratio = np.exp(np.clip(new_log_probs - old_log_probs, -20.0, 20.0))
            return np.mean(np.minimum(
                ratio * advantages,
                np.clip(ratio, 1 - ppo.clip, 1 + ppo.clip) * advantages))

        eps = 1e-6
        numeric = np.zeros(4)
        for i in range(4):
            plus = base.copy()
            plus[i] += eps
            minus = base.copy()
            minus[i] -= eps
            numeric[i] = (objective(plus) - objective(minus)) / (2 * eps)

        ppo.actor.theta = base
        np.testing.assert_allclose(grad, numeric, atol=1e-6)

    def _train_small_system(self):
        dim = 16
        rng = np.random.default_rng(7)
        preferences = make_preferences(rng, n=20, dim=dim)
        system = MicroRLHF(dim, seed=0, std=1.0, beta=0.5)
        system.train_reward_model(
            preferences, learning_rate=0.1, epochs=50, verbose=False)
        train_prompts = rng.standard_normal((8, dim))
        eval_prompts = rng.standard_normal((64, dim))
        return system, train_prompts, eval_prompts

    def test_collect_advantages_use_value_baseline(self):
        """Advantage must be the shaped reward minus V(prompt), not the raw reward."""
        system, train_prompts, _ = self._train_small_system()
        rollout = system.policy.collect(
            train_prompts, system.reward_model, system.rng,
            samples_per_prompt=4)

        shaped = (rollout.raw_rewards
                  - system.policy.beta * rollout.sample_kl)
        expected_advantages = shaped - rollout.values

        np.testing.assert_allclose(rollout.advantages, expected_advantages)
        np.testing.assert_allclose(rollout.returns, shaped)

    def test_alignment_improves_eval_reward(self):
        system, train_prompts, eval_prompts = self._train_small_system()
        before = system.evaluate(eval_prompts)

        history = system.align_policy(
            train_prompts, n_iterations=20, eval_prompts=eval_prompts,
            samples_per_prompt=16, verbose=False)

        after = system.evaluate(eval_prompts)
        self.assertGreater(after, before + 0.5)
        self.assertGreater(history[-1], history[0] + 0.5)
        self.assertLess(system.exact_kl_to_reference(), 5.0)

    def test_alignment_approaches_kl_optimum(self):
        """
        For a linear reward r = w . a and a Gaussian policy, the KL-regularized
        optimum is theta* = theta_ref + std^2 * w / beta. Check we get close.
        """
        system, train_prompts, eval_prompts = self._train_small_system()
        system.align_policy(
            train_prompts, n_iterations=20, eval_prompts=eval_prompts,
            samples_per_prompt=16, verbose=False)

        theta = system.policy.actor.theta
        theta_ref = system.policy.reference.theta
        w = system.reward_model.weights
        beta = system.policy.beta
        std = system.policy.actor.std
        theta_star = theta_ref + (std ** 2) * w / beta

        relative_error = (np.linalg.norm(theta - theta_star)
                          / np.linalg.norm(theta_star))
        self.assertLess(relative_error, 0.25)

    def test_same_seed_is_reproducible(self):
        def run(seed):
            rng = np.random.default_rng(8)
            preferences = make_preferences(rng, n=20, dim=16)
            system = MicroRLHF(16, seed=seed, std=1.0, beta=0.5)
            system.train_reward_model(
                preferences, learning_rate=0.1, epochs=50, verbose=False)
            prompts = rng.standard_normal((8, 16))
            history = system.align_policy(
                prompts, n_iterations=10, samples_per_prompt=16, verbose=False)
            return history, system.policy.actor.theta.copy()

        history_a, theta_a = run(0)
        history_b, theta_b = run(0)

        np.testing.assert_allclose(history_a, history_b)
        np.testing.assert_array_equal(theta_a, theta_b)


if __name__ == "__main__":
    unittest.main()
