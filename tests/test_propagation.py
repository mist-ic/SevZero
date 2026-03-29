"""Tests for queueing theory and propagation."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.propagation import (
    compute_utilisation,
    compute_queueing_latency_multiplier,
    compute_retry_amplification,
    CircuitBreaker,
    BreakerState,
)
import random


class TestQueueingTheory:
    """Little's Law and M/M/c approximations."""

    def test_utilisation_basic(self):
        # L = 100 * 0.05 = 5, T = 50, ρ = 0.1
        rho = compute_utilisation(100.0, 0.05, 50)
        assert abs(rho - 0.1) < 0.001

    def test_utilisation_saturated(self):
        # L = 1000 * 0.1 = 100, T = 50, ρ = 2.0 → capped at 1.0
        rho = compute_utilisation(1000.0, 0.1, 50)
        assert rho == 1.0

    def test_utilisation_zero_traffic(self):
        rho = compute_utilisation(0.0, 0.05, 50)
        assert rho == 0.0

    def test_latency_multiplier_low_utilisation(self):
        mult = compute_queueing_latency_multiplier(0.1)
        assert 1.0 < mult < 2.0  # ~1.11x

    def test_latency_multiplier_high_utilisation(self):
        mult = compute_queueing_latency_multiplier(0.95)
        assert mult >= 10.0

    def test_latency_multiplier_saturated(self):
        mult = compute_queueing_latency_multiplier(0.99)
        assert mult >= 20.0

    def test_retry_amplification_no_failures(self):
        amp = compute_retry_amplification(0.0, 3)
        assert amp == 1.0

    def test_retry_amplification_total_failure(self):
        amp = compute_retry_amplification(1.0, 3)
        assert amp == 4.0  # 1 + 3 retries

    def test_retry_amplification_partial(self):
        amp = compute_retry_amplification(0.5, 3)
        assert 1.0 < amp < 4.0


class TestCircuitBreaker:
    """Circuit breaker state transitions."""

    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == BreakerState.CLOSED

    def test_trips_open_on_high_errors(self):
        cb = CircuitBreaker(error_threshold=0.5, window_size=3)
        rng = random.Random(42)
        for _ in range(5):
            cb.tick(0.8, rng)
        assert cb.state == BreakerState.OPEN

    def test_transitions_to_half_open(self):
        cb = CircuitBreaker(error_threshold=0.5, cooldown_ticks=5, window_size=2)
        rng = random.Random(42)
        # Trip open
        for _ in range(3):
            cb.tick(0.9, rng)
        assert cb.state == BreakerState.OPEN
        # Wait for cooldown
        for _ in range(6):
            cb.tick(0.0, rng)
        assert cb.state in (BreakerState.HALF_OPEN, BreakerState.CLOSED)

    def test_dampening_factor(self):
        cb = CircuitBreaker()
        assert cb.dampening_factor == 1.0  # CLOSED
        cb.state = BreakerState.OPEN
        assert cb.dampening_factor == 0.05
        cb.state = BreakerState.HALF_OPEN
        assert cb.dampening_factor == 0.3
