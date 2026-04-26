"""Curriculum (Tier1) scenario overrides."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.curriculum import Curriculum
from server.failures import FailureType
from server.scenarios import generate_scenario


def test_tier1_weights_bias_worst():
    c = Curriculum()
    c.on_episode_end(0.5, False, [FailureType.CRASH.value, FailureType.BAD_DEPLOY.value])
    c.on_episode_end(0.5, True, [FailureType.CRASH.value])
    o = c.next_scenario_overrides()
    assert "failure_type_weights" in o
    w = o["failure_type_weights"]
    assert w.get(FailureType.CRASH.value, 0) > w.get(FailureType.NETWORK_ERROR.value, 0)


def test_tier1_fallback_no_api():
    c = Curriculum()
    o = c.next_scenario_overrides()
    assert isinstance(o, dict)


def test_scenario_merges_overrides():
    sc = generate_scenario(
        1, "easy", bump_num_failures=1, max_steps_offset=-1,
    )
    assert sc.max_steps >= 3
    # bump adds at least 1 to num_failures in easy=1
    assert len(sc.failure_specs) >= 1
