# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for PR #37307 — scheduler_reserve_full_isl admission gate.

These are CPU-only unit tests. They construct a scheduler whose KV cache is
deliberately too small to hold a single full prompt, but large enough to hold
the first chunked-prefill chunk. The PR's admission gate should refuse the
request; with the gate disabled, the request gets admitted (and would later
cause preemption / KV thrash in a real run).
"""

import pytest

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

# Tuned so that:
#   - prompt requires ceil(2000/16) = 125 KV blocks   (does NOT fit)
#   - first chunk requires ceil(512/16) = 32 blocks   (fits in 40)
BLOCK_SIZE = 16
NUM_BLOCKS = 40
MAX_BATCHED_TOKENS = 512
MAX_MODEL_LEN = 4096
PROMPT_TOKENS = 2000


def _make_scheduler():
    return create_scheduler(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    )


def _set_reserve_flag(scheduler, value: bool) -> None:
    """Flip the admission-gate flag on both the cached attribute and the
    config, so the test is robust to either implementation style."""
    assert hasattr(scheduler, "scheduler_reserve_full_isl"), (
        "Scheduler is missing the `scheduler_reserve_full_isl` attribute. "
        "Did you cache `self.scheduler_config.scheduler_reserve_full_isl` "
        "in Scheduler.__init__?"
    )
    scheduler.scheduler_reserve_full_isl = value
    scheduler.scheduler_config.scheduler_reserve_full_isl = value


def test_sanity_first_chunk_fits_but_full_prompt_does_not():
    """Confirms the chosen sizes actually create the over-admission scenario.
    If this fails, the other two tests don't prove anything — re-tune the
    constants at the top of this file."""
    scheduler = _make_scheduler()
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    blocks_for_full_prompt = -(-PROMPT_TOKENS // BLOCK_SIZE)  # ceil
    blocks_for_first_chunk = -(-MAX_BATCHED_TOKENS // BLOCK_SIZE)
    assert blocks_for_first_chunk <= free, (
        f"first chunk needs {blocks_for_first_chunk} blocks but pool has "
        f"{free} — pick larger NUM_BLOCKS or smaller MAX_BATCHED_TOKENS"
    )
    assert blocks_for_full_prompt > free, (
        f"full prompt needs {blocks_for_full_prompt} blocks and pool has "
        f"{free} — full prompt fits too, scenario invalid"
    )


def test_admission_gate_on_blocks_oversized_request():
    """Default (ON): full ISL doesn't fit → request stays in waiting,
    nothing is scheduled."""
    scheduler = _make_scheduler()
    _set_reserve_flag(scheduler, True)

    (req,) = create_requests(
        num_requests=1, num_tokens=PROMPT_TOKENS, block_size=BLOCK_SIZE
    )
    scheduler.add_request(req)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 0, (
        "Admission gate should refuse a request whose full ISL doesn't fit, "
        f"but scheduled {output.total_num_scheduled_tokens} tokens"
    )
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1, "Request should still be in waiting"


def test_admission_gate_off_admits_partial_request():
    """Flag OFF: only the first chunk is checked, so the request gets
    admitted even though the full ISL won't fit (the legacy behaviour the
    PR fixes)."""
    scheduler = _make_scheduler()
    _set_reserve_flag(scheduler, False)

    (req,) = create_requests(
        num_requests=1, num_tokens=PROMPT_TOKENS, block_size=BLOCK_SIZE
    )
    scheduler.add_request(req)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == MAX_BATCHED_TOKENS, (
        "With the gate off, the first chunk should be admitted; "
        f"got {output.total_num_scheduled_tokens} scheduled tokens"
    )
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0


def test_admission_gate_admits_request_that_fits():
    """Sanity: a short prompt fits → admission gate must NOT block it."""
    scheduler = _make_scheduler()
    _set_reserve_flag(scheduler, True)

    short_tokens = MAX_BATCHED_TOKENS  # exactly one chunk; fits easily
    (req,) = create_requests(
        num_requests=1, num_tokens=short_tokens, block_size=BLOCK_SIZE
    )
    scheduler.add_request(req)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == short_tokens
    assert len(scheduler.running) == 1
