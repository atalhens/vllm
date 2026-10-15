# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repro: prefix-pruning reload should load the surviving suffix end-to-end."""

import pytest

from tests.v1.kv_connector.unit.offloading_connector import utils as offloading_utils
from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.v1.request import RequestStatus


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_prefix_pruning_reload_loads_suffix(request_runner, async_scheduling):
    block_size = 4
    tokens_per_chunk = block_size  # blocks_per_chunk = 1
    num_gpu_blocks = 100

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
    )

    # Phase 1: store all 5 chunks A B C D E.
    runner.new_request(token_ids=[0] * tokens_per_chunk * 5)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2, 3, 4),
    )

    # Phase 2: clear GPU cache; simulate prefix pruning (A,B gone, C,D,E cached).
    runner.scheduler.reset_prefix_cache()
    sched = runner.connector_scheduler
    sched._find_hit_runs = lambda keys, ctx, req, grp, start: [(2, 3)]
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )

    runner.new_request(token_ids=[0] * tokens_per_chunk * 5)
    # CDE (chunks 2,3,4) loaded from storage; AB (0,1) recomputed.
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=(2, 3, 4),
    )


def test_async_suffix_reload_stops_when_decode_output_is_missing(
    request_runner, monkeypatch
):
    block_size = 4
    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=100,
        async_scheduling=True,
    )

    runner.new_request(token_ids=[0] * (block_size * 5))
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3, 4))

    runner.scheduler.reset_prefix_cache()
    runner.connector_scheduler._find_hit_runs = (
        lambda keys, ctx, req, grp, start: [(2, 3)]
    )
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.new_request(token_ids=[0] * (block_size * 5))
    request = runner.scheduler.waiting.peek_request()
    request.max_tokens = 2

    create_model_runner_output = offloading_utils.create_model_runner_output

    def create_output_without_samples(**kwargs):
        output = create_model_runner_output(**kwargs)
        output.sampled_token_ids = []
        return output

    monkeypatch.setattr(
        offloading_utils,
        "create_model_runner_output",
        create_output_without_samples,
    )
    runner.run(decoded_tokens=[0] * 20, expected_loaded=(2, 3, 4))

    assert request.status is RequestStatus.FINISHED_ERROR
    assert request.num_computed_tokens <= (
        request.num_prompt_tokens + request.max_tokens
    )
