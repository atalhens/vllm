# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repro: prefix-pruning reload should load the surviving suffix end-to-end."""

import pytest

from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)


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
