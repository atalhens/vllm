# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch

import vllm.v1.worker.gpu.model_runner as gpu_model_runner
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.gpu.states import RequestState


def test_update_requests_syncs_prefill_cursor_for_cached_suffix(monkeypatch):
    monkeypatch.setattr(gpu_model_runner.envs, "VLLM_SUFFIX_PRESERVING_EVICTION", False)

    req_states = RequestState(
        max_num_reqs=1,
        max_model_len=32,
        max_num_batched_tokens=1,
        num_speculative_steps=0,
        vocab_size=1,
        device=torch.device("cpu"),
    )
    req_states.add_request(
        req_id="req",
        prompt_len=16,
        all_token_ids=list(range(16)),
        num_computed_tokens=8,
        max_tokens=8,
    )
    req_states.apply_staged_writes()

    runner = SimpleNamespace(
        req_states=req_states,
        block_tables=SimpleNamespace(append_block_ids=lambda *args, **kwargs: None),
    )
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["req"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[12],
        num_output_tokens=[0],
    )
    scheduler_output.prefill_cursor_sync_req_ids = ["req"]

    gpu_model_runner.GPUModelRunner.update_requests(runner, scheduler_output)

    req_idx = req_states.req_id_to_index["req"]
    assert req_states.num_computed_tokens.gpu[req_idx].item() == 12
    assert req_states.num_computed_prefill_tokens[req_idx] == 12
