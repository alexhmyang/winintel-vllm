# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Capture the upstream Qwen MTP forward with stable SM70 attention inputs."""

import os
from dataclasses import dataclass

import torch
from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    set_forward_context,
)
from vllm.logger import init_logger

from vllm.config import CUDAGraphMode
from vllm.winintel.sm70.backend import WinIntelSM70Metadata

logger = init_logger(__name__)


@dataclass
class DraftInputs:
    kwargs: dict
    metadata: dict
    slots: dict
    output: object = None
    verified: bool = False


class SM70DraftGraphs:
    def __init__(self, proposer):
        self.proposer = proposer
        self.config = proposer.vllm_config
        self.original = proposer.model.forward
        self.graphs = {}
        self.entries = {}

    def _run(self, entry, tokens):
        query_len = next(iter(entry.metadata.values())).max_query_len
        with set_forward_context(
            entry.metadata,
            self.config,
            num_tokens=tokens,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=BatchDescriptor(num_tokens=tokens, uniform=True),
            slot_mapping=entry.slots,
        ):
            return self.graphs[query_len](**entry.kwargs)

    @torch.inference_mode()
    def capture(self):
        proposer = self.proposer
        names = proposer._draft_attn_layer_names
        if not names or len(proposer.draft_attn_groups) != 1:
            raise NotImplementedError("SM70 MTP graphs require one attention group")
        group = proposer.draft_attn_groups[0]
        builder = group.get_metadata_builder()
        device = proposer.device
        block_size = group.kv_cache_spec.block_size
        blocks = (proposer.max_model_len + block_size - 1) // block_size
        max_seqs = self.config.scheduler_config.max_num_seqs
        dflash = proposer.method == "dflash2"
        query_lengths = (8,) if dflash else (1, 1 + proposer.num_speculative_tokens)
        for query_len in reversed(query_lengths):
            self.graphs[query_len] = CUDAGraphWrapper(
                self.original,
                self.config,
                CUDAGraphMode.FULL,
                CUDAGraphOptions(weak_ref_output=False),
            )
            for batch in range(max_seqs, 0, -1):
                tokens = batch * query_len
                graph_key = (tokens, query_len)
                if graph_key in self.entries:
                    raise ValueError("SM70 draft graph token counts must be distinct")
                metadata = WinIntelSM70Metadata(
                    num_actual_tokens=tokens,
                    max_query_len=query_len,
                    max_seq_len=proposer.max_model_len,
                    block_table=torch.zeros(
                        (batch, blocks), device=device, dtype=torch.int32
                    ),
                    token_seq=torch.arange(
                        batch, device=device, dtype=torch.int32
                    ).repeat_interleave(query_len),
                    token_pos=torch.zeros(tokens, device=device, dtype=torch.int32),
                    fixed_num_partitions=builder.decode_partitions,
                    dflash_window=(
                        proposer.draft_model_config.hf_config.sliding_window + 8
                        if dflash
                        else 0
                    ),
                )
                slots = torch.full((tokens,), -1, device=device, dtype=torch.int64)
                kwargs = dict(
                    positions=torch.zeros_like(proposer._get_positions(tokens)),
                    input_ids=None,
                    inputs_embeds=None,
                )
                if not dflash:
                    kwargs["hidden_states"] = torch.zeros_like(
                        proposer.hidden_states[:tokens]
                    )
                if proposer.supports_mm_inputs:
                    kwargs["inputs_embeds"] = torch.zeros_like(
                        proposer.inputs_embeds[:tokens]
                    )
                else:
                    kwargs["input_ids"] = torch.zeros_like(proposer.input_ids[:tokens])
                entry = DraftInputs(
                    kwargs,
                    {name: metadata for name in names},
                    {name: slots for name in names},
                )
                with set_forward_context(
                    entry.metadata,
                    self.config,
                    num_tokens=tokens,
                    slot_mapping=entry.slots,
                ):
                    for _ in range(2):
                        self.original(**kwargs)
                torch.cuda.synchronize()
                entry.output = self._run(entry, tokens)
                self.entries[graph_key] = entry
        logger.info(
            "Captured SM70 MTP forward graphs for %s tokens", sorted(self.entries)
        )

    @torch.inference_mode()
    def forward(self, *args, **kwargs):
        context = get_forward_context()
        metadata = context.attn_metadata
        if not metadata or args or not self.entries:
            return self.original(*args, **kwargs)
        names = self.proposer._draft_attn_layer_names
        actual = metadata.get(next(iter(names)))
        if (
            not isinstance(actual, WinIntelSM70Metadata)
            or actual.fixed_num_partitions is None
        ):
            return self.original(*args, **kwargs)
        tokens = kwargs["positions"].shape[-1]
        entry = self.entries.get((tokens, actual.max_query_len))
        if entry is None:
            return self.original(*args, **kwargs)
        stable = next(iter(entry.metadata.values()))
        if (
            actual.num_actual_tokens != tokens
            or actual.max_query_len != stable.max_query_len
            or actual.block_table.shape[0] != stable.block_table.shape[0]
            or actual.block_table.shape[1] > stable.block_table.shape[1]
        ):
            return self.original(*args, **kwargs)
        stable.block_table[:, : actual.block_table.shape[1]].copy_(actual.block_table)
        stable.token_seq.copy_(actual.token_seq)
        stable.token_pos.copy_(actual.token_pos)
        for name in names:
            entry.slots[name].copy_(context.slot_mapping[name][:tokens])
        for name, value in entry.kwargs.items():
            incoming = kwargs.get(name)
            if value is not None:
                if incoming is None or value.shape != incoming.shape:
                    return self.original(*args, **kwargs)
                value.copy_(incoming)
        expected = None
        if (
            os.environ.get("WININTEL_SM70_MTP_GRAPH_VERIFY", "1") == "1"
            and not entry.verified
        ):
            expected = self.original(*args, **kwargs).clone()
        result = self._run(entry, tokens)
        if expected is not None:
            if self.proposer.method == "mtp":
                torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(
                    self.proposer.model.get_top_tokens(result),
                    self.proposer.model.get_top_tokens(expected),
                    rtol=0,
                    atol=0,
                )
            else:
                torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
            entry.verified = True
            logger.info("Verified SM70 MTP graph against eager for %s tokens", tokens)
        return result


def install(proposer):
    method = getattr(proposer, "method", None)
    if method not in ("mtp", "dflash2"):
        return
    dflash = method == "dflash2"
    enabled = os.environ.get(
        "WININTEL_SM70_DFLASH_GRAPH" if dflash else "WININTEL_SM70_MTP_GRAPH"
    )
    if enabled != "1":
        return
    if (dflash and proposer.num_speculative_tokens != 7) or (
        not dflash and proposer.num_speculative_tokens not in (3, 5, 7, 9, 11)
    ):
        raise NotImplementedError(
            "SM70 draft graphs require MTP3/5/7/9/11 or block-eight DFlash2"
        )
    if proposer.vllm_config.scheduler_config.max_num_seqs > 16:
        raise NotImplementedError(
            "SM70 draft graphs currently support at most sixteen rows"
        )
    graphs = SM70DraftGraphs(proposer)
    proposer.winintel_draft_graphs = graphs
    proposer.model.forward = graphs.forward


def capture(proposer):
    graphs = getattr(proposer, "winintel_draft_graphs", None)
    if graphs is not None:
        graphs.capture()
