# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Parallel DFlash2 proposer for the official vLLM 0.17 scheduler."""

from dataclasses import replace

import torch
from vllm.forward_context import set_forward_context
from vllm.v1.spec_decode.eagle import EagleProposer, SpecDecodeBaseProposer


class DFlash2Proposer(EagleProposer):
    def __init__(self, vllm_config, device, runner=None):
        SpecDecodeBaseProposer.__init__(
            self, vllm_config, device, pass_hidden_states_to_model=False, runner=runner
        )
        self.supports_mm_inputs = False
        self.eagle3_use_aux_hidden_state = True
        self.block_size = vllm_config.cache_config.block_size
        self.draft_block = 8
        self._verified_batches = set()
        if self.num_speculative_tokens != 7:
            raise ValueError(
                "DFlash2 block-eight checkpoint requires seven draft tokens"
            )
        if self.speculative_config.parallel_drafting:
            raise ValueError("DFlash2 manages its own parallel query construction")

    def _raise_if_multimodal(self):
        pass

    def _raise_if_mrope(self):
        pass

    def load_model(self, target_model):
        super().load_model(target_model)

    def _maybe_share_embeddings(self, target_language_model):
        self.model.model.embed_tokens = target_language_model.model.embed_tokens

    def _maybe_share_lm_head(self, target_language_model):
        self.model.lm_head = target_language_model.lm_head

    def propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu=None,
        slot_mappings=None,
    ):
        cad = common_attn_metadata
        batch = cad.num_reqs
        starts = cad.query_start_loc
        lengths = cad.seq_lens[:batch]
        rejected = (
            torch.zeros_like(lengths)
            if num_rejected_tokens_gpu is None
            else num_rejected_tokens_gpu[:batch]
        )
        accepted_lengths = lengths - rejected
        positions = (
            target_positions[0] if target_positions.ndim == 2 else target_positions
        )
        num_context = target_token_ids.shape[0]
        token_rows = torch.arange(num_context, device=self.device, dtype=torch.int32)
        # searchsorted avoids dynamic repeat_interleave output sizing on the GPU.
        context_seq = torch.searchsorted(
            starts[1:].contiguous(), token_rows, right=True
        )
        valid = positions < accepted_lengths[context_seq]
        columns = torch.div(positions, self.block_size, rounding_mode="floor").long()
        pages = cad.block_table_tensor[context_seq.long(), columns]
        context_slots = pages.long() * self.block_size + positions % self.block_size
        context_slots = torch.where(valid, context_slots, -1)
        with set_forward_context(None, self.vllm_config, num_tokens=num_context):
            self.model.precompute_and_store_context_kv(
                target_hidden_states, positions, context_slots
            )

        offsets = torch.arange(self.draft_block, device=self.device)
        query_positions = accepted_lengths[:, None] + offsets[None, :]
        query_columns = torch.div(
            query_positions, self.block_size, rounding_mode="floor"
        ).long()
        query_pages = cad.block_table_tensor.gather(1, query_columns)
        query_slots = (
            query_pages.long() * self.block_size + query_positions % self.block_size
        ).flatten()
        query_ids = self.input_ids[: batch * self.draft_block].view(
            batch, self.draft_block
        )
        query_ids.fill_(
            self.draft_model_config.hf_config.dflash_config["mask_token_id"]
        )
        query_ids[:, 0].copy_(next_token_ids)
        self.positions[: batch * self.draft_block].copy_(query_positions.flatten())
        query_starts_cpu = torch.arange(batch + 1, dtype=torch.int32) * self.draft_block
        query_cad = replace(
            cad,
            query_start_loc=query_starts_cpu.to(self.device, non_blocking=True),
            query_start_loc_cpu=query_starts_cpu,
            seq_lens=accepted_lengths + self.draft_block,
            num_actual_tokens=batch * self.draft_block,
            max_query_len=self.draft_block,
            max_seq_len=cad.max_seq_len + self.draft_block,
            slot_mapping=query_slots,
            causal=False,
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            _num_computed_tokens_cache=None,
        )
        metadata = {}
        for group in self.draft_attn_groups:
            layer_metadata = group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=query_cad, draft_index=0
            )
            for name in group.layer_names:
                metadata[name] = layer_metadata
        with set_forward_context(
            metadata,
            self.vllm_config,
            num_tokens=batch * self.draft_block,
            slot_mapping={name: query_slots for name in self._draft_attn_layer_names},
        ):
            hidden = self.model(
                input_ids=query_ids.flatten(),
                positions=self.positions[: batch * self.draft_block],
            )
        if batch not in self._verified_batches:
            if not torch.isfinite(hidden).all().item():
                raise FloatingPointError("DFlash2 draft hidden states are not finite")
            self._verified_batches.add(batch)
        return self.model.select_tokens(hidden, next_token_ids)

    def dummy_run(self, num_tokens, **kwargs):
        tokens = min(
            self.vllm_config.scheduler_config.max_num_seqs * self.draft_block,
            max(self.draft_block, num_tokens // self.draft_block * self.draft_block),
        )
        with set_forward_context(None, self.vllm_config, num_tokens=tokens):
            self.model(
                input_ids=self.input_ids[:tokens],
                positions=self.positions[:tokens],
            )
