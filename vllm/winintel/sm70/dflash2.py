# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""DFlash2 checkpoint adapter built on vLLM's Qwen3 tensor-parallel layers."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import Qwen2MLP
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.winintel.sm70.dflash2_layers import (
    CandidateSelector,
    GroupedDynamicCausalConv,
    select_draft_path,
)


class DFlash2DecoderLayer(nn.Module):
    """Keep the BF16 checkpoint's large convolution residuals in FP32."""

    def __init__(self, config, cache_config, prefix):
        super().__init__()
        settings = config.dflash_config
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32
        )
        conv_kwargs = dict(
            block_size=settings["block_size"],
            kernel_size=settings["conv_kernel_size"],
            group_size=settings["conv_group_size"],
        )
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size, **conv_kwargs
        )
        self.mlp_conv = GroupedDynamicCausalConv(config.hidden_size, **conv_kwargs)

    def forward(self, positions, hidden_states):
        residual = hidden_states
        prepared, kernel = self.attention_conv.prepare(
            self.input_layernorm(hidden_states.float()).to(
                self.self_attn.qkv_proj.weight.dtype
            )
        )
        attention_output = self.self_attn(positions, prepared)
        convolved = self.attention_conv.finish(attention_output.float(), kernel.float())
        hidden_states = residual.float() + convolved.float()
        residual = hidden_states
        prepared, kernel = self.mlp_conv.prepare(
            self.post_attention_layernorm(hidden_states).to(
                self.mlp.gate_up_proj.weight.dtype
            )
        )
        mlp_output = self.mlp(prepared)
        convolved = self.mlp_conv.finish(mlp_output.float(), kernel.float())
        return residual.float() + convolved.float()


class DFlash2Model(nn.Module):
    def __init__(self, config, cache_config, prefix):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            DFlash2DecoderLayer(config, cache_config, f"{prefix}.layers.{index}")
            for index in range(config.num_hidden_layers)
        )
        self.fc = ReplicatedLinear(
            config.hidden_size * len(config.dflash_config["target_layer_ids"]),
            config.hidden_size,
            bias=False,
            return_bias=False,
            prefix=f"{prefix}.fc",
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32
        )


class DFlash2DraftModel(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        settings = self.config.dflash_config
        if settings["block_size"] != 8 or self.config.num_hidden_layers != 5:
            raise ValueError("This adapter requires the five-layer block-eight draft")
        if getattr(self.config, "is_causal", True):
            raise ValueError("DFlash2 requires non-causal draft attention")
        self.model = DFlash2Model(
            self.config, vllm_config.cache_config, prefix or "dflash2"
        )
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size)
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.candidate_selector = CandidateSelector(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            rank=settings["selector_rank"],
            top_k=settings["selector_top_k"],
        )

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states):
        return self.model.fc(hidden_states.to(self.model.fc.weight.dtype))

    def precompute_and_store_context_kv(self, hidden_states, positions, slots):
        """Replace speculative KVs with features from the accepted target prefix."""
        combined = self.combine_hidden_states(hidden_states)
        context = self.model.hidden_norm(combined)
        for layer in self.model.layers:
            attention = layer.self_attn
            qkv, _ = attention.qkv_proj(context)
            q, k, v = qkv.split(
                [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
            )
            k = attention.k_norm(
                k.reshape(-1, attention.num_kv_heads, attention.head_dim)
            ).reshape_as(k)
            _, k = attention.rotary_emb(positions, q, k)
            cache_layer = attention.attn
            cache_layer.impl.do_kv_cache_update(
                cache_layer,
                k.view(-1, attention.num_kv_heads, attention.head_dim),
                v.view(-1, attention.num_kv_heads, attention.head_dim),
                cache_layer.kv_cache[get_forward_context().virtual_engine],
                slots,
            )

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        hidden_states = (
            self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        for layer in self.model.layers:
            hidden_states = layer(positions, hidden_states)
        return self.model.norm(hidden_states).to(self.lm_head.weight.dtype)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def select_tokens(self, hidden_states, anchor_ids):
        batch = anchor_ids.shape[0]
        hidden = hidden_states.view(batch, 8, -1)[:, 1:]
        logits = self.compute_logits(hidden.reshape(-1, hidden.shape[-1]))
        return select_draft_path(
            self.candidate_selector,
            logits.view(batch, 7, -1),
            hidden,
            anchor_ids,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        mappings = (
            (".q_proj", ".qkv_proj", "q"),
            (".k_proj", ".qkv_proj", "k"),
            (".v_proj", ".qkv_proj", "v"),
            (".gate_proj", ".gate_up_proj", 0),
            (".up_proj", ".gate_up_proj", 1),
        )
        params = dict(self.named_parameters())
        loaded = set()
        for original, weight in weights:
            name = (
                original
                if original.startswith("candidate_selector.")
                else "model." + original
            )
            for source, dest, shard in mappings:
                if source in name:
                    name = name.replace(source, dest)
                    param = params[name]
                    param.weight_loader(param, weight, shard)
                    break
            else:
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        required = {
            name
            for name in params
            if name not in ("model.embed_tokens.weight", "lm_head.weight")
        }
        missing = required - loaded
        if missing:
            raise ValueError(f"Incomplete DFlash2 checkpoint: {sorted(missing)}")
