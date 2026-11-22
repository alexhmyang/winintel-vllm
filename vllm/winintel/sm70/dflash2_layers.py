# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Z Lab
# Adaptation author: 数萌AI服务器 (winintel)
"""DFlash2 convolutions and selector from Z Lab's public speculators 0.9.0.

Architecture source:
https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn


def grouped_dynamic_conv(
    hidden_states: torch.Tensor,
    delta_kernel: torch.Tensor,
    base_kernel: torch.Tensor,
    *,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Apply the checkpoint's token-conditioned convolution within each block."""
    hidden_size = hidden_states.shape[-1]
    if hidden_size % group_size:
        raise ValueError("Hidden size must be divisible by convolution group size")
    kernel_size = base_kernel.shape[0]
    if kernel_size > block_size:
        raise ValueError("Convolution kernel cannot exceed the draft block")
    num_groups = hidden_size // group_size
    if base_kernel.shape != (kernel_size, hidden_size):
        raise ValueError("Invalid convolution base kernel shape")
    if delta_kernel.shape != (*hidden_states.shape[:-1], kernel_size, num_groups):
        raise ValueError("Invalid convolution delta kernel shape")
    flat_hidden = hidden_states.reshape(-1, num_groups, group_size)
    flat_delta = delta_kernel.reshape(-1, kernel_size, num_groups)
    positions = torch.arange(
        flat_hidden.shape[0], device=hidden_states.device
    ).remainder(block_size)
    output = torch.zeros_like(flat_hidden)
    for tap in range(kernel_size):
        if tap == 0:
            shifted = flat_hidden
        else:
            padding = flat_hidden.new_zeros(tap, num_groups, group_size)
            shifted = torch.cat([padding, flat_hidden[:-tap]], dim=0)
        coefficient = (
            base_kernel[tap].to(flat_hidden.dtype).view(1, num_groups, group_size)
            + flat_delta[:, tap, :, None]
        )
        valid = positions.ge(tap).view(-1, 1, 1)
        output = output + shifted * coefficient * valid
    return output.reshape(hidden_states.shape)


class GroupedDynamicCausalConv(nn.Module):
    """Input/output local convolutions driven by one hidden-state projection."""

    def __init__(
        self,
        hidden_size: int,
        *,
        block_size: int,
        kernel_size: int,
        group_size: int,
    ) -> None:
        super().__init__()
        if hidden_size % group_size or kernel_size > block_size:
            raise ValueError("Invalid grouped convolution dimensions")
        self.block_size = block_size
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * kernel_size * self.num_groups, bias=False
        )
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0].fill_(1.0)
            self.kernel_projection.weight.zero_()

    def prepare(self, hidden_states: torch.Tensor):
        kernels = self.kernel_projection(hidden_states).view(
            *hidden_states.shape[:-1], 2, self.kernel_size, self.num_groups
        )
        return (
            grouped_dynamic_conv(
                hidden_states,
                kernels[..., 0, :, :],
                self.base_kernel[0],
                block_size=self.block_size,
                group_size=self.group_size,
            ),
            kernels[..., 1, :, :],
        )

    def finish(self, hidden_states: torch.Tensor, delta_kernel: torch.Tensor):
        return grouped_dynamic_conv(
            hidden_states,
            delta_kernel,
            self.base_kernel[1],
            block_size=self.block_size,
            group_size=self.group_size,
        )


class CandidateSelector(nn.Module):
    """Bilinear predecessor/hidden/candidate scoring after unary top-k."""

    def __init__(self, *, vocab_size: int, hidden_size: int, rank: int, top_k: int):
        super().__init__()
        if top_k > vocab_size:
            raise ValueError("Selector top-k cannot exceed vocabulary size")
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)

    def context(self, hidden_states, predecessor_ids):
        predecessor = self.predecessor_codebook[predecessor_ids.long()]
        projected_hidden = self.hidden_projection(hidden_states)
        return predecessor * projected_hidden.to(predecessor.dtype)

    def transition_scores(self, hidden_states, predecessor_ids, candidate_ids):
        context = self.context(hidden_states, predecessor_ids)
        successors = self.successor_codebook[candidate_ids.long()]
        return (context.unsqueeze(-2) * successors).sum(dim=-1)

    def score_candidates(
        self, unary_logits, hidden_states, predecessor_ids, candidate_ids
    ):
        unary_scores = unary_logits.gather(-1, candidate_ids)
        transition_scores = self.transition_scores(
            hidden_states, predecessor_ids, candidate_ids
        )
        return unary_scores + transition_scores.to(unary_scores.dtype)

    def select(self, unary_logits, hidden_states, predecessor_ids, top_k=None):
        candidate_ids = unary_logits.topk(top_k or self.top_k, dim=-1).indices
        scores = self.score_candidates(
            unary_logits, hidden_states, predecessor_ids, candidate_ids
        )
        return candidate_ids, scores


def select_draft_path(selector, unary_logits, hidden_states, anchor_ids):
    """Select each successor using the previously selected token as predecessor."""
    candidates = unary_logits.topk(selector.top_k, dim=-1).indices
    unary = unary_logits.gather(-1, candidates)
    projected = selector.hidden_projection(hidden_states)
    successors = selector.successor_codebook[candidates.long()]
    predecessor = anchor_ids
    selected = []
    for position in range(hidden_states.shape[1]):
        context = selector.predecessor_codebook[predecessor.long()] * projected[
            :, position
        ].to(selector.predecessor_codebook.dtype)
        transition = (context.unsqueeze(-2) * successors[:, position]).sum(-1)
        scores = unary[:, position] + transition.to(unary.dtype)
        predecessor = (
            candidates[:, position]
            .gather(-1, scores.argmax(-1, keepdim=True))
            .squeeze(-1)
        )
        selected.append(predecessor)
    return torch.stack(selected, dim=1)
