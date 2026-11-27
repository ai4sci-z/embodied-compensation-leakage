"""The 4-layer transformer decoder head with action chunk 8 (10.4.0).

Eight learned queries, one per step of the chunk, cross-attend over the
observation tokens; four pre-norm layers of self-attention, cross-attention and a
feed-forward block. Read as an action-chunking decoder rather than as a causal
encoder because that is what "decoder head, chunk length 8" names, and because the
chunk then has an explicit carrier: query `i` predicts step `i`, and no
autoregression is introduced that the rollout loop would have to unroll.

The feed-forward block is built by a factory instead of being hard-coded. That is
the single hook E9 needs: level I1 replaces it with the vendored HB-MoE and AS-MoE
without this file knowing anything about mixtures, and `data_mask` is threaded
through with the signature the vendored modules use (`forward(hidden_states,
data_mask)`), so the dense block and the mixture block are interchangeable.

FiLM conditioning is optional and per layer. It is the attachment point for B4's
style token, B6's embodiment embedding and B13's latent extrinsics, which 10.5.6
requires to enter at the same place across those arms so that they are comparable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from torch import nn

from portal_trace.policies.obs import ACTION_DIM, CHUNK

__all__ = [
    "FeedForward",
    "DenseFeedForward",
    "FeedForwardFactory",
    "ChunkDecoderLayer",
    "ActionChunkDecoder",
]


class FeedForward(Protocol):
    """The interface a decoder feed-forward block must satisfy.

    Two arguments, matching the vendored HiMoE modules: the hidden states and the
    action-space descriptor AS-MoE routes on. Dense blocks ignore the second.
    """

    def __call__(
        self, hidden_states: torch.Tensor, data_mask: torch.Tensor | None
    ) -> torch.Tensor: ...


FeedForwardFactory = Callable[[int], nn.Module]
"""Builds the feed-forward block for a given layer index."""


class DenseFeedForward(nn.Module):
    """The shared feed-forward block: E9's I0, and the default everywhere else."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, ffn_dim)
        self.down = nn.Linear(ffn_dim, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, hidden_states: torch.Tensor, data_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        del data_mask  # dense: every embodiment takes the same path
        return self.down(self.dropout(self.act(self.up(hidden_states))))


class _FiLM(nn.Module):
    """Scale and shift generated from a conditioning vector."""

    def __init__(self, cond_dim: int, d_model: int) -> None:
        super().__init__()
        self.to_params = nn.Linear(cond_dim, 2 * d_model)
        # Start at the identity so that adding a conditioning channel does not
        # change the function at initialisation; B6's "token permutation at
        # inference" ablation is only interpretable if the token starts inert.
        nn.init.zeros_(self.to_params.weight)
        nn.init.zeros_(self.to_params.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_params(cond).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ChunkDecoderLayer(nn.Module):
    """Pre-norm self-attention, cross-attention, feed-forward."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        feed_forward: nn.Module,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = feed_forward
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.film = _FiLM(cond_dim, d_model) if cond_dim else None

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        *,
        cond: torch.Tensor | None = None,
        data_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm_self(queries)
        queries = queries + self.dropout(self.self_attn(h, h, h, need_weights=False)[0])
        h = self.norm_cross(queries)
        queries = queries + self.dropout(
            self.cross_attn(h, memory, memory, need_weights=False)[0]
        )
        h = self.norm_ffn(queries)
        if self.film is not None:
            if cond is None:
                raise ValueError(
                    "this decoder was built with FiLM conditioning; the arm must pass "
                    "`cond`"
                )
            h = self.film(h, cond)
        return queries + self.dropout(self.feed_forward(h, data_mask))


class ActionChunkDecoder(nn.Module):
    """The head of 10.4.0: `CHUNK` queries, `n_layers` layers, 7 outputs each."""

    def __init__(
        self,
        d_model: int,
        *,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        cond_dim: int = 0,
        action_dim: int = ACTION_DIM,
        feed_forward_factory: FeedForwardFactory | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.queries = nn.Parameter(torch.zeros(1, CHUNK, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        factory = feed_forward_factory or (
            lambda _idx: DenseFeedForward(d_model, ffn_dim, dropout)
        )
        self.layers = nn.ModuleList(
            ChunkDecoderLayer(d_model, n_heads, dropout, factory(i), cond_dim=cond_dim)
            for i in range(n_layers)
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.to_action = nn.Linear(d_model, action_dim)

    def features(
        self,
        memory: torch.Tensor,
        *,
        cond: torch.Tensor | None = None,
        data_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The per-chunk-step features, before the action projection.

        Exposed because E4 fits its decomposition head on a frozen trunk and needs
        the representation, not the action, and because B4's two-branch heads read
        the same features.
        """
        queries = self.queries.expand(memory.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, memory, cond=cond, data_mask=data_mask)
        return self.norm_out(queries)

    def forward(
        self,
        memory: torch.Tensor,
        *,
        cond: torch.Tensor | None = None,
        data_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(action_chunk, features)`, shapes (B, 8, 7) and (B, 8, d_model)."""
        features = self.features(memory, cond=cond, data_mask=data_mask)
        return self.to_action(features), features
