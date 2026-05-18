# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
import math

import torch
from torch import nn

# 2 GB in bytes; used to decide whether to chunk a weight tensor
_2GB = 2 * 1024 * 1024 * 1024
# Bytes per element for float32
_BYTES_PER_FLOAT32 = 4


def _num_chunks_for_weight(num_rows: int, num_cols: int) -> int:
    """Return the minimum K such that each chunk of (num_rows // K) * num_cols * 4 bytes < 2 GB."""
    total_bytes = num_rows * num_cols * _BYTES_PER_FLOAT32
    if total_bytes <= _2GB:
        return 1
    # Smallest K where ceil(num_rows / K) * num_cols * 4 < 2 GB
    max_rows_per_chunk = _2GB // (num_cols * _BYTES_PER_FLOAT32)
    if max_rows_per_chunk == 0:
        raise ValueError(
            f"A single row of the weight matrix ({num_cols} cols x 4 bytes = {num_cols * 4} bytes) "
            "already exceeds the 2 GB protobuf limit. Cannot split further."
        )
    return math.ceil(num_rows / max_rows_per_chunk)


class QeffChunkedEmbedding(nn.Module):
    """
    Drop-in replacement for ``nn.Embedding`` whose weight matrix would exceed the
    2 GB protobuf limit when exported to ONNX.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        num_chunks = _num_chunks_for_weight(num_embeddings, embedding_dim)
        # Compute per-chunk vocab sizes (last chunk may be smaller)
        base_size, remainder = divmod(num_embeddings, num_chunks)
        chunk_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_chunks)]

        self.chunks = nn.ModuleList([nn.Embedding(size, embedding_dim) for size in chunk_sizes])
        # Cumulative offsets for each chunk (used in forward to remap indices)
        offsets = [0]
        for size in chunk_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self.register_buffer("_chunk_offsets", torch.tensor(offsets, dtype=torch.long))

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "QeffChunkedEmbedding":
        """Construct a ``QeffChunkedEmbedding`` from an existing ``nn.Embedding``."""
        obj = cls(
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            padding_idx=embedding.padding_idx,
        )
        with torch.no_grad():
            offset = 0
            for chunk in obj.chunks:
                size = chunk.num_embeddings
                chunk.weight.copy_(embedding.weight[offset : offset + size])
                offset += size
        return obj

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if len(self.chunks) == 1:
            return self.chunks[0](input_ids)

        # For each chunk, mask out-of-range indices, look up, and zero-out invalid rows.
        # Final result is the sum across chunks (only one chunk will be non-zero per token).
        result = None
        for i, chunk in enumerate(self.chunks):
            offset = self._chunk_offsets[i]
            local_ids = input_ids - offset
            # Clamp to valid range so the embedding lookup never sees OOB indices.
            clamped = local_ids.clamp(0, chunk.num_embeddings - 1)
            embeds = chunk(clamped)
            # Zero out positions that don't belong to this chunk.
            valid = (local_ids >= 0) & (local_ids < chunk.num_embeddings)
            embeds = embeds * valid.unsqueeze(-1).to(embeds.dtype)
            result = embeds if result is None else result + embeds
        return result


class QeffChunkedLMHead(nn.Module):
    """
    Drop-in replacement for ``nn.Linear`` (used as ``lm_head``) whose weight
    matrix would exceed the 2 GB protobuf limit when exported to ONNX.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        num_chunks = _num_chunks_for_weight(out_features, in_features)
        base_size, remainder = divmod(out_features, num_chunks)
        chunk_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_chunks)]

        self.chunks = nn.ModuleList([nn.Linear(in_features, size, bias=bias) for size in chunk_sizes])

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QeffChunkedLMHead":
        """Construct a ``QeffChunkedLMHead`` from an existing ``nn.Linear``."""
        obj = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
        )
        with torch.no_grad():
            offset = 0
            for chunk in obj.chunks:
                size = chunk.out_features
                chunk.weight.copy_(linear.weight[offset : offset + size])
                if linear.bias is not None and chunk.bias is not None:
                    chunk.bias.copy_(linear.bias[offset : offset + size])
                offset += size
        return obj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if len(self.chunks) == 1:
            return self.chunks[0](hidden_states)
        return torch.cat([chunk(hidden_states) for chunk in self.chunks], dim=-1)


def is_large_embed_chunking_applied(model: nn.Module) -> bool:
    """
    Returns True if model contains chunked embedding/lm_head modules with >1 chunks.
    """
    for module in model.modules():
        if isinstance(module, (QeffChunkedEmbedding, QeffChunkedLMHead)) and len(module.chunks) > 1:
            return True
    return False
