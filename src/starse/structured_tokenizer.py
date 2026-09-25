"""Exact batched matching primitives for a differentiable hard tokenizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MatchingResult:
    """Partition statistics and deterministic MAP edges for a padded batch."""

    log_partition: torch.Tensor
    marginals: torch.Tensor
    map_edges: torch.Tensor


def _validate_edges(
    edge_scores: torch.Tensor, edge_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_scores.ndim != 2 or edge_mask.ndim != 2:
        raise ValueError("edge_scores and edge_mask must both have shape [batch, edges]")
    if edge_scores.shape != edge_mask.shape:
        raise ValueError("edge_scores and edge_mask must have identical shapes")
    if not edge_scores.is_floating_point():
        raise TypeError("edge_scores must be floating point")
    if edge_mask.dtype is not torch.bool:
        raise TypeError("edge_mask must be boolean")
    if edge_scores.device != edge_mask.device:
        raise ValueError("edge_scores and edge_mask must use the same device")
    if bool((~torch.isfinite(edge_scores) & edge_mask).any()):
        raise ValueError("valid edge scores must be finite")
    scores = torch.where(edge_mask, edge_scores.float(), torch.zeros_like(edge_scores.float()))
    return scores, edge_mask


def _prefix_log_partitions(
    scores: torch.Tensor, mask: torch.Tensor
) -> list[torch.Tensor]:
    batch = scores.shape[0]
    zero = torch.zeros(batch, dtype=torch.float32, device=scores.device)
    prefix = [zero, zero]
    for edge in range(scores.shape[1]):
        separate = prefix[-1]
        merged = prefix[-2] + scores[:, edge]
        prefix.append(
            torch.where(mask[:, edge], torch.logaddexp(separate, merged), separate)
        )
    return prefix


def _suffix_log_partitions(
    scores: torch.Tensor, mask: torch.Tensor
) -> list[torch.Tensor]:
    batch, edges = scores.shape
    zero = torch.zeros(batch, dtype=torch.float32, device=scores.device)
    suffix = [zero for _ in range(edges + 2)]
    for edge in range(edges - 1, -1, -1):
        separate = suffix[edge + 1]
        merged = scores[:, edge] + suffix[edge + 2]
        suffix[edge] = torch.where(
            mask[:, edge], torch.logaddexp(separate, merged), separate
        )
    return suffix


def _map_matching(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch, edges = scores.shape
    if edges == 0:
        return torch.zeros_like(mask)

    zero = torch.zeros(batch, dtype=torch.float32, device=scores.device)
    best = [zero, zero]
    take_by_edge = torch.zeros_like(mask)
    for edge in range(edges):
        separate = best[-1]
        merged = best[-2] + scores[:, edge]
        take = mask[:, edge] & (merged > separate)
        best.append(torch.where(take, merged, separate))
        take_by_edge[:, edge] = take

    selected = torch.zeros_like(mask)
    vertices = torch.full(
        (batch,), edges + 1, dtype=torch.long, device=scores.device
    )
    rows = torch.arange(batch, device=scores.device)
    for _ in range(edges + 1):
        active = vertices >= 2
        edge = (vertices - 2).clamp(min=0, max=edges - 1)
        take = active & take_by_edge[rows, edge]
        selected[rows, edge] |= take
        vertices = vertices - torch.where(take, 2, 1) * active.to(torch.long)
    return selected


def batched_matching(
    edge_scores: torch.Tensor,
    edge_mask: torch.Tensor,
) -> MatchingResult:
    """Solve independent monomer-dimer CRFs over a padded sentence batch.

    Invalid edges act as fixed token boundaries. Every probabilistic recurrence
    runs in float32 even when the caller is inside BF16 autocast.
    """

    scores, mask = _validate_edges(edge_scores, edge_mask)
    prefix = _prefix_log_partitions(scores, mask)
    log_partition = prefix[-1]

    if scores.shape[1] == 0:
        marginals = torch.empty_like(scores)
    else:
        suffix = _suffix_log_partitions(scores, mask)
        log_marginals = torch.stack(
            [
                prefix[edge]
                + scores[:, edge]
                + suffix[edge + 2]
                - log_partition
                for edge in range(scores.shape[1])
            ],
            dim=1,
        )
        marginals = torch.where(mask, torch.exp(log_marginals), torch.zeros_like(scores))

    map_edges = _map_matching(scores, mask)
    if not bool(torch.isfinite(log_partition).all()):
        raise FloatingPointError("nonfinite matching log partition")
    if not bool(torch.isfinite(marginals).all()):
        raise FloatingPointError("nonfinite matching marginals")
    return MatchingResult(log_partition, marginals, map_edges)


def structured_straight_through(
    marginals: torch.Tensor,
    map_edges: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return hard MAP values whose gradient follows exact edge marginals."""

    if marginals.shape != map_edges.shape:
        raise ValueError("marginals and map_edges must have identical shapes")
    if map_edges.dtype is not torch.bool:
        raise TypeError("map_edges must be boolean")
    if not dtype.is_floating_point:
        raise TypeError("straight-through dtype must be floating point")
    soft = marginals.to(dtype=dtype)
    hard = map_edges.to(dtype=dtype)
    return soft + (hard - soft).detach()
