"""Gradient-cache losses for StaRSE's differentiable static modules."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from typing import Any

import torch
import tqdm
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import (
    RandContext,
)


def _is_static_flattened(features: dict[str, Any]) -> bool:
    input_ids = features.get("input_ids")
    offsets = features.get("offsets")
    return (
        isinstance(input_ids, torch.Tensor)
        and input_ids.ndim == 1
        and isinstance(offsets, torch.Tensor)
        and offsets.ndim == 1
    )


def _slice_static_features(
    features: dict[str, Any], *, begin: int, end: int
) -> dict[str, Any]:
    """Slice StaRSE's flattened token stream by sentence offsets."""

    if not _is_static_flattened(features):
        raise ValueError("static feature slicing requires 1D input_ids and offsets")
    input_ids = features["input_ids"]
    offsets = features["offsets"]
    batch_size = int(offsets.numel())
    begin = max(0, min(int(begin), batch_size))
    end = max(begin, min(int(end), batch_size))
    token_begin = int(offsets[begin].item()) if begin < batch_size else input_ids.numel()
    token_end = (
        int(offsets[end].item()) if end < batch_size else input_ids.numel()
    )
    result: dict[str, Any] = {}
    for key, value in features.items():
        if key == "input_ids":
            result[key] = value[token_begin:token_end]
        elif key == "offsets":
            result[key] = value[begin:end] - token_begin
        elif isinstance(value, torch.Tensor) and value.ndim:
            if value.shape[0] == batch_size:
                result[key] = value[begin:end]
            elif value.shape[0] == input_ids.numel():
                result[key] = value[token_begin:token_end]
            else:
                result[key] = value
        else:
            result[key] = value
    return result


class _StaticTypeGuardProxy:
    """Bypass only the upstream constructor's StaticEmbedding type guard."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getitem__(self, index: int) -> object:
        if index != 0:
            raise IndexError(index)
        return object()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


class StaticCompatibleCachedMultipleNegativesRankingLoss(
    CachedMultipleNegativesRankingLoss
):
    """Use upstream GradCache with a differentiable StaRSE static encoder.

    SentenceTransformers rejects all ``StaticEmbedding`` subclasses because its
    ordinary lookup module has little reason to use GradCache. StaRSE adds a
    differentiable collapse graph whose activation memory does benefit from the
    cache. Construction therefore uses a temporary type-guard proxy, then
    restores the real model before any forward call. All cache, objective and
    replay code remains upstream.
    """

    def __init__(self, model: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(_StaticTypeGuardProxy(model), *args, **kwargs)
        self.model = model

    def _online_table(self) -> Any | None:
        try:
            module = self.model[0]
        except (AttributeError, IndexError, KeyError, TypeError):
            return None
        return getattr(module, "online_collapse", None)

    def _telemetry_window(self) -> Any | None:
        try:
            module = self.model[0]
        except (AttributeError, IndexError, KeyError, TypeError):
            return None
        for name in ("online_collapse", "collapse"):
            table = getattr(module, name, None)
            window = getattr(table, "_vocabulary_telemetry", None)
            if window is not None:
                return window
        return None

    def forward(
        self,
        sentence_features: Iterable[dict[str, torch.Tensor]],
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        table = self._online_table()
        audit = table is not None and torch.is_grad_enabled()
        if audit:
            table.start_cache_audit()
        try:
            return super().forward(sentence_features, labels)
        except BaseException:
            if audit:
                table.abort_cache_audit()
            raise

    def embed_minibatch(
        self,
        sentence_feature: dict[str, torch.Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[torch.Tensor, RandContext | None]:
        if not _is_static_flattened(sentence_feature):
            return super().embed_minibatch(
                sentence_feature,
                begin,
                end,
                with_grad,
                copy_random_state,
                random_state,
            )
        minibatch = _slice_static_features(
            sentence_feature, begin=begin, end=end
        )
        table = self._online_table()
        telemetry = self._telemetry_window()
        if table is not None:
            if with_grad:
                table.begin_cache_replay()
            else:
                table.begin_cache_record()
        telemetry_started = False
        if telemetry is not None:
            if with_grad:
                telemetry.begin_replay()
            else:
                telemetry.begin_record()
            telemetry_started = True
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        try:
            with random_state_context, grad_context():
                copied = (
                    RandContext(*minibatch.values()) if copy_random_state else None
                )
                reps = self.model(minibatch)["sentence_embedding"]
        except BaseException:
            if table is not None:
                table.abort_cache_audit()
            if telemetry is not None:
                telemetry.abort_phase()
                telemetry_started = False
            raise
        finally:
            if table is not None:
                table.end_cache_phase()
            if telemetry is not None and telemetry_started:
                telemetry.end_phase()
        return reps, copied

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, torch.Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[torch.Tensor, RandContext | None]]:
        if not _is_static_flattened(sentence_feature):
            yield from super().embed_minibatch_iter(
                sentence_feature,
                with_grad,
                copy_random_state,
                random_states,
            )
            return
        batch_size = int(sentence_feature["offsets"].numel())
        for index, begin in enumerate(
            tqdm.trange(
                0,
                batch_size,
                self.mini_batch_size,
                desc="Embed static mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            yield self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=begin,
                end=begin + self.mini_batch_size,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=(
                    None if random_states is None else random_states[index]
                ),
            )
