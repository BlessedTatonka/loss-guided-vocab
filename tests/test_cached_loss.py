from __future__ import annotations

import copy

import pytest
import torch
from sentence_transformers.models import StaticEmbedding
from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
)
from tokenizers import Tokenizer, models, pre_tokenizers

from starse.cached_loss import (
    StaticCompatibleCachedMultipleNegativesRankingLoss,
    _slice_static_features,
)
from starse.conditioned_module import ConditionedStaticEmbedding, marker_for
from starse.vocabulary_telemetry import VocabularyWindow


class _StaticSentinel(StaticEmbedding):
    """Satisfy the upstream type guard without invoking a tokenizer."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)


class _ToyStaticSentenceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(3, 2, bias=False)
        self.gate = torch.nn.Parameter(torch.tensor(0.25))
        self.static = _StaticSentinel()
        self.forward_modes: list[tuple[bool, int]] = []
        self.observations: list[tuple[int, torch.Tensor]] = []

    def __getitem__(self, index: int) -> torch.nn.Module:
        if index != 0:
            raise IndexError(index)
        return self.static

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = int(features["x"].shape[0])
        self.forward_modes.append((torch.is_grad_enabled(), batch))
        probability = torch.sigmoid(self.gate)
        if self.training and probability.requires_grad:
            probability.register_hook(
                lambda gradient, size=batch: self.observations.append(
                    (size, gradient.detach().clone())
                )
            )
        encoded = torch.nn.functional.normalize(self.encoder(features["x"]), dim=1)
        return {"sentence_embedding": encoded * probability}


def _features() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "x": torch.tensor(
                [
                    [1.0, 0.0, 0.5],
                    [0.0, 1.0, 0.5],
                    [0.5, 0.0, 1.0],
                    [0.0, 0.5, 1.0],
                ]
            )
        },
        {
            "x": torch.tensor(
                [
                    [0.9, 0.1, 0.5],
                    [0.1, 0.9, 0.5],
                    [0.5, 0.1, 0.9],
                    [0.1, 0.5, 0.9],
                ]
            )
        },
    ]


def test_cached_symmetric_loss_accepts_differentiable_static_subclass() -> None:
    model = _ToyStaticSentenceModel()

    loss = StaticCompatibleCachedMultipleNegativesRankingLoss(
        model=model,
        scale=20.0,
        mini_batch_size=2,
        directions=("query_to_doc", "doc_to_query"),
        partition_mode="per_direction",
    )

    assert loss.model is model


def test_cached_symmetric_loss_matches_uncached_loss_and_gradients() -> None:
    torch.manual_seed(7)
    reference_model = _ToyStaticSentenceModel()
    cached_model = copy.deepcopy(reference_model)
    features = _features()
    reference = MultipleNegativesRankingLoss(
        reference_model,
        scale=20.0,
        directions=("query_to_doc", "doc_to_query"),
        partition_mode="per_direction",
    )
    cached = StaticCompatibleCachedMultipleNegativesRankingLoss(
        cached_model,
        scale=20.0,
        mini_batch_size=2,
        directions=("query_to_doc", "doc_to_query"),
        partition_mode="per_direction",
    )

    reference_value = reference(features, labels=None)
    reference_value.backward()
    cached_value = cached(features, labels=None)
    cached_value.backward()

    assert cached_value.item() == pytest.approx(reference_value.item(), abs=1e-6)
    for cached_parameter, reference_parameter in zip(
        cached_model.parameters(), reference_model.parameters(), strict=True
    ):
        if cached_parameter.grad is None or reference_parameter.grad is None:
            assert cached_parameter.grad is reference_parameter.grad
        else:
            torch.testing.assert_close(
                cached_parameter.grad,
                reference_parameter.grad,
                atol=2e-6,
                rtol=2e-5,
            )


def test_cached_replay_observes_each_example_once_only_with_gradients() -> None:
    model = _ToyStaticSentenceModel()
    loss = StaticCompatibleCachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=2,
        directions=("query_to_doc", "doc_to_query"),
        partition_mode="per_direction",
    )

    loss(_features(), labels=None).backward()

    assert model.forward_modes.count((False, 2)) == 4
    assert model.forward_modes.count((True, 2)) == 4
    assert [batch for batch, _ in model.observations] == [2, 2, 2, 2]


def test_static_feature_slice_uses_offsets_to_slice_flattened_token_stream() -> None:
    features = {
        "input_ids": torch.tensor([10, 11, 12, 20, 21, 30]),
        "offsets": torch.tensor([0, 3, 5]),
    }

    sliced = _slice_static_features(features, begin=1, end=3)

    torch.testing.assert_close(sliced["input_ids"], torch.tensor([20, 21, 30]))
    torch.testing.assert_close(sliced["offsets"], torch.tensor([0, 2]))


class _FlattenedStaticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 3)
        self.static = _StaticSentinel()

    def __getitem__(self, index: int) -> torch.nn.Module:
        if index != 0:
            raise IndexError(index)
        return self.static

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_ids = features["input_ids"]
        offsets = features["offsets"]
        assert int(offsets[-1]) < token_ids.numel()
        ends = torch.cat(
            (offsets[1:], torch.tensor([token_ids.numel()], device=offsets.device))
        )
        rows = [
            self.embedding(token_ids[start:end]).mean(0)
            for start, end in zip(offsets.tolist(), ends.tolist(), strict=True)
        ]
        return {"sentence_embedding": torch.stack(rows)}


def test_cached_loss_handles_starse_flattened_input_ids_and_offsets() -> None:
    model = _FlattenedStaticModel()
    loss = StaticCompatibleCachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=2,
        directions=("query_to_doc", "doc_to_query"),
        partition_mode="per_direction",
    )
    features = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7]),
            "offsets": torch.tensor([0, 2, 4, 6]),
        },
        {
            "input_ids": torch.tensor([1, 2, 3, 8, 5, 6, 9]),
            "offsets": torch.tensor([0, 2, 4, 6]),
        },
    ]

    value = loss(features, labels=None)
    value.backward()

    assert torch.isfinite(value)
    assert model.embedding.weight.grad is not None


class _OnlineSentenceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        languages = ["eng_Latn", "deu_Latn"]
        vocab = {"alpha": 0, "beta": 1, "gamma": 2, "delta": 3}
        for language in languages:
            vocab[marker_for(language)] = len(vocab)
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="alpha"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.module = ConditionedStaticEmbedding(
            tokenizer,
            embedding_weights=torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.7, 0.4, 0.2],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            languages=languages,
            mode="collapse_online",
            ngram_buckets=17,
            online_rank=2,
        )
        table = self.module.online_collapse
        with torch.no_grad():
            table.score_hash.copy_(torch.linspace(-0.7, 0.7, 17))
            table.language_bias.copy_(torch.tensor([-0.2, 0.1]))
            table.merged.copy_(torch.arange(51).reshape(17, 3) / 19)

    def __getitem__(self, index: int) -> ConditionedStaticEmbedding:
        if index != 0:
            raise IndexError(index)
        return self.module

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.module(features)


class _Q15SentenceModel(_OnlineSentenceModel):
    def __init__(self) -> None:
        super().__init__()
        online = self.module
        replacement = ConditionedStaticEmbedding(
            online.tokenizer,
            embedding_weights=online.embedding.weight.detach().clone(),
            languages=online.languages,
            mode="collapse_dynamic",
            ngram_buckets=17,
        )
        with torch.no_grad():
            replacement.collapse.score.copy_(online.online_collapse.score_hash)
            replacement.collapse.language_bias.copy_(
                online.online_collapse.language_bias
            )
            replacement.collapse.merged.copy_(online.online_collapse.merged)
        self.module = replacement


def _online_features() -> list[dict[str, torch.Tensor]]:
    eng_marker = 4
    deu_marker = 5
    return [
        {
            "input_ids": torch.tensor(
                [
                    eng_marker, 0, 1, 2,
                    deu_marker, 1, 2,
                    eng_marker, 3, 0,
                    deu_marker, 2, 3, 1,
                ]
            ),
            "offsets": torch.tensor([0, 4, 7, 10]),
        },
        {
            "input_ids": torch.tensor(
                [
                    eng_marker, 0, 1, 3,
                    deu_marker, 1, 3,
                    eng_marker, 3, 0,
                    deu_marker, 2, 0, 1,
                ]
            ),
            "offsets": torch.tensor([0, 4, 7, 10]),
        },
    ]


def test_cached_online_loss_matches_uncached_loss_gradients_and_map_replay() -> None:
    """A different replay segmentation would corrupt the cached parameter gradient."""

    torch.manual_seed(21)
    reference_model = _OnlineSentenceModel()
    cached_model = copy.deepcopy(reference_model)
    reference = MultipleNegativesRankingLoss(
        reference_model,
        scale=20.0,
        directions=("query_to_doc", "doc_to_query"),
        partition_mode="per_direction",
    )
    cached = StaticCompatibleCachedMultipleNegativesRankingLoss(
        cached_model,
        scale=20.0,
        mini_batch_size=2,
        directions=("query_to_doc", "doc_to_query"),
        partition_mode="per_direction",
    )

    reference_value = reference(_online_features(), labels=None)
    reference_value.backward()
    cached_value = cached(_online_features(), labels=None)
    parameters_before_backward = {
        name: parameter.detach().clone()
        for name, parameter in cached_model.named_parameters()
    }
    cached_value.backward()
    cached_model.module.online_collapse.assert_cache_audit_empty()

    assert cached_value.item() == pytest.approx(reference_value.item(), abs=3e-5)
    for name, cached_parameter in cached_model.named_parameters():
        reference_parameter = dict(reference_model.named_parameters())[name]
        torch.testing.assert_close(
            cached_parameter,
            parameters_before_backward[name],
            atol=0.0,
            rtol=0.0,
        )
        if cached_parameter.grad is None or reference_parameter.grad is None:
            assert cached_parameter.grad is reference_parameter.grad
        else:
            torch.testing.assert_close(
                cached_parameter.grad,
                reference_parameter.grad,
                atol=3e-5,
                rtol=3e-4,
            )


def test_cached_online_loss_raises_when_score_mutation_changes_replay_map() -> None:
    """A mid-cache optimizer mutation must be a fatal segmentation mismatch."""

    model = _OnlineSentenceModel()
    loss = StaticCompatibleCachedMultipleNegativesRankingLoss(
        model,
        scale=20.0,
        mini_batch_size=2,
        directions=("query_to_doc", "doc_to_query"),
        partition_mode="per_direction",
    )

    value = loss(_online_features(), labels=None)
    with torch.no_grad():
        model.module.online_collapse.score_hash.add_(20.0)

    with pytest.raises(RuntimeError, match="cache/replay MAP mismatch"):
        value.backward()
    model.module.online_collapse.assert_cache_audit_empty()


@pytest.mark.parametrize(
    ("model_type", "table_name"),
    [(_OnlineSentenceModel, "online_collapse"), (_Q15SentenceModel, "collapse")],
)
def test_cached_vocabulary_telemetry_counts_record_not_replay(
    model_type: type[torch.nn.Module], table_name: str
) -> None:
    """Removing generic cache phases would double counts or omit Q15 entirely."""

    model = model_type()
    table = getattr(model.module, table_name)
    window = VocabularyWindow()
    table.set_vocabulary_telemetry(window)
    loss = StaticCompatibleCachedMultipleNegativesRankingLoss(
        model,
        scale=20.0,
        mini_batch_size=2,
        directions=("query_to_doc", "doc_to_query"),
        partition_mode="per_direction",
    )

    value = loss(_online_features(), labels=None)
    value.backward()
    if table_name == "online_collapse":
        table.assert_cache_audit_empty()

    metrics = window.snapshot()
    assert metrics["vocab/base_token_occurrences"] == 20
    assert metrics["vocab/eligible_edge_occurrences"] == 12
