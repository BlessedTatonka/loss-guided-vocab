from __future__ import annotations

import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from torch import nn

from scripts.benchmark_static_inference import model_inventory
from starse.collapse import (
    CollapseTable,
    ContrastiveVocabularyDiscovery,
    FrequencyVocabularyDiscovery,
    _weighted_index_add_,
    ablate_language_bias,
)
from starse.conditioned_module import ConditionedStaticEmbedding, marker_for
from starse.recursive_cascade import recursive_pair_key


def test_chunked_weighted_index_add_preserves_values_and_gradients() -> None:
    segment = torch.tensor([0, 1, 0, 1, 1, 0])
    source_values = torch.arange(18, dtype=torch.float64).reshape(6, 3) / 7
    weight_values = torch.tensor(
        [0.25, 1.0, -0.5, 0.75, 1.5, -1.0], dtype=torch.float64
    )

    direct_source = source_values.clone().requires_grad_()
    direct_weight = weight_values.clone().requires_grad_()
    direct = torch.zeros(2, 3, dtype=torch.float64)
    direct.index_add_(0, segment, direct_source * direct_weight.unsqueeze(1))
    direct.square().sum().backward()

    chunked_source = source_values.clone().requires_grad_()
    chunked_weight = weight_values.clone().requires_grad_()
    chunked = torch.zeros(2, 3, dtype=torch.float64)
    _weighted_index_add_(
        chunked,
        segment,
        chunked_source,
        chunked_weight,
        max_chunk_bytes=48,
    )
    chunked.square().sum().backward()

    torch.testing.assert_close(chunked, direct)
    torch.testing.assert_close(chunked_source.grad, direct_source.grad)
    torch.testing.assert_close(chunked_weight.grad, direct_weight.grad)


def test_mean_language_bias_ablation_preserves_mean_and_removes_variation() -> None:
    model = nn.Sequential(CollapseTable(buckets=8, dim=4, n_languages=3))
    table = model[0]
    with torch.no_grad():
        table.language_bias.copy_(torch.tensor([-3.0, -1.0, 1.0]))
        score_before = table.score.clone()
        merged_before = table.merged.clone()

    ablate_language_bias(model, "mean")

    torch.testing.assert_close(table.language_bias, torch.tensor([-1.0, -1.0, -1.0]))
    torch.testing.assert_close(table.score, score_before)
    torch.testing.assert_close(table.merged, merged_before)


def test_language_bias_ablation_rejects_noncollapse_model() -> None:
    with pytest.raises(ValueError, match="exactly one CollapseTable"):
        ablate_language_bias(nn.Linear(2, 2), "mean")


def test_shared_collapse_is_invariant_to_language_marker() -> None:
    languages = ["eng_Latn", "tel_Telu"]
    vocab = {"alpha": 0, "beta": 1}
    for language in languages:
        vocab[marker_for(language)] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="alpha"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    module = ConditionedStaticEmbedding(
        tokenizer,
        embedding_weights=torch.randn(len(vocab), 4),
        languages=languages,
        mode="collapse_shared",
        ngram_buckets=8,
    )
    ids = torch.tensor(
        [
            tokenizer.token_to_id(marker_for(languages[0])),
            0,
            1,
            tokenizer.token_to_id(marker_for(languages[1])),
            0,
            1,
        ]
    )
    output = module({"input_ids": ids, "offsets": torch.tensor([0, 3])})[
        "sentence_embedding"
    ]
    torch.testing.assert_close(output[0], output[1])


def _online_conditioned_module(
    *,
    mode: str = "collapse_online",
    compiled_pair_keys: list[int] | None = None,
    compiled_pair_scores: list[float] | None = None,
) -> tuple[ConditionedStaticEmbedding, Tokenizer]:
    languages = ["eng_Latn", "tel_Telu"]
    vocab = {"alpha": 0, "beta": 1, "gamma": 2}
    for language in languages:
        vocab[marker_for(language)] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="alpha"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    module = ConditionedStaticEmbedding(
        tokenizer,
        embedding_weights=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        languages=languages,
        mode=mode,
        ngram_buckets=19,
        online_rank=3,
        compiled_pair_keys=compiled_pair_keys,
        compiled_pair_scores=compiled_pair_scores,
    )
    return module, tokenizer


def _marked_features(tokenizer: Tokenizer) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(
            [
                tokenizer.token_to_id(marker_for("eng_Latn")),
                0,
                1,
                2,
                tokenizer.token_to_id(marker_for("tel_Telu")),
                1,
                2,
            ]
        ),
        "offsets": torch.tensor([0, 4]),
    }


def test_online_conditioned_mode_round_trips_embeddings_and_maps(tmp_path) -> None:
    """Omitting online tensors from save/load would silently reset the tokenizer."""

    module, tokenizer = _online_conditioned_module()
    table = module.online_collapse
    assert table is not None
    hash_01 = int((0 * 2654435761 + 1 * 40503) % table.buckets)
    with torch.no_grad():
        table.score_hash.fill_(-8.0)
        table.score_hash[hash_01] = 2.0
        table.language_bias.copy_(torch.tensor([0.3, -0.4]))
        table.merged[hash_01] = torch.tensor([2.0, 3.0, 0.5])
        table.interaction_weight.copy_(torch.tensor([0.2, -0.1, 0.4]))
    module.eval()
    expected = module(_marked_features(tokenizer))["sentence_embedding"]
    expected_map = table.last_map_edges.clone()

    module.save(str(tmp_path))
    loaded = ConditionedStaticEmbedding.load(str(tmp_path)).eval()
    observed = loaded(_marked_features(loaded.tokenizer))["sentence_embedding"]
    config = json.loads((tmp_path / "language_conditioning.json").read_text())

    torch.testing.assert_close(observed, expected)
    assert torch.equal(loaded.online_collapse.last_map_edges, expected_map)
    assert config["mode"] == "collapse_online"
    assert config["online_rank"] == 3
    assert config["compiled_pair_keys"] is None
    assert config["compiled_pair_scores"] is None
    for name, value in module.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_compiled_online_mode_routes_only_sorted_registered_exact_pairs(tmp_path) -> None:
    """Hash fallback in a compiled model would reintroduce excluded pair types."""

    key_01 = 1
    module, tokenizer = _online_conditioned_module(
        mode="collapse_online_compiled",
        compiled_pair_keys=[key_01],
        compiled_pair_scores=[1.5],
    )
    table = module.online_collapse
    assert table is not None
    with torch.no_grad():
        table.language_bias.copy_(torch.tensor([0.0, 0.0]))
        table.merged[0] = torch.tensor([3.0, 2.0, 0.0])
    module.eval()

    expected = module(_marked_features(tokenizer))["sentence_embedding"]
    assert torch.equal(
        table.last_map_edges,
        torch.tensor([[True, False], [False, False]]),
    )
    assert not hasattr(table, "score_hash")
    assert not hasattr(table, "left_projection")
    assert not hasattr(table, "right_projection")
    assert not hasattr(table, "interaction_weight")

    module.save(str(tmp_path))
    loaded = ConditionedStaticEmbedding.load(str(tmp_path)).eval()
    observed = loaded(_marked_features(loaded.tokenizer))["sentence_embedding"]

    torch.testing.assert_close(observed, expected)
    assert loaded.online_collapse.compiled_pair_keys.tolist() == [key_01]
    assert loaded.online_collapse.compiled_pair_scores.tolist() == pytest.approx([1.5])


@pytest.mark.parametrize(
    ("keys", "scores", "message"),
    [
        ([3, 2], [0.1, 0.2], "strictly increasing"),
        ([2], [0.1, 0.2], "same length"),
    ],
)
def test_compiled_online_mode_rejects_malformed_exact_table(
    keys: list[int], scores: list[float], message: str
) -> None:
    """An ambiguous routing table must fail before model inference."""

    with pytest.raises(ValueError, match=message):
        _online_conditioned_module(
            mode="collapse_online_compiled",
            compiled_pair_keys=keys,
            compiled_pair_scores=scores,
        )


def test_exact_vocabulary_collapses_only_promoted_pair() -> None:
    table = CollapseTable(
        buckets=4,
        dim=1,
        n_languages=1,
        bias_init=0.0,
        pair_key_base=10,
        pair_keys=[12],
        pair_slots=[0],
    )
    with torch.no_grad():
        table.merged[0] = 10.0
        table.score[0] = 20.0
    vectors = torch.tensor([[2.0], [4.0], [2.0], [6.0]])
    content = torch.tensor([1, 2, 1, 3])
    segment = torch.tensor([0, 0, 1, 1])
    language = torch.tensor([0, 0])

    output = table.pool(vectors, content, segment, language, n_sentences=2)

    torch.testing.assert_close(output[0], torch.tensor([13.0]))
    torch.testing.assert_close(output[1], torch.tensor([4.0]))


def test_two_tier_collapse_routes_exact_and_unmatched_pairs_separately() -> None:
    """Dropping unmatched pairs after the first growth step stops future growth."""

    table = CollapseTable(
        buckets=1,
        dim=1,
        n_languages=1,
        bias_init=0.0,
        pair_key_base=10,
        pair_keys=[12],
        pair_slots=[0],
        residual_buckets=5,
    )
    residual_slot = int(table._hash_residual(torch.tensor([1]), torch.tensor([3]))[0])
    with torch.no_grad():
        table.merged[0] = 10.0
        table.score[0] = 20.0
        table.residual_merged[residual_slot] = 14.0
        table.residual_score[residual_slot] = 20.0
    vectors = torch.tensor([[2.0], [4.0], [2.0], [6.0]], requires_grad=True)
    content = torch.tensor([1, 2, 1, 3])
    segment = torch.tensor([0, 0, 1, 1])

    output = table.pool(vectors, content, segment, torch.tensor([0, 0]), 2)
    output.sum().backward()

    torch.testing.assert_close(output[:, 0], torch.tensor([13.0, 18.0]))
    assert table.merged.grad is not None
    assert float(table.merged.grad[0]) > 0
    assert table.residual_merged.grad is not None
    assert float(table.residual_merged.grad[residual_slot]) > 0


def test_two_tier_discovery_observes_only_unmatched_residual_pairs() -> None:
    """Re-observing exact rows would repeatedly promote the same dictionary item."""

    table = CollapseTable(
        buckets=1,
        dim=2,
        n_languages=1,
        pair_key_base=10,
        pair_keys=[12],
        pair_slots=[0],
        residual_buckets=5,
    )
    table.configure_discovery(candidate_capacity=8, top_per_forward=8)
    vectors = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 1.0]],
        requires_grad=True,
    )

    table.pool(
        vectors,
        torch.tensor([1, 2, 1, 3]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 0]),
        2,
    ).square().sum().backward()

    assert table.discovery is not None
    rows = table.discovery.snapshot_records()
    assert [row["pair_key"] for row in rows] == [13]


def test_first_growth_migrates_exact_and_folded_residual_adamw_state() -> None:
    """Replacing parameters without their Adam moments changes the training run."""

    table = CollapseTable(
        buckets=8,
        dim=2,
        n_languages=1,
        pair_key_base=10,
    )
    optimizer = torch.optim.AdamW(table.parameters(), lr=0.01)
    (table.merged.sum() + table.score.sum() + table.language_bias.sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    old_merged = table.merged
    old_score = table.score
    with torch.no_grad():
        table.merged.copy_(torch.arange(16, dtype=torch.float32).reshape(8, 2))
        table.score.copy_(torch.arange(8, dtype=torch.float32) / 10)
        optimizer.state[old_merged]["exp_avg"].copy_(
            torch.arange(16, dtype=torch.float32).reshape(8, 2) / 100
        )
        optimizer.state[old_merged]["exp_avg_sq"].copy_(
            torch.arange(16, dtype=torch.float32).reshape(8, 2) / 1000
        )
        optimizer.state[old_score]["exp_avg"].copy_(
            torch.arange(8, dtype=torch.float32) / 100
        )
        optimizer.state[old_score]["exp_avg_sq"].copy_(
            torch.arange(8, dtype=torch.float32) / 1000
        )
    keys = (12, 13)
    source = table._hash(torch.tensor([1, 1]), torch.tensor([2, 3])).tolist()
    expected_exact = table.merged.detach()[source].clone()
    expected_exact_moment = optimizer.state[old_merged]["exp_avg"][source].clone()
    expected_residual = torch.stack(
        [table.merged.detach()[[row, row + 4]].mean(0) for row in range(4)]
    )
    expected_residual_moment = torch.stack(
        [
            optimizer.state[old_merged]["exp_avg"][[row, row + 4]].mean(0)
            for row in range(4)
        ]
    )

    report = table.grow_exact_vocabulary(
        keys,
        optimizer=optimizer,
        residual_buckets=4,
    )

    assert report == {
        "old_exact_rows": 0,
        "added_exact_rows": 2,
        "active_exact_rows": 2,
        "residual_rows": 4,
    }
    assert table.pair_keys.tolist() == [12, 13]
    assert table.pair_slots.tolist() == [0, 1]
    torch.testing.assert_close(table.merged, expected_exact)
    torch.testing.assert_close(table.residual_merged, expected_residual)
    torch.testing.assert_close(
        optimizer.state[table.merged]["exp_avg"], expected_exact_moment
    )
    torch.testing.assert_close(
        optimizer.state[table.residual_merged]["exp_avg"],
        expected_residual_moment,
    )
    assert old_merged not in optimizer.state
    assert old_score not in optimizer.state
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert sum(parameter is table.merged for parameter in parameters) == 1
    assert sum(parameter is table.score for parameter in parameters) == 1
    assert sum(parameter is table.residual_merged for parameter in parameters) == 1
    assert sum(parameter is table.residual_score for parameter in parameters) == 1


def test_first_growth_attaches_residual_parameters_removed_for_resume() -> None:
    """A collector resume omits empty residual tensors until growth makes them real."""

    table = CollapseTable(buckets=8, dim=2, n_languages=1, pair_key_base=10)
    table.residual_merged.requires_grad_(False)
    table.residual_score.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in table.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    (table.merged.sum() + table.score.sum() + table.language_bias.sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    report = table.grow_exact_vocabulary(
        (12, 13), optimizer=optimizer, residual_buckets=4
    )

    assert report["active_exact_rows"] == 2
    assert table.residual_merged.requires_grad
    assert table.residual_score.requires_grad
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert sum(parameter is table.residual_merged for parameter in parameters) == 1
    assert sum(parameter is table.residual_score for parameter in parameters) == 1
    tracked = [
        parameter for parameter in parameters
        if parameter in {
            table.merged, table.score, table.residual_merged, table.residual_score
        }
    ]
    assert tracked == [
        table.merged, table.score, table.residual_merged, table.residual_score
    ]
    assert optimizer.state[table.residual_merged]["exp_avg"].shape == (4, 2)
    assert optimizer.state[table.residual_score]["exp_avg"].shape == (4,)


def test_later_growth_preserves_existing_rows_and_appends_in_key_order() -> None:
    """Appending a lower key must not detach or misalign existing exact rows."""

    table = CollapseTable(buckets=8, dim=2, n_languages=1, pair_key_base=10)
    optimizer = torch.optim.AdamW(table.parameters(), lr=0.01)
    (table.merged.sum() + table.score.sum() + table.language_bias.sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    table.grow_exact_vocabulary((12, 13), optimizer=optimizer, residual_buckets=4)
    before_rows = {
        key: table.merged[index].detach().clone()
        for index, key in enumerate(table.pair_keys.tolist())
    }
    before_moments = {
        key: optimizer.state[table.merged]["exp_avg"][index].detach().clone()
        for index, key in enumerate(table.pair_keys.tolist())
    }
    residual_before = table.residual_merged.detach().clone()
    new_source = int(table._hash_residual(torch.tensor([1]), torch.tensor([1]))[0])
    expected_new = table.residual_merged[new_source].detach().clone()
    expected_new_moment = optimizer.state[table.residual_merged]["exp_avg"][
        new_source
    ].detach().clone()

    report = table.grow_exact_vocabulary(
        (11,), optimizer=optimizer, residual_buckets=4
    )

    assert report["old_exact_rows"] == 2
    assert report["active_exact_rows"] == 3
    assert table.pair_keys.tolist() == [11, 12, 13]
    torch.testing.assert_close(table.merged[0], expected_new)
    torch.testing.assert_close(optimizer.state[table.merged]["exp_avg"][0], expected_new_moment)
    for index, key in enumerate((12, 13), start=1):
        torch.testing.assert_close(table.merged[index], before_rows[key])
        torch.testing.assert_close(
            optimizer.state[table.merged]["exp_avg"][index], before_moments[key]
        )
    torch.testing.assert_close(table.residual_merged, residual_before)

    output = table.pool(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
        torch.tensor([1, 1]),
        torch.tensor([0, 0]),
        torch.tensor([0]),
        1,
    )
    output.square().sum().backward()
    assert table.merged.grad is not None
    assert bool(torch.isfinite(table.merged.grad).all())


def test_discovery_promotion_obeys_hard_budget_and_language_reservation() -> None:
    discovery = ContrastiveVocabularyDiscovery(
        n_languages=2,
        candidate_capacity=8,
        top_per_forward=8,
    )
    # Negative gradients give positive -p*dL/dp utility. Pair 10 belongs to
    # language 0, pair 20 to language 1, and pair 30 is the global filler.
    discovery.observe(
        torch.tensor([10] * 4 + [20] * 4 + [30] * 4),
        torch.tensor([0] * 4 + [1] * 4 + [0] * 4),
        torch.ones(12),
        torch.tensor([-3.0] * 4 + [-2.0] * 4 + [-1.0] * 4),
    )

    promoted = discovery.promote(
        budget=2,
        min_support=2,
        per_language_quota=1,
        per_language_min_support=2,
    )

    assert len(promoted.pair_keys) == 2
    assert set(promoted.pair_keys) == {10, 20}
    assert promoted.reservation_memberships == ((10, 0), (20, 1))
    assert all(row["reservation_language_count"] == 1 for row in promoted.records)
    assert all(row["max_reservation_support"] == 4 for row in promoted.records)


def test_discovery_promotion_uses_utility_threshold_before_safety_ceiling() -> None:
    discovery = ContrastiveVocabularyDiscovery(
        n_languages=1,
        candidate_capacity=8,
        top_per_forward=8,
    )
    discovery.observe(
        torch.tensor([10, 20, 30]),
        torch.zeros(3, dtype=torch.long),
        torch.ones(3),
        torch.tensor([-3e-4, -2.5e-4, -1e-4]),
    )

    promoted = discovery.promote(
        budget=8,
        min_support=1,
        per_language_quota=0,
        per_language_min_support=1,
        min_utility=2e-4,
    )

    assert promoted.pair_keys == (10, 20)
    assert len(promoted.pair_keys) < 8
    assert min(row["utility"] for row in promoted.records) >= 2e-4


def test_promotion_audit_measures_quota_utility_displacement() -> None:
    discovery = ContrastiveVocabularyDiscovery(
        n_languages=2,
        candidate_capacity=8,
        top_per_forward=8,
    )
    discovery.observe(
        torch.tensor([10] * 4 + [20] * 4 + [30] * 4),
        torch.tensor([0] * 4 + [1] * 4 + [0] * 4),
        torch.ones(12),
        torch.tensor([-3.0] * 4 + [-1.0] * 4 + [-2.5] * 4),
    )
    promoted = discovery.promote(
        budget=2,
        min_support=2,
        per_language_quota=1,
        per_language_min_support=2,
    )

    audit = discovery.audit_promotion(
        promoted,
        budget=2,
        min_support=2,
        per_language_quota=1,
        per_language_min_support=2,
    )

    assert audit["candidate_table"] == {
        "capacity": 8,
        "pair_language_records": 3,
        "saturation": pytest.approx(0.375),
        "candidate_pairs": 3,
        "initializer_languages_total": 2,
        "languages_total": 2,
        "trainable_language_indices": [0, 1],
        "languages_with_candidates": 2,
        "records_per_language": [2, 1],
        "reservation_eligible_records_per_language": [2, 1],
        "min_records_per_language": 1,
        "median_records_per_language": pytest.approx(1.5),
        "max_records_per_language": 2,
        "admission_pressure": {
            "unique_records_before_transfer": 3,
            "dropped_by_top_per_forward": 0,
            "transferred_records": 3,
            "retained_updates": 0,
            "admitted_into_free_slots": 3,
            "capacity_replacements": 0,
            "capacity_rejections": 0,
        },
    }
    assert audit["reservation"] == {
        "quota_per_language": 1,
        "membership_capacity": 2,
        "memberships": 2,
        "membership_occupancy": pytest.approx(1.0),
        "unique_reserved_rows": 2,
        "languages_with_reservations": 2,
        "memberships_per_language": [1, 1],
    }
    assert audit["global_only_counterfactual"] == {
        "global_selected_rows": 2,
        "global_selected_utility": pytest.approx(22.0),
        "reservation_selected_rows": 2,
        "reservation_selected_utility": pytest.approx(16.0),
        "utility_displaced_by_reservations": pytest.approx(6.0),
        "relative_utility_displaced_by_reservations": pytest.approx(6.0 / 22.0),
        "reservation_only_rows": 1,
        "global_only_rows": 1,
    }


def test_global_only_deployment_can_audit_scaled_reservation_counterfactual() -> None:
    discovery = ContrastiveVocabularyDiscovery(
        n_languages=3,
        candidate_capacity=8,
        top_per_forward=8,
    )
    discovery.observe(
        torch.tensor([10] * 4 + [20] * 4 + [30] * 4),
        torch.tensor([0] * 4 + [1] * 4 + [0] * 4),
        torch.ones(12),
        torch.tensor([-3.0] * 4 + [-1.0] * 4 + [-2.5] * 4),
    )
    global_only = discovery.promote(
        budget=2,
        min_support=2,
        per_language_quota=0,
        per_language_min_support=2,
    )

    audit = discovery.audit_promotion(
        global_only,
        budget=2,
        min_support=2,
        per_language_quota=0,
        per_language_min_support=2,
        reservation_audit_quota=1,
        trainable_language_indices=(0, 1),
    )

    assert audit["deployed_selector"] == {
        "per_language_quota": 0,
        "matches_global_only": True,
        "reservation_audit_quota": 1,
    }
    assert audit["reservation"]["memberships"] == 2
    assert audit["reservation"]["membership_capacity"] == 2
    assert audit["reservation"]["membership_occupancy"] == pytest.approx(1.0)
    assert audit["candidate_table"]["initializer_languages_total"] == 3
    assert audit["candidate_table"]["languages_total"] == 2
    assert audit["candidate_table"]["records_per_language"] == [2, 1]
    assert audit["global_only_counterfactual"][
        "utility_displaced_by_reservations"
    ] == pytest.approx(6.0)


def test_promotion_audit_records_transfer_and_capacity_pressure() -> None:
    discovery = ContrastiveVocabularyDiscovery(
        n_languages=1,
        candidate_capacity=1,
        top_per_forward=1,
    )
    discovery.observe(
        torch.tensor([10]),
        torch.tensor([0]),
        torch.ones(1),
        torch.tensor([-100.0]),
    )
    discovery.observe(
        torch.tensor([20, 21]),
        torch.tensor([0, 0]),
        torch.ones(2),
        torch.tensor([-1.0, -0.5]),
    )
    discovery.observe(
        torch.tensor([30]),
        torch.tensor([0]),
        torch.ones(1),
        torch.tensor([-200.0]),
    )
    promoted = discovery.promote(
        budget=1,
        min_support=1,
        per_language_quota=0,
        per_language_min_support=1,
    )

    audit = discovery.audit_promotion(
        promoted,
        budget=1,
        min_support=1,
        per_language_quota=0,
        per_language_min_support=1,
    )

    assert audit["candidate_table"]["admission_pressure"] == {
        "unique_records_before_transfer": 4,
        "dropped_by_top_per_forward": 1,
        "transferred_records": 3,
        "retained_updates": 0,
        "admitted_into_free_slots": 1,
        "capacity_replacements": 1,
        "capacity_rejections": 1,
    }


def test_language_reservation_cannot_promote_globally_negative_pair() -> None:
    discovery = ContrastiveVocabularyDiscovery(
        n_languages=2,
        candidate_capacity=8,
        top_per_forward=8,
    )
    # Pair 10 looks useful in language 0 but is harmful in aggregate. Pair 20
    # is globally positive and is therefore the only eligible reservation.
    discovery.observe(
        torch.tensor([10, 10, 20]),
        torch.tensor([0, 1, 1]),
        torch.ones(3),
        torch.tensor([-1.0, 3.0, -0.5]),
    )

    promoted = discovery.promote(
        budget=2,
        min_support=1,
        per_language_quota=1,
        per_language_min_support=1,
    )

    assert promoted.pair_keys == (20,)
    assert promoted.records[0]["utility"] > 0


def test_discovery_retained_records_never_exceed_candidate_capacity() -> None:
    discovery = ContrastiveVocabularyDiscovery(
        n_languages=1,
        candidate_capacity=4,
        top_per_forward=8,
    )
    discovery.observe(
        torch.arange(8),
        torch.zeros(8, dtype=torch.long),
        torch.ones(8),
        -torch.arange(1, 9, dtype=torch.float32),
    )

    assert len(discovery.records) == 4
    retained_pairs = {composite for composite in discovery.records}
    assert retained_pairs == {4, 5, 6, 7}


def test_countsketch_admits_late_recurring_utility_over_early_candidate() -> None:
    discovery = ContrastiveVocabularyDiscovery(
        n_languages=1,
        candidate_capacity=1,
        top_per_forward=1,
    )

    def observe(pair: int, utility: float) -> None:
        discovery.observe(
            torch.tensor([pair]),
            torch.tensor([0]),
            torch.ones(1),
            torch.tensor([-utility]),
        )

    observe(10, 1.0)
    observe(20, 0.6)
    observe(20, 0.6)

    assert set(discovery.records) == {20}
    assert float(discovery.records[20][0]) == pytest.approx(0.6)


def test_discovery_lazy_heap_metadata_is_bounded_after_repeated_updates() -> None:
    discovery = ContrastiveVocabularyDiscovery(
        n_languages=1,
        candidate_capacity=4,
        top_per_forward=4,
    )
    keys = torch.arange(4)
    for _ in range(20):
        discovery.observe(
            keys,
            torch.zeros(4, dtype=torch.long),
            torch.ones(4),
            -torch.arange(1, 5, dtype=torch.float32),
        )

    assert len(discovery.records) == 4
    assert len(discovery._heap) <= 2 * discovery.candidate_capacity
    promoted = discovery.promote(
        budget=4,
        min_support=1,
        per_language_quota=0,
        per_language_min_support=1,
    )
    assert promoted.pair_keys == (0, 1, 2, 3)


def test_discovery_snapshot_records_is_complete_sorted_and_decoded() -> None:
    discovery = ContrastiveVocabularyDiscovery(
        n_languages=2,
        candidate_capacity=8,
        top_per_forward=8,
    )
    discovery.observe(
        torch.tensor([20, 10, 20]),
        torch.tensor([1, 0, 1]),
        torch.tensor([0.25, 0.5, 0.75]),
        torch.tensor([-2.0, -1.0, -1.0]),
    )

    rows = discovery.snapshot_records()

    assert [row["composite_key"] for row in rows] == [20, 41]
    assert rows[0] == {
        "composite_key": 20,
        "pair_key": 10,
        "language_index": 0,
        "utility": pytest.approx(0.5),
        "probability_sum": pytest.approx(0.5),
        "captured_support": 1,
        "mean_probability": pytest.approx(0.5),
    }
    assert rows[1]["pair_key"] == 20
    assert rows[1]["language_index"] == 1
    assert rows[1]["captured_support"] == 2
    assert rows[1]["probability_sum"] == pytest.approx(1.0)


def test_discovery_state_round_trip_preserves_promotion_and_future_updates() -> None:
    original = ContrastiveVocabularyDiscovery(
        n_languages=3,
        candidate_capacity=8,
        top_per_forward=4,
    )
    original.observe(
        torch.tensor([10, 10, 20, 30]),
        torch.tensor([0, 1, 2, 1]),
        torch.tensor([0.2, 0.4, 0.6, 0.8]),
        torch.tensor([-1.0, -2.0, -3.0, 1.0]),
    )
    restored = ContrastiveVocabularyDiscovery(
        n_languages=3,
        candidate_capacity=8,
        top_per_forward=4,
    )

    restored.load_state_dict(original.state_dict())

    assert restored.snapshot_records() == original.snapshot_records()
    torch.testing.assert_close(restored._utility_sketch, original._utility_sketch)
    assert restored.observed_occurrences == original.observed_occurrences
    assert restored.transferred_records == original.transferred_records
    assert restored.admission_pressure() == original.admission_pressure()
    for discovery in (original, restored):
        discovery.observe(
            torch.tensor([10, 40]),
            torch.tensor([0, 2]),
            torch.ones(2),
            torch.tensor([-0.5, -4.0]),
        )
    assert restored.promote(
        budget=3,
        min_support=1,
        per_language_quota=1,
        per_language_min_support=1,
    ) == original.promote(
        budget=3,
        min_support=1,
        per_language_quota=1,
        per_language_min_support=1,
    )


def test_discovery_state_rejects_incompatible_capacity() -> None:
    source = ContrastiveVocabularyDiscovery(
        n_languages=2,
        candidate_capacity=8,
        top_per_forward=4,
    )
    target = ContrastiveVocabularyDiscovery(
        n_languages=2,
        candidate_capacity=4,
        top_per_forward=4,
    )

    with pytest.raises(ValueError, match="candidate capacity"):
        target.load_state_dict(source.state_dict())


def test_frequency_discovery_ranks_global_support_without_reading_utility() -> None:
    discovery = FrequencyVocabularyDiscovery(
        n_languages=3,
        candidate_capacity=8,
        top_per_forward=8,
    )
    discovery.observe(
        torch.tensor([10, 10, 10, 20, 20, 30]),
        torch.tensor([0, 1, 2, 0, 0, 1]),
        torch.tensor([0.1, 0.9, 0.4, 0.3, 0.7, 0.5]),
        # Pair 10 has negative utility and pair 30 has the largest positive
        # utility.  Neither value is allowed to affect frequency membership.
        torch.tensor([5.0, 5.0, 5.0, -1.0, -1.0, -100.0]),
    )

    rows = discovery.snapshot_records()

    assert [row["pair_key"] for row in rows] == [10, 20, 30]
    assert {row["pair_key"]: row["captured_support"] for row in rows} == {
        10: 3,
        20: 2,
        30: 1,
    }
    assert all(row["language_index"] == -1 for row in rows)
    assert all(row["utility"] == 0.0 for row in rows)


def test_frequency_discovery_count_min_admits_late_recurring_pair() -> None:
    discovery = FrequencyVocabularyDiscovery(
        n_languages=1,
        candidate_capacity=1,
        top_per_forward=1,
    )

    def observe(pair: int, occurrences: int) -> None:
        discovery.observe(
            torch.tensor([pair] * occurrences),
            torch.zeros(occurrences, dtype=torch.long),
            torch.ones(occurrences),
            torch.zeros(occurrences),
        )

    observe(10, 2)
    observe(20, 1)
    observe(20, 2)

    assert set(discovery.records) == {20}
    assert int(discovery.records[20][2]) == 3


def test_frequency_discovery_state_round_trip_preserves_future_ranking() -> None:
    original = FrequencyVocabularyDiscovery(
        n_languages=2,
        candidate_capacity=4,
        top_per_forward=2,
    )
    original.observe(
        torch.tensor([10, 10, 20, 30]),
        torch.tensor([0, 1, 0, 1]),
        torch.ones(4),
        torch.tensor([1.0, -1.0, 3.0, -7.0]),
    )
    restored = FrequencyVocabularyDiscovery(
        n_languages=2,
        candidate_capacity=4,
        top_per_forward=2,
    )

    restored.load_state_dict(original.state_dict())

    assert restored.snapshot_records() == original.snapshot_records()
    assert restored.admission_pressure() == original.admission_pressure()
    for discovery in (original, restored):
        discovery.observe(
            torch.tensor([20, 20, 40]),
            torch.tensor([0, 1, 0]),
            torch.ones(3),
            torch.tensor([-100.0, 100.0, -5.0]),
        )
    assert restored.snapshot_records() == original.snapshot_records()
    torch.testing.assert_close(
        restored._frequency_sketch, original._frequency_sketch
    )


def test_collapse_can_attach_frequency_only_discovery() -> None:
    table = CollapseTable(
        buckets=8,
        dim=2,
        n_languages=1,
        pair_key_base=16,
    )
    table.configure_discovery(
        candidate_capacity=8,
        top_per_forward=8,
        admission_mode="frequency",
    )
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

    table.pool(
        vectors,
        torch.tensor([3, 4]),
        torch.tensor([0, 0]),
        torch.tensor([0]),
        n_sentences=1,
    ).square().sum().backward()

    assert isinstance(table.discovery, FrequencyVocabularyDiscovery)
    assert table.discovery.snapshot_records()[0]["pair_key"] == 3 * 16 + 4


def test_collapse_backward_populates_discovery_sketch() -> None:
    table = CollapseTable(
        buckets=8,
        dim=2,
        n_languages=1,
        pair_key_base=16,
    )
    table.configure_discovery(candidate_capacity=8, top_per_forward=8)
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    output = table.pool(
        vectors,
        torch.tensor([3, 4]),
        torch.tensor([0, 0]),
        torch.tensor([0]),
        n_sentences=1,
    )

    output.square().sum().backward()

    assert table.discovery is not None
    assert table.discovery.observed_occurrences == 1
    assert table.discovery.transferred_records == 1
    assert len(table.discovery.records) == 1


def test_window_discovery_resets_without_resetting_cumulative() -> None:
    table = CollapseTable(
        buckets=16, dim=3, n_languages=2, pair_key_base=100
    )
    table.configure_discovery(candidate_capacity=16, top_per_forward=16)
    table.configure_diagnostic_discovery(candidate_capacity=16, top_per_forward=16)
    vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    content = torch.tensor([1, 2])
    segment = torch.tensor([0, 0])
    language = torch.tensor([0])

    table.pool(vectors, content, segment, language, 1).sum().backward()

    assert table.discovery is not None
    assert table.diagnostic_discovery is not None
    assert table.discovery.snapshot_records() == table.diagnostic_discovery.snapshot_records()
    table.configure_diagnostic_discovery(candidate_capacity=16, top_per_forward=16)
    table.zero_grad(set_to_none=True)
    table.pool(vectors, content, segment, language, 1).sum().backward()
    cumulative = table.discovery.snapshot_records()
    window = table.diagnostic_discovery.snapshot_records()
    assert cumulative[0]["captured_support"] == 2
    assert window[0]["captured_support"] == 1


def test_collision_promotion_allocates_distinct_rows_and_compacts() -> None:
    table = CollapseTable(
        buckets=8,
        dim=2,
        n_languages=1,
        pair_key_base=16,
    )
    pairs_by_hash: dict[int, list[int]] = {}
    for left in range(1, 8):
        for right in range(1, 8):
            key = left * 16 + right
            bucket = int(table._hash(torch.tensor([left]), torch.tensor([right]))[0])
            pairs_by_hash.setdefault(bucket, []).append(key)
    collision = next(sorted(values)[:2] for values in pairs_by_hash.values() if len(values) >= 2)
    source = int(table._hash(
        torch.tensor([collision[0] // 16]),
        torch.tensor([collision[0] % 16]),
    )[0])
    with torch.no_grad():
        table.merged[source].copy_(torch.tensor([2.0, 3.0]))
        table.score[source] = 0.75

    report = table.promote_exact_vocabulary(tuple(collision))

    assert report["active_pairs"] == 2
    assert report["promoted_hash_collisions"] == 1
    assert len(set(table.pair_slots.tolist())) == 2
    torch.testing.assert_close(table.merged[table.pair_slots[0]], table.merged[table.pair_slots[1]])
    compact = table.compact_exact_vocabulary()
    assert compact == {"old_rows": 8, "active_rows": 2, "dropped_rows": 6}
    assert table.merged.shape == (2, 2)
    assert table.pair_slots.tolist() == [0, 1]


def test_dynamic_vocabulary_round_trips_compact_model(tmp_path) -> None:
    language = "eng_Latn"
    vocab = {"alpha": 0, "beta": 1, marker_for(language): 2}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="alpha"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    module = ConditionedStaticEmbedding(
        tokenizer,
        embedding_weights=torch.randn(len(vocab), 4),
        languages=[language],
        mode="collapse_dynamic",
        ngram_buckets=8,
    )
    key = 0 * len(vocab) + 1
    module.promote_dynamic_vocabulary([key])
    module.compact_dynamic_vocabulary()
    module.save(str(tmp_path))

    loaded = ConditionedStaticEmbedding.load(str(tmp_path))

    assert loaded.mode == "collapse_dynamic"
    assert loaded.ngram_buckets == 1
    assert loaded.collapse.merged.shape[0] == 1
    assert loaded.collapse.pair_keys.tolist() == [key]


def test_growing_vocabulary_round_trips_exact_and_residual_tiers(tmp_path) -> None:
    """Omitting residual metadata makes a growing checkpoint impossible to resume."""

    language = "eng_Latn"
    vocab = {"alpha": 0, "beta": 1, "gamma": 2, marker_for(language): 3}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="alpha"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    module = ConditionedStaticEmbedding(
        tokenizer,
        embedding_weights=torch.randn(len(vocab), 4),
        languages=[language],
        mode="collapse_dynamic",
        ngram_buckets=8,
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    sum(parameter.sum() for parameter in module.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    key = 0 * len(vocab) + 1

    module.grow_dynamic_vocabulary(
        [key], optimizer=optimizer, residual_buckets=4
    )
    features = {
        "input_ids": torch.tensor(
            [tokenizer.token_to_id(marker_for(language)), 0, 1, 2]
        ),
        "offsets": torch.tensor([0]),
    }
    expected = module(features)["sentence_embedding"]
    module.save(str(tmp_path))

    loaded = ConditionedStaticEmbedding.load(str(tmp_path))
    observed = loaded(features)["sentence_embedding"]
    config = json.loads((tmp_path / "language_conditioning.json").read_text())

    torch.testing.assert_close(observed, expected)
    assert loaded.ngram_buckets == 1
    assert loaded.growth_residual_buckets == 4
    assert loaded.collapse.merged.shape == (1, 4)
    assert loaded.collapse.residual_merged.shape == (4, 4)
    assert config["growth_residual_buckets"] == 4


def test_recursive_cascade_round_trips_dense_rows_and_rule_dag(tmp_path) -> None:
    language = "eng_Latn"
    vocab = {"alpha": 0, "beta": 1, "gamma": 2, marker_for(language): 3}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="alpha"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    module = ConditionedStaticEmbedding(
        tokenizer,
        embedding_weights=torch.randn(len(vocab), 4),
        languages=[language],
        mode="collapse_recursive",
        ngram_buckets=8,
        recursive_max_span_length=4,
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    sum(parameter.square().sum() for parameter in module.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    module.grow_recursive_vocabulary(
        [recursive_pair_key(0, 1)], optimizer=optimizer
    )
    features = {
        "input_ids": torch.tensor([3, 0, 1, 2]),
        "offsets": torch.tensor([0]),
    }
    expected = module(dict(features))
    assert expected["recursive_token_ids"].tolist() == [4, 2]
    module.save(str(tmp_path))

    loaded = ConditionedStaticEmbedding.load(str(tmp_path))
    observed = loaded(dict(features))

    torch.testing.assert_close(
        observed["sentence_embedding"], expected["sentence_embedding"]
    )
    assert observed["recursive_token_ids"].tolist() == [4, 2]
    assert loaded.recursive_cascade.learned.shape == (1, 4)
    assert loaded.recursive_cascade.token_span(4) == (0, 1)
    config = json.loads((tmp_path / "language_conditioning.json").read_text())
    assert config["recursive_learned_count"] == 1
    assert config["recursive_max_span_length"] == 4


def test_reclaimed_collapse_expands_token_and_reads_merged_vector_from_base(tmp_path) -> None:
    language = "eng_Latn"
    vocab = {"alpha": 0, "beta": 1, "gamma": 2, marker_for(language): 3}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="alpha"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    weights = torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [3.0, 4.0], [0.0, 0.0],
    ])
    module = ConditionedStaticEmbedding(
        tokenizer,
        embedding_weights=weights,
        languages=[language],
        mode="collapse_reclaimed",
        ngram_buckets=1,
        dynamic_pair_keys=[1],  # alpha + beta under key base 4
        dynamic_pair_slots=[0],
        reclaimed_token_ids=[2],
        reclaimed_left_ids=[0],
        reclaimed_right_ids=[1],
    )
    with torch.no_grad():
        module.collapse.score[0] = 0.7
        module.collapse.language_bias[0] = -0.2
    marker = vocab[marker_for(language)]
    reclaimed_features = {
        "input_ids": torch.tensor([marker, 2]), "offsets": torch.tensor([0]),
    }
    expanded_features = {
        "input_ids": torch.tensor([marker, 0, 1]), "offsets": torch.tensor([0]),
    }

    reclaimed_output = module(reclaimed_features.copy())["sentence_embedding"]
    expanded_output = module(expanded_features.copy())["sentence_embedding"]
    torch.testing.assert_close(reclaimed_output, expanded_output)
    assert module.collapse.merged.shape == (0, 2)

    module.save(str(tmp_path))
    loaded = ConditionedStaticEmbedding.load(str(tmp_path))
    loaded_output = loaded(reclaimed_features.copy())["sentence_embedding"]
    torch.testing.assert_close(loaded_output, reclaimed_output)
    assert loaded.mode == "collapse_reclaimed"
    assert loaded.collapse.merged.numel() == 0
    assert loaded.collapse.reclaimed_token_ids.tolist() == [2]


def test_direct_reclaimed_promotion_maps_adam_state_and_round_trips(tmp_path) -> None:
    language = "eng_Latn"
    vocab = {
        "alpha": 0, "beta": 1, "left": 2, "right": 3, "unused": 4,
        marker_for(language): 5,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="alpha"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    weights = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10

    def prepared():
        module = ConditionedStaticEmbedding(
            tokenizer,
            embedding_weights=weights.clone(),
            languages=[language],
            mode="collapse_dynamic",
            ngram_buckets=8,
        )
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
        marker = vocab[marker_for(language)]
        features = {
            "input_ids": torch.tensor([marker, 0, 1]),
            "offsets": torch.tensor([0]),
        }
        module(features)["sentence_embedding"].square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return module, optimizer, features

    standard, standard_optimizer, features = prepared()
    direct, direct_optimizer, _ = prepared()
    key = 1  # alpha + beta under key base 6
    standard.promote_dynamic_vocabulary([key], optimizer=standard_optimizer)
    direct_nonreclaimed = direct.embedding.weight.detach().clone()[[0, 1, 2, 3, 5]]
    direct_state_nonreclaimed = direct_optimizer.state[direct.embedding.weight][
        "exp_avg"
    ].detach().clone()[[0, 1, 2, 3, 5]]
    report = direct.promote_dynamic_vocabulary_to_reclaimed(
        [key], [4], [2], [3], optimizer=direct_optimizer
    )

    standard_slot = int(standard.collapse.pair_slots[0])
    torch.testing.assert_close(
        direct.embedding.weight[4], standard.collapse.merged[standard_slot]
    )
    torch.testing.assert_close(direct.collapse.score, standard.collapse.score[[standard_slot]])
    torch.testing.assert_close(
        direct_optimizer.state[direct.embedding.weight]["exp_avg"][4],
        standard_optimizer.state[standard.collapse.merged]["exp_avg"][standard_slot],
    )
    torch.testing.assert_close(
        direct_optimizer.state[direct.collapse.score]["exp_avg"],
        standard_optimizer.state[standard.collapse.score]["exp_avg"][[standard_slot]],
    )
    torch.testing.assert_close(
        direct.embedding.weight[[0, 1, 2, 3, 5]], direct_nonreclaimed
    )
    torch.testing.assert_close(
        direct_optimizer.state[direct.embedding.weight]["exp_avg"][[0, 1, 2, 3, 5]],
        direct_state_nonreclaimed,
    )
    torch.testing.assert_close(
        direct(features.copy())["sentence_embedding"],
        standard(features.copy())["sentence_embedding"],
    )
    assert report["released_merged_rows"] == 8
    assert direct.mode == "collapse_reclaimed"
    assert direct.collapse.merged.shape == (0, 4)
    assert direct.collapse.reclaimed_token_ids.tolist() == [4]
    assert all(
        tuple(value.shape) != (8, 4)
        for state in direct_optimizer.state.values()
        for value in state.values() if torch.is_tensor(value)
    )

    direct.save(str(tmp_path))
    loaded = ConditionedStaticEmbedding.load(str(tmp_path))
    torch.testing.assert_close(
        loaded(features.copy())["sentence_embedding"],
        direct(features.copy())["sentence_embedding"],
    )


def test_static_cost_inventory_reads_collapse_parameter_shape(tmp_path) -> None:
    language = "eng_Latn"
    vocab = {"alpha": 0, "beta": 1, marker_for(language): 2}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="alpha"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    module = ConditionedStaticEmbedding(
        tokenizer,
        embedding_weights=torch.randn(len(vocab), 4),
        languages=[language],
        mode="collapse_dynamic",
        ngram_buckets=8,
    )
    module.promote_dynamic_vocabulary([1])
    module.compact_dynamic_vocabulary()
    module.save(str(tmp_path))

    inventory = model_inventory(nn.Sequential(module), tmp_path)

    assert inventory["auxiliary_rows"] == 1
    assert inventory["auxiliary_dimension"] == 4


def test_external_vocabulary_selection_validates_discovery_membership() -> None:
    discovery = ContrastiveVocabularyDiscovery(
        n_languages=2, candidate_capacity=8, top_per_forward=8
    )
    discovery.records = {
        10 * 2: [1.0, 0.5, 4],
        10 * 2 + 1: [0.5, 0.25, 2],
        20 * 2: [2.0, 1.0, 3],
    }

    selected = discovery.select_external([10, 20], minimum_support=2)

    assert selected.pair_keys == (10, 20)
    assert selected.candidate_count == 2
    assert [row["captured_support"] for row in selected.records] == [6, 3]
    assert [row["language_count"] for row in selected.records] == [2, 1]
    with pytest.raises(ValueError, match="sorted and unique"):
        discovery.select_external([20, 10], minimum_support=2)
    with pytest.raises(ValueError, match="absent from discovery"):
        discovery.select_external([10, 30], minimum_support=2)
    with pytest.raises(ValueError, match="utility/support eligibility"):
        discovery.select_external([20], minimum_support=4)
