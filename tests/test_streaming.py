import random
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from starse.streaming import (
    _batch_no_duplicates,
    _LemmaOverlapFilter,
    _queued_pairs,
    _random_pairs,
    _sample_pair_columns,
    discover_shards,
)
from starse.train_multilingual import _resolve_resume_scheduler_start_batch


def _write_pairs(
    path, anchor_column: str, positive_column: str, prefix: str, rows: int = 12
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            anchor_column: [f"{prefix}-anchor-{index}" for index in range(rows)],
            positive_column: [f"{prefix}-positive-{index}" for index in range(rows)],
        }
    )
    pq.write_table(table, path, row_group_size=4)


def test_pair_column_sampling_is_bounded_and_reproducible() -> None:
    table = pa.table(
        {
            "anchor": [f"a-{index}" for index in range(100)],
            "positive": [f"p-{index}" for index in range(100)],
        }
    )

    first = _sample_pair_columns(table, random.Random(11), fragment_rows=8)
    second = _sample_pair_columns(table, random.Random(11), fragment_rows=8)

    assert first == second
    assert len(first[0]) == len(first[1]) == 8
    assert len(set(first[0])) == 8


def test_random_pair_stream_is_mixed_reproducible_and_prefetched(tmp_path) -> None:
    fineweb = tmp_path / "fineweb.parquet"
    parallel = tmp_path / "parallel.parquet"
    _write_pairs(fineweb, "sentence_1", "sentence_2", "fineweb")
    _write_pairs(parallel, "english", "non_english", "parallel")
    shards = [
        {
            "source": "fineweb2",
            "path": str(fineweb),
            "anchor_column": "sentence_1",
            "positive_column": "sentence_2",
        },
        {
            "source": "ccmatrix",
            "path": str(parallel),
            "anchor_column": "english",
            "positive_column": "non_english",
        },
    ]
    settings = {
        "seed": 7,
        "source_weights": {"fineweb2": 0.5, "ccmatrix": 0.5},
        "fragment_rows": 3,
        "max_chars": 512,
        "shuffle_buffer_size": 4,
        "queue_size": 2,
    }

    first = list(islice(_random_pairs(shards, settings), 30))
    second = list(islice(_random_pairs(shards, settings), 30))
    prefetched = list(islice(_queued_pairs(shards, settings), 10))

    assert first == second
    assert any(row["anchor"].startswith("fineweb") for row in first)
    assert any(row["anchor"].startswith("parallel") for row in first)
    assert prefetched == first[:10]


def test_monolingual_batches_keep_source_specific_language_markers(tmp_path) -> None:
    """FineWeb l->l and CCMatrix en->l must not share one random marker."""
    fineweb = tmp_path / "fineweb.parquet"
    parallel = tmp_path / "parallel.parquet"
    _write_pairs(fineweb, "sentence_1", "sentence_2", "fineweb")
    _write_pairs(parallel, "english", "non_english", "parallel")
    shards = [
        {
            "source": "fineweb2",
            "path": str(fineweb),
            "anchor_column": "sentence_1",
            "positive_column": "sentence_2",
            "sampling_weight": 1.0,
            "language_id": "rus_Cyrl",
            "anchor_language_id": "rus_Cyrl",
        },
        {
            "source": "ccmatrix",
            "path": str(parallel),
            "anchor_column": "english",
            "positive_column": "non_english",
            "sampling_weight": 1.0,
            "language_id": "rus_Cyrl",
            "anchor_language_id": "eng_Latn",
        },
    ]
    settings = {
        "seed": 7,
        "source_weights": {"fineweb2": 0.5, "ccmatrix": 0.5},
        "fragment_rows": 3,
        "max_chars": 512,
        "batch_size": 4,
        "batch_no_duplicates": True,
        "monolingual_batches": True,
        "language_markers": True,
        "holdout_row_groups": 0,
    }

    rows = list(islice(_random_pairs(shards, settings), 40))
    assert any("fineweb-" in row["anchor"] for row in rows)
    assert any("parallel-" in row["anchor"] for row in rows)
    for row in rows:
        if "fineweb-" in row["anchor"]:
            assert row["anchor"].startswith("__rus_Cyrl__ ")
            assert row["positive"].startswith("__rus_Cyrl__ ")
        else:
            assert row["anchor"].startswith("__eng_Latn__ ")
            assert row["positive"].startswith("__rus_Cyrl__ ")
    # Every batch is also source-homogeneous, so negatives do not mix the
    # adjacent-sentence and translation objectives accidentally.
    for start in range(0, len(rows), settings["batch_size"]):
        kinds = {"fineweb" if "fineweb-" in row["anchor"] else "parallel"
                 for row in rows[start : start + settings["batch_size"]]}
        assert len(kinds) == 1


def test_smooth_hierarchy_applies_theta_without_partition_source_bias(tmp_path) -> None:
    large_a = tmp_path / "large-a.parquet"
    large_b = tmp_path / "large-b.parquet"
    small = tmp_path / "small.parquet"
    _write_pairs(large_a, "sentence_1", "sentence_2", "large-a", rows=36)
    _write_pairs(large_b, "sentence_1", "sentence_2", "large-b", rows=36)
    _write_pairs(small, "sentence_1", "sentence_2", "small", rows=8)
    shards = [
        {
            "source": "large",
            "path": str(large_a),
            "anchor_column": "sentence_1",
            "positive_column": "sentence_2",
            "language_id": "aaa_Latn",
            "anchor_language_id": "aaa_Latn",
            "usable_rows": 36,
        },
        {
            "source": "large",
            "path": str(large_b),
            "anchor_column": "sentence_1",
            "positive_column": "sentence_2",
            "language_id": "bbb_Latn",
            "anchor_language_id": "bbb_Latn",
            "usable_rows": 36,
        },
        {
            "source": "small",
            "path": str(small),
            "anchor_column": "sentence_1",
            "positive_column": "sentence_2",
            "language_id": "ccc_Latn",
            "anchor_language_id": "ccc_Latn",
            "usable_rows": 8,
        },
    ]
    settings = {
        "seed": 19,
        "source_weights": {"large": 1.0, "small": 1.0},
        "fragment_rows": 8,
        "max_chars": 512,
        "batch_size": 2,
        "monolingual_batches": True,
        "smooth_weighted_round_robin": True,
        "source_temperature": 0.5,
        "language_temperature": 0.5,
        "strict_unique_batches": True,
        "debug_language": True,
        "holdout_row_groups": 0,
    }

    rows = list(islice(_random_pairs(shards, settings), 200))
    batches = [rows[start : start + 2] for start in range(0, len(rows), 2)]

    assert all(len({row["source"] for row in batch}) == 1 for batch in batches)
    assert all(len({row["language"] for row in batch}) == 1 for batch in batches)
    assert {
        source: sum(batch[0]["source"] == source for batch in batches)
        for source in ("large", "small")
    } == {"large": 75, "small": 25}


def test_smooth_hierarchy_resumes_the_exact_group_sequence(tmp_path) -> None:
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    _write_pairs(first_path, "sentence_1", "sentence_2", "first", rows=16)
    _write_pairs(second_path, "sentence_1", "sentence_2", "second", rows=16)
    shards = [
        {
            "source": source,
            "path": str(path),
            "anchor_column": "sentence_1",
            "positive_column": "sentence_2",
            "language_id": language,
            "usable_rows": 16,
        }
        for source, path, language in (
            ("first", first_path, "aaa_Latn"),
            ("second", second_path, "bbb_Latn"),
        )
    ]
    base = {
        "seed": 23,
        "source_weights": {"first": 3.0, "second": 1.0},
        "fragment_rows": 8,
        "max_chars": 512,
        "batch_size": 2,
        "monolingual_batches": True,
        "smooth_weighted_round_robin": True,
        "source_temperature": 0.0,
        "language_temperature": 0.0,
        "strict_unique_batches": True,
        "debug_language": True,
        "holdout_row_groups": 0,
    }

    uninterrupted = list(islice(_random_pairs(shards, base), 32))
    resumed = list(
        islice(_random_pairs(shards, {**base, "scheduler_start_batch": 8}), 16)
    )

    assert resumed == uninterrupted[16:32]


def test_online_optimizer_step_resume_starts_at_the_exact_next_batch(tmp_path) -> None:
    """Using examples rather than complete batches for the cursor would skip data."""

    first_path = tmp_path / "first-online.parquet"
    second_path = tmp_path / "second-online.parquet"
    _write_pairs(first_path, "sentence_1", "sentence_2", "first", rows=16)
    _write_pairs(second_path, "sentence_1", "sentence_2", "second", rows=16)
    shards = [
        {
            "source": source,
            "path": str(path),
            "anchor_column": "sentence_1",
            "positive_column": "sentence_2",
            "language_id": language,
            "usable_rows": 16,
        }
        for source, path, language in (
            ("first", first_path, "aaa_Latn"),
            ("second", second_path, "bbb_Latn"),
        )
    ]
    base = {
        "seed": 41,
        "source_weights": {"first": 2.0, "second": 1.0},
        "fragment_rows": 8,
        "max_chars": 512,
        "batch_size": 2,
        "monolingual_batches": True,
        "smooth_weighted_round_robin": True,
        "source_temperature": 0.4,
        "language_temperature": 0.4,
        "strict_unique_batches": True,
        "debug_language": True,
        "holdout_row_groups": 0,
    }
    completed_steps = 7
    scheduler_start = _resolve_resume_scheduler_start_batch(
        {"scheduler_start_batch": 0},
        {"gradient_accumulation_steps": 1},
        global_step=completed_steps,
    )

    uninterrupted = list(islice(_random_pairs(shards, base), 20))
    resumed = list(
        islice(
            _random_pairs(
                shards,
                {**base, "scheduler_start_batch": scheduler_start},
            ),
            6,
        )
    )

    assert resumed == uninterrupted[completed_steps * 2 : completed_steps * 2 + 6]


def test_strict_unique_monolingual_batch_never_pads_duplicates(tmp_path) -> None:
    tiny = tmp_path / "tiny.parquet"
    _write_pairs(tiny, "sentence_1", "sentence_2", "tiny", rows=1)
    shards = [
        {
            "source": "tiny",
            "path": str(tiny),
            "anchor_column": "sentence_1",
            "positive_column": "sentence_2",
            "language_id": "aaa_Latn",
            "usable_rows": 1,
        }
    ]
    settings = {
        "seed": 29,
        "source_weights": {"tiny": 1.0},
        "fragment_rows": 1,
        "max_chars": 512,
        "batch_size": 2,
        "monolingual_batches": True,
        "smooth_weighted_round_robin": True,
        "source_temperature": 0.7,
        "language_temperature": 0.7,
        "strict_unique_batches": True,
        "holdout_row_groups": 0,
    }

    with pytest.raises(RuntimeError, match="unique batch"):
        list(islice(_random_pairs(shards, settings), 2))


def test_language_temperature_sampling_selects_largest_languages_and_sets_mass(tmp_path) -> None:
    root = tmp_path / "fineweb"
    _write_pairs(root / "large" / "train" / "part-0.parquet", "sentence_1", "sentence_2", "large-0")
    _write_pairs(root / "large" / "train" / "part-1.parquet", "sentence_1", "sentence_2", "large-1")
    _write_pairs(root / "medium" / "train" / "part-0.parquet", "sentence_1", "sentence_2", "medium")
    _write_pairs(root / "small" / "train" / "part-0.parquet", "sentence_1", "sentence_2", "s")

    shards, counts = discover_shards(
        [
            {
                "name": "fineweb2",
                "root": "fineweb",
                "parquet": "**/*.parquet",
                "anchor_column": "sentence_1",
                "positive_column": "sentence_2",
                "language_sampling": {"temperature": 0.7, "max_languages": 2},
            }
        ],
        {"fineweb": root},
        seed=7,
    )

    languages = {Path(shard["path"]).relative_to(root).parts[0] for shard in shards}
    assert languages == {"large", "medium"}
    assert counts == {"fineweb2": 3}
    assert all(float(shard["sampling_weight"]) > 0 for shard in shards)


def test_language_sampling_can_exclude_mmbert_non_decay_codes(tmp_path) -> None:
    root = tmp_path / "fineweb"
    _write_pairs(root / "eng_Latn" / "part-0.parquet", "sentence_1", "sentence_2", "eng")
    _write_pairs(root / "und_Latn" / "part-0.parquet", "sentence_1", "sentence_2", "und")

    shards, counts = discover_shards(
        [
            {
                "name": "fineweb2",
                "root": "fineweb",
                "parquet": "**/*.parquet",
                "anchor_column": "sentence_1",
                "positive_column": "sentence_2",
                "language_sampling": {"temperature": 0.3, "exclude_language_codes": ["und"]},
            }
        ],
        {"fineweb": root},
        seed=7,
    )

    assert counts == {"fineweb2": 1}
    assert {shard["language_id"] for shard in shards} == {"eng_Latn"}


def test_lemma_overlap_surface_fallback_and_cjk_bigrams() -> None:
    pair_filter = _LemmaOverlapFilter(
        {
            "enabled": True,
            "min_overlap": 0.3,
            "strict": True,
            "remove_stopwords": False,
            "unsupported_language_fallback": "surface",
        }
    )

    assert pair_filter.passes("alpha beta", "alpha gamma", "zzz_Latn")
    assert not pair_filter.passes("alpha beta", "gamma delta", "zzz_Latn")
    assert pair_filter.passes("東京都の天気", "東京都の人口", "jpn_Jpan")
    assert pair_filter.passes("猫が公園を走った", "猫は公園で走る", "jpn_Jpan")


def test_no_duplicate_texts_are_rejected_within_each_batch() -> None:
    pairs = iter(
        [
            {"anchor": "same", "positive": "positive-1"},
            {"anchor": "same", "positive": "positive-2"},
            {"anchor": "anchor-3", "positive": "positive-3"},
            {"anchor": "same", "positive": "positive-4"},
        ]
    )

    selected = list(_batch_no_duplicates(pairs, batch_size=2))

    assert selected == [
        {"anchor": "same", "positive": "positive-1"},
        {"anchor": "anchor-3", "positive": "positive-3"},
        {"anchor": "same", "positive": "positive-4"},
    ]
