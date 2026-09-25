#!/usr/bin/env python3
"""Evaluate a local SentenceTransformer model task-by-task on an MTEB benchmark.

The runner is intentionally independent from training: benchmark datasets are
downloaded only by this process and can never enter the training sampler.
Completed per-task results are reusable, so interrupted full JMTEB runs resume
without repeating successful tasks.
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

PREFERRED_SPLITS = ("test", "dev", "validation")
BITEXT_MINING_TASK_TYPE = "BitextMining"
STS_TASK_TYPE = "STS"
RETRIEVAL_TASK_TYPE = "Retrieval"

# MTEB and the Q15 training inventory occasionally use different ISO 639
# macrolanguage or ISO 15924 spellings for the same deployed marker.  Keep the
# mapping explicit: an approximate/fuzzy fallback would silently route text to
# an unrelated language.
MTEB_TO_STARSE_LANGUAGE_ALIASES = {
    "ara_Arab": "arb_Arab",
    "aze_Latn": "azj_Latn",
    "cmn_Hans": "cmn_Hani",
    "cmn_Hant": "cmn_Hani",
    "est_Latn": "ekk_Latn",
    "jpn_Hira": "jpn_Jpan",
    "kor_Kore": "kor_Hang",
    "kur_Arab": "ckb_Arab",
    "kur_Latn": "kmr_Latn",
    "lav_Latn": "lvs_Latn",
    "mon_Cyrl": "khk_Cyrl",
    "msa_Latn": "zsm_Latn",
    "nep_Deva": "npi_Deva",
    "nor_Latn": "nob_Latn",
    "pes_Arab": "fas_Arab",
    "rom_Latn": "rmy_Latn",
    "sqi_Latn": "als_Latn",
    "swa_Latn": "swh_Latn",
    "tgl_Latn": "fil_Latn",
    "zho_Hans": "cmn_Hani",
    "zho_Hant": "cmn_Hani",
}

def call_supported(function: Any, kwargs: dict[str, Any]) -> Any:
    """Call a version-sensitive API with only accepted keyword arguments."""

    signature = inspect.signature(function)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return function(**kwargs)
    return function(**{key: value for key, value in kwargs.items() if key in signature.parameters})


def task_name(task: Any, index: int = 0) -> str:
    metadata = getattr(task, "metadata", None)
    name = getattr(metadata, "name", None)
    if isinstance(name, str) and name:
        return name
    name = getattr(task, "name", None)
    return name if isinstance(name, str) and name else f"task_{index}"


def safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def task_result_path(raw_dir: Path, name: str) -> Path:
    """Return an order-independent result path while reusing legacy indexed files."""

    slug = safe_slug(name)
    stable = raw_dir / f"{slug}.json"
    if stable.exists():
        return stable
    legacy = sorted(raw_dir.glob(f"[0-9][0-9][0-9]_{slug}.json"))
    return legacy[0] if legacy else stable


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def normalize_results(results: Any) -> dict[str, Any]:
    if isinstance(results, dict):
        return to_jsonable(results)
    if isinstance(results, list):
        normalized: dict[str, Any] = {}
        for index, item in enumerate(results):
            normalized[task_name(item, index)] = to_jsonable(item)
        return normalized
    return {"result": to_jsonable(results)}


def prune_result_subsets(value: Any, allowed_subsets: set[str]) -> Any:
    """Remove cached MTEB score rows outside the current matched subset set."""

    if isinstance(value, dict):
        return {key: prune_result_subsets(item, allowed_subsets) for key, item in value.items()}
    if isinstance(value, list):
        return [
            prune_result_subsets(item, allowed_subsets)
            for item in value
            if not isinstance(item, dict)
            or "hf_subset" not in item
            or item["hf_subset"] in allowed_subsets
        ]
    return value


def _language_match(item: Any, languages: tuple[str, ...]) -> bool:
    if not isinstance(item, dict) or not languages:
        return False
    requested = {language.lower() for language in languages}
    markers: list[str] = []
    raw_languages = item.get("languages")
    if isinstance(raw_languages, str):
        markers.append(raw_languages)
    elif isinstance(raw_languages, list):
        markers.extend(str(value) for value in raw_languages)
    for key in ("hf_subset", "subset"):
        if isinstance(item.get(key), str):
            markers.append(item[key])
    aliases = {"jpn": {"jpn", "jpn-jpan", "ja", "jp"}}
    expanded = set(requested)
    for language in requested:
        expanded.update(aliases.get(language, set()))
    normalized_markers = set()
    for marker in markers:
        normalized = marker.lower().replace("_", "-")
        normalized_markers.add(normalized)
        normalized_markers.update(normalized.split("-"))
    return bool(normalized_markers & expanded)


def _score_items(obj: Any, languages: tuple[str, ...]) -> list[float]:
    if isinstance(obj, dict) and len(obj) == 1 and "scores" not in obj:
        return _score_items(next(iter(obj.values())), languages)
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        splits = obj["scores"]
        chosen = next((split for split in PREFERRED_SPLITS if split in splits), next(iter(splits), None))
        raw_items = splits.get(chosen) if chosen is not None else None
        items = raw_items if isinstance(raw_items, list) else ([raw_items] if raw_items is not None else [])
        language_items = [item for item in items if _language_match(item, languages)]
        selected = language_items or items
        scores = [item.get("main_score") for item in selected if isinstance(item, dict)]
        valid = [float(score) for score in scores if isinstance(score, (int, float)) and math.isfinite(score)]
        if valid:
            return valid
    scores: list[float] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "main_score" and isinstance(value, (int, float)) and math.isfinite(value):
                scores.append(float(value))
            elif key != "scores_per_experiment":
                scores.extend(_score_items(value, languages))
    elif isinstance(obj, list):
        for item in obj:
            scores.extend(_score_items(item, languages))
    return scores


def read_names(value: str | None, file_path: Path | None) -> list[str]:
    names = [item.strip() for item in (value or "").split(",") if item.strip()]
    if file_path:
        names.extend(
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return list(dict.fromkeys(names))


def resolve_tasks(benchmark_name: str, requested: list[str], languages: tuple[str, ...]) -> list[Any]:
    import mteb

    benchmark = call_supported(
        mteb.get_benchmark,
        {"benchmark_name": benchmark_name, "benchmark": benchmark_name},
    )
    benchmark_names = [task_name(task, index) for index, task in enumerate(benchmark.tasks)]
    selected_names = requested or benchmark_names
    missing = sorted(set(selected_names) - set(benchmark_names))
    if missing:
        raise ValueError(f"Tasks are not part of {benchmark_name}: {missing}")
    return [
        call_supported(mteb.get_task, {"task_name": name, "task": name, "languages": list(languages)})
        for name in selected_names
    ]


def mteb_language_to_starse(language: str) -> str:
    """Convert MTEB's ISO-script spelling to StaRSE's marker spelling."""

    normalized = language.replace("-", "_")
    return MTEB_TO_STARSE_LANGUAGE_ALIASES.get(normalized, normalized)


def task_subset_languages(task_metadata: Any, hf_subset: str) -> tuple[str, ...]:
    """Return the exact StaRSE marker inventory declared for one MTEB subset."""

    eval_langs = getattr(task_metadata, "eval_langs", None)
    if isinstance(eval_langs, dict):
        languages = eval_langs.get(hf_subset)
    elif isinstance(eval_langs, (list, tuple)):
        languages = eval_langs
    else:
        languages = None
    if not isinstance(languages, (list, tuple)) or not languages:
        raise ValueError(f"No language metadata for subset {hf_subset!r}")
    return tuple(mteb_language_to_starse(str(language)) for language in languages)


def generic_conditioned_language(
    task_metadata: Any,
    hf_subset: str,
    prompt_type: Any = None,
) -> str:
    """Resolve a non-Bitext/non-STS encoder call without target leakage.

    Monolingual subsets have one marker.  Cross-lingual retrieval subsets use
    the first metadata language for queries and the second for documents.  A
    few classification/reranking subsets intentionally mix languages, including
    language-identification benchmarks.  Their gold per-row language is not
    available at inference and would reveal the target, so every row receives
    one task-level marker fixed in advance: the first metadata language.
    """

    languages = task_subset_languages(task_metadata, hf_subset)
    task_type = getattr(task_metadata, "type", None)
    if task_type == RETRIEVAL_TASK_TYPE and len(languages) == 2:
        prompt_value = getattr(prompt_type, "value", prompt_type)
        if prompt_value == "query":
            return languages[0]
        if prompt_value == "document":
            return languages[1]
        raise ValueError(
            f"Cross-lingual retrieval subset {hf_subset!r} requires query/document prompt_type, "
            f"got {prompt_type!r}"
        )
    return languages[0]


def bitext_column_language(task_metadata: Any, hf_subset: str, column: str) -> str:
    """Resolve the language of one side of a non-parallel MTEB bitext subset.

    Tatoeba and BUCC store the two languages in ``metadata.eval_langs`` while
    naming the actual text columns ``sentence1`` and ``sentence2``.  MTEB's
    stock evaluator does not pass the side identity to the encoder, so a
    language-conditioned model otherwise receives the same (or no) marker for
    both sides.
    """

    languages = task_subset_languages(task_metadata, hf_subset)
    if len(languages) not in (1, 2):
        raise ValueError(f"Expected one or two languages for {hf_subset!r}, got {languages!r}")
    try:
        side = ("sentence1", "sentence2").index(column)
    except ValueError as exc:
        raise ValueError(f"Unsupported bitext column {column!r} in {hf_subset!r}") from exc
    return languages[0] if len(languages) == 1 else languages[side]


def sts_side_language(task_metadata: Any, hf_subset: str, side: int) -> str:
    """Resolve one side of a monolingual or cross-lingual STS subset."""

    eval_langs = getattr(task_metadata, "eval_langs", None)
    if isinstance(eval_langs, dict):
        languages = eval_langs.get(hf_subset)
    elif isinstance(eval_langs, (list, tuple)):
        # MTEB represents a single-subset monolingual task such as FinParaSTS
        # with one task-level language list rather than a subset mapping.
        languages = eval_langs
    else:
        languages = None
    if languages is None:
        raise ValueError(f"No language metadata for STS subset {hf_subset!r}")
    if not isinstance(languages, (list, tuple)) or len(languages) not in (1, 2):
        raise ValueError(f"Expected one or two STS languages for {hf_subset!r}, got {languages!r}")
    language = languages[0] if len(languages) == 1 else languages[side]
    return mteb_language_to_starse(str(language))


def filter_conditioned_bitext_subsets(task: Any, languages: tuple[str, ...]) -> list[str]:
    """Restrict a conditioned bitext task to pairs supported by the checkpoint."""

    metadata = getattr(task, "metadata", None)
    if getattr(metadata, "type", None) != BITEXT_MINING_TASK_TYPE:
        raise ValueError(
            f"Conditioned MTEB evaluation currently supports BitextMining only; "
            f"{task_name(task)!r} has type {getattr(metadata, 'type', None)!r}"
        )
    known = set(languages)
    supported = []
    for subset in getattr(task, "hf_subsets", ()):
        pair = task_subset_languages(metadata, subset)
        if len(pair) in (1, 2) and set(pair) <= known:
            supported.append(subset)
    if not supported:
        raise ValueError(
            f"No {task_name(task)} subsets are covered by checkpoint languages {sorted(known)}"
        )
    task.hf_subsets = supported
    return supported


def filter_conditioned_sts_subsets(task: Any, languages: tuple[str, ...]) -> list[str]:
    """Restrict an STS task to mono/cross subsets covered by the checkpoint."""

    metadata = getattr(task, "metadata", None)
    if getattr(metadata, "type", None) != STS_TASK_TYPE:
        raise ValueError(f"{task_name(task)!r} is not an STS task")
    known = set(languages)
    supported = []
    available = list(getattr(task, "hf_subsets", ()))
    eval_langs = getattr(metadata, "eval_langs", None)
    for subset in available:
        if isinstance(eval_langs, dict):
            subset_languages = eval_langs.get(subset, ())
        elif isinstance(eval_langs, (list, tuple)) and len(available) == 1:
            subset_languages = eval_langs
        else:
            subset_languages = ()
        normalized = {mteb_language_to_starse(str(language)) for language in subset_languages}
        if len(subset_languages) in (1, 2) and normalized <= known:
            supported.append(subset)
    if not supported:
        raise ValueError(f"No {task_name(task)} subsets are covered by the checkpoint")
    task.hf_subsets = supported
    return supported


def filter_conditioned_generic_subsets(task: Any, languages: tuple[str, ...]) -> list[str]:
    """Keep generic subsets whose inference-time marker policy is supported."""

    metadata = getattr(task, "metadata", None)
    known = set(languages)
    supported = []
    for subset in getattr(task, "hf_subsets", ()):
        subset_languages = task_subset_languages(metadata, subset)
        if getattr(metadata, "type", None) == RETRIEVAL_TASK_TYPE and len(subset_languages) == 2:
            required = set(subset_languages)
        else:
            # Mixed-language classification/reranking uses one fixed marker for
            # the entire subset; requiring every gold language would incorrectly
            # suggest that labels are consulted during encoding.
            required = {subset_languages[0]}
        if required <= known:
            supported.append(subset)
    if not supported:
        raise ValueError(f"No {task_name(task)} subsets are covered by the checkpoint")
    task.hf_subsets = supported
    return supported


def filter_task_subsets(task: Any, requested_subsets: tuple[str, ...]) -> list[str]:
    """Apply an explicit, task-local subset intersection for matched evaluation."""

    available = list(getattr(task, "hf_subsets", ()))
    if not requested_subsets:
        return available
    requested = set(requested_subsets)
    selected = [subset for subset in available if subset in requested]
    if not selected:
        raise ValueError(
            f"None of the requested subsets belong to {task_name(task)}; "
            f"requested={sorted(requested)}, available={available}"
        )
    task.hf_subsets = selected
    return selected


def subsets_for_task(
    name: str,
    global_subsets: tuple[str, ...],
    task_subsets: dict[str, list[str]],
) -> tuple[str, ...]:
    return tuple(task_subsets.get(name, global_subsets))


def conditioning_config(model_path: str) -> dict[str, Any] | None:
    """Read a local conditioned checkpoint manifest, if one is present."""

    path = Path(model_path).expanduser() / "language_conditioning.json"
    if not path.is_file():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    languages = config.get("languages")
    if not isinstance(languages, list) or not languages or not all(isinstance(item, str) for item in languages):
        raise ValueError(f"Invalid languages in {path}")
    return config


def prepend_language_marker(inputs: Any, language: str) -> Iterator[dict[str, Any]]:
    """Yield MTEB batches with the StaRSE marker prepended to every text."""

    marker = f"__{language}__ "
    for batch in inputs:
        yield {**batch, "text": [marker + text for text in batch["text"]]}


def wrap_conditioned_model(model: Any, languages: tuple[str, ...]) -> Any:
    """Wrap SentenceTransformer so every encoded text gets an explicit language marker."""

    from mteb.models import SentenceTransformerEncoderWrapper

    class ConditionedSentenceTransformerEncoderWrapper(SentenceTransformerEncoderWrapper):
        starse_languages = frozenset(languages)

        def encode(self, inputs: Any, *, starse_language: str | None = None, **kwargs: Any) -> Any:
            if starse_language is None:
                task_metadata = kwargs.get("task_metadata")
                hf_subset = kwargs.get("hf_subset")
                if task_metadata is None or not isinstance(hf_subset, str):
                    raise ValueError(
                        "Conditioned StaRSE evaluation requires either starse_language or "
                        "task_metadata plus hf_subset"
                    )
                starse_language = generic_conditioned_language(
                    task_metadata,
                    hf_subset,
                    kwargs.get("prompt_type"),
                )
            if starse_language not in self.starse_languages:
                raise ValueError(f"Checkpoint has no language marker for {starse_language!r}")
            marked_inputs = prepend_language_marker(inputs, starse_language)
            return super().encode(marked_inputs, **kwargs)

    return ConditionedSentenceTransformerEncoderWrapper(model)


@contextmanager
def conditioned_bitext_evaluator() -> Iterator[None]:
    """Temporarily make MTEB pass the correct language for each bitext side."""

    import mteb.abstasks.text.bitext_mining as bitext_module
    from mteb._create_dataloaders import _create_dataloader_from_texts
    from mteb._evaluators.text.bitext_mining_evaluator import BitextMiningEvaluator as BaseEvaluator
    from tqdm.auto import tqdm

    class ConditionedBitextMiningEvaluator(BaseEvaluator):
        def __call__(self, model: Any, *, encode_kwargs: dict[str, Any], num_proc: int | None = None) -> Any:
            pair_elements = {element for pair in self.pairs for element in pair}
            if hasattr(self.sentences, "features"):
                subsets = [column for column in self.sentences.features if column in pair_elements]
            else:
                subsets = list(pair_elements)

            embeddings = {}
            for column in tqdm(subsets):
                if self.hf_subset == "parallel":
                    language = mteb_language_to_starse(column)
                    encode_subset = column
                else:
                    language = bitext_column_language(self.task_metadata, self.hf_subset, column)
                    encode_subset = self.hf_subset
                dataloader = _create_dataloader_from_texts(
                    self.sentences[column], num_proc=num_proc, **encode_kwargs
                )
                embeddings[column] = model.encode(
                    dataloader,
                    task_metadata=self.task_metadata,
                    hf_subset=encode_subset,
                    hf_split=self.hf_split,
                    starse_language=language,
                    **encode_kwargs,
                )

            neighbours = {}
            for key1, key2 in tqdm(self.pairs, desc="Matching sentences"):
                neighbours[f"{key1}-{key2}"] = self._similarity_search(
                    embeddings[key1], embeddings[key2], model
                )
            return neighbours

    original = bitext_module.BitextMiningEvaluator
    bitext_module.BitextMiningEvaluator = ConditionedBitextMiningEvaluator
    try:
        yield
    finally:
        bitext_module.BitextMiningEvaluator = original


@contextmanager
def conditioned_sts_evaluator() -> Iterator[None]:
    """Temporarily make MTEB pass a language marker for each STS side."""

    import mteb.abstasks.sts as sts_module
    from mteb._create_dataloaders import create_dataloader
    from mteb._evaluators.any_sts_evaluator import AnySTSEvaluator as BaseEvaluator
    from mteb.similarity_functions import compute_pairwise_similarity
    from sklearn.metrics.pairwise import (
        paired_cosine_distances,
        paired_euclidean_distances,
        paired_manhattan_distances,
    )

    class ConditionedSTSEvaluator(BaseEvaluator):
        def __call__(self, model: Any, *, encode_kwargs: dict[str, Any], num_proc: int | None = None) -> Any:
            if not isinstance(self.input_columns[0], str) or not isinstance(self.input_columns[1], str):
                raise ValueError("Conditioned multimodal STS is not supported")
            embeddings = []
            prompt_types = (self.input1_prompt_type, self.input2_prompt_type)
            for side, column in enumerate(self.input_columns):
                language = sts_side_language(self.task_metadata, self.hf_subset, side)
                dataloader = create_dataloader(
                    self.dataset.select_columns(column).rename_column(
                        column, self.task_metadata.modalities[0]
                    ),
                    task_metadata=self.task_metadata,
                    num_proc=num_proc,
                    **encode_kwargs,
                )
                embeddings.append(
                    model.encode(
                        dataloader,
                        task_metadata=self.task_metadata,
                        hf_split=self.hf_split,
                        hf_subset=self.hf_subset,
                        prompt_type=prompt_types[side],
                        starse_language=language,
                        **encode_kwargs,
                    )
                )
            embeddings1, embeddings2 = embeddings
            cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
            manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
            euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
            similarity_scores = compute_pairwise_similarity(model, embeddings1, embeddings2)
            return {
                "cosine_scores": cosine_scores.tolist(),
                "manhattan_distances": manhattan_distances.tolist(),
                "euclidean_distances": euclidean_distances.tolist(),
                "similarity_scores": similarity_scores.tolist(),
            }

    original = sts_module.AnySTSEvaluator
    sts_module.AnySTSEvaluator = ConditionedSTSEvaluator
    try:
        yield
    finally:
        sts_module.AnySTSEvaluator = original


def conditioned_evaluator_for(task: Any) -> Any:
    """Select the scoped evaluator patch for a conditioned task."""

    task_type = getattr(getattr(task, "metadata", None), "type", None)
    if task_type == BITEXT_MINING_TASK_TYPE:
        return conditioned_bitext_evaluator()
    if task_type == STS_TASK_TYPE:
        return conditioned_sts_evaluator()
    return nullcontext()


def prepare_memory_efficient_legacy_retrieval(task: Any) -> bool:
    """Keep MrTyDi's seven-million-row corpus Arrow-backed instead of copying it to a dict.

    MTEB 2.12.30 ships MrTidyRetrieval with a legacy loader that materializes
    every document as nested Python objects before converting the same data back
    to a Dataset.  The standard retrieval evaluator already accepts an Arrow
    Dataset and performs chunked search, so supplying that representation
    directly preserves the benchmark while avoiding tens of gigabytes of peak
    host memory.
    """

    if task_name(task) != "MrTidyRetrieval" or getattr(task, "data_loaded", False):
        return False

    import datasets

    metadata = task.metadata
    dataset_spec = metadata.dataset
    path = dataset_spec["path"]
    revision = dataset_spec["revision"]
    subsets = list(getattr(task, "hf_subsets", ()))
    splits = list(getattr(metadata, "eval_splits", ("test",)))
    if not subsets or splits != ["test"]:
        raise ValueError(f"Unexpected MrTidy layout: subsets={subsets}, splits={splits}")

    task.dataset = {}
    for subset in subsets:
        qrels = datasets.load_dataset(path, name=f"{subset}-qrels", revision=revision)["test"]
        relevant_docs: dict[str, dict[str, int | float]] = {}
        for row in qrels:
            relevant_docs.setdefault(row["query-id"], {})[row["corpus-id"]] = row["score"]

        corpus = datasets.load_dataset(path, name=f"{subset}-corpus", revision=revision)["train"]
        queries = datasets.load_dataset(path, name=f"{subset}-queries", revision=revision)["test"]
        if "_id" in corpus.column_names:
            corpus = corpus.rename_column("_id", "id")
        if "_id" in queries.column_names:
            queries = queries.rename_column("_id", "id")
        task.dataset[subset] = {
            "test": {
                "corpus": corpus,
                "queries": queries,
                "relevant_docs": relevant_docs,
                "top_ranked": None,
            }
        }

    task.data_loaded = True
    return True


def release_task_data(task: Any) -> None:
    """Release benchmark payloads between tasks in long resumable runs."""

    for attribute in ("dataset", "corpus", "queries", "relevant_docs", "instructions", "top_ranked"):
        if hasattr(task, attribute):
            delattr(task, attribute)
    if hasattr(task, "data_loaded"):
        task.data_loaded = False
    gc.collect()


def load_model(
    model_path: str,
    device: str | None,
    trust_remote_code: bool,
    truncate_dim: int | None = None,
    revision: str | None = None,
) -> Any:
    from sentence_transformers import SentenceTransformer

    kwargs = {"device": device} if device else {}
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim
    return SentenceTransformer(
        model_path, revision=revision, trust_remote_code=trust_remote_code, **kwargs
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mteb

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw_task_results"
    mteb_dir = output_dir / "mteb_results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mteb_dir.mkdir(parents=True, exist_ok=True)
    requested = read_names(args.tasks, args.tasks_file)
    languages = tuple(item.strip() for item in args.languages.split(",") if item.strip())
    requested_subsets = tuple(item.strip() for item in args.subsets.split(",") if item.strip())
    task_subset_overrides = dict(getattr(args, "task_subsets", {}) or {})
    tasks = resolve_tasks(args.benchmark, requested, languages)
    condition = conditioning_config(args.model)
    conditioned_languages = tuple(condition["languages"]) if condition else ()
    task_subsets: dict[str, list[str]] = {}
    for task in tasks:
        name = task_name(task)
        filter_task_subsets(
            task,
            subsets_for_task(name, requested_subsets, task_subset_overrides),
        )
        if condition:
            task_type = getattr(getattr(task, "metadata", None), "type", None)
            if task_type == BITEXT_MINING_TASK_TYPE:
                selected_subsets = filter_conditioned_bitext_subsets(task, conditioned_languages)
            elif task_type == STS_TASK_TYPE:
                selected_subsets = filter_conditioned_sts_subsets(task, conditioned_languages)
            else:
                selected_subsets = filter_conditioned_generic_subsets(task, conditioned_languages)
            task_subsets[task_name(task)] = selected_subsets
        else:
            task_subsets[task_name(task)] = list(getattr(task, "hf_subsets", ()))
    names = [task_name(task, index) for index, task in enumerate(tasks)]
    write_json(
        output_dir / "eval_config.json",
        {
            "model": args.model,
            "model_revision": args.revision,
            "benchmark": args.benchmark,
            "tasks": names,
            "task_subsets": task_subsets,
            "requested_subsets": requested_subsets,
            "task_subset_overrides": task_subset_overrides,
            "languages": languages,
            "conditioned_languages": conditioned_languages,
            "language_aware_bitext": bool(condition),
            "conditioning_policy": (
                "subset marker; query/document markers for cross-lingual retrieval; "
                "first metadata marker fixed for mixed-language subsets"
                if condition
                else None
            ),
            "collapse_bias_ablation": args.collapse_bias_ablation,
            "batch_size": args.batch_size,
            "truncate_dim": args.truncate_dim,
            "device": args.device,
            "normalize_embeddings": args.normalize_embeddings,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    model = load_model(
        args.model, args.device, args.trust_remote_code, args.truncate_dim, args.revision
    )
    if condition:
        from starse.collapse import ablate_language_bias

        ablate_language_bias(model, args.collapse_bias_ablation)
        model = wrap_conditioned_model(model, conditioned_languages)
    raw_results: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for index, task in enumerate(tasks):
        name = names[index]
        raw_path = task_result_path(raw_dir, name)
        started = time.perf_counter()
        if raw_path.exists() and not args.overwrite:
            result = json.loads(raw_path.read_text(encoding="utf-8"))
            raw_results.update(result)
            record = {"task": name, "status": "reused", "wall_seconds": 0.0, "error": None}
            records.append(record)
            print(json.dumps({"event": "task_reused", **record}, ensure_ascii=False), flush=True)
            continue
        if prepare_memory_efficient_legacy_retrieval(task):
            print(json.dumps({"event": "memory_efficient_task_data", "task": name}), flush=True)
        status, error, result = "completed", None, {}
        try:
            runner = mteb.MTEB(tasks=[task])
            evaluator_context = conditioned_evaluator_for(task) if condition else nullcontext()
            with evaluator_context:
                task_result = call_supported(
                    runner.run,
                    {
                        "model": model,
                        "output_folder": str(mteb_dir / f"{index:03d}_{safe_slug(name)}"),
                        "overwrite_results": args.overwrite,
                        "verbosity": args.verbosity,
                        "encode_kwargs": {
                            "batch_size": args.batch_size,
                            "normalize_embeddings": args.normalize_embeddings,
                        },
                        "co2_tracker": False,
                        "raise_error": args.fail_on_error,
                    },
                )
            result = normalize_results(task_result)
            if name not in result and len(result) == 1:
                result = {name: next(iter(result.values()))}
            if name not in result:
                raise RuntimeError(f"MTEB returned no result for {name}")
            result[name] = prune_result_subsets(
                result[name], set(getattr(task, "hf_subsets", ()))
            )
            write_json(raw_path, {name: result[name]})
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            if args.fail_on_error:
                raise
        finally:
            release_task_data(task)
        raw_results.update(result)
        record = {"task": name, "status": status, "wall_seconds": time.perf_counter() - started, "error": error}
        records.append(record)
        print(json.dumps({"event": "task_done", **record}, ensure_ascii=False), flush=True)

    scores: dict[str, float | None] = {}
    for name in names:
        values = _score_items(raw_results.get(name), languages)
        scores[name] = sum(values) / len(values) if values else None
    valid = [score for score in scores.values() if score is not None]
    failed = [record for record in records if record["status"] == "failed"]
    summary = {
        "status": "ok" if not failed else "partial",
        "model": args.model,
        "benchmark": args.benchmark,
        "languages": languages,
        "task_count": len(names),
        "completed_count": len(names) - len(failed),
        "failed_count": len(failed),
        "mean_main_score": sum(valid) / len(valid) if valid else None,
        "scores": scores,
        "task_records": records,
        "total_wall_seconds": time.perf_counter() - started_all,
    }
    write_json(output_dir / "mteb_eval_results.json", raw_results)
    write_json(output_dir / "mteb_eval_summary.json", summary)
    with (output_dir / "mteb_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "main_score"])
        writer.writeheader()
        writer.writerows({"task": name, "main_score": scores[name]} for name in names)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default="JMTEB(v2)")
    parser.add_argument("--tasks")
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--languages", default="jpn")
    parser.add_argument(
        "--subsets",
        default="",
        help="comma-separated exact hf_subset allowlist, intersected separately with each task",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--truncate-dim", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--collapse-bias-ablation",
        choices=("none", "mean"),
        default="none",
        help="evaluation-only ablation of a conditioned collapse checkpoint",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbosity", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
