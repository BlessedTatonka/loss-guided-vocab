"""Random, background-prefetched Parquet streams for multilingual training."""

from __future__ import annotations

import bisect
import hashlib
import os
import queue
import random
import re
import threading
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from starse.weighted_schedule import SmoothWeightedRoundRobin


@dataclass(frozen=True)
class PairShard:
    source: str
    path: str
    anchor_column: str
    positive_column: str
    sampling_weight: float = 1.0
    language_id: str = ""
    # For a parallel stream the two sides are in different languages, so each
    # needs its own marker. Defaults to language_id, i.e. a monolingual pair.
    anchor_language_id: str = ""
    usable_rows: int = 0


_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_PUNCTUATION_RE = re.compile(r"^[^\w]+$", re.UNICODE)
_UNSEGMENTED_SCRIPTS = {"Hani", "Hans", "Hant", "Jpan", "Khmr", "Laoo", "Mymr", "Thai"}

# simplemma uses BCP-47/ISO-639-1 for most languages while FineWeb2 directory
# names use Glottocode-style ISO-639-3 language identifiers.
_SIMPLEMMA_LANGUAGE = {
    "ast": "ast",
    "bul": "bg",
    "cat": "ca",
    "ces": "cs",
    "cym": "cy",
    "dan": "da",
    "deu": "de",
    "ekk": "et",
    "ell": "el",
    "eng": "en",
    "enm": "enm",
    "epo": "eo",
    "est": "et",
    "fas": "fa",
    "fil": "tl",
    "fin": "fi",
    "fra": "fr",
    "gla": "gd",
    "gle": "ga",
    "glg": "gl",
    "glv": "gv",
    "hbs": "hbs",
    "hin": "hi",
    "hrv": "hbs",
    "hun": "hu",
    "hye": "hy",
    "ind": "id",
    "isl": "is",
    "ita": "it",
    "kat": "ka",
    "lat": "la",
    "lit": "lt",
    "ltz": "lb",
    "lvs": "lv",
    "mkd": "mk",
    "nld": "nl",
    "nno": "nn",
    "nob": "nb",
    "pol": "pl",
    "por": "pt",
    "ron": "ro",
    "rus": "ru",
    "slk": "sk",
    "slv": "sl",
    "sme": "se",
    "spa": "es",
    "sqi": "sq",
    "srp": "hbs",
    "swe": "sv",
    "swh": "sw",
    "tur": "tr",
    "ukr": "uk",
    "zsm": "ms",
}

# Extra mappings are useful for stop-word removal even when simplemma has no
# dictionary for a language.
_STOPWORD_LANGUAGE = {
    **_SIMPLEMMA_LANGUAGE,
    "afr": "af",
    "arb": "ar",
    "ben": "bn",
    "eus": "eu",
    "heb": "he",
    "jpn": "ja",
    "kor": "ko",
    "tha": "th",
    "vie": "vi",
    "yue": "zh",
    "cmn": "zh",
}


def _language_code(language_id: str) -> str:
    return language_id.split("_", 1)[0].casefold()


def _canonical_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class _LemmaOverlapFilter:
    """StaRSE-style Dice overlap over lemma sets, computed online per pair."""

    def __init__(self, settings: dict[str, Any] | None) -> None:
        settings = dict(settings or {})
        self.enabled = bool(settings.get("enabled", False))
        self.minimum = float(settings.get("min_overlap", 0.0))
        self.strict = bool(settings.get("strict", True))
        self.remove_stopwords = bool(settings.get("remove_stopwords", True))
        self.fallback = str(settings.get("unsupported_language_fallback", "surface"))
        if not 0.0 <= self.minimum <= 1.0:
            raise ValueError("lemma_filter.min_overlap must be between 0 and 1")
        if self.fallback not in {"surface", "reject"}:
            raise ValueError("lemma_filter.unsupported_language_fallback must be surface or reject")
        self._lemmatizer: Any | None = None
        self._russian_morph: Any | None = None
        self._russian_lemma_cache: dict[str, str] = {}
        self._japanese_tokenizer: Any | None = None
        self._japanese_split_mode: Any | None = None
        self._stopwords: dict[str, frozenset[str]] = {}

    def _get_lemmatizer(self) -> Any:
        if self._lemmatizer is None:
            from simplemma import Lemmatizer
            from simplemma.strategies import DefaultDictionaryFactory, DefaultStrategy

            # One dictionary per worker bounds RAM. Rows arrive in shard-sized
            # fragments, so dictionary loading is amortized over many pairs.
            dictionary_factory = DefaultDictionaryFactory(cache_max_size=1)
            self._lemmatizer = Lemmatizer(
                cache_max_size=131_072,
                lemmatization_strategy=DefaultStrategy(dictionary_factory=dictionary_factory),
            )
        return self._lemmatizer

    def _get_stopwords(self, language_code: str) -> frozenset[str]:
        stopword_language = _STOPWORD_LANGUAGE.get(language_code, "")
        if not self.remove_stopwords or not stopword_language:
            return frozenset()
        if stopword_language not in self._stopwords:
            from stopwordsiso import stopwords

            words = (unicodedata.normalize("NFKC", word).casefold() for word in stopwords(stopword_language))
            self._stopwords[stopword_language] = frozenset(words)
        return self._stopwords[stopword_language]

    def _russian_lemma(self, token: str) -> str:
        cached = self._russian_lemma_cache.get(token)
        if cached is not None:
            return cached
        if self._russian_morph is None:
            import pymorphy3

            self._russian_morph = pymorphy3.MorphAnalyzer()
        parsed = self._russian_morph.parse(token)
        lemma = parsed[0].normal_form.casefold().replace("ё", "е") if parsed else token
        if len(self._russian_lemma_cache) < 2_000_000:
            self._russian_lemma_cache[token] = lemma
        return lemma

    def _japanese_terms(self, text: str) -> set[str]:
        if self._japanese_tokenizer is None:
            from sudachipy import dictionary, tokenizer

            self._japanese_tokenizer = dictionary.Dictionary().create()
            self._japanese_split_mode = tokenizer.Tokenizer.SplitMode.C

        stopwords = self._get_stopwords("jpn")
        ignored_pos = {"助詞", "助動詞", "補助記号", "空白"}
        terms: set[str] = set()
        normalized = unicodedata.normalize("NFKC", text)
        for morpheme in self._japanese_tokenizer.tokenize(normalized, self._japanese_split_mode):
            if morpheme.part_of_speech()[0] in ignored_pos:
                continue
            surface = morpheme.surface().casefold().strip()
            lemma = morpheme.dictionary_form().casefold().strip()
            if not lemma or lemma == "*":
                lemma = surface
            if lemma and lemma not in stopwords and not _PUNCTUATION_RE.fullmatch(lemma):
                terms.add(lemma)
        return terms

    @staticmethod
    def _character_bigrams(text: str) -> set[str]:
        letters = [
            character
            for character in unicodedata.normalize("NFKC", text).casefold()
            if unicodedata.category(character).startswith("L")
        ]
        if len(letters) < 2:
            return set(letters)
        return {"".join(letters[index : index + 2]) for index in range(len(letters) - 1)}

    def _terms(self, text: str, language_id: str) -> set[str]:
        language_code, _, script = language_id.partition("_")
        language_code = language_code.casefold()
        if language_code == "jpn":
            return self._japanese_terms(text)
        if script in _UNSEGMENTED_SCRIPTS:
            return self._character_bigrams(text)

        lemmatizer_language = _SIMPLEMMA_LANGUAGE.get(language_code)
        if lemmatizer_language is None and self.fallback == "reject":
            return set()
        lemmatizer = self._get_lemmatizer() if lemmatizer_language and language_code != "rus" else None
        stopwords = self._get_stopwords(language_code)
        terms: set[str] = set()
        for match in _WORD_RE.finditer(unicodedata.normalize("NFKC", text)):
            token = match.group(0).casefold()
            if len(token) < 2 or token in stopwords:
                continue
            if language_code == "rus":
                lemma = self._russian_lemma(token)
            else:
                lemma = lemmatizer.lemmatize(token, lemmatizer_language).casefold() if lemmatizer else token
            if len(lemma) >= 2 and lemma not in stopwords:
                terms.add(lemma)
        return terms

    def passes(self, anchor: str, positive: str, language_id: str) -> bool:
        if not self.enabled:
            return True
        anchor_terms = self._terms(anchor, language_id)
        positive_terms = self._terms(positive, language_id)
        denominator = len(anchor_terms) + len(positive_terms)
        overlap = 0.0 if denominator == 0 else 2.0 * len(anchor_terms & positive_terms) / denominator
        return overlap > self.minimum if self.strict else overlap >= self.minimum


def _language_sampling_weights(
    files: Sequence[Path],
    root: Path,
    settings: dict[str, Any],
) -> tuple[list[Path], dict[Path, float]]:
    """Select languages and assign shard weights so P(lang) is proportional to size**temperature."""

    component = int(settings.get("path_component", 0))
    temperature = float(settings.get("temperature", 1.0))
    max_languages = int(settings.get("max_languages", 0))
    included_codes = {_language_code(str(value)) for value in settings.get("include_language_codes", [])}
    # Codes ignore the script suffix, so "tel" admits both tel_Telu and tel_Latn.
    # include_languages matches the directory name exactly, which is what you want
    # whenever the model has a fixed language list.
    included_names = {str(value) for value in settings.get("include_languages", [])}
    excluded_codes = {_language_code(str(value)) for value in settings.get("exclude_language_codes", [])}
    if component < 0:
        raise ValueError("language_sampling.path_component must be non-negative")
    if not 0.0 <= temperature <= 1.0:
        raise ValueError("language_sampling.temperature must be between 0 and 1")
    if included_codes & excluded_codes:
        raise ValueError("language_sampling include/exclude language codes overlap")

    language_files: dict[str, list[Path]] = {}
    for path in files:
        relative_parts = path.relative_to(root).parts
        if component >= len(relative_parts):
            raise ValueError(f"Cannot infer a language from {path} using path component {component}")
        language = relative_parts[component]
        code = _language_code(language)
        if included_names and language not in included_names:
            continue
        if included_codes and code not in included_codes:
            continue
        if code in excluded_codes:
            continue
        language_files.setdefault(language, []).append(path)

    language_sizes = {
        language: sum(path.stat().st_size for path in paths) for language, paths in language_files.items()
    }
    ranked_languages = sorted(language_sizes, key=lambda language: (-language_sizes[language], language))
    if max_languages > 0:
        ranked_languages = ranked_languages[:max_languages]
    selected_languages = set(ranked_languages)
    selected_files = [path for path in files if path.relative_to(root).parts[component] in selected_languages]

    raw_weights: dict[Path, float] = {}
    for path in selected_files:
        language = path.relative_to(root).parts[component]
        language_size = float(language_sizes[language])
        # File-size weighting within a language makes the total language mass
        # exactly size(language)**temperature.
        raw_weights[path] = float(path.stat().st_size) * language_size ** (temperature - 1.0)
    if not raw_weights:
        return [], {}
    maximum = max(raw_weights.values())
    return selected_files, {path: weight / maximum for path, weight in raw_weights.items()}


def _resolve_language(
    spec: dict[str, Any],
    path: Path,
    root: Path,
    component: int | None,
) -> str:
    """Name the language of a shard's positive side.

    A parallel corpus names its directories after a language *pair* (``en-de``),
    so `language_map` translates that into the same identifier the rest of the
    pipeline and the language markers use.
    """
    if component is None:
        return str(spec.get("language_id", ""))
    raw = path.relative_to(root).parts[component]
    mapping = dict(spec.get("language_map") or {})
    return str(mapping.get(raw, raw))


def discover_shards(
    stream_specs: Sequence[dict[str, Any]],
    roots: dict[str, Path],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Resolve configured Parquet globs and validate their pair columns."""

    shards: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for spec in stream_specs:
        name = str(spec["name"])
        root_name = str(spec["root"])
        if root_name not in roots:
            raise ValueError(f"No dataset root was provided for {root_name!r}")
        root = roots[root_name].expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        pattern = str(spec.get("parquet", "**/*.parquet"))
        files = sorted(path.resolve() for path in root.glob(pattern) if path.is_file())
        if not files:
            raise FileNotFoundError(f"No Parquet files matched {pattern!r} under {root}")

        sampling_weights: dict[Path, float] = {}
        language_sampling = spec.get("language_sampling")
        if language_sampling:
            files, sampling_weights = _language_sampling_weights(files, root, dict(language_sampling))
            if not files:
                raise ValueError(f"Language sampling removed every file from stream {name!r}")

        anchor_column = str(spec["anchor_column"])
        positive_column = str(spec["positive_column"])
        columns = set(pq.ParquetFile(files[0]).schema_arrow.names)
        missing = {anchor_column, positive_column} - columns
        if missing:
            raise ValueError(f"Stream {name!r} is missing columns {sorted(missing)} in {files[0].name}")

        counts[name] = len(files)
        language_component = int(dict(language_sampling).get("path_component", 0)) if language_sampling else None
        shards.extend(
            {
                "source": name,
                "path": str(path),
                "anchor_column": anchor_column,
                "positive_column": positive_column,
                "sampling_weight": sampling_weights.get(path, 1.0),
                "language_id": _resolve_language(spec, path, root, language_component),
                "anchor_language_id": str(spec.get("anchor_language", "")) or
                    _resolve_language(spec, path, root, language_component),
                "usable_rows": int(pq.ParquetFile(path).metadata.num_rows),
            }
            for path in files
        )

    random.Random(seed).shuffle(shards)
    return shards, counts


class _SourceRows:
    def __init__(
        self,
        shards: Sequence[PairShard],
        *,
        rng: random.Random,
        fragment_rows: int,
        max_chars: int,
        lemma_filter: dict[str, Any] | None = None,
        holdout_row_groups: int = 0,
    ) -> None:
        if not shards:
            raise ValueError("A streaming source has no Parquet shards")
        self.shards = list(shards)
        self.rng = rng
        self.fragment_rows = int(fragment_rows)
        self.max_chars = int(max_chars)
        # Trailing row groups reserved for evaluation. Training never reads them,
        # so a held-out score cannot be inflated by memorisation — which matters
        # here because arms differ in how often they revisit a given language.
        self.holdout_row_groups = max(0, int(holdout_row_groups))
        self.lemma_filter = _LemmaOverlapFilter(lemma_filter)
        self.shard_weights = [float(shard.sampling_weight) for shard in self.shards]
        self.weighted_shards = any(weight != self.shard_weights[0] for weight in self.shard_weights[1:])
        self.deck: list[PairShard] = []
        self.fragment: Iterator[tuple[str, str]] = iter(())

    def _next_shard(self) -> PairShard:
        if self.weighted_shards:
            return self.rng.choices(self.shards, weights=self.shard_weights, k=1)[0]
        if not self.deck:
            self.deck = self.shards.copy()
            self.rng.shuffle(self.deck)
        return self.deck.pop()

    def _read_fragment(self) -> Iterator[tuple[str, str]]:
        shard = self._next_shard()
        parquet = pq.ParquetFile(shard.path)
        if parquet.metadata.num_row_groups == 0:
            return iter(())

        trainable_groups = parquet.metadata.num_row_groups - self.holdout_row_groups
        if trainable_groups < 1:
            # Never starve a shard that is too small to hold anything back; such
            # a language simply has no held-out split.
            trainable_groups = parquet.metadata.num_row_groups
        row_group = self.rng.randrange(trainable_groups)
        table = parquet.read_row_group(
            row_group,
            columns=[shard.anchor_column, shard.positive_column],
            use_threads=False,
        )
        anchors, positives = _sample_pair_columns(
            table, self.rng, fragment_rows=self.fragment_rows
        )

        def rows() -> Iterator[tuple[str, str]]:
            for anchor, positive in zip(anchors, positives, strict=True):
                if not isinstance(anchor, str) or not isinstance(positive, str):
                    continue
                anchor = anchor.strip()
                positive = positive.strip()
                if not anchor or not positive:
                    continue
                if self.max_chars > 0 and (len(anchor) > self.max_chars or len(positive) > self.max_chars):
                    continue
                if _canonical_text(anchor) == _canonical_text(positive):
                    continue
                if not self.lemma_filter.passes(anchor, positive, shard.language_id):
                    continue
                yield anchor, positive

        return rows()

    def next(self) -> tuple[str, str]:
        while True:
            try:
                return next(self.fragment)
            except StopIteration:
                self.fragment = self._read_fragment()


def _stable_group_batch_seed(
    seed: int, group: tuple[str, str], occurrence: int
) -> int:
    payload = f"{seed}\0{group[0]}\0{group[1]}\0{occurrence}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _sample_pair_columns(
    table: Any, rng: random.Random, *, fragment_rows: int
) -> tuple[list[Any], list[Any]]:
    """Convert only the sampled rows instead of every string in a row group."""

    rows = int(table.num_rows)
    if rows <= 0:
        return [], []
    limit = rows if fragment_rows <= 0 else min(rows, int(fragment_rows))
    generator = np.random.default_rng(rng.getrandbits(64))
    if limit < rows:
        indices = generator.choice(rows, size=limit, replace=False, shuffle=True)
    else:
        indices = np.arange(rows, dtype=np.int64)
        generator.shuffle(indices)
    sampled = table.take(indices)
    return sampled.column(0).to_pylist(), sampled.column(1).to_pylist()


class _DeterministicGroupBatches:
    """Produce a batch from its group-local index, independent of cursor state."""

    def __init__(
        self,
        shards: Sequence[PairShard],
        *,
        group: tuple[str, str],
        seed: int,
        fragment_rows: int,
        max_chars: int,
        lemma_filter: dict[str, Any] | None,
        holdout_row_groups: int,
    ) -> None:
        self.group = group
        self.seed = int(seed)
        self.fragment_rows = int(fragment_rows)
        self.max_chars = int(max_chars)
        self.lemma_filter = _LemmaOverlapFilter(lemma_filter)
        self.fragments: list[tuple[PairShard, int]] = []
        self.fragment_weights: list[float] = []
        for shard in shards:
            parquet = pq.ParquetFile(shard.path)
            trainable_groups = parquet.metadata.num_row_groups - holdout_row_groups
            if trainable_groups < 1:
                trainable_groups = parquet.metadata.num_row_groups
            row_counts = [
                int(parquet.metadata.row_group(index).num_rows)
                for index in range(trainable_groups)
            ]
            total = sum(row_counts)
            if total <= 0:
                continue
            for index, count in enumerate(row_counts):
                self.fragments.append((shard, index))
                self.fragment_weights.append(
                    float(shard.sampling_weight) * count / total
                )
        if not self.fragments or not any(weight > 0 for weight in self.fragment_weights):
            raise ValueError(f"deterministic group {group!r} has no trainable rows")

    def _read_fragment(self, rng: random.Random) -> Iterator[tuple[str, str]]:
        shard, row_group = rng.choices(
            self.fragments, weights=self.fragment_weights, k=1
        )[0]
        table = pq.ParquetFile(shard.path).read_row_group(
            row_group,
            columns=[shard.anchor_column, shard.positive_column],
            use_threads=False,
        )
        anchors, positives = _sample_pair_columns(
            table, rng, fragment_rows=self.fragment_rows
        )
        for anchor, positive in zip(anchors, positives, strict=True):
            if not isinstance(anchor, str) or not isinstance(positive, str):
                continue
            anchor = anchor.strip()
            positive = positive.strip()
            if not anchor or not positive:
                continue
            if self.max_chars > 0 and (
                len(anchor) > self.max_chars or len(positive) > self.max_chars
            ):
                continue
            if _canonical_text(anchor) == _canonical_text(positive):
                continue
            if not self.lemma_filter.passes(anchor, positive, shard.language_id):
                continue
            yield anchor, positive

    def batch(
        self, occurrence: int, *, batch_size: int, strict_unique: bool
    ) -> list[tuple[str, str]]:
        rng = random.Random(
            _stable_group_batch_seed(self.seed, self.group, occurrence)
        )
        seen: set[str] = set()
        pairs: list[tuple[str, str]] = []
        attempts = 0
        while len(pairs) < batch_size and attempts < batch_size * 20:
            for anchor, positive in self._read_fragment(rng):
                attempts += 1
                keys = (_canonical_text(anchor), _canonical_text(positive))
                if any(key in seen for key in keys):
                    if attempts >= batch_size * 20:
                        break
                    continue
                seen.update(keys)
                pairs.append((anchor, positive))
                if len(pairs) >= batch_size or attempts >= batch_size * 20:
                    break
        if len(pairs) < batch_size and strict_unique:
            raise RuntimeError(
                f"could not construct a unique batch of {batch_size} pairs for "
                f"source={self.group[0]!r}, language={self.group[1]!r}"
            )
        while len(pairs) < batch_size:
            fragment = list(self._read_fragment(rng))
            if fragment:
                pairs.extend(fragment[: batch_size - len(pairs)])
        return pairs


def _weighted_source_picker(
    sources: Sequence[str],
    weights: dict[str, float],
    rng: random.Random,
) -> Iterator[str]:
    values = [float(weights.get(source, 1.0)) for source in sources]
    if any(value < 0 for value in values) or not any(value > 0 for value in values):
        raise ValueError("Stream weights must be non-negative and at least one must be positive")
    cumulative = np.cumsum(values).tolist()
    total = cumulative[-1]
    while True:
        yield sources[bisect.bisect_left(cumulative, rng.random() * total)]


def _batch_no_duplicates(
    pairs: Iterator[dict[str, str]],
    *,
    batch_size: int,
) -> Iterator[dict[str, str]]:
    """Reject repeated anchor/positive texts inside each emitted batch."""

    seen: set[str] = set()
    emitted_in_batch = 0
    for pair in pairs:
        keys = (_canonical_text(pair["anchor"]), _canonical_text(pair["positive"]))
        if any(key in seen for key in keys):
            continue
        seen.update(keys)
        yield pair
        emitted_in_batch += 1
        if emitted_in_batch >= batch_size:
            seen.clear()
            emitted_in_batch = 0


def _monolingual_batches(
    by_source: dict[str, list[PairShard]],
    *,
    settings: dict[str, Any],
    rng: random.Random,
    batch_size: int,
    holdout_row_groups: int,
) -> Iterator[dict[str, str]]:
    """Emit runs of ``batch_size`` pairs drawn from a single language.

    In-batch negatives are the whole contrastive signal here. A language-mixed
    batch of 16,384 over 60 languages leaves only a few hundred same-language
    negatives; the rest are separable on script alone and teach nothing. Keeping
    a batch monolingual makes every negative a real one at no extra cost.

    Languages are picked with probability proportional to the shard sampling
    weights, which already encode ``size**temperature``, so this changes only
    how pairs are grouped, never how often a language is seen.
    """

    # A group must be homogeneous in both source and target language. FineWeb2
    # and CCMatrix can share the same positive-side language but not the same
    # anchor language: a Russian FineWeb pair is ru->ru while a CCMatrix pair is
    # en->ru. Grouping by language alone used the first shuffled shard's marker
    # for both sources, silently mistagging one side of every other source.
    by_group: dict[tuple[str, str], list[PairShard]] = {}
    for shards in by_source.values():
        for shard in shards:
            by_group.setdefault((shard.source, shard.language_id), []).append(shard)
    if not by_group:
        raise ValueError("Monolingual batching found no language-tagged shards")

    groups = sorted(by_group)
    source_weights = dict(settings.get("source_weights") or {})
    if any(float(value) < 0 for value in source_weights.values()):
        raise ValueError("Stream weights must be non-negative")
    group_mass = {
        group: sum(float(shard.sampling_weight) for shard in shards)
        for group, shards in by_group.items()
    }
    source_mass: dict[str, float] = {}
    for (source_name, _), mass in group_mass.items():
        source_mass[source_name] = source_mass.get(source_name, 0.0) + mass
    # First normalise languages inside a source, then assign the configured
    # source mass. This makes a 0.5/0.5 stream mixture a realised 0.5/0.5 batch
    # mixture regardless of how many shards either source owns.
    weights = [
        group_mass[group] / source_mass[group[0]]
        * float(source_weights.get(group[0], 1.0))
        for group in groups
    ]
    if not any(weight > 0 for weight in weights):
        raise ValueError("Monolingual batching needs at least one positive language weight")

    smooth = bool(settings.get("smooth_weighted_round_robin", False))
    source_scheduler: SmoothWeightedRoundRobin | None = None
    language_schedulers: dict[str, SmoothWeightedRoundRobin] = {}
    group_occurrences = {group: 0 for group in groups}
    if smooth:
        source_temperature = float(settings.get("source_temperature", 1.0))
        language_temperature = float(settings.get("language_temperature", 1.0))
        if not 0.0 <= source_temperature <= 1.0:
            raise ValueError("source_temperature must be between zero and one")
        if not 0.0 <= language_temperature <= 1.0:
            raise ValueError("language_temperature must be between zero and one")

        def usable_rows(shard: PairShard) -> int:
            if holdout_row_groups <= 0 and shard.usable_rows > 0:
                return int(shard.usable_rows)
            parquet = pq.ParquetFile(shard.path)
            trainable_groups = parquet.metadata.num_row_groups - holdout_row_groups
            if trainable_groups < 1:
                trainable_groups = parquet.metadata.num_row_groups
            return sum(
                int(parquet.metadata.row_group(index).num_rows)
                for index in range(trainable_groups)
            )

        group_rows = {
            group: sum(usable_rows(shard) for shard in shards)
            for group, shards in by_group.items()
        }
        if any(count <= 0 for count in group_rows.values()):
            raise ValueError("smooth scheduling requires positive usable rows per group")
        source_rows: dict[str, int] = {}
        for (source_name, _), count in group_rows.items():
            source_rows[source_name] = source_rows.get(source_name, 0) + count
        sources = sorted(source_rows)
        source_schedule_weights = [
            float(source_weights.get(source_name, 1.0))
            * float(source_rows[source_name]) ** source_temperature
            for source_name in sources
        ]
        source_scheduler = SmoothWeightedRoundRobin(
            sources,
            source_schedule_weights,
            seed=rng.getrandbits(64),
        )
        for source_name in sources:
            languages = sorted(
                language for source, language in groups if source == source_name
            )
            language_schedulers[source_name] = SmoothWeightedRoundRobin(
                languages,
                [
                    float(group_rows[source_name, language]) ** language_temperature
                    for language in languages
                ],
                seed=rng.getrandbits(64),
            )

        for _ in range(int(settings.get("scheduler_start_batch", 0))):
            source_name = source_scheduler.next_item()
            group = (source_name, language_schedulers[source_name].next_item())
            group_occurrences[group] += 1

    rows = {
        group: (
            _DeterministicGroupBatches(
                by_group[group],
                group=group,
                seed=int(settings["seed"]),
                fragment_rows=int(settings["fragment_rows"]),
                max_chars=int(settings["max_chars"]),
                lemma_filter=dict(settings.get("lemma_filter") or {}),
                holdout_row_groups=holdout_row_groups,
            )
            if smooth
            else _SourceRows(
                by_group[group],
                rng=random.Random(rng.getrandbits(64)),
                fragment_rows=int(settings["fragment_rows"]),
                max_chars=int(settings["max_chars"]),
                lemma_filter=dict(settings.get("lemma_filter") or {}),
                holdout_row_groups=holdout_row_groups,
            )
        )
        for group in groups
    }

    # Tests need to see which language produced a pair; training must not, because
    # the dataset features are fixed to anchor/positive.
    debug = bool(settings.get("debug_language", False))

    def emit(
        anchor: str, positive: str, source_name: str, language: str
    ) -> dict[str, str]:
        # The language is carried inside the text as a marker token, the way NLLB
        # is told its source language. That keeps the dataset schema at two string
        # columns, so nothing in the collator or trainer has to know about it, and
        # each side of a pair can carry a different language.
        pair = {"anchor": anchor_tag + anchor, "positive": positive_tag + positive}
        if debug:
            pair["source"] = source_name
            pair["language"] = language
        return pair

    marker = bool(settings.get("language_markers", False))
    strict_unique = bool(settings.get("strict_unique_batches", False))

    while True:
        if source_scheduler is None:
            group = rng.choices(groups, weights=weights, k=1)[0]
        else:
            source_name = source_scheduler.next_item()
            group = (source_name, language_schedulers[source_name].next_item())
        source = rows[group]
        first = by_group[group][0]
        language = first.language_id
        anchor_tag = f"__{first.anchor_language_id or language}__ " if marker else ""
        positive_tag = f"__{language}__ " if marker else ""
        if isinstance(source, _DeterministicGroupBatches):
            occurrence = group_occurrences[group]
            group_occurrences[group] += 1
            for anchor, positive in source.batch(
                occurrence,
                batch_size=batch_size,
                strict_unique=strict_unique,
            ):
                yield emit(anchor, positive, group[0], language)
            continue
        seen: set[str] = set()
        emitted = 0
        attempts = 0
        while emitted < batch_size and attempts < batch_size * 20:
            attempts += 1
            anchor, positive = source.next()
            keys = (_canonical_text(anchor), _canonical_text(positive))
            if any(key in seen for key in keys):
                continue
            seen.update(keys)
            emitted += 1
            yield emit(anchor, positive, group[0], language)
        if emitted < batch_size and strict_unique:
            raise RuntimeError(
                f"could not construct a unique batch of {batch_size} pairs for "
                f"source={group[0]!r}, language={language!r}"
            )
        while emitted < batch_size:
            # Never emit a short batch: a ragged tail would silently change the
            # number of in-batch negatives and confound the comparison.
            anchor, positive = source.next()
            emitted += 1
            yield emit(anchor, positive, group[0], language)


def _random_pairs(shards: Sequence[dict[str, Any]], settings: dict[str, Any]) -> Iterator[dict[str, str]]:
    worker_id = 0
    try:
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
    except ImportError:
        pass
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    seed = int(settings["seed"]) + rank * 1_000_003 + worker_id * 10_007
    rng = random.Random(seed)

    by_source: dict[str, list[PairShard]] = {}
    for raw in shards:
        shard = PairShard(**raw)
        by_source.setdefault(shard.source, []).append(shard)
    if not by_source:
        raise ValueError("This worker received no Parquet shards")

    holdout_row_groups = int(settings.get("holdout_row_groups", 0))
    source_rows = {
        source: _SourceRows(
            source_shards,
            rng=random.Random(rng.getrandbits(64)),
            fragment_rows=int(settings["fragment_rows"]),
            max_chars=int(settings["max_chars"]),
            lemma_filter=dict(settings.get("lemma_filter") or {}),
            holdout_row_groups=holdout_row_groups,
        )
        for source, source_shards in by_source.items()
    }
    sources = sorted(source_rows)
    source_picker = _weighted_source_picker(sources, dict(settings["source_weights"]), rng)

    def mixed_pairs() -> Iterator[dict[str, str]]:
        while True:
            source = next(source_picker)
            anchor, positive = source_rows[source].next()
            yield {"anchor": anchor, "positive": positive}

    batch_size = int(settings.get("batch_size", 0))
    if bool(settings.get("monolingual_batches", False)):
        if batch_size <= 1:
            raise ValueError("monolingual_batches requires streaming.batch_size")
        yield from _monolingual_batches(
            by_source,
            settings=settings,
            rng=rng,
            batch_size=batch_size,
            holdout_row_groups=holdout_row_groups,
        )
        return

    stream: Iterator[dict[str, str]] = mixed_pairs()
    shuffle_buffer_size = int(settings["shuffle_buffer_size"])
    if shuffle_buffer_size > 1:
        unshuffled = stream

        def shuffled_pairs() -> Iterator[dict[str, str]]:
            buffer = [next(unshuffled) for _ in range(shuffle_buffer_size)]
            while True:
                index = rng.randrange(shuffle_buffer_size)
                result = buffer[index]
                buffer[index] = next(unshuffled)
                yield result

        stream = shuffled_pairs()

    if not bool(settings.get("batch_no_duplicates", False)) or batch_size <= 1:
        yield from stream
        return

    yield from _batch_no_duplicates(stream, batch_size=batch_size)


@dataclass(frozen=True)
class _ProducerFailure:
    error: BaseException


def _queued_pairs(shards: Sequence[dict[str, Any]], settings: dict[str, Any]) -> Iterator[dict[str, str]]:
    pair_queue: queue.Queue[dict[str, str] | _ProducerFailure] = queue.Queue(maxsize=int(settings["queue_size"]))
    stopped = threading.Event()

    def put(item: dict[str, str] | _ProducerFailure) -> None:
        while not stopped.is_set():
            try:
                pair_queue.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def produce() -> None:
        try:
            for pair in _random_pairs(shards, settings):
                if stopped.is_set():
                    return
                put(pair)
        except BaseException as error:
            put(_ProducerFailure(error))

    producer = threading.Thread(target=produce, name="starse-parquet-prefetch", daemon=True)
    producer.start()
    try:
        while True:
            item = pair_queue.get()
            if isinstance(item, _ProducerFailure):
                raise RuntimeError("The Parquet prefetch worker failed") from item.error
            yield item
    finally:
        stopped.set()


def _queued_pairs_from_replica(
    shard_replicas: Sequence[Sequence[dict[str, Any]]],
    settings: dict[str, Any],
) -> Iterator[dict[str, str]]:
    """Give every DataLoader worker the full weighted shard population."""

    if len(shard_replicas) != 1:
        raise ValueError("Each dataset shard must contain exactly one shard replica")
    yield from _queued_pairs(shard_replicas[0], settings)


def build_streaming_dataset(
    stream_specs: Sequence[dict[str, Any]],
    roots: dict[str, Path],
    streaming_settings: dict[str, Any],
) -> tuple[Any, dict[str, int]]:
    """Create an infinite Hugging Face IterableDataset with known pair features."""

    from datasets import Features, IterableDataset, Value

    seed = int(streaming_settings["seed"])
    shards, counts = discover_shards(stream_specs, roots, seed=seed)
    source_weights = {str(spec["name"]): float(spec.get("weight", 1.0)) for spec in stream_specs}
    settings = {
        "seed": seed,
        "source_weights": source_weights,
        "fragment_rows": int(streaming_settings.get("fragment_rows", 2048)),
        "max_chars": int(streaming_settings.get("max_chars", 512)),
        "shuffle_buffer_size": int(streaming_settings.get("shuffle_buffer_size", 65536)),
        "queue_size": int(streaming_settings.get("queue_size", 8192)),
        "lemma_filter": dict(streaming_settings.get("lemma_filter") or {}),
        "batch_size": int(streaming_settings.get("batch_size", 0)),
        "batch_no_duplicates": bool(streaming_settings.get("batch_no_duplicates", False)),
        "monolingual_batches": bool(streaming_settings.get("monolingual_batches", False)),
        "language_markers": bool(streaming_settings.get("language_markers", False)),
        "holdout_row_groups": int(streaming_settings.get("holdout_row_groups", 0)),
        "smooth_weighted_round_robin": bool(
            streaming_settings.get("smooth_weighted_round_robin", False)
        ),
        "source_temperature": float(streaming_settings.get("source_temperature", 1.0)),
        "language_temperature": float(
            streaming_settings.get("language_temperature", 1.0)
        ),
        "scheduler_start_batch": int(
            streaming_settings.get("scheduler_start_batch", 0)
        ),
        "strict_unique_batches": bool(
            streaming_settings.get("strict_unique_batches", False)
        ),
    }
    features = Features({"anchor": Value("string"), "positive": Value("string")})
    worker_replicas = int(streaming_settings.get("worker_replicas", 0))
    if worker_replicas > 0:
        dataset = IterableDataset.from_generator(
            _queued_pairs_from_replica,
            gen_kwargs={"shard_replicas": [tuple(shards)] * worker_replicas, "settings": settings},
            features=features,
        )
    else:
        dataset = IterableDataset.from_generator(
            _queued_pairs,
            gen_kwargs={"shards": shards, "settings": settings},
            features=features,
        )
    return dataset, counts
