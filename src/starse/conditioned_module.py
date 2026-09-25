"""SentenceTransformer module for a language-conditioned static encoder.

Subclasses ``StaticEmbedding`` so tokenisation, saving and loading are inherited;
only ``forward`` changes. The language arrives as a marker token at the front of
each sequence, which ``forward`` peels off before pooling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from sentence_transformers.sentence_transformer.modules.static_embedding import StaticEmbedding
from tokenizers import Tokenizer

from starse.collapse import CollapseTable
from starse.language_conditioned import GATING_MODES, LanguageConditioner, split_markers
from starse.ngram_table import NgramTable
from starse.online_tokenizer import OnlineMergeTable
from starse.recursive_cascade import DenseRecursiveCascade
from starse.split_vocabulary import SplitVocabulary
from starse.vocab_assignment import VocabAssignment

CONFIG_NAME = "language_conditioning.json"


def marker_for(language: str) -> str:
    return f"__{language}__"


class ConditionedStaticEmbedding(StaticEmbedding):
    def __init__(
        self,
        tokenizer: Tokenizer,
        embedding_weights=None,
        embedding_dim: int | None = None,
        *,
        languages: list[str] | None = None,
        mode: str = "none",
        code_dim: int = 16,
        n_senses: int = 64,
        rank: int = 8,
        split_index: str | None = None,
        ngram_buckets: int = 0,
        n_vocabs: int = 4,
        weight_index: str | None = None,
        dynamic_pair_keys: list[int] | None = None,
        dynamic_pair_slots: list[int] | None = None,
        reclaimed_token_ids: list[int] | None = None,
        reclaimed_left_ids: list[int] | None = None,
        reclaimed_right_ids: list[int] | None = None,
        growth_residual_buckets: int = 0,
        online_rank: int = 32,
        compiled_pair_keys: list[int] | None = None,
        compiled_pair_scores: list[float] | None = None,
        recursive_max_span_length: int = 32,
        recursive_restored_state: dict[str, torch.Tensor] | None = None,
        ignored_token_ids: list[int] | tuple[int, ...] | None = None,
        default_language: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, embedding_weights=embedding_weights,
                         embedding_dim=embedding_dim, **kwargs)
        self.languages = list(languages or [])
        self.mode = mode
        vocab_size = self.embedding.weight.shape[0]
        dim = self.embedding.weight.shape[1]
        ignored = tuple(sorted({int(value) for value in (ignored_token_ids or ())}))
        if any(value < 0 or value >= vocab_size for value in ignored):
            raise ValueError("ignored token id is outside the conditioned vocabulary")
        self.ignored_token_ids = list(ignored)
        ignored_lookup = torch.zeros(vocab_size, dtype=torch.bool)
        if ignored:
            ignored_lookup[torch.tensor(ignored, dtype=torch.long)] = True
        self.register_buffer("ignored_token_lookup", ignored_lookup, persistent=False)
        if default_language is not None and default_language not in self.languages:
            raise ValueError("default language is absent from the conditioned language list")
        self.default_language = default_language

        # marker token id -> language index; -1 for every ordinary token, which
        # makes a missing marker fail loudly instead of silently picking language 0
        lookup = torch.full((vocab_size,), -1, dtype=torch.long)
        for index, language in enumerate(self.languages):
            token_id = self.tokenizer.token_to_id(marker_for(language))
            if token_id is None:
                raise ValueError(f"tokenizer has no marker for {language!r}")
            lookup[token_id] = index
        self.register_buffer("marker_lookup", lookup, persistent=False)

        self.split_index = split_index
        self.split = None
        if mode in ("split", "fuse"):
            if not split_index:
                raise ValueError("mode 'split' needs split_index")
            self.split = SplitVocabulary(split_index, vocab_size, dim, len(self.languages))
            self.split.seed_from(self.embedding.weight.data)

        self.n_vocabs = int(n_vocabs)
        self.assign = None
        if mode == "vocabmoe":
            if not split_index:
                raise ValueError("mode 'vocabmoe' needs split_index")
            self.assign = VocabAssignment(split_index, vocab_size, dim,
                                          len(self.languages), self.n_vocabs)

        self.ngram_buckets = int(ngram_buckets)
        self.growth_residual_buckets = int(growth_residual_buckets)
        self.ngram = None
        self.collapse = None
        self.online_rank = int(online_rank)
        self.online_collapse = None
        self.recursive_max_span_length = int(recursive_max_span_length)
        self.recursive_cascade = None
        if mode in ("collapse", "collapse_shared", "collapse_dynamic", "collapse_reclaimed"):
            if not self.ngram_buckets:
                raise ValueError(f"mode {mode!r} needs ngram_buckets")
            bias_languages = 1 if mode == "collapse_shared" else len(self.languages)
            self.collapse = CollapseTable(
                self.ngram_buckets,
                dim,
                bias_languages,
                pair_key_base=vocab_size,
                pair_keys=dynamic_pair_keys,
                pair_slots=dynamic_pair_slots,
                reclaimed_token_ids=(
                    reclaimed_token_ids if mode == "collapse_reclaimed" else None
                ),
                residual_buckets=self.growth_residual_buckets,
            )
        if mode in ("collapse_online", "collapse_online_compiled"):
            if mode == "collapse_online" and not self.ngram_buckets:
                raise ValueError("mode 'collapse_online' needs ngram_buckets")
            if mode == "collapse_online_compiled" and (
                compiled_pair_keys is None or compiled_pair_scores is None
            ):
                raise ValueError(
                    "mode 'collapse_online_compiled' needs exact pair keys and scores"
                )
            self.online_collapse = OnlineMergeTable(
                max(1, self.ngram_buckets),
                dim,
                len(self.languages),
                rank=self.online_rank,
                pair_key_base=vocab_size,
                compiled_pair_keys=(
                    torch.as_tensor(compiled_pair_keys, dtype=torch.long)
                    if mode == "collapse_online_compiled" else None
                ),
                compiled_pair_scores=(
                    torch.as_tensor(compiled_pair_scores, dtype=torch.float32)
                    if mode == "collapse_online_compiled" else None
                ),
            )
        if mode == "collapse_recursive":
            if not self.ngram_buckets:
                raise ValueError("mode 'collapse_recursive' needs proposal buckets")
            self.recursive_cascade = DenseRecursiveCascade(
                base_size=vocab_size,
                dim=dim,
                n_languages=len(self.languages),
                residual_buckets=self.ngram_buckets,
                max_span_length=self.recursive_max_span_length,
                restored_state=recursive_restored_state,
            )

        self.reclaimed_token_ids = list(reclaimed_token_ids or [])
        self.reclaimed_left_ids = list(reclaimed_left_ids or [])
        self.reclaimed_right_ids = list(reclaimed_right_ids or [])
        reclaim_left = torch.full((vocab_size,), -1, dtype=torch.long)
        reclaim_right = torch.full((vocab_size,), -1, dtype=torch.long)
        if mode == "collapse_reclaimed":
            if not (len(self.reclaimed_token_ids) == len(self.reclaimed_left_ids)
                    == len(self.reclaimed_right_ids)):
                raise ValueError("reclaimed token and parent arrays must align")
            reclaimed = torch.tensor(self.reclaimed_token_ids, dtype=torch.long)
            left_parent = torch.tensor(self.reclaimed_left_ids, dtype=torch.long)
            right_parent = torch.tensor(self.reclaimed_right_ids, dtype=torch.long)
            if reclaimed.numel() != self.ngram_buckets:
                raise ValueError("collapse_reclaimed needs one physical row per exact pair")
            if reclaimed.numel() and (
                reclaimed.unique().numel() != reclaimed.numel()
                or int(reclaimed.min()) < 0 or int(reclaimed.max()) >= vocab_size
                or int(left_parent.min()) < 0 or int(left_parent.max()) >= vocab_size
                or int(right_parent.min()) < 0 or int(right_parent.max()) >= vocab_size
            ):
                raise ValueError("invalid reclaimed token or parent id")
            selected = set(self.reclaimed_token_ids)
            if selected & set(self.reclaimed_left_ids + self.reclaimed_right_ids):
                raise ValueError("reclaimed split map contains direct recursion")
            reclaim_left[reclaimed] = left_parent
            reclaim_right[reclaimed] = right_parent
        self.register_buffer("reclaim_left", reclaim_left, persistent=False)
        self.register_buffer("reclaim_right", reclaim_right, persistent=False)

        self.weight_index = weight_index
        if mode == "idfpool":
            if not weight_index:
                raise ValueError("mode 'idfpool' needs weight_index")
            import numpy as np

            data = np.load(weight_index)
            stored = [str(name) for name in data["languages"]]
            if stored != self.languages:
                raise ValueError(
                    "weight table languages do not match the model's, in order: "
                    f"{stored[:3]}... vs {self.languages[:3]}...")
            table = torch.from_numpy(data["weight"].astype("float32"))
            if table.shape[1] < vocab_size:
                # Marker rows are appended after the weight table was built. They
                # are stripped before pooling, so their weight is never read, but
                # the buffer still has to be indexable by any id in the table.
                pad = torch.ones(table.shape[0], vocab_size - table.shape[1])
                table = torch.cat([table, pad], dim=1)
            elif table.shape[1] > vocab_size:
                raise ValueError(
                    f"weight table has {table.shape[1]} columns, vocabulary is {vocab_size}")
            self.register_buffer("pool_weight", table, persistent=True)

        if mode == "wordpool":
            # A token that starts with the SentencePiece boundary marker opens a
            # new word, so a running sum of that flag inside a sentence gives
            # word ids without touching the tokenizer.
            starts = torch.zeros(vocab_size, dtype=torch.bool)
            for token_id in range(vocab_size):
                piece = self.tokenizer.id_to_token(token_id)
                if piece is None or piece.startswith("\u2581"):
                    starts[token_id] = True
            self.register_buffer("word_start", starts, persistent=False)

        if mode == "ngram_gate":
            # One scalar per language: how much n-gram signal this language wants.
            # It starts open enough that the n-gram rows receive gradient, and
            # what it converges to is itself the result — a language that
            # tokenises into words should learn to shut it.
            self.ngram_gate = torch.nn.Parameter(torch.zeros(len(self.languages)))

        if mode in ("ngram", "ngram3", "ngram_gate"):
            if not self.ngram_buckets:
                raise ValueError(f"mode {mode!r} needs ngram_buckets")
            orders = (2, 3) if mode == "ngram3" else (2,)
            self.ngram = NgramTable(self.ngram_buckets, dim, orders)

        self.conditioner = LanguageConditioner(
            mode=mode, dim=dim, vocab_size=vocab_size,
            n_languages=len(self.languages), code_dim=code_dim,
            n_senses=n_senses, rank=rank,
        )

    def preprocess(self, inputs: list[str], prompt: str | None = None, **kwargs):
        """Tokenize texts, optionally supplying the sole deployment language.

        Multilingual training continues to carry explicit markers in the input
        stream. A monolingual model may instead register ``default_language``;
        this keeps ordinary ``encode()`` and benchmark calls marker-safe without
        requiring an evaluation-specific text wrapper.
        """

        if prompt:
            inputs = self._prepend_prompt(inputs, prompt)
        if self.default_language is not None:
            markers = tuple(f"{marker_for(language)} " for language in self.languages)
            prefix = f"{marker_for(self.default_language)} "
            inputs = [
                text if text.startswith(markers) else prefix + text
                for text in inputs
            ]
        return super().preprocess(inputs, prompt=None, **kwargs)

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        input_ids = features["input_ids"]
        offsets = features["offsets"]
        # Only the "marker" mode wants the marker inside the mean; there its
        # presence IS the mechanism. Everywhere else it must be stripped, so the
        # control is exactly the unconditioned model and the marker row never
        # receives a gradient.
        keep_marker = self.mode == "marker"
        content, segment, lengths, language, new_offsets = split_markers(
            input_ids, offsets, self.marker_lookup, keep_marker)
        if (language < 0).any():
            raise ValueError(
                "a sequence did not start with a language marker; the stream must "
                "set streaming.language_markers and the evaluator must prepend one")
        if self.ignored_token_ids and content.numel():
            keep = ~self.ignored_token_lookup[content]
            content = content[keep]
            segment = segment[keep]
            lengths = torch.bincount(segment, minlength=language.shape[0])
            new_offsets = torch.cat([
                torch.zeros(1, dtype=torch.long, device=offsets.device),
                lengths.cumsum(0)[:-1],
            ])
        if self.mode in ("collapse_online", "collapse_online_compiled"):
            vectors = self.embedding.weight[content]
            pooled, _ = self.online_collapse.pool(
                vectors,
                content,
                segment,
                language,
                language.shape[0],
                embedding_weight=self.embedding.weight,
            )
            features["sentence_embedding"] = pooled
            return features
        if self.mode == "collapse_recursive":
            pooled, emitted, emitted_segment = self.recursive_cascade.pool(
                self.embedding.weight,
                content,
                segment,
                language,
                language.shape[0],
            )
            features["sentence_embedding"] = pooled
            # Training callbacks and lifecycle smokes may inspect this detached
            # stream; SentenceTransformers ignores additional feature entries.
            features["recursive_token_ids"] = emitted.detach()
            features["recursive_token_segment"] = emitted_segment.detach()
            return features
        if self.mode in ("collapse", "collapse_shared", "collapse_dynamic", "collapse_reclaimed"):
            if self.mode == "collapse_reclaimed" and content.numel():
                selected = self.reclaim_left[content] >= 0
                if bool(selected.any()):
                    lengths_per_token = 1 + selected.to(torch.long)
                    starts = lengths_per_token.cumsum(0) - lengths_per_token
                    expanded = torch.empty(
                        int(lengths_per_token.sum()), dtype=torch.long, device=content.device
                    )
                    first = content.clone()
                    first[selected] = self.reclaim_left[content[selected]]
                    expanded[starts] = first
                    expanded[starts[selected] + 1] = self.reclaim_right[content[selected]]
                    segment = torch.repeat_interleave(segment, lengths_per_token)
                    split_per_sentence = torch.zeros_like(lengths)
                    split_per_sentence.index_add_(0, segment[starts], selected.to(torch.long))
                    lengths = lengths + split_per_sentence
                    content = expanded
            vectors = self.embedding.weight[content]
            collapse_language = torch.zeros_like(language) if self.mode == "collapse_shared" else language
            features["sentence_embedding"] = self.collapse.pool(
                vectors, content, segment, collapse_language, language.shape[0],
                embedding_weight=(
                    self.embedding.weight if self.mode == "collapse_reclaimed" else None
                ),
            )
            return features

        if self.mode == "idfpool":
            # A weighted mean with a fixed per-(language, token) weight. The
            # denominator carries the same weights, so a sentence of common
            # tokens is not simply scaled down — it is the *relative* weight
            # inside a sentence that changes.
            vectors = self.embedding.weight[content]
            weight = self.pool_weight[language[segment], content].to(vectors.dtype)
            numerator = torch.zeros(language.shape[0], vectors.shape[1],
                                    dtype=vectors.dtype, device=vectors.device)
            numerator.index_add_(0, segment, vectors * weight.unsqueeze(1))
            denominator = torch.zeros(language.shape[0], dtype=vectors.dtype,
                                      device=vectors.device)
            denominator.index_add_(0, segment, weight)
            features["sentence_embedding"] = numerator / denominator.clamp(min=1e-6).unsqueeze(1)
            return features

        if self.mode == "vocabmoe":
            shared = self.embedding.weight[content]
            vectors = self.assign(content, language[segment], shared)
            numerator = torch.zeros(language.shape[0], vectors.shape[1],
                                    dtype=vectors.dtype, device=vectors.device)
            numerator.index_add_(0, segment, vectors)
            features["sentence_embedding"] = numerator / lengths.clamp(min=1).unsqueeze(1).to(vectors.dtype)
            return features

        if self.mode == "wordpool":
            vectors = self.embedding.weight[content]
            starts = self.word_start[content].to(torch.long)
            # Word index inside the sentence: restart the running sum at each
            # sentence so words never merge across the batch.
            within = torch.cumsum(starts, 0)
            offset = torch.zeros_like(within)
            first = torch.zeros(language.shape[0], dtype=within.dtype, device=within.device)
            first.scatter_reduce_(0, segment, within, reduce="amin", include_self=False)
            offset = first[segment]
            word = (within - offset)
            key = segment * (word.max() + 1) + word
            uniq, inverse = torch.unique(key, return_inverse=True)
            wsum = torch.zeros(uniq.numel(), vectors.shape[1],
                               dtype=vectors.dtype, device=vectors.device)
            wsum.index_add_(0, inverse, vectors)
            wcount = torch.zeros(uniq.numel(), dtype=vectors.dtype, device=vectors.device)
            wcount.index_add_(0, inverse, torch.ones_like(inverse, dtype=vectors.dtype))
            word_vectors = wsum / wcount.clamp(min=1.0).unsqueeze(1)
            word_segment = torch.zeros(uniq.numel(), dtype=torch.long, device=vectors.device)
            word_segment.scatter_(0, inverse, segment)
            numerator = torch.zeros(language.shape[0], vectors.shape[1],
                                    dtype=vectors.dtype, device=vectors.device)
            numerator.index_add_(0, word_segment, word_vectors)
            counts = torch.zeros(language.shape[0], dtype=vectors.dtype, device=vectors.device)
            counts.index_add_(0, word_segment, torch.ones_like(word_segment, dtype=vectors.dtype))
            features["sentence_embedding"] = numerator / counts.clamp(min=1.0).unsqueeze(1)
            return features

        if self.ngram is not None:
            vectors = self.embedding.weight[content]
            extra, extra_segment = self.ngram.gather(content, segment)
            weight = None
            if self.mode == "ngram_gate" and extra.numel():
                weight = torch.sigmoid(self.ngram_gate)[language[extra_segment]].unsqueeze(1)
                extra = extra * weight
            allv = torch.cat([vectors, extra]) if extra.numel() else vectors
            alls = torch.cat([segment, extra_segment]) if extra.numel() else segment
            numerator = torch.zeros(language.shape[0], allv.shape[1],
                                    dtype=allv.dtype, device=allv.device)
            numerator.index_add_(0, alls, allv)
            counts = torch.zeros(language.shape[0], dtype=allv.dtype, device=allv.device)
            unit = torch.ones_like(alls, dtype=allv.dtype)
            if weight is not None:
                unit[vectors.shape[0]:] = weight.squeeze(1)
            counts.index_add_(0, alls, unit)
            features["sentence_embedding"] = numerator / counts.clamp(min=1.0).unsqueeze(1)
            return features
        if self.mode in ("split", "fuse"):
            lang_per_token = language[segment]
            shared = self.embedding.weight[content]
            vectors = self.split(content, lang_per_token, shared,
                                 use_gate=self.mode == "split")
            numerator = torch.zeros(language.shape[0], vectors.shape[1],
                                    dtype=vectors.dtype, device=vectors.device)
            numerator.index_add_(0, segment, vectors)
            pooled = numerator / lengths.clamp(min=1).unsqueeze(1).to(vectors.dtype)
            features["sentence_embedding"] = pooled
            return features
        if self.mode in GATING_MODES:
            # Gating reweights tokens, so pooling has to happen here rather than
            # in the EmbeddingBag: a weighted mean is not a mean of weighted rows
            # unless the denominator carries the same weights.
            vectors = self.embedding.weight[content]
            weight = self.conditioner.token_gate(content, language[segment])
            numerator = torch.zeros(language.shape[0], vectors.shape[1],
                                    dtype=vectors.dtype, device=vectors.device)
            numerator.index_add_(0, segment, vectors * weight.unsqueeze(1))
            denominator = torch.zeros(language.shape[0], dtype=vectors.dtype,
                                      device=vectors.device)
            denominator.index_add_(0, segment, weight)
            pooled = numerator / denominator.clamp(min=1e-6).unsqueeze(1)
        else:
            pooled = self.embedding(content, new_offsets)
        features["sentence_embedding"] = self.conditioner(
            pooled, language, content, segment, lengths)
        return features

    def configure_dynamic_vocabulary_discovery(
        self,
        *,
        candidate_capacity: int,
        top_per_forward: int,
        admission_mode: str = "utility",
    ) -> None:
        if self.mode != "collapse_dynamic" or self.collapse is None:
            raise ValueError("dynamic vocabulary discovery requires collapse_dynamic mode")
        self.collapse.configure_discovery(
            candidate_capacity=candidate_capacity,
            top_per_forward=top_per_forward,
            admission_mode=admission_mode,
        )

    def configure_dynamic_vocabulary_diagnostic_discovery(
        self,
        *,
        candidate_capacity: int,
        top_per_forward: int,
        admission_mode: str = "utility",
    ) -> None:
        if self.mode != "collapse_dynamic" or self.collapse is None:
            raise ValueError("diagnostic discovery requires collapse_dynamic mode")
        self.collapse.configure_diagnostic_discovery(
            candidate_capacity=candidate_capacity,
            top_per_forward=top_per_forward,
            admission_mode=admission_mode,
        )

    def promote_dynamic_vocabulary(
        self,
        pair_keys: tuple[int, ...] | list[int],
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, int]:
        if self.mode != "collapse_dynamic" or self.collapse is None:
            raise ValueError("dynamic vocabulary promotion requires collapse_dynamic mode")
        return self.collapse.promote_exact_vocabulary(pair_keys, optimizer=optimizer)

    def grow_dynamic_vocabulary(
        self,
        pair_keys: tuple[int, ...] | list[int],
        *,
        optimizer: torch.optim.Optimizer,
        residual_buckets: int,
    ) -> dict[str, int]:
        if self.mode != "collapse_dynamic" or self.collapse is None:
            raise ValueError("dynamic vocabulary growth requires collapse_dynamic mode")
        report = self.collapse.grow_exact_vocabulary(
            pair_keys,
            optimizer=optimizer,
            residual_buckets=residual_buckets,
        )
        self.ngram_buckets = int(report["active_exact_rows"])
        self.growth_residual_buckets = int(report["residual_rows"])
        return report

    def promote_dynamic_vocabulary_to_reclaimed(
        self,
        pair_keys: tuple[int, ...] | list[int],
        reclaimed_token_ids: tuple[int, ...] | list[int],
        reclaimed_left_ids: tuple[int, ...] | list[int],
        reclaimed_right_ids: tuple[int, ...] | list[int],
        *,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, int]:
        if self.mode != "collapse_dynamic" or self.collapse is None:
            raise ValueError("direct reclaimed promotion requires collapse_dynamic mode")
        if not (len(pair_keys) == len(reclaimed_token_ids) == len(reclaimed_left_ids)
                == len(reclaimed_right_ids)):
            raise ValueError("direct reclaimed promotion arrays must align")
        reclaimed = torch.as_tensor(
            reclaimed_token_ids, dtype=torch.long, device=self.embedding.weight.device
        )
        left = torch.as_tensor(
            reclaimed_left_ids, dtype=torch.long, device=self.embedding.weight.device
        )
        right = torch.as_tensor(
            reclaimed_right_ids, dtype=torch.long, device=self.embedding.weight.device
        )
        if reclaimed.numel() and (
            reclaimed.unique().numel() != reclaimed.numel()
            or int(reclaimed.min()) < 0 or int(reclaimed.max()) >= self.embedding.weight.shape[0]
            or int(left.min()) < 0 or int(left.max()) >= self.embedding.weight.shape[0]
            or int(right.min()) < 0 or int(right.max()) >= self.embedding.weight.shape[0]
        ):
            raise ValueError("invalid direct reclaimed token or parent id")
        selected = set(int(value) for value in reclaimed_token_ids)
        if selected & set(int(value) for value in (*reclaimed_left_ids, *reclaimed_right_ids)):
            raise ValueError("direct reclaimed split map contains recursion")
        key_base = int(self.collapse.pair_key_base)
        endpoints = {
            endpoint for key in pair_keys
            for endpoint in (int(key) // key_base, int(key) % key_base)
        }
        if selected & endpoints:
            raise ValueError("a reclaimed token is an active exact-pair endpoint")

        report = self.collapse.promote_exact_vocabulary_to_reclaimed(
            pair_keys, reclaimed, self.embedding.weight, optimizer=optimizer
        )
        self.reclaimed_token_ids = [int(value) for value in reclaimed_token_ids]
        self.reclaimed_left_ids = [int(value) for value in reclaimed_left_ids]
        self.reclaimed_right_ids = [int(value) for value in reclaimed_right_ids]
        self.reclaim_left.fill_(-1)
        self.reclaim_right.fill_(-1)
        self.reclaim_left[reclaimed] = left
        self.reclaim_right[reclaimed] = right
        self.ngram_buckets = int(reclaimed.numel())
        self.mode = "collapse_reclaimed"
        self.conditioner.mode = "collapse_reclaimed"
        return report

    def compact_dynamic_vocabulary(self) -> dict[str, int]:
        if self.mode != "collapse_dynamic" or self.collapse is None:
            raise ValueError("dynamic vocabulary compaction requires collapse_dynamic mode")
        report = self.collapse.compact_exact_vocabulary()
        self.ngram_buckets = self.collapse.buckets
        return report

    def configure_recursive_vocabulary_discovery(self) -> None:
        if self.mode != "collapse_recursive" or self.recursive_cascade is None:
            raise ValueError("recursive discovery requires collapse_recursive mode")
        self.recursive_cascade.configure_discovery()

    def grow_recursive_vocabulary(
        self,
        pair_keys: tuple[int, ...] | list[int],
        *,
        optimizer: torch.optim.Optimizer,
        initialization: str = "parent_mean",
    ) -> dict[str, object]:
        if self.mode != "collapse_recursive" or self.recursive_cascade is None:
            raise ValueError("recursive growth requires collapse_recursive mode")
        return self.recursive_cascade.grow(
            pair_keys,
            base_embedding=self.embedding.weight,
            optimizer=optimizer,
            initialization=initialization,
        )

    def deactivate_unobserved_recursive_vocabulary(
        self, observed_token_ids: tuple[int, ...] | list[int]
    ) -> dict[str, object]:
        if self.mode != "collapse_recursive" or self.recursive_cascade is None:
            raise ValueError("recursive deactivation requires collapse_recursive mode")
        return self.recursive_cascade.deactivate_unobserved(observed_token_ids)

    def reactivate_recursive_vocabulary(
        self, pair_keys: tuple[int, ...] | list[int]
    ) -> dict[str, object]:
        if self.mode != "collapse_recursive" or self.recursive_cascade is None:
            raise ValueError("recursive reactivation requires collapse_recursive mode")
        return self.recursive_cascade.reactivate_pair_keys(pair_keys)

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        super().save(output_path, *args, safe_serialization=safe_serialization, **kwargs)
        Path(output_path, CONFIG_NAME).write_text(json.dumps({
            "mode": self.mode,
            "languages": self.languages,
            "code_dim": self.conditioner.code_dim,
            "n_senses": self.conditioner.n_senses,
            "rank": self.conditioner.rank,
            "split_index": self.split_index,
            "ngram_buckets": self.ngram_buckets,
            "growth_residual_buckets": self.growth_residual_buckets,
            "n_vocabs": self.n_vocabs,
            "weight_index": self.weight_index,
            "dynamic_pair_keys": (
                self.collapse.pair_keys.detach().cpu().tolist()
                if self.mode in ("collapse_dynamic", "collapse_reclaimed")
                and self.collapse is not None else None
            ),
            "dynamic_pair_slots": (
                self.collapse.pair_slots.detach().cpu().tolist()
                if self.mode in ("collapse_dynamic", "collapse_reclaimed")
                and self.collapse is not None else None
            ),
            "reclaimed_token_ids": self.reclaimed_token_ids,
            "reclaimed_left_ids": self.reclaimed_left_ids,
            "reclaimed_right_ids": self.reclaimed_right_ids,
            "online_rank": self.online_rank,
            "compiled_pair_keys": None,
            "compiled_pair_scores": None,
            "compiled_pair_count": (
                int(self.online_collapse.compiled_pair_keys.numel())
                if self.mode == "collapse_online_compiled"
                and self.online_collapse is not None else None
            ),
            "recursive_max_span_length": self.recursive_max_span_length,
            "recursive_learned_count": (
                self.recursive_cascade.learned_count
                if self.mode == "collapse_recursive"
                and self.recursive_cascade is not None else None
            ),
            "recursive_active_count": (
                self.recursive_cascade.active_count
                if self.mode == "collapse_recursive"
                and self.recursive_cascade is not None else None
            ),
            "ignored_token_ids": self.ignored_token_ids,
            "default_language": self.default_language,
        }, ensure_ascii=False, indent=2))

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        from safetensors.torch import load_file

        root_path = cls.load_dir_path(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if root_path is None:
            raise FileNotFoundError(
                f"could not resolve conditioned static model {model_name_or_path!r}"
            )
        root = Path(root_path)
        config = json.loads((root / CONFIG_NAME).read_text())
        tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        state = load_file(str(root / "model.safetensors"))
        weights = state["embedding.weight"]
        compiled_pair_keys = config.get("compiled_pair_keys")
        compiled_pair_scores = config.get("compiled_pair_scores")
        if config["mode"] == "collapse_online_compiled" and compiled_pair_keys is None:
            compiled_pair_keys = state["online_collapse.compiled_pair_keys"]
            compiled_pair_scores = state["online_collapse.compiled_pair_scores"]
            expected_count = config.get("compiled_pair_count")
            if expected_count is not None and int(expected_count) != compiled_pair_keys.numel():
                raise ValueError("compiled pair count disagrees with model state")
        recursive_restored_state = None
        if config["mode"] == "collapse_recursive":
            prefix = "recursive_cascade."
            recursive_names = {
                "learned", "rule_left", "rule_right", "rule_generation",
                "span_offsets", "span_values",
            }
            if f"{prefix}rule_active" in state:
                recursive_names.add("rule_active")
            recursive_restored_state = {
                name: state[f"{prefix}{name}"] for name in recursive_names
            }
            expected_count = config.get("recursive_learned_count")
            if (
                expected_count is not None
                and int(expected_count) != recursive_restored_state["learned"].shape[0]
            ):
                raise ValueError("recursive learned count disagrees with model state")
            expected_active = config.get("recursive_active_count")
            active = recursive_restored_state.get(
                "rule_active",
                torch.ones(
                    recursive_restored_state["learned"].shape[0], dtype=torch.bool
                ),
            )
            if expected_active is not None and int(expected_active) != int(active.sum()):
                raise ValueError("recursive active count disagrees with model state")
        module = cls(tokenizer, embedding_weights=weights,
                     languages=config["languages"], mode=config["mode"],
                     code_dim=config["code_dim"], n_senses=config["n_senses"],
                     rank=config["rank"], split_index=config.get("split_index"),
                     ngram_buckets=config.get("ngram_buckets", 0),
                     n_vocabs=config.get("n_vocabs", 4),
                     weight_index=config.get("weight_index"),
                     dynamic_pair_keys=config.get("dynamic_pair_keys"),
                     dynamic_pair_slots=config.get("dynamic_pair_slots"),
                     growth_residual_buckets=config.get("growth_residual_buckets", 0),
                     reclaimed_token_ids=config.get("reclaimed_token_ids"),
                     reclaimed_left_ids=config.get("reclaimed_left_ids"),
                     reclaimed_right_ids=config.get("reclaimed_right_ids"),
                     online_rank=config.get("online_rank", 32),
                     compiled_pair_keys=compiled_pair_keys,
                     compiled_pair_scores=compiled_pair_scores,
                     recursive_max_span_length=config.get(
                         "recursive_max_span_length", 32
                     ),
                     recursive_restored_state=recursive_restored_state,
                     ignored_token_ids=config.get("ignored_token_ids", ()),
                     default_language=config.get("default_language"))
        module.load_state_dict(state, strict=False)
        return module


def build(
    base_dir: str | os.PathLike,
    languages: list[str],
    mode: str,
    *,
    code_dim: int = 16,
    n_senses: int = 64,
    rank: int = 8,
    split_index: str | None = None,
    ngram_buckets: int = 0,
    n_vocabs: int = 4,
    weight_index: str | None = None,
    online_rank: int = 32,
    ignored_token_ids: list[int] | tuple[int, ...] | None = None,
    default_language: str | None = None,
) -> ConditionedStaticEmbedding:
    """Extend an unconditioned StaRSE checkpoint with language markers.

    The marker rows are appended to the table and initialised to zero, and every
    conditioning parameter is initialised so the conditioner is the identity, so
    the model starts numerically equal to the checkpoint it came from. Any later
    difference is attributable to the mechanism rather than to a different
    starting point.
    """
    from safetensors.torch import load_file

    base = Path(base_dir)
    tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
    weights = load_file(str(base / "model.safetensors"))["embedding.weight"]

    markers = [marker_for(language) for language in languages]
    added = tokenizer.add_special_tokens(markers)
    if added:
        extra = torch.zeros(added, weights.shape[1], dtype=weights.dtype)
        weights = torch.cat([weights, extra], dim=0)
    return ConditionedStaticEmbedding(
        tokenizer, embedding_weights=weights, languages=languages, mode=mode,
        code_dim=code_dim, n_senses=n_senses, rank=rank, split_index=split_index,
        ngram_buckets=ngram_buckets, n_vocabs=n_vocabs, weight_index=weight_index,
        online_rank=online_rank, ignored_token_ids=ignored_token_ids,
        default_language=default_language)
