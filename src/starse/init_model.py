"""Create a StaRSE static-table initialization from one or more encoders."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from tokenizers import Tokenizer as BackendTokenizer
from transformers import AutoModel, AutoTokenizer, LlamaTokenizerFast, PreTrainedTokenizerFast

REPRESENTATION_CHOICES = (
    "input",
    "swe_mean",
    "hidden_middle_token",
    "hidden_last_token",
    "hidden_last4_mean_token",
)


def build_initial_matrix(
    matrix: np.ndarray,
    *,
    target_dim: int = 512,
    sif_a: float = 5e-5,
    zipf_exponent: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the PCA + SIF/Zipf recipe used for the released model."""

    vocab_size, hidden_dim = matrix.shape
    if target_dim > hidden_dim:
        raise ValueError(f"target_dim={target_dim} exceeds source dimension {hidden_dim}")

    components_to_remove = math.ceil(target_dim / 100)
    n_components = min(hidden_dim, target_dim + components_to_remove)
    components_to_remove = max(0, n_components - target_dim)
    # These token tables are extremely tall. The covariance solver is exact
    # and avoids sklearn's width-based switch to a much slower randomized SVD
    # when several aligned representation tables are concatenated.
    pca_solver = "covariance_eigh" if vocab_size >= 10 * hidden_dim else "auto"
    pca = PCA(n_components=n_components, svd_solver=pca_solver, whiten=False, random_state=42)
    reduced = pca.fit_transform(matrix)
    if components_to_remove:
        reduced = reduced[:, components_to_remove:]
    reduced = reduced[:, :target_dim]

    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    zipf = 1.0 / np.power(ranks, zipf_exponent)
    probabilities = zipf / zipf.sum()
    weights = sif_a / (sif_a + probabilities)
    initialized = np.ascontiguousarray((reduced * weights[:, None]).astype(np.float32))

    metadata = {
        "source": "AutoModel.get_input_embeddings().weight",
        "source_shape": [int(vocab_size), int(hidden_dim)],
        "final_shape": list(initialized.shape),
        "pca_components": int(n_components),
        "pca_solver": pca_solver,
        "leading_components_removed": int(components_to_remove),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "sif_a": float(sif_a),
        "zipf_exponent": float(zipf_exponent),
    }
    return initialized, metadata


def load_static_tokenizer(
    model_name: str,
    *,
    trust_remote_code: bool,
    revision: str | None = None,
) -> Any:
    """Load the Rust-backed tokenizer required by StaticEmbedding."""

    auto_error: Exception | None = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        if getattr(tokenizer, "is_fast", False):
            return tokenizer
    except Exception as exc:
        auto_error = exc

    try:
        backend = BackendTokenizer.from_pretrained(model_name, revision=revision or "main")
        return PreTrainedTokenizerFast(tokenizer_object=backend)
    except Exception:
        try:
            return LlamaTokenizerFast.from_pretrained(model_name, revision=revision)
        except Exception as fallback_error:
            detail = f" AutoTokenizer failed with: {auto_error}" if auto_error is not None else ""
            raise TypeError(
                f"{model_name!r} has no usable Rust-backed tokenizer.{detail} "
                f"Fast tokenizer fallback failed with: {fallback_error}"
            ) from fallback_error


def _special_token_id(tokenizer: Any, model: Any, name: str, fallback_token: str) -> int:
    token_id = getattr(tokenizer, f"{name}_token_id", None)
    if token_id is None:
        token_id = getattr(model.config, f"{name}_token_id", None)
    if token_id is None:
        token_id = tokenizer.convert_tokens_to_ids(fallback_token)
    if token_id is None or int(token_id) < 0:
        raise ValueError(f"Could not resolve {name} token ID for {model.name_or_path!r}")
    return int(token_id)


def _select_contextual_representation(outputs: Any, representation: str) -> torch.Tensor:
    """Select a Model2Vec-style or layer-wise one-token representation."""

    if representation == "swe_mean":
        return outputs.last_hidden_state.mean(dim=1)
    if representation == "hidden_last_token":
        return outputs.last_hidden_state[:, 1, :]

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError(f"{representation!r} requires output_hidden_states=True")
    if representation == "hidden_middle_token":
        # hidden_states[0] is the embedding output; all later entries are
        # completed Transformer layers. Select the central completed layer.
        middle_index = max(1, len(hidden_states) // 2)
        return hidden_states[middle_index][:, 1, :]
    if representation == "hidden_last4_mean_token":
        return torch.stack(hidden_states[-4:], dim=0).mean(dim=0)[:, 1, :]
    raise ValueError(f"Unsupported contextual representation: {representation!r}")


def extract_representation_tables(
    *,
    model: Any,
    tokenizer: Any,
    representations: Sequence[str],
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Extract aligned input, SWE/output, or hidden-state token tables.

    ``swe_mean`` matches Model2Vec output distillation before PCA/SIF: each
    vocabulary item is forwarded as ``[BOS, token, EOS]`` and the final hidden
    states are mean pooled. Hidden variants retain only the contextualized
    vocabulary-token position.
    """

    requested = tuple(dict.fromkeys(representations))
    unknown = sorted(set(requested) - set(REPRESENTATION_CHOICES))
    if unknown:
        raise ValueError(f"Unknown representations: {unknown}")
    if batch_size <= 0:
        raise ValueError("contextual batch size must be positive")

    input_table = model.get_input_embeddings().weight.detach().to(dtype=torch.float32, device="cpu").numpy()
    vocab_size, hidden_dim = input_table.shape
    tables: dict[str, np.ndarray] = {}
    if "input" in requested:
        tables["input"] = np.ascontiguousarray(input_table)

    contextual = [name for name in requested if name != "input"]
    if not contextual:
        return tables

    bos_id = _special_token_id(tokenizer, model, "bos", "<s>")
    eos_id = _special_token_id(tokenizer, model, "eos", "</s>")
    arrays = {name: np.empty((vocab_size, hidden_dim), dtype=np.float32) for name in contextual}
    needs_hidden_states = any(
        name in {"hidden_middle_token", "hidden_last4_mean_token"} for name in contextual
    )

    model = model.eval().to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, vocab_size, batch_size):
            stop = min(vocab_size, start + batch_size)
            token_ids = torch.arange(start, stop, dtype=torch.long, device=device)
            input_ids = torch.stack(
                (
                    torch.full_like(token_ids, bos_id),
                    token_ids,
                    torch.full_like(token_ids, eos_id),
                ),
                dim=1,
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_hidden_states=needs_hidden_states,
                return_dict=True,
            )
            for name in contextual:
                selected = _select_contextual_representation(outputs, name)
                arrays[name][start:stop] = selected.detach().to(dtype=torch.float32, device="cpu").numpy()

    tables.update({name: np.ascontiguousarray(array) for name, array in arrays.items()})
    model.to("cpu")
    return tables


def initialize_model(
    *,
    base_models: Sequence[str],
    base_revisions: Sequence[str | None] | None,
    output_dir: Path,
    target_dim: int,
    sif_a: float,
    zipf_exponent: float,
    trust_remote_code: bool,
    representations: Sequence[str] = ("input",),
    device: str = "cpu",
    contextual_batch_size: int = 2048,
) -> None:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding

    if not base_models:
        raise ValueError("At least one base model is required")

    revisions = list(base_revisions or [None] * len(base_models))
    if len(revisions) != len(base_models):
        raise ValueError("--base-revision must be repeated once per --base-model")
    tokenizer = load_static_tokenizer(
        base_models[0],
        revision=revisions[0],
        trust_remote_code=trust_remote_code,
    )
    reference_vocab = tokenizer.get_vocab()
    reference_special_ids = tuple(tokenizer.all_special_ids)
    source_tables: list[np.ndarray] = []
    source_components: list[dict[str, Any]] = []
    for model_name, revision in zip(base_models, revisions, strict=True):
        candidate_tokenizer = load_static_tokenizer(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        if candidate_tokenizer.get_vocab() != reference_vocab:
            raise ValueError(f"Tokenizer vocabulary or token IDs differ for {model_name!r}")
        if tuple(candidate_tokenizer.all_special_ids) != reference_special_ids:
            raise ValueError(f"Special token IDs differ for {model_name!r}")

        model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        tables = extract_representation_tables(
            model=model,
            tokenizer=candidate_tokenizer,
            representations=representations,
            device=device,
            batch_size=contextual_batch_size,
        )
        for representation in representations:
            table = tables[representation]
            if table.shape[0] != len(reference_vocab):
                raise ValueError(
                    f"Embedding rows for {model_name!r}/{representation} ({table.shape[0]}) "
                    f"do not match tokenizer size ({len(reference_vocab)})"
                )
            source_tables.append(table)
            source_components.append(
                {
                    "model": model_name,
                    "revision": revision,
                    "representation": representation,
                    "shape": [int(value) for value in table.shape],
                }
            )
        del model

    source = np.ascontiguousarray(np.concatenate(source_tables, axis=1))
    matrix, metadata = build_initial_matrix(
        source,
        target_dim=target_dim,
        sif_a=sif_a,
        zipf_exponent=zipf_exponent,
    )
    concatenated = len(source_tables) > 1
    del source_tables, source

    static_model = SentenceTransformer(modules=[StaticEmbedding(tokenizer=tokenizer, embedding_weights=matrix)])
    output_dir.mkdir(parents=True, exist_ok=True)
    static_model.save_pretrained(str(output_dir), create_model_card=False)
    metadata.update(
        {
            "base_models": list(base_models),
            "base_revisions": revisions,
            "representations": list(representations),
            "source_components": source_components,
            "source": "aligned token representations concatenated before PCA",
            "concatenated": concatenated,
            "target_dim": target_dim,
            "contextual_inference_dtype": (
                "float32" if any(representation != "input" for representation in representations) else None
            ),
        }
    )
    (output_dir / "initialization.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        dest="base_models",
        action="append",
        help="Encoder supplying tokenizer and input embeddings; repeat to concatenate aligned token tables",
    )
    parser.add_argument(
        "--base-revision",
        dest="base_revisions",
        action="append",
        help="Pinned Hub commit for the matching --base-model; repeat in the same order",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-dim", type=int, default=512)
    parser.add_argument("--sif-a", type=float, default=5e-5)
    parser.add_argument("--zipf-exponent", type=float, default=1.0)
    parser.add_argument(
        "--representation",
        dest="representations",
        action="append",
        choices=REPRESENTATION_CHOICES,
        help="Aligned table to extract from every base model; repeat to fuse representations (default: input)",
    )
    parser.add_argument("--device", default="cpu", help="Device used only while extracting contextual tables")
    parser.add_argument("--contextual-batch-size", type=int, default=2048)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    initialize_model(
        base_models=args.base_models or ["ai-forever/ruBert-base"],
        base_revisions=args.base_revisions,
        output_dir=args.output_dir.resolve(),
        target_dim=args.target_dim,
        sif_a=args.sif_a,
        zipf_exponent=args.zipf_exponent,
        trust_remote_code=args.trust_remote_code,
        representations=args.representations or ["input"],
        device=args.device,
        contextual_batch_size=args.contextual_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
