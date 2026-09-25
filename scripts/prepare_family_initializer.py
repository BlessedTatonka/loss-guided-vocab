#!/usr/bin/env python3
"""Build a static-table initializer from the input embeddings of a pretrained multilingual encoder.

The tokenizer-reachable rows of the base model's input embedding matrix are copied verbatim and
zero-initialized language-marker rows are appended. Two layouts are written: a plain table
("no-pairs" control) and a discovery table with empty pair proposals.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

from starse.conditioned_module import ConditionedStaticEmbedding, marker_for

FAMILIES = {
    "xlm-roberta-base": {
        "model_id": "FacebookAI/xlm-roberta-base",
        "revision": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        "weight_key": "roberta.embeddings.word_embeddings.weight",
    },
    "multilingual-e5-small": {
        "model_id": "intfloat/multilingual-e5-small",
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "weight_key": "embeddings.word_embeddings.weight",
    },
    "multilingual-e5-base": {
        "model_id": "intfloat/multilingual-e5-base",
        "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
        "weight_key": "embeddings.word_embeddings.weight",
    },
    "labse": {
        "model_id": "sentence-transformers/LaBSE",
        "revision": "836121a0533e5664b21c7aacc5d22951f2b8b25b",
        "weight_key": "embeddings.word_embeddings.weight",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--languages", type=Path, default=Path("configs/languages.txt"))
    parser.add_argument("--ngram-buckets", type=int, default=262_144)
    args = parser.parse_args()

    spec = FAMILIES[args.family]
    languages = [line.strip() for line in args.languages.read_text(encoding="utf-8").splitlines() if line.strip()]
    source = Path(snapshot_download(spec["model_id"], revision=spec["revision"],
                                    allow_patterns=["*.json", "model.safetensors", "*.model", "vocab.txt"]))
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    rows = tokenizer.get_vocab_size(with_added_tokens=True)
    with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as tensors:
        weights = tensors.get_tensor(spec["weight_key"])[:rows].detach().cpu().to(torch.float32).clone()
    added = tokenizer.add_special_tokens([marker_for(language) for language in languages])
    if added != len(languages):
        raise ValueError("language marker collision with the base vocabulary")
    weights = torch.cat([weights, torch.zeros(added, weights.shape[1], dtype=weights.dtype)], dim=0)

    control = args.output_dir / "no-pairs"
    discovery = args.output_dir / "discovery"
    for path in (control, discovery):
        if path.exists():
            raise FileExistsError(path)
    module = ConditionedStaticEmbedding(tokenizer, embedding_weights=weights, languages=languages, mode="none")
    SentenceTransformer(modules=[module], device="cpu").save(str(control))
    del module
    gc.collect()
    module = ConditionedStaticEmbedding(tokenizer, embedding_weights=weights, languages=languages,
                                        mode="collapse_dynamic", ngram_buckets=args.ngram_buckets)
    SentenceTransformer(modules=[module], device="cpu").save(str(discovery))
    manifest = {
        "family": args.family, "model_id": spec["model_id"], "revision": spec["revision"],
        "transferred_rows": rows, "language_marker_rows": added, "embedding_dim": int(weights.shape[1]),
        "no_pairs": str(control), "discovery": str(discovery),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
