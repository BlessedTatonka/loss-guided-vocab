# Loss-Guided Vocabulary Discovery for Multilingual Static Encoders

A static sentence encoder is an embedding table with mean pooling. Its vocabulary is normally fixed by the tokenizer before training. Loss-guided vocabulary discovery lets the sentence-level contrastive objective extend it: during a short discovery stage, every adjacent token pair is scored by a first-order estimate of how much the loss would rise without it, pairs above a threshold receive their own embedding rows, and training continues with the extended table. The tokenizer never changes and inference stays a table lookup.

## Results

Cross-lingual macro F1 (%) at checkpoint 2,000, mmBERT base table, mean over three seeds:

| Setting | Added pairs | Tatoeba | BUCC | Flores |
|---|---|---|---|---|
| No pairs | 0 | 50.33 | 97.56 | 56.59 |
| Frequency selector | 56,071 | 50.54 | 97.61 | 57.19 |
| Loss-guided | 56,044 | **52.24** | **97.73** | **57.81** |

The full 176-language model (20,000 continuation updates) reaches 50.13 on MTEB(Multilingual, v2).

## Installation

Python 3.11. Dependencies are pinned in `pyproject.toml`:

```bash
uv sync --extra dev --extra selection
uv run pytest
```

## Reproducing the experiments

Raw training data is not distributed. Copy `configs/data-roots.example.json` to `configs/data-roots.local.json` and point every entry to a local Parquet root (the 22 sources are named in the configuration files).

**1. Initialization.** Build the mmBERT static table (concatenated mmBERT-base and mmBERT-small input embeddings, PCA to 1,024 dimensions, Zipf–SIF weighting, 1,914 language-marker rows):

```bash
uv run python scripts/prepare_initialization.py --output-dir outputs/init-mmbert --device cuda:0
```

**2. Discovery (1,000 updates).** Adjacent pairs are scored and promoted at the end of the run:

```bash
uv run python scripts/train.py --config configs/mmbert/discovery.yaml \
  --roots configs/data-roots.local.json --model outputs/init-mmbert \
  --output-dir outputs/mmbert-discovery --no-comet
```

**3. Continuation.** Continue training with the extended table (20,000 updates for the full model; use `--max-steps 2000` for the comparison above):

```bash
uv run python scripts/train.py --config configs/mmbert/continuation.yaml \
  --roots configs/data-roots.local.json --model outputs/mmbert-discovery \
  --output-dir outputs/mmbert-continuation --no-comet
```

**4. Controls.** The no-pair control uses `configs/mmbert/no-pairs-prefill.yaml` then `configs/mmbert/no-pairs-continuation.yaml`. The frequency control collects pair counts with `configs/mmbert/frequency-collector.yaml`, selects the most frequent pairs, materializes them as rows, and continues with `configs/mmbert/frequency-continuation.yaml`:

```bash
uv run python scripts/select_frequency_pairs.py \
  --candidates outputs/mmbert-frequency/dynamic_vocabulary_candidates.csv --k 56071 \
  --output outputs/mmbert-frequency/selection.csv
uv run python scripts/materialize_frequency_control.py \
  --discovery-model outputs/mmbert-frequency --selection outputs/mmbert-frequency/selection.csv \
  --output outputs/mmbert-frequency-promoted
```

**5. Evaluation.** Tatoeba, BUCC, and Flores (106 / 4 / 193 language-pair subsets) with MTEB:

```bash
uv run python scripts/evaluate_crosslingual.py --model outputs/mmbert-continuation/checkpoint-2000 \
  --suite configs/eval/crosslingual.json --output results/mmbert-loss-guided --device cpu --batch-size 512
uv run python scripts/paired_bootstrap.py results/mmbert-loss-guided results/mmbert-no-pairs
```

The full MTEB(Multilingual, v2) run uses `scripts/run_mteb.py`.

**Other base-model families.** `configs/families/<family>/` holds the discovery, continuation, and no-pair configurations for XLM-R base, multilingual-E5 small and base, and LaBSE (batch 2,048, matched update budgets). Their initializers come from the base model's input embeddings:

```bash
uv run python scripts/prepare_family_initializer.py --family xlm-roberta-base --output-dir outputs/init-xlm-roberta-base
```

## Layout

- `src/starse/` — static encoder module, pair discovery and promotion, streaming data pipeline, trainer.
- `configs/` — training configurations, evaluation suite, language inventory.
- `scripts/` — initialization, training, controls, evaluation, statistics.
- `tests/` — unit tests for the loss, discovery table, initializer, streaming, and evaluation code.

## License

Apache 2.0
