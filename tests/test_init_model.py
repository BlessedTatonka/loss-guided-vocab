from types import SimpleNamespace

import numpy as np
import torch

from starse.init_model import _select_contextual_representation, build_initial_matrix, parse_args


def test_initial_matrix_shape_and_dtype() -> None:
    rng = np.random.default_rng(42)
    source = rng.normal(size=(128, 32)).astype(np.float32)
    matrix, metadata = build_initial_matrix(source, target_dim=16)
    assert matrix.shape == (128, 16)
    assert matrix.dtype == np.float32
    assert metadata["leading_components_removed"] == 1


def test_multiple_base_models_can_be_concatenated() -> None:
    args = parse_args(
        [
            "--base-model",
            "jhu-clsp/mmBERT-base",
            "--base-model",
            "jhu-clsp/mmBERT-small",
            "--base-revision",
            "base-commit",
            "--base-revision",
            "small-commit",
            "--target-dim",
            "1024",
            "--output-dir",
            "output",
        ]
    )

    assert args.base_models == ["jhu-clsp/mmBERT-base", "jhu-clsp/mmBERT-small"]
    assert args.base_revisions == ["base-commit", "small-commit"]
    assert args.target_dim == 1024


def test_multiple_representations_can_be_fused() -> None:
    args = parse_args(
        [
            "--base-model",
            "model-30m",
            "--base-model",
            "model-70m",
            "--representation",
            "input",
            "--representation",
            "swe_mean",
            "--output-dir",
            "out",
        ]
    )
    assert args.representations == ["input", "swe_mean"]


def test_contextual_representation_selection() -> None:
    hidden_states = tuple(torch.full((2, 3, 4), float(index)) for index in range(11))
    outputs = SimpleNamespace(last_hidden_state=hidden_states[-1], hidden_states=hidden_states)
    assert torch.all(_select_contextual_representation(outputs, "swe_mean") == 10)
    assert torch.all(_select_contextual_representation(outputs, "hidden_middle_token") == 5)
    assert torch.all(_select_contextual_representation(outputs, "hidden_last_token") == 10)
    assert torch.all(_select_contextual_representation(outputs, "hidden_last4_mean_token") == 8.5)
