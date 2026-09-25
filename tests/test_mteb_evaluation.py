from __future__ import annotations

import pytest

from starse.mteb_evaluation import PeriodicMTEBSettings, task_main_score


def test_task_main_score_prefers_test_split() -> None:
    result = {
        "scores": {
            "validation": [{"main_score": 0.2}],
            "test": [{"main_score": 0.7}],
        }
    }
    assert task_main_score(result) == pytest.approx(0.7)


def test_periodic_eval_must_align_with_reloadable_checkpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="multiple of save_steps"):
        PeriodicMTEBSettings.from_config(
            {"task": "RuBQRetrieval", "eval_steps": 1500},
            training_settings={"save_strategy": "steps", "save_steps": 1000},
            output_dir=tmp_path,
        )


def test_periodic_eval_resolves_output_and_defaults(tmp_path) -> None:
    settings = PeriodicMTEBSettings.from_config(
        {"task": "RuBQRetrieval", "eval_steps": 1000},
        training_settings={"save_strategy": "steps", "save_steps": 1000},
        output_dir=tmp_path,
    )
    assert settings.evaluate_at_start is True
    assert settings.batch_size == 1024
    assert settings.output_dir == tmp_path / "evaluation" / "rubq_retrieval"
