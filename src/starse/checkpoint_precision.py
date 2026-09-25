"""Atomic BF16 model persistence and model/resume-state retention policy."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import TrainerCallback

_CHECKPOINT = re.compile(r"checkpoint-(\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rewrite_safetensors_bf16(path: Path) -> dict[str, object]:
    """Replace one safetensors file only after a complete BF16 verification."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before_sha256 = _sha256(path)
    tensors = load_file(str(path), device="cpu")
    inventory: dict[str, dict[str, object]] = {}
    converted: dict[str, torch.Tensor] = {}
    for name, tensor in sorted(tensors.items()):
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"nonfinite checkpoint tensor: {name}")
        target = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
        converted[name] = target.contiguous()
        inventory[name] = {
            "shape": list(tensor.shape),
            "source_dtype": str(tensor.dtype).removeprefix("torch."),
            "stored_dtype": str(target.dtype).removeprefix("torch."),
        }

    temporary = path.with_suffix(path.suffix + ".bf16.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        save_file(converted, str(temporary))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        verified = load_file(str(temporary), device="cpu")
        if set(verified) != set(converted):
            raise RuntimeError("BF16 checkpoint tensor inventory changed")
        for name, tensor in verified.items():
            source = tensors[name]
            if tensor.shape != source.shape:
                raise RuntimeError(f"BF16 checkpoint shape changed: {name}")
            expected_dtype = torch.bfloat16 if source.is_floating_point() else source.dtype
            if tensor.dtype != expected_dtype:
                raise RuntimeError(f"BF16 checkpoint dtype mismatch: {name}")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"nonfinite converted checkpoint tensor: {name}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "protocol": "bf16-model-safetensors-v1",
        "path": str(path),
        "before_sha256": before_sha256,
        "after_sha256": _sha256(path),
        "stored_floating_dtype": "bfloat16",
        "tensor_count": len(inventory),
        "tensors": inventory,
    }


class BFloat16CheckpointCallback(TrainerCallback):
    """Quantize every model snapshot and retain resume files only on the newest."""

    def __init__(self, output_dir: Path, *, keep_resume_checkpoints: int = 1) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        if keep_resume_checkpoints <= 0:
            raise ValueError("keep_resume_checkpoints must be positive")
        self.keep_resume_checkpoints = int(keep_resume_checkpoints)
        self.completed_saves = 0

    @staticmethod
    def _step(path: Path) -> int:
        match = _CHECKPOINT.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid Trainer checkpoint name: {path.name}")
        return int(match.group(1))

    def _write_manifest(self, checkpoint: Path, report: dict[str, object]) -> None:
        destination = checkpoint / "checkpoint_precision.json"
        temporary = checkpoint / "checkpoint_precision.json.tmp"
        if destination.exists() or temporary.exists():
            raise FileExistsError(destination)
        payload: dict[str, Any] = {
            "protocol": "bf16-checkpoint-precision-v1",
            "checkpoint_step": self._step(checkpoint),
            "stored_floating_dtype": "bfloat16",
            "model_sha256": report["after_sha256"],
            "rewrite": report,
        }
        data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(checkpoint)

    @staticmethod
    def _remove_resume_files(checkpoint: Path) -> None:
        targets = {
            checkpoint / "optimizer.pt",
            checkpoint / "scheduler.pt",
            checkpoint / "scaler.pt",
            checkpoint / "rng_state.pth",
            *checkpoint.glob("rng_state_*.pth"),
        }
        for target in sorted(targets):
            if target.is_file():
                target.unlink()
        _fsync_directory(checkpoint)

    def on_save(self, args, state, control, **kwargs):
        del kwargs
        if Path(args.output_dir).expanduser().resolve() != self.output_dir:
            raise ValueError("BF16 callback output directory differs from Trainer")
        checkpoint = self.output_dir / f"checkpoint-{int(state.global_step)}"
        model_path = checkpoint / "model.safetensors"
        report = rewrite_safetensors_bf16(model_path)
        self._write_manifest(checkpoint, report)

        verified = sorted(
            (
                path for path in self.output_dir.glob("checkpoint-*")
                if path.is_dir() and (path / "checkpoint_precision.json").is_file()
            ),
            key=self._step,
        )
        retained = set(verified[-self.keep_resume_checkpoints :])
        for older in verified:
            if older not in retained:
                self._remove_resume_files(older)
        self.completed_saves += 1
        return control
