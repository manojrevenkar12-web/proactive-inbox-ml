"""Export the PyTorch classifier to ONNX and quantise it to int8 for serving."""

from __future__ import annotations

from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from .model import NgramClassifier


class _Probs(torch.nn.Module):
    """Serving graph: ids -> logits. Temperature and threshold stay outside so they can be re-tuned without re-export."""

    def __init__(self, m: NgramClassifier):
        super().__init__()
        self.m = m

    def forward(self, ids):
        return self.m(ids)


def to_onnx(model: NgramClassifier, path: Path) -> Path:
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.ones((2, 16), dtype=torch.int64)
    torch.onnx.export(
        _Probs(model),
        (example,),
        str(path),
        input_names=["ids"],
        output_names=["logits"],
        dynamic_axes={"ids": {0: "batch", 1: "length"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path


def quantise(src: Path, dst: Path) -> Path:
    # Embedding table (Gather) and dense layers (MatMul) to int8; activations stay float.
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])
    return dst
