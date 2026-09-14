from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


class FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True

    def requires_grad_(self, value: bool):
        self.requires_grad = value
        return self


class FakeModule:
    def __init__(self, count: int) -> None:
        self.values = [FakeParameter() for _ in range(count)]

    def parameters(self):
        return iter(self.values)


class FakeBackbone:
    def __init__(self) -> None:
        self.blocks = [FakeModule(2) for _ in range(6)]
        self.norm = FakeModule(1)
        self.other = FakeModule(3)

    def parameters(self):
        for module in [*self.blocks, self.norm, self.other]:
            yield from module.parameters()


def load_finetune_module():
    path = Path(__file__).parents[1] / "scripts" / "run_labram_finetune_v13.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("run_labram_finetune_v13", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_trainable_tail_freezes_non_tail_parameters() -> None:
    module = load_finetune_module()
    backbone = FakeBackbone()

    module.select_trainable_tail(backbone, 4)

    assert all(not p.requires_grad for block in backbone.blocks[:2] for p in block.values)
    assert all(p.requires_grad for block in backbone.blocks[2:] for p in block.values)
    assert all(p.requires_grad for p in backbone.norm.values)
    assert all(not p.requires_grad for p in backbone.other.values)


@pytest.mark.parametrize("count", [0, 7])
def test_select_trainable_tail_rejects_invalid_block_count(count: int) -> None:
    module = load_finetune_module()
    with pytest.raises(ValueError, match="final_blocks"):
        module.select_trainable_tail(FakeBackbone(), count)
