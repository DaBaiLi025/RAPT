from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class ModelConfig:
    veg_hidden: int = 128
    prod_hidden: int = 64
    cli_hidden: int = 96
    num_layers: int = 1
    aux_hidden: int = 16
    fusion_hidden: int = 384
    head_hidden: int = 64
    dropout: float = 0.27551200549764077

    use_unit_embedding: bool = True
    unit_embedding_dim: int = 32
    shared_head: bool = False

    use_modality_gating: bool = False
    gating_dim: int = 96

    loss_name: str = "uw_mse"  # tb_mse | tb_huber | uw_mse | uw_huber
    huber_delta: float = 1.0

    balanced_task_sampler: bool = True
    samples_per_task: int = 8

    optimizer_name: str = "adamw"
    lr: float = 1.49899410035487e-4
    weight_decay: float = 4.211582075404216e-5
    grad_clip_norm: float = 2.0
    scheduler_factor: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass
class TrainConfig:
    max_epochs: int = 350
    patience: int = 40
    scheduler_patience: int = 12
    min_lr: float = 1e-6
    num_workers: int = 0
    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
