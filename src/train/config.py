"""One TrainConfig per run. Load from a JSON file in configs/ and override from the command line:

    python scripts/train.py --config configs/e3_stage2_lora.json --seed 1 --output_dir outputs/e3_s1
"""

import json
from dataclasses import asdict, dataclass, fields
from typing import Optional


@dataclass
class TrainConfig:
    # --- run ---
    name: str = "run"
    output_dir: str = "outputs/run"      # on Vertex this can be a /gcs/<bucket>/... path
    seed: int = 0

    # --- data (see src/data/loader.py) ---
    stage: int = 1                       # 1 = alignment (captions), 2 = instruction tuning
    split: str = "pilot"                 # "overfit" (100) | "pilot" (10K) | "full"
    subset_size: Optional[int] = None    # random N examples, fixed by seed (e.g. 200000 for the Stage 1 fallback)
    data_root: str = "data"
    text_only: bool = False              # E0: no images, no image tokens
    num_workers: int = 4

    # --- model ---
    pool_grid: int = 13                  # 13 -> 169 visual tokens (9 -> 81, 27 -> 729)
    lora: bool = False                   # False = E1/E2 (projector only); True = E3, E0, rank ablation
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    init_projector: Optional[str] = None  # Stage 1 projector.pt to start Stage 2 from
    gradient_checkpointing: bool = False
    dtype: str = "bfloat16"              # "float32" on CPU
    device: str = "auto"                 # "auto" = cuda if available, else cpu

    # --- optimisation ---
    epochs: int = 1
    max_steps: Optional[int] = None      # overrides epochs (overfit test, pilot)
    micro_batch_size: int = 16
    grad_accum: int = 8                  # effective batch = micro_batch_size * grad_accum
    lr_projector: float = 1e-3
    lr_lora: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # --- logging and checkpoints ---
    log_every: int = 10
    save_every_steps: int = 200
    save_every_minutes: float = 15.0
    keep_last: int = 2
    gpu_hourly_usd: float = 4.43         # for the cost estimate in summary.json

    # --- overfit test: after training, compare loss with correct vs shuffled images ---
    image_shuffle_check: bool = False

    def effective_batch(self):
        return self.micro_batch_size * self.grad_accum

    @classmethod
    def from_json(cls, path, **overrides):
        data = json.load(open(path))
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)
