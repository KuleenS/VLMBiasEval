from typing import List

from dataclasses import dataclass, field

@dataclass
class SafetyTunedLLaMaConfig:

    base_model: str

    data_path: str 

    output_dir: str 

    batch_size: int = field(default=128)

    micro_batch_size: int = field(default=4)
    num_epochs: int = field(default=3)
    learning_rate: float= field(default=3e-4)
    cutoff_len: int = field(default=256)
    val_set_size: int = field(default=200)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(default=["q_proj","v_proj"])

    train_on_inputs: bool = field(default=True)

    add_eos_token: bool = field(default=False)

    group_by_length: bool = field(default=False)

    wandb_project: str = field(default="")

    wandb_run_name: str = field(default="")

    wandb_watch: str = field(default="")

    wandb_log_model: str= field(default="")

    resume_from_checkpoint: str = field(default=None)

    template_path: str = field(default="configs/alpaca.json")