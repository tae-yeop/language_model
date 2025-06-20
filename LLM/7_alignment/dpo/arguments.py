# https://github.dev/2U1/Llama3.2-Vision-Finetune/blob/master/src/train/train_sft.py
from dataclass import dataclass, field
from typing import Optional
from transformers import TrainingArguments as HFTrainingArguments
from trl import DPOConfig as DPOConfigTRL

@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="meta-llama/Llama-3.2-11B-Vision-Instruct")

@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_img_projector: bool = field(default=True)
    max_seq_length: int = field(
        default=131072,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    projector_lr: Optional[float] = None
    vision_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger: bool = True


@dataclass
class DPOArguments(DPOConfigTRL):
    # Directory to download & cache the model/tokenizer
    cache_dir: Optional[str] = field(default=None)
    # Optimizer backend; 'adamw_torch' = fused AdamW in PyTorch
    optim: str = field(default="adamw_torch")
    # β₁ in AdamW
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)
    # If True, stop back-prop into the vision encoder
    freeze_vision_tower: bool = field(default=False)
    # If True, freeze all LLM transformer layers
    freeze_llm: bool = field(default=False)
    # If True, freeze the linear img->text projector (saves VRAM)
    freeze_img_projector: bool = field(default=True)
    # # 128 k + 약간 여유
    max_seq_length: int = field(
        default=131072, # This is the default max_length for phi3-vision-128k-instruct
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    # -------------------------------------------------
    # 4-bit/NF4 양자화 설정
    # -------------------------------------------------
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    # -------------------------------------------------
    # LoRA / DoRA / Liger 어댑터 관련
    # -------------------------------------------------
    # 전체 LoRA on/off
    lora_enable: bool = False 
    # Vision tower에도 LoRA 삽입?
    vision_lora: bool = False
    # DoRA(Δ-rank) 사용 여부
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    # 'none'/'all'/'lora_only'
    lora_bias: str = "none"
    projector_lr: Optional[float] = None
    vision_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    use_liger: bool = True
    # -1=auto; else top-k modules only
    num_lora_modules: int = -1
    beta: float = field(
        default=0.1,
        metadata={"help": "The beta value for DPO."}
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={"help": "Whether to precompute the reference log probabilities."}
    )
    dpo_loss:str = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."}
    )



@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    max_num_frames: int = 10