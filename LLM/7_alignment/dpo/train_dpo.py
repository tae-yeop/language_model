import os
import torch
import transformers
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, MllamaForConditionalGeneration

# triton kernel을 사용해서 메모리 사용을 줄임
from liger_kernel.transformers import apply_liger_kernel_to_mllama
from arguments import DataArguments, ModelArguments, TrainingArguments


def set_requires_grad(parameters, requires_grad):
    for param in parameters:
        param.requires_grad = requires_grad

if __name__ == '__main__':
    parser = transformers.HfArgumentParser(
        (DataArguments, ModelArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.use_liger:
        apply_liger_kernel_to_mllama()

    
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16 if training_args.fp16 else (
            torch.bfloat16 if training_args.bf16 else torch.float32
        )
    )

    # BitsAndBytesConfig 설정
    # 가중치를 8비트 혹은 4비트로 모델을 양자화
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"":training_args.device},  # 빈문자열 key는 모듈 전체를 의미
                quantization_config=BitsAndBytesConfig(
                    # 모든 Linear를 bnb.nn.Linear4bit or bnb.nn.Linear8bitLt로 교체
                    load_in_4bit=training_args.bits==4, # 4비트 양자화할지
                    load_in_8bit=training_args.bits==8, # 8비트 양자화할지
                    llm_int8_skip_modules=["multi_modal_projector", "vision_model"], # 양자화하지 않을 부분, 양자화가 불안정한 부분은 스킵
                    llm_int8_threshold=6.0, # 이 임계값보다 큰 activation을 fp16으로 처리
                    llm_int8_has_fp16_weight=False, # 8-bit 로드 시 fp16 본가중치를 함께 보관, True면 역전파 시 가중치 재변환이 필요 없어 LoRA·미세조정이 편해짐
                    bnb_4bit_compute_dtype=compute_dtype, # 연산(곱셈) 시 사용할 dtype (저장은 4-bit)
                    bnb_4bit_use_double_quant=training_args.double_quant, # Double Quantization(중첩 양자화) 사용 여부. 4-bit 가중치 자체를 한 번 더 양자화해 메모리 추가 절감 & 일부 정확도 개선.
                    bnb_4bit_quant_type=training_args.quant_type, #4-bit 데이터 타입. "nf4"
                ),


            )
        )


    model = MllamaForConditionalGeneration.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation="sdpa",
        **bnb_model_from_pretrained_args
    )

    model_to_configure = model
    
    llm_params = model.language_model.parameters()
    # Freeze True해버리면 requires_grad가 False로 설정
    set_requires_grad(llm_params, not training_args.freeze_llm)

    vision_tower = model.vision_model
    # 비전 모델은 왜 따로 dtype, device를 설정하지? 위에서 한거 아닌가?
    vision_tower.to(dtype=compute_dtype, device=training_args.device)

    # Projector requires_grad 설정
    img_projection_params = model.multi_modal_projector.parameters()
    set_requires_grad(img_projection_params, not training_args.freeze_img_projector)

    # 비전 타워 requires_grad 설정
    vision_model_params = vision_tower.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    if training_args.bits in [4, 8]:
        model.multi_model_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.hidden_size = model.config.text_config.hidden_size
    model.config.text_config.use_cache = False
    model.config.use_cache = False
    


