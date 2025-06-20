
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import wandb 

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.53.0.dev0")

# require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# 내가 직접 정의해서 trasnformers의 Argument 클래스와 합칠 수 있음
@dataclass
class CustomArguments:
    # 데이터 관련
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None # subset으로 나눠진 경우 해당 이름
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    streaming: bool = False
    block_size: Optional[int] = None # 토큰화 이후 인풋 시퀀스 길이
    overwrite_cache: bool = False
    validation_split_percentage: int = 5
    preprocessing_num_workers: Optional[int] = None
    keep_linebreaks: bool = True # txt 파일 쓸 때 라인 브레이크 유지할지

    # 모델·토크나이저 관련
    model_name_or_path: Optional[str] = None
    model_type: Optional[str] = None
    config_overrides: Optional[str] = None
    config_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    use_fast_tokenizer: bool = True
    model_revision: str = "main"
    trust_remote_code: bool = True
    token: Optional[str] = None
    torch_dtype: Optional[str] = "bfloat16"

    # 로그
    wandb_project: str = "clm"
    wandb_run_name: str = "run1"
    wandb_entity: Optional[str] = "ailab"
    wandb_host: Optional[str] = "http://wandb.artfacestudio.com"
    wandb_key: Optional[str] = None

def main():
    parser = HfArgumentParser((CustomArguments, TrainingArguments))
    # 그냥 json 파일 하나만 전달하는것도 처리
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        custom_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        custom_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # 
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    

    if training_args.local_process_index == 0:
        wandb.login(key=custom_args.wandb_key, host=custom_args.wandb_host)
        wandb.init(
            project=custom_args.wandb_project,
            name=custom_args.wandb_run_name,
            config={**vars(custom_args), **training_args.to_dict()}
        )
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        # 아웃풋 폴더에 체크포인트가 있는지 확인하고 마지막 체크포인트 얻기
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        # 체크포인트는 있지만 리쥼하는게 아님
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 허깅 페이스에서 받는 데이터셋 이름 잇는 경우
    if custom_args.dataset_name is not None:
        raw_datasets = load_dataset(
            path=custom_args.dataset_name, #
            name= custom_args.dataset_config_name, # 허깅페이스 상에서 바로 아래 하위폴더? 서브셋?,
            cache_dir=custom_args.cache_dir,
            streaming=custom_args.streaming,
            trust_remote_code=custom_args.trust_remote_code,
        )

        # train만 있는 경우
        # train에서 일부를 잘라서 validation으로 사용
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                path=custom_args.dataset_name, #
                name= custom_args.dataset_config_name,
                split=f"train[:{custom_args.validation_split_percentage}%]",
                cache_dir=custom_args.cache_dir,
                streaming=custom_args.streaming,
                trust_remote_code=custom_args.trust_remote_code,
            )

            raw_datasets["train"] = load_dataset(
                path=custom_args.dataset_name, #
                name= custom_args.dataset_config_name,
                split=f"train[{custom_args.validation_split_percentage}%:]",
                cache_dir=custom_args.cache_dir,
                streaming=custom_args.streaming,
                trust_remote_code=custom_args.trust_remote_code,
            )

    # 허깅페이스 허브에 올라간 데이터셋이 아닐 경우 
    # 직접 로컬 위치를 dict에 넣어서 전달
    else:
        data_files = {}
        dataset_args = {}

        if custom_args.train_file is not None:
            data_files["train"] = custom_args.train_file
        if custom_args.validation_file is not None:
            data_files["validation"] = custom_args.validation_file

        # 확장자
        extension = (
            custom_args.train_file.split(".")[-1] if custom_args.train_file is not None
            else custom_args.validation_file.split(".")[-1]
        )
   
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = custom_args.keep_linebreaks

        # 일단 한번 이렇게 올려놓으면 기존의 datasetdict 쓰듯이 쓸 수 있음
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=custom_args.cache_dir,
            **dataset_args
        )

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{custom_args.validation_split_percentage}%]",
                cache_dir=custom_args.cache_dir,
                **dataset_args
            )

            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{custom_args.validation_split_percentage}%:]",
                cache_dir=custom_args.cache_dir,
                **dataset_args
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": custom_args.cache_dir,
        "revision": custom_args.model_revision,
        "trust_remote_code": custom_args.trust_remote_code,
    }

    config = AutoConfig.from_pretrained(custom_args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = {
        "cache_dir":  custom_args.cache_dir,
        "use_fast": custom_args.use_fast_tokenizer,
        "revision": custom_args.model_revision,
        "trust_remote_code": custom_args.trust_remote_code,
    }

    tokenizer = AutoTokenizer.from_pretrained(custom_args.model_name_or_path, **tokenizer_kwargs)

    torch_dtype = (custom_args.torch_dtype if custom_args.torch_dtype in ["auto", None] else getattr(torch, custom_args.torch_dtype))

    model = AutoModelForCausalLM.from_pretrained(
        custom_args.model_name_or_path,
        from_tf=bool(".ckpt" in custom_args.model_name_or_path),
        config=config,
        cache_dir=custom_args.cache_dir,
        revision=custom_args.model_revision,
        trust_remote_code=custom_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")

    # 만약 모델 임베딩의 vocab과 사이즈가 안맞으면
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        

    # 전처리
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    text_column_name = "text" if "text" in column_names else column_names[0]


    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        # 멀티프로세싱은 스트리밍아닐때만 사용 캐시에서 불러와서
        if not custom_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=custom_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not custom_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names
            )

    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024

    if custom_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024

    else:
        block_size = min(custom_args.block_size, tokenizer.model_max_length)


    def group_texts(examples):
        # 전체 텍스트를 하나로 concat함
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # block size에 맞게 dropout
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        if not custom_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=custom_args.preprocessing_num_workers,
                load_from_cache_file=not custom_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True
            )

    if training_args.do_train:
        train_dataset = lm_datasets["train"]
        if custom_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), custom_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        eval_dataset = lm_datasets["validation"]
        if custom_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), custom_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            # model(..., return_dict=False)의 경우 (logits, past_key_values, ...) 튜플이 리턴
            if isinstance(logits, tuple):
                # 튜플에서 logits만 선택
                logits = logits[0]
            return logits.argmax(dim=-1) # 마지막 vocab 차원에서 가장 높은거 : [B, L, C] -> [B, L]
        
        metric = evaluate.load("accuracy", cache_dir=custom_args.cache_dir)

        def compute_metrics(eval_preds):
            """
            구분	labels (정답)	모델 입력	모델이 예측해야 할 것	preds(argmax 후)
            토큰 0	        <BOS>	    <BOS>	        토큰 1	    — (아직 예측 안 함)
            토큰 1	        토큰 1	    <BOS> 토큰 1	    토큰 2	        토큰 1
            토큰 2	        토큰 2	    <BOS> 토큰 1 토큰 2	    토큰 3	    토큰 2
            토큰 3	        토큰 3	    …	                끝(예측 없음)	토큰 3
            """
            preds, labels = eval_preds # [B, L], 실제 정답 [B, L]
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            # evaluate.load("accuracy")는 1차원 배열받기 때문에 reshpae로 flatten을 시킴
            return metric.compute(predictions=preds, references=labels)
        

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model() # 토크나이저도 같이 저장

        metrics = train_result.metrics

        max_train_samples = (
            custom_args.max_train_samples if custom_args.max_train_samples is not None else len(train_dataset)
        )

        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    # Eval
    if training_args.do_eval:
        metrics = trainer.evaluate()

        max_eval_samples = custom_args.max_eval_samples if custom_args.max_eval_samples is not None else len(eval_dataset)

        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # push
    kwargs = {"finetuned_from": custom_args.model_name_or_path, "tasks": "text-generation"}
    if custom_args.dataset_name is not None:
        kwargs["dataset_tags"] = custom_args.dataset_name
        if custom_args.dataset_config_name is not None:
            kwargs["dataset_args"] = custom_args.dataset_config_name
            kwargs["dataset"] = f"{custom_args.dataset_name} {custom_args.dataset_config_name}"
        else:
            kwargs["dataset"] = custom_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()