# https://medium.com/@lucamassaron/fine-tuning-gemma-3-1b-it-for-financial-sentiment-analysis-a-step-by-step-guide-1a025d2fc75d
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import torch
import torch.nn as nn

import transformers
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments, # Note: SFTConfig from TRL is used later
                          pipeline,
                          logging)


# Explicitly import Gemma3ForCausalLM
from transformers.models.gemma3 import Gemma3ForCausalLM

from datasets import Dataset
from peft import LoraConfig, PeftConfig, PeftModel
from trl import SFTTrainer, SFTConfig # Use SFTConfig from TRL
import bitsandbytes as bnb

from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)

from sklearn.model_selection import train_test_split


def default_device_map():
    if torch.cuda.is_available():
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        return {"": local_rank} 
    else:
        return {"": "cpu"}   


def evaluate(y_true, y_pred):
    # Define sentiment label mapping to numeric for scikit-learn metrics
    label_mapping = {'positive':2, 'neutral':1, 'negative':0}

    # Handle 'none' predictions (map to neutral, or handle as error if preferred)
    y_true_num = np.array([label_mapping.get(label, 1) for label in y_true])
    y_pred_num = np.array([label_mapping.get(label, 1) for label in y_pred])

    # Calculate overall accuracy
    accuracy = accuracy_score(y_true_num, y_pred_num)
    print(f'Overall Accuracy: {accuracy:.3f}')

    # Compute accuracy for each sentiment label
    unique_labels = np.unique(y_true_num) # Get unique numeric labels

    # Map numeric back to string for printing
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    for label_num in unique_labels:
        label_mask = y_true_num == label_num # Mask for current class
        label_accuracy = accuracy_score(y_true_num[label_mask], y_pred_num[label_mask])
        print(f'Accuracy for label {label_num} ({reverse_label_mapping.get(label_num, "unknown")}): {label_accuracy:.3f}')

    # Generate classification report using string labels for clarity
    class_report = classification_report(y_true, y_pred, labels=["negative", "neutral", "positive"], zero_division=0)
    print('\nClassification Report:\n', class_report)

    # Compute and display confusion matrix (using numeric labels)
    # Ensure labels are ordered correctly: negative(0), neutral(1), positive(2)
    conf_matrix = confusion_matrix(y_true_num, y_pred_num, labels=[0, 1, 2])
    print('\nConfusion Matrix (Rows: True, Cols: Pred) [Neg, Neu, Pos]:\n', conf_matrix)

def predict(X_test_df, model_to_use, tokenizer_to_use, device_to_use=device, max_new_tokens=5, temperature=0.0):
    """
    Predict the sentiment of news headlines using the provided model and tokenizer.
    """

    y_pred = [] # List to store predicted sentiment labels
    model_to_use.eval() # Set model to evaluation mode

    # Iterate through each headline in the test DataFrame
    for i in tqdm(range(len(X_test_df)), desc="Predicting Sentiments"):
        prompt = X_test_df.iloc[i]["text"] # Extract the prompt text

        # Tokenize the prompt and move tensors to the correct device
        input_ids = tokenizer_to_use(prompt, return_tensors="pt").to(device_to_use)

        # Generate output from the model
        with torch.no_grad(): # Disable gradient calculations for inference
             outputs = model_to_use.generate(**input_ids,
                                      max_new_tokens=max_new_tokens,
                                      temperature=temperature,
                                      pad_token_id=tokenizer_to_use.eos_token_id # Avoid warning
                                     )

        # Decode the generated tokens (excluding the input prompt)
        # Find the start of the generated part by looking after the prompt structure
        prompt_end_marker = "]= "
        full_decoded_text = tokenizer_to_use.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part after the prompt marker
        try:
            generated_text = full_decoded_text.split(prompt_end_marker)[1].strip().lower()
        except IndexError:
            generated_text = "" # Handle cases where the marker isn't found

        # Extract the first predicted sentiment label
        if "positive" in generated_text:
            y_pred.append("positive")
        elif "negative" in generated_text:
            y_pred.append("negative")
        elif "neutral" in generated_text:
            y_pred.append("neutral")
        else:
            # Fallback if no clear label is found in the short generation
            y_pred.append("none")
            # print(f"Warning: Could not parse sentiment from: '{generated_text}' derived from '{full_decoded_text}'")

    return y_pred

@dataclass
class MySFTConfig(SFTConfig):
    output_dir: str = field(
        default="logs",
        metadata={"help": "Directory to save logs and checkpoints"},
    )
    num_train_epochs: int = field(
        default=4,
        metadata={"help": "Number of training epochs"},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size per GPU (keep small for large models/limited VRAM)"
        },
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={
            "help": "Accumulate gradients over N steps (effective batch = batch_size*N)"
        },
    )
    optim: str = field(
        default="adamw_torch_fused",
        metadata={"help": "Use fused AdamW optimizer (efficient)"},
    )
    save_steps: int = field(
        default=112,
        metadata={"help": "Save a checkpoint every N steps"},
    )
    logging_steps: int = field(
        default=25,
        metadata={"help": "Log training metrics every N steps"},
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "Learning rate"},
    )
    weight_decay: float = field(
        default=1e-3,
        metadata={"help": "Weight decay for regularization"},
    )
    fp16: bool = field(
        default=(compute_dtype == torch.float16),
        metadata={"help": "Enable mixed-precision FP16 if available"},
    )
    bf16: bool = field(
        default=(compute_dtype == torch.bfloat16),
        metadata={"help": "Enable mixed-precision BF16 if available"},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={"help": "Gradient clipping threshold"},
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "-1 means use num_train_epochs"},
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Proportion of training steps for LR warm-up"},
    )
    group_by_length: bool = field(
        default=False,
        metadata={"help": "Don't group sequences by length (can sometimes speed up)"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning-rate scheduler type"},
    )
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "Report metrics to TensorBoard"},
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={
            "help": "Evaluate during training at specified step intervals"
        },
    )
    eval_steps: int = field(
        default=112,
        metadata={"help": "Evaluate every N steps"},
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Load the best model checkpoint at the end of training"},
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing to save memory"},
    )
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False},
        metadata={
            "help": "Recommended setting for newer PyTorch versions (no re-entrant)"
        },
    )

    # --------------- SFT 전용 필드 --------------------------
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Name of the text field in the dataset"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"},
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Pack multiple sequences into one input (False = off)"},
    )
    dataset_kwargs: dict = field(
        default_factory=lambda: {
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        metadata={"help": "Arguments for dataset processing"},
    )

if __name__ == '__main__':

    parser = transformers.HfArgumentParser(MySFTConfig)
    (cfg,) = parser.parse_args_into_dataclasses()

    device = default_device_map()
    # 언어 모델만
    model = Gemma3ForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        attn_implementation="eager",
        low_cpu_mem_usage=True,      # Reduces CPU RAM usage during loading
        device_map=device, 
    )

    # Define maximum sequence length for the tokenizer
    max_seq_length = 8192 # Gemma 3 supports long contexts

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3-1b-it",
        max_seq_length=max_seq_length,
        device_map=device
    )

    EOS_TOKEN = tokenizer.eos_token

    # FinancialPhraseBank Dataset 
    # approximately 5,000 sentences.
    filename = "v/all-data.csv"
    df = pd.read_csv(
        filename,
        names=["sentiment", "text"],
        encoding="utf-8", 
        encoding_errors="replace"
    )

    X_train, X_test = [], []
    y_true_list = [] # To store true labels for the test set separately


    # Stratified train-test split (300 per sentiment)
    for sentiment in ["positive", "neutral", "negative"]:
        # Split data for the current sentiment
        train, test = train_test_split(df[df.sentiment == sentiment],
                                   train_size=300,
                                   test_size=300,
                                   random_state=42,
                                   # Stratify within the sentiment group (though less critical here)
                                   stratify=df[df.sentiment == sentiment]["sentiment"])

        X_train.append(train)
        X_test.append(test)

    # Combine splits from all sentiments
    X_train = pd.concat(X_train).sample(frac=1, random_state=10).reset_index(drop=True)
    X_test_full = pd.concat(X_test).reset_index(drop=True) # Keep full test data temporarily


    # Extract true labels before creating prompts
    y_true = X_test_full["sentiment"]

    # Prepare the test set text data (without labels)
    X_test = X_test_full[['text']] # Keep only the text column for test prompt generation

    # -- Prepare Evaluation Data --
    # Identify indices used in train or test sets
    train_indices = set(X_train.index)
    test_indices = set(X_test_full.index) # Use index from the full test set before dropping columns
    selected_indices = train_indices | test_indices

    # Create evaluation set from data not in train or test
    X_eval = df.loc[~df.index.isin(selected_indices)].copy()

    # Resample evaluation data for balance (50 per class)
    # Use 'replace=True' allows sampling with replacement if a class has < 50 samples
    X_eval = X_eval.groupby('sentiment', group_keys=False).apply(
        lambda x: x.sample(n=50, random_state=10, replace=True)
    ).reset_index(drop=True)

    # -- Prompt Generation Functions --

    # Function to generate training prompts (with label)
    def generate_train_prompt(data_point):
        return f"""
        Analyze the sentiment of the news headline enclosed in square brackets.
        Determine if it is positive, neutral, or negative, and return the corresponding sentiment label: "positive", "neutral", or "negative".

        [{data_point["text"]}] = {data_point["sentiment"]}
        """.strip() + EOS_TOKEN # Add EOS token

    # Function to generate test prompts (without label)
    def generate_test_prompt(data_point):
        return f"""
        Analyze the sentiment of the news headline enclosed in square brackets.
        Determine if it is positive, neutral, or negative, and return the corresponding sentiment label: "positive", "neutral", or "negative".

        [{data_point["text"]}] = """.strip() # No label or EOS token needed here for generation


    # -- Apply Prompts and Convert to Dataset --

    # Apply prompt generation to create the final text column for training and evaluation
    X_train = pd.DataFrame(X_train.apply(generate_train_prompt, axis=1), columns=["text"])
    X_eval = pd.DataFrame(X_eval.apply(generate_train_prompt, axis=1), columns=["text"])

    # Apply prompt generation for the test set
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])

    # Convert pandas DataFrames to Hugging Face Dataset objects
    train_data = Dataset.from_pandas(X_train)
    eval_data = Dataset.from_pandas(X_eval)
    # Note: X_test remains a DataFrame for the predict function, y_true holds labels


    # Generate predictions using the base model
    y_pred_base = predict(X_test, model, tokenizer)

    # Evaluate the baseline predictions
    print("--- Baseline Model Evaluation ---")
    evaluate(y_true, y_pred_base)


    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=-0.05,
        r=64,
        bias="none", # Whether to train bias parameters ('none', 'all', or 'lora_only')
        task_type="CAUSAL_LM",
        target_modules="all-linear" # Apply LoRA to all linear layers
    )
    
    # Disable caching for training, re-enable for inference later
    # 왜 cache를 끄지?
    model.config.use_cache = False

    # Set pretraining_tp if relevant for distributed training (usually 1 for single GPU)
    # 이건 왜 
    model.config.pretraining_tp = 1

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=cfg,
        # processing_class=tokenizer -> Not needed if tokenizer passed directly
    )

    train_result = trainer.train()

    print("Training Metrics:", metrics)

    # Define directory to save LoRA adapter and tokenizer
    lora_directory = "LoRA-Gemma3-1B-Financial-Sentiment"

    # Save the LoRA adapter weights
    trainer.model.save_pretrained(lora_directory)
    print(f"LoRA adapter saved to {lora_directory}")

    # Save the tokenizer associated with the training
    trainer.tokenizer.save_pretrained(lora_directory)
    print(f"Tokenizer saved to {lora_directory}")


    # Generate predictions using the fine-tuned model from the trainer
    print("Predicting with fine-tuned model...")
    y_pred_tuned = predict(X_test, trainer.model, tokenizer)

    # Evaluate the fine-tuned predictions
    print("\n--- Fine-Tuned Model Evaluation ---")
    evaluate(y_true, y_pred_tuned)