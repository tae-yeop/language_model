import evaluate, torch, math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator

ckpt_dir = '/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/LLM/4_pretraining/clm/hugging_clm/out'

device = "cuda"
batch_size = 8  

tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
model     = AutoModelForCausalLM.from_pretrained(ckpt_dir).to("cuda").eval()

# 같은 validation split 다시 로드
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

# 토큰화 -> padding
def tok(ex):
    return tokenizer(ex['text'], truncation=True, return_attention_mask=False)

ds = ds.map(tok, batched=True, remove_columns=["text"])

ds.set_format(type="torch", columns=["input_ids"])

ppl_metric = evaluate.load("perplexity")

for i in range(0, len(ds), batch_size):
    batch = ds[i : i + batch_size]
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss

    ppl_metric.add_batch(predictions=[loss.item()], references=[0])  # dummy ref

print("Perplexity:", math.exp(ppl_metric.compute()["mean_perplexity"]))