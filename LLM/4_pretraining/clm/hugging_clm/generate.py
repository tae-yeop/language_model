from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

ckpt_dir = "/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/LLM/4_pretraining/clm/hugging_clm/out"


tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
model     = AutoModelForCausalLM.from_pretrained(ckpt_dir).to("cuda")  # GPU 사용


generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,                       # CPU라면 생략
)

prompts = [
    "Deep learning is transforming",
    "Once upon a time in Seoul,",
]

for p in prompts:
    out = generator(p, max_new_tokens=80, do_sample=True, top_p=0.9)[0]["generated_text"]
    print("=" * 80)
    print(out)
    