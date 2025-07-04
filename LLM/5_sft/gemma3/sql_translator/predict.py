import torch
from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process dataset for PaLiGemma')
    parser.add_argument('--hf_token', type=str, required=True, help='Hugging Face access token')
    args = parser.parse_args()
    
    # model_id = "gemma-text-to-sql"
    model_id = "/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/LLM/5_sft/gemma3/sql_translator/merged_model"
    # Load Model with PEFT adapter
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    from random import randint
    import re

    # Load the model and tokenizer into the pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


    dataset = load_dataset("ty-kim/sql_translator", token=args.hf_token, split="test")

    # Load a random sample from the test dataset
    rand_idx = randint(0, len(dataset))
    test_sample = dataset[rand_idx]


    print('message', test_sample["messages"][:1]) # 여기까지가 user의 content dict

    print('---------------------------')

    print('message', test_sample["messages"])

    print('---------------------------')
    # Convert as test example into a prompt with the Gemma template
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:1], 
                                                tokenize=False, 
                                                add_generation_prompt=True)
    

    print('prompt', prompt)

    print('---------------------------')

    # Generate our SQL query.
    outputs = pipe(prompt, 
                   max_new_tokens=256, 
                   do_sample=False, 
                   temperature=0.1, 
                   top_k=50, top_p=0.1, 
                   eos_token_id=tokenizer.eos_token_id, 
                   disable_compile=True)

    # Extract the user query and original answer
    print(f"Context:\n", re.search(r'<SCHEMA>\n(.*?)\n</SCHEMA>', test_sample['messages'][0]['content'], re.DOTALL).group(1).strip())
    print(f"Query:\n", re.search(r'<USER_QUERY>\n(.*?)\n</USER_QUERY>', test_sample['messages'][0]['content'], re.DOTALL).group(1).strip())
    print(f"Original Answer:\n{test_sample['messages'][1]['content']}")
    print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")