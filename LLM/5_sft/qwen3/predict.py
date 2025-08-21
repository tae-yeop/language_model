
if __name__ == '__main__':
    question = dataset[10]['Question']
    inputs = tokenizer(
        [inference_prompt_style.format(question) + tokenizer.eos_token],
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(response[0].split("### Response:")[1])


    question = dataset[100]['Question']
    inputs = tokenizer(
        [inference_prompt_style.format(question) + tokenizer.eos_token],
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(response[0].split("### Response:")[1])