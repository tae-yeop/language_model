from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

from PIL import Image
import requests

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.to(device)

    # 모델마다 prompt template이 다르다 => 성능 영향이 있음
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

    inputs = processor(prompt, image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100)

    print(processor.decode(output[0], skip_special_tokens=True))
    