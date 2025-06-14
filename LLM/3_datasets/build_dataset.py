# 데이터 전처리하고 허브에 푸쉬하기 (멀티프로세싱)

import torch
import os
import json
import datasets
from datasets import Dataset, load_dataset
from facenet_pytorch import MTCNN
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from transformers import AutoTokenizer
import argparse
import concurrent.futures


def load_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            caption = json_data.get('caption', '')
            image_path = os.path.join(os.path.dirname(json_path), os.path.basename(json_path).replace('.json', '.jpg'))
            return {'image_path': image_path, 'caption': caption}
    except json.JSONDecodeError:
        print(f"Error decoding JSON file: {json_path}")
    except UnicodeDecodeError:
        print(f"Error reading file due to Unicode decode error: {json_path}")
    return None


class FaceProcessor:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=512, margin=10, device='cuda', post_process=False)
        self.to_pil = transforms.ToPILImage()

    def __call__(self, examples):
        image = examples['image']
        img_cropped = self.mtcnn(image)
        if img_cropped is not None:
            examples['face_image'] = self.to_pil(img_cropped.to(torch.uint8))
        else:
            width, height = image.size

            # Calculate the coordinates for a center crop
            crop_size = 300
            left = (width - crop_size) / 2
            top = (height - crop_size) / 2
            right = left + crop_size  # Corrected
            bottom = top + crop_size  # Corrected

            # Crop and resize the image
            image_cropped = image.crop((left, top, right, bottom))
            image_resized = image_cropped.resize((512, 512))

            examples['face_image'] = image_resized

        print('examples', examples)
        examples['face_image'].save('/purestorage/project/tyk/project24/face_dataset/test.png')
        return examples




if __name__ == "__main__":
    print("Starting dataset processing...")
    parser = argparse.ArgumentParser(description="Process and upload dataset to Hugging Face Hub")
    parser.add_argument('--root_path', type=str, default='/ffhq_wild_files' help='Root path of the dataset')
    paser.add_argument('--hub_token', type=str, required=True, help='Hugging Face Hub token for authentication')
    parser.add_argument('hub_repo', type=str, default='ty-kim/myface', help='Hugging Face Hub repository to push the dataset')
    parser.add_argument('--num_processors', type=int, default=16, help='Number of processors to use for parallel processing')
    args = parser.parse_args()


    all_items = os.listdir(args.root_path)
    subdirectories = [os.path.join(args.root_path, item) for item in all_items if os.path.isdir(os.path.join(args.root_path, item))]

    data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_processors) as executor:
        future_to_json = {executor.submit(load_json, os.path.join(args.root_path, folder, file_name)): file_name
                        for folder in subdirectories
                        for file_name in os.listdir(os.path.join(args.root_path, folder))
                        if file_name.endswith('.json')}
        for future in concurrent.futures.as_completed(future_to_json):
            result = future.result()
            if result:
                data.append(result)


    processor = FaceProcessor()


    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict({'image': [item['image_path'] for item in data], 'caption': [item['caption'] for item in data]}).cast_column('image', datasets.Image())


    dataset = dataset.map(processor)

    print(dataset[0])
    dataset.push_to_hub(args.hub_repo, token=args.hub_token, private=True)
