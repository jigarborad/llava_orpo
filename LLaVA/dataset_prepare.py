from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import io
import os
import json
import base64

def download_and_prepare_dataset(dataset_name, base_dir='playground/data'):
    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name)

    # Prepare the output directories
    dataset_dir = os.path.join(base_dir, dataset_name.split('/')[-1])
    image_dir = os.path.join(dataset_dir, 'images')
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Prepare the output JSON file
    output_json_path = os.path.join(dataset_dir, f'{dataset_name.split("/")[-1]}_data.json')

    # Convert the dataset to the required format
    converted_data = []
    for split in dataset.keys():
        for idx, item in enumerate(tqdm(dataset[split], desc=f"Processing {split}")):
          if idx<100:
            # Prepare conversation format
            chosen_conv = [
                {"from": "human", "value": item['question']},
                {"from": "gpt", "value": item['chosen']}
            ]
            rejected_conv = [
                {"from": "human", "value": item['question']},
                {"from": "gpt", "value": item['rejected']}
            ]

            # Handle image
            image = item['image']
            if isinstance(image, dict) and 'bytes' in image:
                image = Image.open(io.BytesIO(image['bytes']))
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unexpected image format for item {idx} in {split}")

            # Save image
            image_filename = f"{split}_{idx}.png"
            image_path = os.path.join(image_dir, image_filename)
            image.save(image_path)

            # Store relative path to image
            relative_image_path = os.path.relpath(image_path, dataset_dir)

            converted_item = {
                "chosen": chosen_conv,
                "rejected": rejected_conv,
                "image": relative_image_path
            }
            converted_data.append(converted_item)
          else:
            break

    # Save the converted data as JSON
    with open(output_json_path, 'w') as f:
        json.dump(converted_data, f)

    print(f"Dataset downloaded and prepared. Saved to {output_json_path}")

    return output_json_path

if __name__ == '__main__':
    download_and_prepare_dataset('openbmb/RLAIF-V-Dataset', base_dir='playground/data')