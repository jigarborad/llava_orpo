from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import io
import os
import json

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
            converted_item = {
                'ds_name': dataset_name,
                'question': item['question'],
                'chosen': item['chosen'],
                'rejected': item['rejected'],
                'origin_dataset': item.get('origin_dataset', ''),
                'origin_split': item.get('origin_split', ''),
                'idx': idx
            }

            # Save the image
            image = item['image']
            if isinstance(image, dict) and 'bytes' in image:
                image = Image.open(io.BytesIO(image['bytes']))
            elif isinstance(image, Image.Image):
                image = image
            else:
                raise ValueError(f"Unexpected image format for item {idx} in {split}")

            image_filename = f"{split}_{idx}.png"
            image_path = os.path.join(image_dir, image_filename)
            image.save(image_path)

            # Update the image field to contain the relative path to the image
            converted_item['image'] = os.path.relpath(image_path, start=dataset_dir)

            converted_data.append(converted_item)

    # Save the converted data as JSON
    with open(output_json_path, 'w') as f:
        json.dump(converted_data, f)

    print(f"Dataset downloaded and prepared. Saved to {output_json_path}")
    print(f"Images saved to {image_dir}")
    
    return output_json_path, image_dir


if __name__ == '__main__':
    download_and_prepare_dataset('openbmb/RLAIF-V-Dataset', base_dir='playground/data')