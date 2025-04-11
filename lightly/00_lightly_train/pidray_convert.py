import argparse
import json
import shutil
import os
from tqdm import tqdm


def convert_mmcv_to_ultralytics(mmcv_label, image_width, image_height):
    # Extract values from MMCV format
    x_min, y_min, width, height = map(float, mmcv_label)
    
    # Calculate center coordinates
    x_center = (x_min + width / 2) / image_width
    y_center = (y_min + height / 2) / image_height
    
    # Normalize width and height
    norm_width = width / image_width
    norm_height = height / image_height
    
    # Return in Ultralytics format
    return [x_center, y_center, norm_width, norm_height]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="datasets/pidray")
    parser.add_argument("--output", type=str, default="datasets/pidray_converted")
    args = parser.parse_args()

    mapping = {
        "train": "train",
        "test": "valid"
    }
    for old, new in mapping.items():
        with open(f"{args.input}/annotations/{old}.json") as f:
            data = json.load(f)
        annotations = data["annotations"]
        images = data["images"]
        image_to_id = {image["id"]: image for image in images}
        for annotation in tqdm(annotations, desc=f"Processing {old} annotations"):
            image_id = annotation["image_id"]
            image = image_to_id[image_id]
            height = image["height"]
            width = image["width"]
            ultralytics_box = convert_mmcv_to_ultralytics(annotation["bbox"], width, height)
            label_text = " ".join(map(str, [annotation["category_id"] - 1] + ultralytics_box))
            image_path = f"{args.input}/{old}/{image['file_name']}"
            new_image_path = f"{args.output}/{new}/images/{image['file_name']}"
            label_filename = image["file_name"].replace(".png", ".txt")
            label_path = f"{args.output}/{new}/labels/{label_filename}"
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(label_path, "a") as file:
                print(label_text, file=file)
            os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
            shutil.copy2(image_path, new_image_path)

if __name__ == "__main__":
    main()