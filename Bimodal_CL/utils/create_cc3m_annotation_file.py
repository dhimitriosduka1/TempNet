# This script reads a directory of images and captions and creates a json file with the following format:
# {
#     "image_id": "image_id",
#     "caption": "caption"
#     "image": "image_path"
# }

# Stored image format: sample1276285_3534586286.image.pth
# Stored caption format: sample1276285_3534586286.metadata.pyd

import os
import json
import argparse


def create_cc3m_annotation_file(root_dir, output_file):
    # Get all image files in the directory
    image_files = [f for f in os.listdir(root_dir) if f.endswith((".image.pth"))]

    print(f"Found {len(image_files)} image files")

    # Create a list to store the annotations
    annotations = []

    # Process each image file
    for image_file in image_files:
        # Get the image ID from the file name
        image_id = image_file.split(".")[0]

        annotation = {
            "image_id": image_id,
            "image": image_file,
            "caption": "{0}{1}".format(image_id, ".metadata.pyd"),
        }

        annotations.append(annotation)

    # Write the annotations to a json file
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=4)

    print(f"Saved {len(annotations)} annotations to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    root_dir = args.root_dir
    output_file = args.output_file

    create_cc3m_annotation_file(root_dir, output_file)
