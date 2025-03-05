import tarfile
import json
import os
import pickle
import argparse

from tqdm import tqdm


def extract_captions_and_save_json(tar_path, output_dir, i):
    captions = {}

    # Extract the base name of the tar file (without extension)
    tar_name = os.path.basename(tar_path).split("/")[-1].split(".")[0]

    # Create a directory with the same name as the tar file
    os.makedirs(output_dir, exist_ok=True)

    # Open the tar file
    with tarfile.open(tar_path, "r:*") as tar:
        # Iterate through each member in the tar file
        for member in tqdm(tar.getmembers(), desc=f"Processing tar file: {tar_path}"):
            # Check if the file is a .pyd file
            if member.name.endswith(".pyd"):
                # Extract the file's content
                with tar.extractfile(member) as f:
                    binary_data = f.read()
                    try:
                        # Attempt to deserialize the data using pickle
                        content = pickle.loads(binary_data)
                        caption = content["caption"]

                        # Use the filename (without extension) as the key
                        filename = f"{i}_{os.path.basename(member.name).split('.')[0]}"
                        captions[filename] = caption
                    except pickle.PickleError as e:
                        print(f"Failed to deserialize {member.name}: {e}")

    # Save the captions to a JSON file in the created directory
    json_path = os.path.join(output_dir, f"{tar_name}_captions.json")
    with open(json_path, "w") as json_file:
        json.dump(captions, json_file, indent=4)

    print(f"Captions saved to {json_path}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Extract captions from .pyd files in a tar archive."
    )
    parser.add_argument("--i", type=int, required=True)

    # Parse the arguments
    args = parser.parse_args()

    args.tar_path = f"/BS/databases23/CC3M_tar/training/{args.i}.tar"
    args.output_dir = f"/BS/dduka/work/projects/TempNet/Bimodal_CL/cc3m_extracted/{args.i}"

    print(f"Processing tar: {args.tar_path}")
    # Call the function with the provided arguments
    extract_captions_and_save_json(args.tar_path, args.output_dir, args.i)


if __name__ == "__main__":
    main()
