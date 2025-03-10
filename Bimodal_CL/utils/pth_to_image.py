import torch
import PIL.Image as Image
import argparse


def decoder_pth(path, save_path):
    image = torch.load(path, map_location="cpu")
    image = Image.fromarray(image.numpy()).convert("RGB")
    image.save(save_path)


def main():
    parser = argparse.ArgumentParser(description="Convert PTH image files to PNG")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input PTH file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save output PNG file"
    )

    args = parser.parse_args()

    decoder_pth(args.input, args.output)


if __name__ == "__main__":
    main()
