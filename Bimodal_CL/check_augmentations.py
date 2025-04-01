# The purpose of this script is to visualize the image augmentations applied to the dataset.
import argparse
import os
from dataset.randaugment import RandomAugment
from torchvision import transforms
from PIL import Image

OUTPUT_DIR = "/BS/dduka/work/projects/TempNet/Bimodal_CL/images/augmentations/"

normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            256, scale=(0.5, 1.0), interpolation=Image.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        RandomAugment(
            2,
            7,
            isPIL=True,
            augs=[
                "Identity",
                "AutoContrast",
                "Equalize",
                "Brightness",
                "Sharpness",
                "ShearX",
                "ShearY",
                "TranslateX",
                "TranslateY",
                "Rotate",
            ],
        ),
        transforms.ToTensor(),
        normalize,
    ]
)


def save_augmented_images(image_path, output_dir):
    """
    Loads an image, applies augmentations twice, and saves the original and augmented images.
    Args:
        image_path (str): Path to the image.
        output_dir (str): Path to the directory where images will be saved.
    """
    # Load the image
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at '{image_path}'")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the original image
    original_filename = (
        os.path.splitext(os.path.basename(image_path))[0] + "_original.jpg"
    )
    original_save_path = os.path.join(output_dir, original_filename)
    image.save(original_save_path)
    print(f"Saved original image to: {original_save_path}")

    # Apply the augmentations twice and save
    for i in range(2):
        augmented_image_tensor = train_transform(image)
        augmented_image_pil = transforms.ToPILImage()(augmented_image_tensor)
        augmented_filename = (
            os.path.splitext(os.path.basename(image_path))[0] + f"_augmented_{i+1}.jpg"
        )
        augmented_save_path = os.path.join(output_dir, augmented_filename)
        augmented_image_pil.save(augmented_save_path)
        print(f"Saved augmented image {i+1} to: {augmented_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize and save image augmentations."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Path to the directory where augmented images will be saved.",
    )
    args = parser.parse_args()

    save_augmented_images(args.image_path, args.output_dir)
