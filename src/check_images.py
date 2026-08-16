from pathlib import Path
import cv2

image_folder = Path("data/test_images")

image_paths = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpeg"))

print(f"Number of images found: {len(image_paths)}")

for image_path in image_paths:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    height, width, channels = image.shape
    print(f"{image_path.name}: width={width}, height={height}, channels={channels}")