from pathlib import Path
import cv2
import shutil

# =========================================================
# SETTINGS
# =========================================================

TEST_LIMIT = None


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_IMAGES = PROJECT_ROOT / "data" / "raw" / "DUO" / "images"

SOURCE_ANNOTATIONS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "DUO_coco"
    / "annotations"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "DUO_enhanced"
)

OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_ANNOTATIONS = OUTPUT_ROOT / "annotations"


# =========================================================
# IMAGE ENHANCEMENT
# =========================================================

def enhance_image(image):
    """
    Apply CLAHE contrast enhancement to the luminance channel.

    This improves local contrast while preserving the original
    image dimensions and bounding-box coordinates.
    """

    # Convert from BGR to LAB colour space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Separate luminance and colour channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Create CLAHE enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # Apply CLAHE only to luminance
    enhanced_l = clahe.apply(l_channel)

    # Merge the channels again
    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    # Convert back to normal BGR image format
    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_image


# =========================================================
# PROCESS IMAGE FOLDER
# =========================================================

def process_folder(split):
    """
    Enhance all images within the specified DUO split.
    """

    input_folder = SOURCE_IMAGES / split
    output_folder = OUTPUT_IMAGES / split

    # Create destination folder if it does not exist
    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find all JPG images
    image_files = sorted(
        input_folder.glob("*.jpg")
    )

    # Apply testing limit if enabled
    if TEST_LIMIT is not None:
        image_files = image_files[:TEST_LIMIT]

    print(
        f"\nProcessing {split}: "
        f"{len(image_files)} images"
    )

    for i, image_path in enumerate(
        image_files,
        start=1
    ):

        # Load image
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"Warning: could not read "
                f"{image_path}"
            )
            continue

        # Enhance image
        enhanced_image = enhance_image(
            image
        )

        # Save using same filename
        output_path = (
            output_folder
            / image_path.name
        )

        success = cv2.imwrite(
            str(output_path),
            enhanced_image
        )

        if not success:
            print(
                f"Warning: could not save "
                f"{output_path}"
            )

        # Progress message
        if (
            i % 500 == 0
            or i == len(image_files)
        ):
            print(
                f"{split}: processed "
                f"{i}/{len(image_files)}"
            )

    print(
        f"Finished processing {split}."
    )


# =========================================================
# COPY ANNOTATIONS
# =========================================================

def copy_annotations():
    """
    Copy the existing COCO annotation files.

    Image enhancement does not alter image dimensions,
    so bounding-box annotations remain unchanged.
    """

    OUTPUT_ANNOTATIONS.mkdir(
        parents=True,
        exist_ok=True
    )

    annotation_files = list(
        SOURCE_ANNOTATIONS.glob("*.json")
    )

    for annotation_file in annotation_files:

        destination = (
            OUTPUT_ANNOTATIONS
            / annotation_file.name
        )

        shutil.copy2(
            annotation_file,
            destination
        )

    print(
        f"\nCopied {len(annotation_files)} "
        f"annotation files."
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "Starting DUO image enhancement..."
    )

    print(
        f"Source images: "
        f"{SOURCE_IMAGES}"
    )

    print(
        f"Output location: "
        f"{OUTPUT_ROOT}"
    )

    if TEST_LIMIT is not None:
        print(
            f"\nTEST MODE: processing only "
            f"{TEST_LIMIT} images per split."
        )
    else:
        print(
            "\nFULL DATASET MODE"
        )

    # Enhance train images
    process_folder("train")

    # Enhance official test images
    process_folder("test")

    # Copy original COCO annotations
    copy_annotations()

    print(
        "\nEnhancement complete."
    )

    print(
        f"Enhanced dataset saved to:\n"
        f"{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()