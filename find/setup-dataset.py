from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

def load_yolo_dataset(data_dir: str) -> dict:
    """
    Load YOLO v1.1 format dataset from a directory.
    Each image should have a corresponding .txt label file.
    Filters out images without any labels.

    Args:
        data_dir: Directory containing .jpg images and .txt labels

    Returns:
        Dictionary mapping image names to their label content
    """
    data_path = Path(data_dir)
    annotations = {}

    # Get all image files
    image_files = list(data_path.glob("*.jpg"))
    print(f"Found {len(image_files)} total images")

    unlabeled_count = 0
    for image_path in image_files:
        label_path = image_path.with_suffix(".txt")

        # Skip if label file doesn't exist
        if not label_path.exists():
            unlabeled_count += 1
            continue

        # Read label file
        with open(label_path, "r") as f:
            labels = f.read().strip()

        # Skip if label file is empty (no annotations)
        if not labels:
            unlabeled_count += 1
            continue

        annotations[image_path.name] = labels

    print(f"Filtered out {unlabeled_count} unlabeled images")
    print(f"Keeping {len(annotations)} labeled images")

    return annotations


def load_class_names(names_file: str) -> list:
    """
    Load class names from YOLO .names file.

    Args:
        names_file: Path to .names file

    Returns:
        List of class names
    """
    with open(names_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def create_yolo_dataset(
        annotations: dict,
        source_dir: str,
        output_dir: str,
        class_names: list,
        val_split: float = 0.2,
        random_seed: int = 42,
):
    """
    Create YOLO dataset structure with train/val split.

    Args:
        annotations: Dictionary of image names to label content
        source_dir: Directory containing source images and labels
        output_dir: Output directory for YOLO dataset
        class_names: List of class names
        val_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create directories
    train_images_dir = output_path / "images" / "train"
    val_images_dir = output_path / "images" / "val"
    train_labels_dir = output_path / "labels" / "train"
    val_labels_dir = output_path / "labels" / "val"

    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_dir)

    # Get list of images
    image_names = list(annotations.keys())

    # Verify all images exist in source
    existing_images = [name for name in image_names if (source_path / name).exists()]

    if len(existing_images) < len(image_names):
        missing = set(image_names) - set(existing_images)
        print(f"Warning: {len(missing)} images not found in source:")
        for name in missing:
            print(f"  - {name}")

    if not existing_images:
        raise ValueError(f"No images found in {source_dir}")

    # Split into train and validation
    train_names, val_names = train_test_split(
        existing_images,
        test_size=val_split,
        random_state=random_seed,
    )

    print(f"Dataset split: {len(train_names)} train, {len(val_names)} val")

    # Copy images and label files
    for image_name in train_names:
        # Copy image
        src_img_path = source_path / image_name
        dst_img_path = train_images_dir / image_name
        shutil.copy2(src_img_path, dst_img_path)

        # Copy label file
        label_name = Path(image_name).stem + ".txt"
        src_label_path = source_path / label_name
        dst_label_path = train_labels_dir / label_name
        shutil.copy2(src_label_path, dst_label_path)

    for image_name in val_names:
        # Copy image
        src_img_path = source_path / image_name
        dst_img_path = val_images_dir / image_name
        shutil.copy2(src_img_path, dst_img_path)

        # Copy label file
        label_name = Path(image_name).stem + ".txt"
        src_label_path = source_path / label_name
        dst_label_path = val_labels_dir / label_name
        shutil.copy2(src_label_path, dst_label_path)

    # Create data.yaml for YOLO
    # Format class names as Python list
    names_str = str(class_names)

    data_yaml = f"""path: {output_path.absolute()}
train: images/train
val: images/val

nc: {len(class_names)}
names: {names_str}
"""

    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(data_yaml)

    print(f"Dataset created at: {output_path}")
    print(f"Config file: {yaml_path}")

    return yaml_path

if __name__ == "__main__":
    SOURCE_DIR = "yolo/input/apriltags/obj_train_data/extracted_frames"
    NAMES_FILE = "yolo/input/apriltags/obj.names"
    OUTPUT_DIR = "yolo/dataset"
    VAL_SPLIT = 0.2
    RANDOM_SEED = 67

    print("Loading class names...")
    class_names = load_class_names(NAMES_FILE)
    print(f"Classes: {class_names}")

    print("\nLoading YOLO dataset...")
    annotations = load_yolo_dataset(SOURCE_DIR)

    print("\nCreating YOLO dataset structure...")
    data_yaml_path = create_yolo_dataset(
        annotations,
        SOURCE_DIR,
        OUTPUT_DIR,
        class_names,
        val_split=VAL_SPLIT,
        random_seed=RANDOM_SEED,
    )

    # Example:
    # model = YOLO(f"yolov8{model_size}.pt")
    # results = model.train(data=data_yaml)